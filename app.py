import os
import json
import boto3
import pandas as pd
from dotenv import load_dotenv
from botocore.config import Config
from botocore.exceptions import ClientError
from langchain_community.document_loaders import AmazonTextractPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_aws import ChatBedrock
from metaprompt import json_template
import streamlit as st
import uuid

# Part 1 (Setup)
# Load environment variables (AWS credentials)
load_dotenv()

# Configuring Boto3 for retries
retry_config = Config(
    region_name=os.environ.get("AWS_DEFAULT_REGION"),
    retries={
        'max_attempts': 10,
        'mode': 'standard'
    }
)

# Create a boto3 session for accessing Bedrock and Textract
session = boto3.Session()
client = session.client(service_name='bedrock-runtime',
                        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
                        aws_secret_access_key=os.environ.get(
                            "AWS_SECRET_ACCESS_KEY"),
                        config=retry_config)
textract = session.client('textract', config=retry_config)

model = ChatBedrock(client=client, model_id=os.environ.get("BEDROCK_MODEL"))


def upload_document(uploaded_file):
    """Save the uploaded document to the 'uploaded_files' folder and return its file path."""
    if uploaded_file is not None:
        try:
            # Create the folder if it doesn't exist
            upload_folder = "./uploaded_files"
            if not os.path.exists(upload_folder):
                os.makedirs(upload_folder)

            # Get the original file name and path
            file_path = os.path.join(upload_folder, uploaded_file.name)

            # If the file already exists, rename it with a unique ID
            if os.path.exists(file_path):
                file_name, file_extension = os.path.splitext(
                    uploaded_file.name)
                unique_id = str(uuid.uuid4())[:8]  # Generate a short unique ID
                file_path = os.path.join(
                    upload_folder, f"{file_name}_{unique_id}{file_extension}")

            # Write the file to the folder
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            return file_path
        except Exception as e:
            st.error(f"Error uploading file: {e}")
            return None
    return None


# Function to process the document (Extract text using Textract and split it)
def process_document(file_path, file_type):
    """Extract text from the document (TIFF, PDF, JPG, JPEG) and split it into manageable chunks."""

    # Extract text from TIFF or image (JPG, JPEG)
    def extract_text_from_image(file_path):
        try:
            with open(file_path, 'rb') as document:
                response = textract.detect_document_text(
                    Document={'Bytes': document.read()})

            text = ""
            for item in response["Blocks"]:
                if item["BlockType"] == "LINE":
                    text += item["Text"] + "\n"
            return text
        except ClientError as e:
            st.error(
                f"Amazon Textract error: {e.response['Error']['Message']}")
            return None
        except Exception as e:
            st.error(f"Error extracting text: {e}")
            return None

    # Extract text from PDF using Textract's `analyze_document` API
    def extract_text_from_pdf(file_path):
        try:
            loader = AmazonTextractPDFLoader(file_path)
            response = loader.load()
            response = response[0].page_content
            return response
        except ClientError as e:
            st.error(
                f"Amazon Textract error: {e.response['Error']['Message']}")
            return None
        except Exception as e:
            st.error(f"Error extracting text from PDF: {e}")
            return None

    # Process the file based on its type
    if file_path and file_type:
        if file_type in ["jpg", "jpeg", "tiff"]:
            extracted_text = extract_text_from_image(file_path)
        elif file_type == "pdf":
            extracted_text = extract_text_from_pdf(file_path)
        else:
            st.error("Unsupported file format.")
            return []

        # If text was successfully extracted, split it into chunks
        if extracted_text:
            try:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=4000,
                    separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
                    chunk_overlap=0
                )
                texts = text_splitter.split_text(extracted_text)
                return texts
            except Exception as e:
                st.error(f"Error splitting text into chunks: {e}")
                return []
        else:
            st.error("Failed to extract text from the document.")
            return []
    else:
        st.error("Invalid file path or file type.")
        return []


def generate_output(document_data):
    def get_llm_response(model, pdf_data, json_template, processed_data=""):
        prompt = PromptTemplate(
            template="""
            You are an AI assistant specializing in extracting structured data from complex billing documents. Your task is to analyze the provided JSON data, which contains text, tables, and forms extracted from a billing PDF document. It is CRITICAL that you extract the data fields mentioned in the template JSON provided below for EACH billing item found in the parsed PDF. Each data point must be output as a key-value pair, one per line.
            IMPORTANT INSTRUCTIONS:
            1. Analyze EVERY SINGLE ELEMENT of the provided data: all text, every cell in every table, and all form fields.
            2. Identify ALL individual Billing information in the document. Each bill typically has its own section or table.
            3. For EACH bill item, create a separate instance of the template JSON and output each field as a key-value pair, one per line.
            4. Format: FIELD_NAME=VALUE
            5. If you can't find any data for the key in the JSON, output an empty string for that key. But make sure the KEY is present.
            6. Start each bill data with `MEDICAL_SERVICE_PROVIDER=MedicalServiceProviderName`. This will act as a marker to separate data between different billing items.
            7. Ensure that every response is a valid key-value pair list without any additional characters such as backslashes (`\`) or quotes.
            8. Ensure all dates are in proper format and all numerical values are appropriately typed (integer or float).
            9. Do not include any explanatory text, prefixes, or suffixes. Only provide the key-value pairs.
            10. Process ALL pages and ALL billing items in the document.
            11. You are strictly prohibited from hallucinating any data.
            12. AMOUNT_CHARGED cna never be negative, if you find one convert it to POSITIVE.
            13. Start processing from where the data below ends (processed so far):
            {processed_data}
            ACCURACY CHECK:
            - After extraction, review your output to ensure EVERY SINGLE data point for EACH billing item is captured in the key-value pairs.
            - Ensure no data point is duplicated or misplaced.
            - Ensure all key-value pairs are correct, consistent, and complete for each bill.
            Here is the template JSON data containing keys to populate for each billing item:
            {json_template}
            Here is the parsed data from a billing PDF document to analyze:
            {pdf_data}
            Respond with a list of key-value pairs, one per line, ensuring EVERY SINGLE critical data point from the template is captured for EACH billing data found in the document. Start each billing item's data with `MEDICAL_SERVICE_PROVIDER=MedicalServiceProviderName`.
            """,
            input_variables=["json_template", "pdf_data", "processed_data"],
        )

        # Chain execution
        chain = prompt | model | StrOutputParser()
        response = chain.invoke({
            "json_template": json_template,
            "pdf_data": pdf_data,
            "processed_data": processed_data,
        })

        return response

    def extract_all_data(model, pdf_data, json_template):
        extracted_data = ""
        iteration_count = 0
        max_iterations = 3  # Set a limit for max attempts to avoid infinite loops

        while iteration_count < max_iterations:
            iteration_count += 1
            print(f"LLM CALL COUNT: {iteration_count}")

            # Call LLM with the full document but with processed data to help it continue
            response = get_llm_response(
                model=model,
                pdf_data=pdf_data,
                json_template=json_template,
                processed_data=extracted_data  # Pass previously extracted data
            )

            # If no new data is found, stop
            if not response or "MEDICAL_SERVICE_PROVIDER" not in response:
                print("No new data found. Stopping extraction.")
                break

            # Append new response to extracted data
            extracted_data += response

        return extracted_data

    # Usage example
    result = extract_all_data(model, pdf_data=document_data,
                              json_template=json_template)
    return result


def generate_output_file(result):
    # Split the string into individual records
    records = result.split('\n\n')

    # Convert each record into a dictionary
    json_list = []
    for record in records:
        entry = {}
        lines = record.split('\n')
        for line in lines:
            if '=' in line:
                key, value = line.split('=', 1)
                entry[key.strip()] = value.strip()
        json_list.append(entry)

    # Convert the list of dictionaries to JSON format
    json_output = json.dumps(json_list, indent=4)

    entries = json.loads(json_output)

    # Remove duplicates and incomplete entries
    unique_entries = []
    seen_entries = set()

    required_keys = {
        "MEDICAL_SERVICE_PROVIDER",
        "DATE_OF_SERVICE",
        "PAGE_NO",
        "DESCRIPTION",
        "CPT_CODE",
        "ICD_CODE",
        "AMOUNT_CHARGED",
        "INSURANCE_PAID",
        "INSURANCE_ADJUSTMENT",
        "PLAINTIFF_PAID"
    }
    for entry in entries:
        # Check if all required keys are present
        if all(key in entry for key in required_keys):
            entry_tuple = tuple(entry.items())
            if entry_tuple not in seen_entries:
                seen_entries.add(entry_tuple)
                unique_entries.append(entry)

    len(unique_entries)

    df = pd.DataFrame(unique_entries)
    df.head(20)

    output_file = "billing_data.xlsx"
    df.to_excel(output_file, index=False)

    st.download_button(label="Download Excel File", data=open(output_file, 'rb').read(),
                       file_name=output_file, mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


# Function to generate summary using the Claude Model
def generate_summary(documents):
    """Generate summary from the provided document chunks using the Claude model."""
    try:
        summaries = []
        for chunk in documents:
            # Prepare the request body for the Claude model
            messages = [
                {"role": "user", "content": "I have a medical document that I'd like summarized. Output should have short description with the Patient's Complaint, History and Observations."},
                {"role": "assistant", "content": "Sure, I can help with that. Please provide the text of the medical document."},
                {"role": "user", "content": chunk}
            ]

            body = json.dumps({
                "max_tokens": 1000,
                "messages": messages,
                "anthropic_version": "bedrock-2023-05-31"
            })

            try:
                # Call the Claude model for summarization
                response = client.invoke_model(
                    body=body, modelId=os.environ.get("BEDROCK_MODEL"))
                response_body = json.loads(response.get("body").read())
                output = response_body.get("content", "No summary generated")
                summaries.append(output[0]["text"])
            except ClientError as e:
                st.error(
                    f"Amazon Bedrock error: {e.response['Error']['Message']}")
                return None
            except Exception as e:
                st.error(f"Error invoking the Claude model: {e}")
                return None

        return " ".join(summaries)
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return None


# Streamlit UI
# Streamlit Tabs for multi-program handling
tab1, tab2 = st.tabs(["Medical Document Summarization", "Billing Document Summarization"])


with tab1:
    st.title("Medical Document Summarization")

    # Step 1: Upload the document
    uploaded_file = st.file_uploader(
        "Upload a medical document (PDF, TIFF, JPG, JPEG)", type=["pdf", "tiff", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Step 2: Process the document
        file_type = uploaded_file.name.split(".")[-1].lower()
        file_path = upload_document(uploaded_file)

        if file_path:
            st.write(f"Processing document: {uploaded_file.name}")
            texts = process_document(file_path, file_type)

            if texts:
                # Step 3: Generate summary
                st.write("Generating summary...")
                summary = generate_summary(texts)

                if summary:
                    st.subheader("Generated Summary")
                    st.write(summary)
                else:
                    st.error("Failed to generate summary.")
        else:
            st.error("Failed to upload the document.")
    else:
        st.info("Please upload a medical document to get started.")


with tab2:
    st.title("Bill Document Extractor")

    # Step 1: Upload the document
    uploaded_file = st.file_uploader(
        "Upload a bill document (PDF, TIFF, JPG, JPEG)", type=["pdf", "tiff", "jpg", "jpeg"])

    if uploaded_file is not None:
        # Step 2: Process the document
        file_type = uploaded_file.name.split(".")[-1].lower()
        file_path = upload_document(uploaded_file)

        if file_path:
            st.write(f"Processing document: {uploaded_file.name}")
            texts = process_document(file_path, file_type)

            if texts:
                # Step 3: Generate summary
                st.write("Extracting document...")
                summary = generate_output(texts)

                if summary:
                    st.subheader("Generated Summary")
                    st.write("Generating output file")
                    generate_output_file(summary)
                    st.write("File generated")
                else:
                    st.error("Failed to generate summary.")
        else:
            st.error("Failed to upload the document.")
    else:
        st.info("Please upload a billing document to get started.")

