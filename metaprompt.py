json_template = {
    "MEDICAL_SERVICE_PROVIDER": "",
    "DATE_OF_SERVICE": "",
    "PAGE_NO": "",
    "DESCRIPTION": "",
    "CPT_CODE": "",
    "ICD_CODE": "",
    "AMOUNT_CHARGED": "",
    "INSURANCE_PAID": "",
    "INSURANCE_ADJUSTMENT": "",
    "PLAINTIFF_PAID": ""
}

json_data = {
    "MEDICAL_SERVICE_PROVIDER": "UNC HEALTHCARE SYSTEM",
    "DATE_OF_SERVICE": "09/26/2019",
    "PAGE_NO": "Pg. 110",
    "DESCRIPTION": "FOOT RT PORT",
    "CPT_CODE": "80336",
    "ICD_CODE": "M54.06,M54.07,G89.4",
    "AMOUNT_CHARGED": "18.79",
    "INSURANCE_PAID": "352.23",
    "INSURANCE_ADJUSTMENT": "63.45",
    "PLAINTIFF_PAID": "25.00"
}

meta_prompt = f"""
You are an AI assistant specializing in extracting structured data from complex billing documents. Your task is to analyze the provided JSON data, which contains text, tables, and forms extracted from a billing PDF document. It is CRITICAL that you extract the data fields mentioned in the template JSON provided below for EACH billing item found in the parsed PDF. Each data point must be output as a key-value pair, one per line.

IMPORTANT INSTRUCTIONS:
1. Analyze EVERY SINGLE ELEMENT of the provided data: all text, every cell in every table, and all form fields.
2. Identify ALL individual Billing information in the document. Each bill typically has their own section or table.
3. For EACH bill item, create a separate instance of the template JSON and output each field as a key-value pair, one per line.
4. Format: FIELD_NAME=VALUE
5. Start each bill data with `MEDICAL_SERVICE_PROVIDER=MedicalServiceProviderName`. This will act as a marker to separate data between different billing item.
6. Ensure that every response is a valid key-value pair list without any additional characters such as backslashes (`\`) or quotes.
7. Ensure all dates are in proper format and all numerical values are appropriately typed (integer or float).
8. Do not include any explanatory text, prefixes, or suffixes. Only provide the key-value pairs.
9. Process ALL pages and ALL billing items in the document.

ACCURACY CHECK:
- After extraction, review your output to ensure EVERY SINGLE data point for EACH billing item is captured in the key-value pairs.
- Ensure no data point is duplicated or misplaced.
- Ensure all key-value pairs are correct, consistent, and complete for each bill.

Here is the template JSON data containing keys to populate for each billing item:

{json_template}

Here is the parsed JSON data from a billing PDF document to analyze:

{json_data}

Respond with a list of key-value pairs, one per line, ensuring EVERY SINGLE critical data point from the template is captured for EACH billing data found in the document. Start each billing item's data with `MEDICAL_SERVICE_PROVIDER=MedicalServiceProviderName`.

"""
