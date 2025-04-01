# Financial Document OCR + LLM Analysis API

This Flask application provides an API endpoint to process financial documents using OCR (Optical Character Recognition) and LLM (Large Language Model) analysis.

## Setup

1. Make sure you have all the required dependencies installed:
   ```
   pip install -r requirements.txt
   ```

2. Start the LM Studio server (or other compatible LLM server) on port 1234 (default)

3. Run the Flask application:
   ```
   python app.py
   ```

The server will start on `http://localhost:1234`.

## API Endpoint

### POST /api/process-document

This endpoint accepts financial document images and returns OCR text and LLM analysis.

#### Input Methods

**Method 1: File Upload**
```
POST /api/process-document
Content-Type: multipart/form-data

Form data:
- document: [file]
```

**Method 2: Base64 Encoded Image**
```
POST /api/process-document
Content-Type: application/json

{
  "image": "base64_encoded_image_data"
}
```

#### Response Format

**Success Response:**
```json
{
  "success": true,
  "ocr_text": "Extracted text from the document...",
  "llm_analysis": "LLM analysis of the financial document...",
  "files": {
    "raw_text": "filename_results.txt"
  }
}
```

**Error Response:**
```json
{
  "success": false,
  "error": "Error message",
  "ocr_text": "Extracted text (if OCR was successful)",
  "files": {
    "text": "filename.txt"
  }
}
```

## Output Files

The API produces a raw text output file:

**Raw Text Results File (_results.txt)**: Contains only the LLM analysis in a simple text format

All files are saved in the output directory.

## Example Usage

### Using cURL

#### File Upload:
```bash
curl -X POST -F "document=@path/to/financial_document.png" http://localhost:1234/api/process-document
```

#### Base64 Image:
```bash
curl -X POST -H "Content-Type: application/json" \
  -d '{"image":"BASE64_ENCODED_IMAGE_DATA"}' \
  http://localhost:1234/api/process-document
```

### Using Python Requests

```python
import requests
import base64

# Method 1: File Upload
files = {'document': open('path/to/financial_document.png', 'rb')}
response = requests.post('http://localhost:1234/api/process-document', files=files)

# Method 2: Base64 Encoded Image
with open('path/to/financial_document.png', 'rb') as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

data = {'image': encoded_image}
response = requests.post('http://localhost:1234/api/process-document', json=data)

# Process the response
result = response.json()
if result['success']:
    print("OCR Text:", result['ocr_text'][:100] + "...")  # First 100 chars
    print("LLM Analysis:", result['llm_analysis'])
    print("Raw Text File:", result['files']['raw_text'])
else:
    print("Error:", result['error'])
```

## Notes

- Make sure the LM Studio server (or other compatible LLM server) is running before making API requests.
- For large documents, the processing might take some time due to the OCR and LLM analysis.
- The OCR results and LLM analysis are saved in the output directory. 