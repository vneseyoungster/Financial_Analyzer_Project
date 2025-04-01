#!/usr/bin/env python
# Flask API for Financial Document OCR + LLM Processing

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import base64
import tempfile
import re
from werkzeug.utils import secure_filename
from financial_document_parser import FinancialDocumentParser
from LLM_Request import LLMRequest, Financial_Agent

app = Flask(__name__)
# Enable CORS for all routes
CORS(app)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure output folder
OUTPUT_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Initialize parser and LLMs
parser = FinancialDocumentParser(lang='en')
llm = LLMRequest(default_timeout=90)
financial_agent = Financial_Agent(default_timeout=120)

def save_to_raw_text(llm_analysis, output_path):
    """
    Save only LLM analysis to a raw text file
    
    Args:
        llm_analysis: Analysis from LLM
        output_path: Path to save the raw text file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(llm_analysis)
    
    return output_path

def extract_json_from_text(text_content, output_base_path=None):
    """
    Extract JSON content from analysis text and optionally save to a file
    
    Args:
        text_content: Content from Financial_Agent analysis
        output_base_path: Optional base path for saving the JSON file
    
    Returns:
        dict: Extracted JSON data or None if extraction failed
        str: Path to saved JSON file (if output_base_path provided) or None
    """
    # Try to find JSON content within the text using regex
    json_pattern = r'```json\s*({[\s\S]*?})\s*```'
    json_match = re.search(json_pattern, text_content)
    
    if not json_match:
        # Try to find content between triple backticks without explicit json tag
        json_pattern = r'```\s*({[\s\S]*?})\s*```'
        json_match = re.search(json_pattern, text_content)
    
    if not json_match:
        # Try a more general approach if the specific pattern doesn't match
        json_pattern = r'({[\s\S]*?})'
        json_match = re.search(json_pattern, text_content)
    
    json_data = None
    json_path = None
    
    if json_match:
        json_str = json_match.group(1).strip()
        
        try:
            # First try to parse as is
            json_data = json.loads(json_str)
        except json.JSONDecodeError:
            # If failed, try to fix common issues in the JSON structure
            try:
                # Clean any trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                
                # Replace single quotes with double quotes for keys and string values
                json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
                json_str = re.sub(r':\s*\'([^\']*)\'', r': "\1"', json_str)
                
                # Fix comma issues in values (assuming these are numbers with commas as thousand separators)
                json_str = re.sub(r'"value":\s*([0-9,]+),\s*(["}])', r'"value": \1\2', json_str)
                
                # Fix specific issue with missing quotes around dates
                json_str = re.sub(r'"from":\s*(\d{4}-\d{2}-\d{2})', r'"from": "\1"', json_str)
                json_str = re.sub(r'"to":\s*(\d{4}-\d{2}-\d{2})', r'"to": "\1"', json_str)
                
                # Parse the fixed JSON
                json_data = json.loads(json_str)
                
                # Post-process the JSON to fix number formatting issues
                for key, value in json_data.items():
                    if isinstance(value, dict) and "value" in value:
                        value_str = str(value["value"])
                        # If the value contains commas, remove them and convert to an integer
                        if isinstance(value_str, str) and ',' in value_str:
                            value["value"] = int(value_str.replace(',', ''))
            except Exception:
                # As a last resort, try to extract and reformat manually
                try:
                    # Manual extraction for specific format
                    manual_json = {}
                    field_pattern = r'"([^"]+)":\s*{([^}]+)}'
                    fields = re.finditer(field_pattern, json_str)
                    
                    for field in fields:
                        field_name = field.group(1)
                        field_content = field.group(2)
                        
                        # Extract value, from, and to
                        value_match = re.search(r'"value":\s*([^,"\n]+(?:,[^,"\n]+)*)', field_content)
                        from_match = re.search(r'"from":\s*([^,\n]+)', field_content)
                        to_match = re.search(r'"to":\s*([^,\n]+)', field_content)
                        
                        if value_match:
                            # Create a nested dictionary for this field
                            manual_json[field_name] = {}
                            
                            # Extract and clean the value (removing commas but keeping all digits)
                            value_str = value_match.group(1).strip()
                            if value_str.endswith(','):
                                value_str = value_str[:-1]
                            
                            # Now remove commas between digits while preserving the full number
                            value_str = value_str.replace(',', '')
                            
                            try:
                                # Try to convert to number if possible
                                manual_json[field_name]["value"] = int(value_str)
                            except ValueError:
                                try:
                                    manual_json[field_name]["value"] = float(value_str)
                                except ValueError:
                                    manual_json[field_name]["value"] = value_str.strip('"\'')
                            
                            # Add from and to dates if available
                            if from_match:
                                manual_json[field_name]["from"] = from_match.group(1).strip().strip('"\'')
                            if to_match:
                                manual_json[field_name]["to"] = to_match.group(1).strip().strip('"\'')
                    
                    if manual_json:
                        json_data = manual_json
                except Exception:
                    json_data = None
    
    # If output_base_path is provided and we have JSON data, save it
    if output_base_path and json_data:
        json_path = f"{output_base_path.rsplit('.', 1)[0]}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2)
    
    return json_data, json_path

@app.route('/api/process-document', methods=['POST'])
def process_document():
    """
    Process a financial document image through all steps: OCR, parsing, analysis, and JSON extraction
    
    Expected POST data:
    - JSON with 'image' field containing base64 encoded image
    - OR file upload with 'document' as the file field name
    
    Returns:
    - JSON with extracted financial data
    """
    try:
        # Check if LLM servers are running
        if not llm.check_server():
            return jsonify({
                'success': False,
                'error': 'LLM server for parsing is not running. Please start the server first.'
            }), 503
        
        if not financial_agent.check_server():
            return jsonify({
                'success': False,
                'error': 'LLM server for financial analysis is not running. Please start the server first.'
            }), 503
        
        # Determine input method (base64 or file upload)
        if request.is_json:
            # Handle base64 encoded image
            data = request.json
            if 'image' not in data:
                return jsonify({'success': False, 'error': 'No image data provided'}), 400
                
            # Create a temporary file for the image
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                img_data = base64.b64decode(data['image'])
                temp_file.write(img_data)
                img_path = temp_file.name
                filename = 'temp_image'
        else:
            # Handle file upload
            if 'document' not in request.files:
                return jsonify({'success': False, 'error': 'No document file provided'}), 400
                
            file = request.files['document']
            if file.filename == '':
                return jsonify({'success': False, 'error': 'Empty file name'}), 400
                
            filename = secure_filename(file.filename)
            img_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(img_path)
        
        # Step 1: Process document with OCR
        print(f"Processing document with OCR: {img_path}")
        try:
            output_files, financial_structure = parser.process_document(
                img_path, 
                output_dir=OUTPUT_FOLDER
            )
        except Exception as e:
            return jsonify({'success': False, 'error': f"Error during OCR processing: {str(e)}"}), 500
        
        # Get the OCR text content
        with open(output_files['text'], 'r', encoding='utf-8') as f:
            ocr_text = f.read()
        
        # Step 2: Process with LLMRequest for initial parsing
        print("Parsing financial document with LLMRequest...")
        llm_result = llm.process_text(
            text=ocr_text,
            max_retries=3,
            timeout=300  # 5 minutes timeout for large documents
        )
        
        if not llm_result["success"]:
            return jsonify({
                'success': False,
                'error': f"Error during initial parsing: {llm_result['error']}"
            }), 500
        
        # Save the initial parsing results
        base_name = os.path.splitext(filename)[0]
        raw_text_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_results.txt")
        save_to_raw_text(llm_result["content"], raw_text_path)
        
        # Step 3: Process with Financial_Agent for detailed analysis
        print("Performing financial analysis with Financial_Agent...")
        agent_result = financial_agent.analyze_financial_data(
            text=ocr_text,
            max_retries=3,
            timeout=360  # 6 minutes timeout for initial attempt
        )
        
        if not agent_result["success"]:
            return jsonify({
                'success': False,
                'error': f"Error during financial analysis: {agent_result['error']}"
            }), 500
        
        # Save the financial analysis results
        analysis_path = os.path.join(OUTPUT_FOLDER, f"{base_name}_financial_analysis.txt")
        save_to_raw_text(agent_result["content"], analysis_path)
        
        # Step 4: Extract JSON data from the financial analysis
        json_data, json_path = extract_json_from_text(
            agent_result["content"],
            output_base_path=analysis_path
        )
        
        # Clean up temporary file if used
        if 'temp' in img_path:
            try:
                os.unlink(img_path)
            except:
                pass
        
        # If no JSON data was extracted, return an error
        if not json_data:
            return jsonify({
                'success': False,
                'error': "Failed to extract structured financial data from the analysis"
            }), 500
        
        # Return only the financial data JSON
        return jsonify({
            'success': True,
            'financial_data': json_data,
            'file_path': os.path.basename(json_path) if json_path else None
        })
            
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 