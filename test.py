from financial_document_parser import FinancialDocumentParser
from LLM_Request import LLMRequest, Financial_Agent
import json
import os
import sys
import glob
import re
from datetime import datetime

def save_to_raw_text(llm_analysis, output_path, filename, append=False):
    """
    Save only LLM analysis to a raw text file
    
    Args:
        llm_analysis: Analysis from LLM
        output_path: Path to save the raw text file
        filename: Name of the original file
        append: Whether to append to existing file
    """
    mode = 'a' if append else 'w'
    with open(output_path, mode, encoding='utf-8') as f:
        # Add a divider with timestamp and filename
        if append:
            f.write("\n\n" + "="*50 + "\n")
            f.write(f"Document: {filename}\n")
            f.write(f"Analyzed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*50 + "\n\n")
        f.write(llm_analysis)
    
    print(f"- {'Appended to' if append else 'Created'} raw text file: {output_path}")
    return output_path

def extract_and_save_json(text_content, output_path):
    """
    Extract JSON content from analysis text and save to a JSON file
    
    Args:
        text_content: Content from Financial_Agent analysis
        output_path: Base path for the output file (without extension)
    
    Returns:
        str: Path to the saved JSON file or None if no JSON found
    """
    # Try to find JSON content within the text using regex
    # This pattern looks for content between triple backticks with json
    json_pattern = r'```json\s*({[\s\S]*?})\s*```'
    json_match = re.search(json_pattern, text_content)
    
    if not json_match:
        # Try to find content between triple backticks without explicit json tag
        json_pattern = r'```\s*({[\s\S]*?})\s*```'
        json_match = re.search(json_pattern, text_content)
    
    if not json_match:
        # Try a more general approach if the specific pattern doesn't match
        # Look for content that starts with { and ends with }
        json_pattern = r'({[\s\S]*?})'
        json_match = re.search(json_pattern, text_content)
    
    if json_match:
        json_str = json_match.group(1).strip()
        
        try:
            # First try to parse as is
            json_data = json.loads(json_str)
            
            # Create JSON output path
            json_path = f"{output_path.rsplit('.', 1)[0]}.json"
            
            # Save to JSON file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2)
                
            print(f"- Extracted and saved JSON data to: {json_path}")
            return json_path
        except json.JSONDecodeError:
            # If failed, try to fix common issues in the JSON structure
            print("Initial JSON parsing failed. Attempting to fix the format...")
            
            # Fix for 4_results_financial_analysis.txt - clean and reformat the JSON
            try:
                # Clean any trailing commas
                json_str = re.sub(r',\s*}', '}', json_str)
                json_str = re.sub(r',\s*]', ']', json_str)
                
                # Replace single quotes with double quotes for keys and string values
                json_str = re.sub(r"'([^']*)':", r'"\1":', json_str)
                json_str = re.sub(r':\s*\'([^\']*)\'', r': "\1"', json_str)
                
                # Fix comma issues in values (assuming these are numbers with commas as thousand separators)
                # This needs careful handling to preserve the full numeric value
                # First, handle cases where the value ends with a comma (like in "42,129,819,")
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
                
                # Create JSON output path
                json_path = f"{output_path.rsplit('.', 1)[0]}.json"
                
                # Save to JSON file
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data, f, indent=2)
                    
                print(f"- Fixed and saved JSON data to: {json_path}")
                return json_path
            except Exception as e:
                print(f"Warning: Could not fix JSON format: {e}")
                
                # As a last resort, try to extract and reformat manually
                try:
                    # Manual extraction for the specific format in 4_results_financial_analysis.txt
                    # Create a properly formatted JSON object manually
                    manual_json = {}
                    
                    # Look for patterns like "Revenue": { ... }
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
                            # Remove any trailing comma (like in "42,129,819,")
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
                                    # If not a number, keep as string but remove quotes
                                    manual_json[field_name]["value"] = value_str.strip('"\'')
                            
                            # Add from and to dates if available
                            if from_match:
                                manual_json[field_name]["from"] = from_match.group(1).strip().strip('"\'')
                            if to_match:
                                manual_json[field_name]["to"] = to_match.group(1).strip().strip('"\'')
                    
                    if manual_json:
                        # Create JSON output path
                        json_path = f"{output_path.rsplit('.', 1)[0]}.json"
                        
                        # Save to JSON file
                        with open(json_path, 'w', encoding='utf-8') as f:
                            json.dump(manual_json, f, indent=2)
                            
                        print(f"- Manually extracted and saved JSON data to: {json_path}")
                        return json_path
                    else:
                        print("Warning: Manual extraction failed to find valid data.")
                        return None
                except Exception as e:
                    print(f"Warning: Manual extraction failed: {e}")
                    return None
    else:
        print("Warning: No JSON content found in the analysis.")
        return None

def process_financial_document(img_path, output_file=None, append=False):
    """
    Process a financial document image through OCR and LLM analysis
    
    Args:
        img_path: Path to the image file
        output_file: Path to output file (optional)
        append: Whether to append to existing file
    
    Returns:
        bool: Success status
    """
    # Initialize the parser and LLM
    parser = FinancialDocumentParser(lang='en')
    llm = LLMRequest(default_timeout=90)  # Using longer default timeout
    
    # Check if LLM server is running
    if not llm.check_server():
        print("Error: LLM server is not running. Please start the server first.")
        return False
    
    # Process document with OCR
    print(f"Processing document with OCR: {img_path}")
    try:
        output_files, financial_structure = parser.process_document(img_path)
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        return False
        
    # Get the OCR text content
    with open(output_files['text'], 'r', encoding='utf-8') as f:
        ocr_text = f.read()
    
    # Process the OCR text with LLM
    print("Analyzing text with LLM...")
    print("This may take some time for large documents. Please be patient.")
    
    # Process with longer timeout and retries
    llm_result = llm.process_text(
        text=ocr_text,
        max_retries=3,
        timeout=300  # 5 minutes timeout for initial attempt
    )
    
    if llm_result["success"]:
        # Save to raw text file
        if output_file is None:
            output_dir = os.path.dirname(output_files['text'])
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            output_file = os.path.join(output_dir, f"{base_name}_results.txt")
            
        save_to_raw_text(
            llm_result["content"], 
            output_file,
            os.path.basename(img_path),
            append
        )
        
        print(f"Successfully processed {img_path}")
        return True
    else:
        print(f"Error during LLM analysis: {llm_result['error']}")
        print("\nTroubleshooting tips:")
        print("1. Check that the LM Studio server is still running")
        print("2. Try restarting the LM Studio server")
        print("3. For very large documents, consider processing smaller sections separately")
        return False

def analyze_financial_document(path, output_file=None, append=False):
    """
    Analyze a financial document with Financial_Agent
    
    Args:
        path: Path to the text file or financial document
        output_file: Path to output file (optional)
        append: Whether to append to existing file
    
    Returns:
        bool: Success status
    """
    # Initialize Financial_Agent
    financial_agent = Financial_Agent(default_timeout=120)
    
    # Check if LLM server is running
    if not financial_agent.check_server():
        print("Error: LLM server is not running. Please start the server first.")
        return False
    
    # Check if the input is a file and read its content
    if os.path.isfile(path):
        file_ext = os.path.splitext(path)[1].lower()
        
        # If it's a text file, read it directly
        if file_ext in ['.txt', '.json', '.csv']:
            print(f"Reading financial data from: {path}")
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    financial_text = f.read()
            except Exception as e:
                print(f"Error reading file: {e}")
                return False
        # If it's an image, we need to process it with OCR first
        elif file_ext in ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']:
            print(f"Image file detected, processing with OCR first: {path}")
            # Initialize the parser for OCR
            parser = FinancialDocumentParser(lang='en')
            
            try:
                output_files, _ = parser.process_document(path)
                with open(output_files['text'], 'r', encoding='utf-8') as f:
                    financial_text = f.read()
                print("OCR processing completed.")
            except Exception as e:
                print(f"Error during OCR processing: {e}")
                return False
        else:
            print(f"Unsupported file format: {file_ext}")
            print("Supported formats: .txt, .json, .csv (for direct analysis)")
            print("                   .jpg, .jpeg, .png, .tif, .tiff, .bmp (for OCR processing)")
            return False
    else:
        print(f"Error: {path} is not a valid file.")
        return False
    
    # Process the text with Financial_Agent
    print("Performing financial analysis with Financial_Agent...")
    print("This may take some time for comprehensive analysis. Please be patient.")
    
    # Process with longer timeout and retries
    agent_result = financial_agent.analyze_financial_data(
        text=financial_text,
        max_retries=3,
        timeout=360  # 6 minutes timeout for initial attempt
    )
    
    if agent_result["success"]:
        # Save to raw text file
        if output_file is None:
            # Create output directory if needed
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'financial_data')
            os.makedirs(output_dir, exist_ok=True)
            
            base_name = os.path.splitext(os.path.basename(path))[0]
            output_file = os.path.join(output_dir, f"{base_name}_financial_analysis.txt")
            
        # Save the full analysis to text file
        save_to_raw_text(
            agent_result["content"], 
            output_file,
            os.path.basename(path),
            append
        )
        
        # Extract and save JSON content for visualization
        extract_and_save_json(agent_result["content"], output_file)
        
        print(f"Successfully analyzed {path}")
        return True
    else:
        print(f"Error during financial analysis: {agent_result['error']}")
        print("\nTroubleshooting tips:")
        print("1. Check that the LLM server is still running")
        print("2. Try restarting the LLM server")
        print("3. For very large documents, consider processing smaller sections separately")
        return False

def process_image_folder(folder_path, output_file=None):
    """
    Process all images in a folder with OCR and LLM parsing
    
    Args:
        folder_path: Path to the folder containing images
        output_file: Path to output file (optional)
    """
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return
        
    # Get all image files
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
    if not image_files:
        print(f"No image files found in {folder_path}")
        return
        
    print(f"Found {len(image_files)} image files to process.")
    
    # Create output file path if not provided
    if output_file is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'financial_data')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f"batch_analysis_{timestamp}.txt")
    
    # Process each image
    success_count = 0
    for i, img_path in enumerate(image_files):
        print(f"\nProcessing image {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
        
        success = process_financial_document(
            img_path, 
            output_file=output_file,
            append=(i > 0)  # append for all files except the first one
        )
            
        if success:
            success_count += 1
    
    print(f"\nBatch processing complete: {success_count}/{len(image_files)} files processed successfully.")
    print(f"Results saved to: {output_file}")

def analyze_financial_folder(folder_path, output_file=None):
    """
    Analyze all financial documents in a folder using Financial_Agent
    
    Args:
        folder_path: Path to the folder containing financial documents
        output_file: Path to output file (optional)
    """
    if not os.path.isdir(folder_path):
        print(f"Error: {folder_path} is not a valid directory.")
        return
        
    # Get all supported files
    text_extensions = ['*.txt', '*.json', '*.csv']
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.tif', '*.tiff', '*.bmp']
    all_files = []
    
    for ext in text_extensions + image_extensions:
        all_files.extend(glob.glob(os.path.join(folder_path, ext)))
        
    if not all_files:
        print(f"No supported files found in {folder_path}")
        return
        
    print(f"Found {len(all_files)} files to analyze.")
    
    # Create output file path if not provided
    if output_file is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'financial_data')
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f"financial_analysis_{timestamp}.txt")
    
    # Process each file
    success_count = 0
    for i, file_path in enumerate(all_files):
        print(f"\nAnalyzing file {i+1}/{len(all_files)}: {os.path.basename(file_path)}")
        
        success = analyze_financial_document(
            file_path, 
            output_file=output_file,
            append=(i > 0)  # append for all files except the first one
        )
            
        if success:
            success_count += 1
    
    print(f"\nBatch analysis complete: {success_count}/{len(all_files)} files analyzed successfully.")
    print(f"Results saved to: {output_file}")

def process_existing_analysis(analysis_file):
    """
    Process an existing financial analysis file to extract JSON data
    
    Args:
        analysis_file: Path to the existing analysis text file
    
    Returns:
        bool: Success status
    """
    if not os.path.isfile(analysis_file):
        print(f"Error: {analysis_file} is not a valid file.")
        return False
    
    print(f"Processing existing analysis file: {analysis_file}")
    
    try:
        with open(analysis_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract and save JSON
        json_path = extract_and_save_json(content, analysis_file)
        
        if json_path:
            print(f"Successfully extracted JSON data from {analysis_file}")
            return True
        else:
            print(f"No JSON data found in {analysis_file}")
            return False
            
    except Exception as e:
        print(f"Error processing analysis file: {e}")
        return False

def get_user_choice():
    """
    Get user choice for processing mode
    
    Returns:
        str: "parse", "analyze", or "extract_json"
    """
    print("\n==== Financial Document Processing ====")
    print("1. Parse financial document (extract structured data)")
    print("2. Analyze financial document (comprehensive financial analysis)")
    print("3. Extract JSON from existing analysis file")
    
    while True:
        try:
            choice = input("\nEnter your choice (1, 2, or 3): ").strip()
            if choice == "1":
                return "parse"
            elif choice == "2":
                return "analyze"
            elif choice == "3":
                return "extract_json"
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            sys.exit(0)

if __name__ == "__main__":
    # Get user processing mode choice
    processing_mode = get_user_choice()
    
    # Check command-line arguments
    if len(sys.argv) > 1:
        path = sys.argv[1]
        
        # Handle optional output file path
        output_file = None
        if len(sys.argv) > 2:
            output_file = sys.argv[2]
            
        # Check if path is a directory or a file
        if os.path.isdir(path):
            print(f"Processing folder: {path}")
            if processing_mode == "analyze":
                analyze_financial_folder(path, output_file)
            elif processing_mode == "parse":
                process_image_folder(path, output_file)
            else:  # extract_json mode
                print("Error: For JSON extraction, please specify a single analysis file.")
        else:
            print(f"Processing single file: {path}")
            if processing_mode == "analyze":
                analyze_financial_document(path, output_file)
            elif processing_mode == "parse":
                process_financial_document(path, output_file)
            else:  # extract_json mode
                process_existing_analysis(path)
    else:
        # Prompt for path if not provided
        path = input("\nEnter path to financial document or folder: ").strip()
        
        if not path:
            # Default behavior: look for "imgs" folder in current directory
            imgs_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'imgs')
            if os.path.isdir(imgs_folder) and processing_mode != "extract_json":
                print(f"Using default images folder: {imgs_folder}")
                if processing_mode == "analyze":
                    analyze_financial_folder(imgs_folder)
                else:
                    process_image_folder(imgs_folder)
            else:
                if processing_mode == "extract_json":
                    print("Error: Please provide the path to an existing analysis file.")
                else:
                    print("No path provided and default 'imgs' folder not found.")
                sys.exit(1)
        elif os.path.isdir(path):
            print(f"Processing folder: {path}")
            if processing_mode == "analyze":
                analyze_financial_folder(path)
            elif processing_mode == "parse":
                process_image_folder(path)
            else:  # extract_json mode
                print("Error: For JSON extraction, please specify a single analysis file.")
        elif os.path.isfile(path):
            print(f"Processing single file: {path}")
            if processing_mode == "analyze":
                analyze_financial_document(path)
            elif processing_mode == "parse":
                process_financial_document(path)
            else:  # extract_json mode
                process_existing_analysis(path)
        else:
            print(f"Error: {path} is not a valid file or directory.")

