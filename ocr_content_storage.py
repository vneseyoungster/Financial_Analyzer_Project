#!/usr/bin/env python
# Optimal storage formats for PaddleOCR content

import os
import json
import csv
import pandas as pd
from paddleocr import PaddleOCR

def extract_and_store_ocr_content(img_path, output_dir='./ocr_output'):
    """
    Extract OCR content from an image and store it in multiple formats
    
    Args:
        img_path (str): Path to the input image
        output_dir (str): Directory to save output files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize OCR engine
    ocr = PaddleOCR(lang='en', use_angle_cls=True, use_gpu=False)
    
    # Process the image
    print(f"Processing image: {img_path}")
    result = ocr.ocr(img_path, cls=False)
    
    # Prepare filename base for output files
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # 1. Simple Text Format - Just the extracted text
    simple_output = []
    
    # 2. Structured Format - Content with position and confidence
    structured_output = []
    
    # Process the results
    if result and len(result) > 0:
        for idx in range(len(result)):
            page_results = result[idx]
            
            for line in page_results:
                if len(line) >= 2:
                    bbox = line[0]  # The bounding box coordinates
                    text_info = line[1]  # Text and confidence
                    
                    if isinstance(text_info, tuple) and len(text_info) >= 1:
                        text = text_info[0]
                        confidence = text_info[1] if len(text_info) > 1 else None
                        
                        # Add to simple output (just text)
                        simple_output.append(text)
                        
                        # Add to structured output
                        structured_output.append({
                            'text': text,
                            'confidence': confidence,
                            'bbox': bbox,
                            'page': idx
                        })
    
    # Save in different formats
    
    # 1. Plain Text (simplest, just content)
    txt_path = os.path.join(output_dir, f"{base_name}_content.txt")
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(simple_output))
    
    # 2. JSON (structured, with metadata)
    json_path = os.path.join(output_dir, f"{base_name}_full.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(structured_output, f, ensure_ascii=False, indent=2)
    
    # 3. CSV (tabular format)
    csv_path = os.path.join(output_dir, f"{base_name}_table.csv")
    with open(csv_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['text', 'confidence', 'page', 'bbox_x1', 'bbox_y1', 'bbox_x2', 'bbox_y2', 'bbox_x3', 'bbox_y3', 'bbox_x4', 'bbox_y4'])
        
        for item in structured_output:
            row = [
                item['text'], 
                item['confidence'], 
                item['page']
            ]
            # Add flattened bbox coordinates
            for point in item['bbox']:
                row.extend(point)
            writer.writerow(row)
    
    # 4. Excel (for financial data, easy to open in spreadsheet apps)
    excel_path = os.path.join(output_dir, f"{base_name}_data.xlsx")
    df = pd.DataFrame([{'text': item['text'], 'confidence': item['confidence']} for item in structured_output])
    df.to_excel(excel_path, index=False)
    
    # 5. Content-Only JSON (middle ground between txt and full JSON)
    content_json_path = os.path.join(output_dir, f"{base_name}_content.json")
    with open(content_json_path, 'w', encoding='utf-8') as f:
        json.dump(simple_output, f, ensure_ascii=False, indent=2)
    
    print(f"OCR content extracted and saved in multiple formats:")
    print(f"- Text: {txt_path}")
    print(f"- Full JSON: {json_path}")
    print(f"- CSV: {csv_path}")
    print(f"- Excel: {excel_path}")
    print(f"- Content JSON: {content_json_path}")
    
    return {
        'text': txt_path,
        'json': json_path,
        'csv': csv_path,
        'excel': excel_path,
        'content_json': content_json_path
    }

def extract_financial_data(ocr_result):
    """
    Specialized function to extract and structure financial data from OCR results
    
    Args:
        ocr_result: The result from PaddleOCR
    
    Returns:
        dict: Structured financial data
    """
    financial_data = {
        'line_items': [],
        'values': [],
        'totals': {}
    }
    
    # Process the OCR results
    if ocr_result and len(ocr_result) > 0:
        for idx in range(len(ocr_result)):
            page_results = ocr_result[idx]
            
            for line in page_results:
                if len(line) >= 2:
                    text = line[1][0] if isinstance(line[1], tuple) and len(line[1]) > 0 else ""
                    
                    # Check if it's a total (look for "TOTAL" keyword)
                    if "TOTAL" in text.upper():
                        # Extract the total value - this is a simplistic approach
                        # In a real application, you'd need to find the associated value
                        parts = text.split()
                        if len(parts) > 1 and any(c.isdigit() for c in parts[-1]):
                            financial_data['totals'][' '.join(parts[:-1])] = parts[-1]
                        else:
                            financial_data['totals'][text] = "N/A"
                    
                    # Check if it's likely a financial value (contains digits and commas)
                    elif any(c.isdigit() for c in text) and "," in text:
                        financial_data['values'].append(text)
                    
                    # Otherwise, consider it a line item description
                    else:
                        financial_data['line_items'].append(text)
    
    return financial_data

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract and store OCR content in optimal formats")
    parser.add_argument("image_path", help="Path to the input image")
    parser.add_argument("--output", "-o", default="./ocr_output", 
                        help="Directory to save output files")
    
    args = parser.parse_args()
    
    extract_and_store_ocr_content(args.image_path, args.output) 