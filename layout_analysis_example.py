#!/usr/bin/env python
# Enhanced Layout Analysis Example - Direct Approach

import os
import cv2
import numpy as np
import platform
import sys
import subprocess
from PIL import Image, ImageDraw

# Check platform and PaddlePaddle installation before importing
def check_paddle_installation():
    """
    Check if PaddlePaddle is properly installed, especially for macOS environments.
    Returns True if everything is fine, False if there might be issues.
    """
    system_platform = platform.platform()
    print(f"Current platform: {system_platform}")
    
    is_macos = 'macOS' in system_platform
    is_arm64 = 'arm64' in system_platform
    
    if is_macos:
        print("macOS detected, checking PaddlePaddle installation...")
        
        # Try to import paddle to see if it works
        try:
            import paddle
            print("Paddle import successful!")
            
            # Try to run a quick paddle check
            try:
                import paddle.utils
                if hasattr(paddle.utils, 'run_check'):
                    result = paddle.utils.run_check()
                    print("PaddlePaddle check completed successfully!")
                    return True
            except Exception as e:
                print(f"Paddle check failed: {e}")
                
            # If we get here, paddle imported but run_check couldn't be performed
            # Let's continue anyway since import worked
            return True
            
        except ImportError as e:
            print(f"Failed to import paddle: {e}")
            print("PaddlePaddle might not be installed correctly.")
            
            if is_arm64:
                print("\nRecommendation for macOS ARM64 (Apple Silicon):")
                print("Try installing the development version of PaddlePaddle:")
                print("python -m pip install paddlepaddle==0.0.0 -f https://www.paddlepaddle.org.cn/whl/mac/cpu/develop.html")
            else:
                print("\nRecommendation for macOS Intel:")
                print("Make sure you have the correct version of PaddlePaddle installed.")
            
            # Ask user if they want to try auto-installing the correct version
            if is_arm64:
                user_input = input("\nWould you like to automatically try installing the ARM64 development version? (y/n): ")
                if user_input.lower() == 'y':
                    print("Attempting to install PaddlePaddle for ARM64...")
                    try:
                        subprocess.check_call([
                            sys.executable, "-m", "pip", "install", "paddlepaddle==0.0.0", 
                            "-f", "https://www.paddlepaddle.org.cn/whl/mac/cpu/develop.html"
                        ])
                        print("Installation completed. Please restart this script.")
                    except subprocess.CalledProcessError as e:
                        print(f"Installation failed: {e}")
                    
                    # Exit after installation attempt
                    sys.exit(0)
            
            return False
    
    # For non-macOS platforms or if not ARM64, assume paddle works fine
    return True

# Direct paddleocr command-line approach
def run_paddleocr_command(image_path, output_dir):
    """
    Run paddleocr command-line tool directly (which works successfully)
    """
    cmd = [
        'paddleocr', 
        '--image_dir', image_path,
        '--output', output_dir,
        '--use_angle_cls', 'true',
        '--use_gpu', 'false',
        '--lang', 'en',
        '--type', 'structure',
        '--recovery', 'true'
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    try:
        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT, universal_newlines=True)
        print("Command output:")
        print(output)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        print(f"Output: {e.output}")
        return False

# Check paddle installation before importing paddle-related modules
paddle_ok = check_paddle_installation()
if not paddle_ok:
    print("Warning: Proceeding without validating PaddlePaddle installation.")
    print("The script might hang when processing images.")
    proceed = input("Do you want to continue anyway? (y/n): ")
    if proceed.lower() != 'y':
        sys.exit(0)

# Now import PaddleOCR modules with appropriate error handling
try:
    from paddleocr import PaddleOCR, PPStructure, draw_structure_result, save_structure_res
    from paddleocr.ppstructure.recovery.recovery_to_doc import sorted_layout_boxes
    from paddleocr.ppstructure.recovery.recovery_to_markdown import convert_info_markdown
except ImportError as e:
    print(f"Error importing PaddleOCR modules: {e}")
    print("Please make sure PaddleOCR is correctly installed:")
    print("pip install paddleocr")
    sys.exit(1)

def analyze_layout(image_path, output_path='result.jpg', font_path='/Users/phuong.nguyen24/Downloads/UNIVERSITY/Honour Project/OCR_Method/Code/fonts/simfang.ttf'):
    """
    Analyze the layout of a document image and visualize the results.
    
    Args:
        image_path (str): Path to the input image
        output_path (str): Path to save the output visualization
        font_path (str): Path to the font file for visualization
        
    Returns:
        tuple: (visualization_path, summary_path, output_dir)
    """
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    if not output_dir:
        output_dir = './output'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Read the input image first to check if it's valid
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read image at {image_path}")
    
    # First try using the direct command-line approach which is known to work
    print("Trying direct paddleocr command-line approach (which was previously successful)...")
    command_success = run_paddleocr_command(image_path, output_dir)
    
    if command_success:
        print("Direct paddleocr command completed successfully.")
        
        # Even though we used the command-line tool, let's try to load the results for our summary
        try:
            # Check if we have result files in the output directory from the command
            ocr_files = [f for f in os.listdir(output_dir) if f.endswith('.txt') or f.endswith('.xlsx')]
            
            # Create a basic summary file
            summary_path = os.path.splitext(output_path)[0] + "_summary.txt"
            with open(summary_path, 'w') as f:
                f.write("DOCUMENT ANALYSIS SUMMARY\n")
                f.write("========================\n\n")
                
                # Document title from filename
                doc_title = os.path.basename(image_path)
                f.write(f"Document Title: {doc_title}\n\n")
                
                # Platform information
                f.write(f"System Platform: {platform.platform()}\n")
                f.write(f"Python Version: {platform.python_version()}\n\n")
                
                # List files generated
                f.write("GENERATED FILES:\n")
                for file in ocr_files:
                    f.write(f"- {file}\n")
                
                # Note about method used
                f.write("\nNOTE: Used direct paddleocr command-line tool for processing.\n")
            
            return output_path, summary_path, output_dir
            
        except Exception as e:
            print(f"Error creating summary from command-line results: {e}")
            # Continue with the regular approach as fallback
    
    print("Trying PaddleOCR Python API approach...")
    
    # Initialize PaddleOCR using similar parameters to the working command-line
    print("Initializing PaddleOCR...")
    try:
        ocr_engine = PaddleOCR(
            use_angle_cls=True, 
            lang="en",
            use_gpu=False,
            show_log=True,
            enable_mkldnn=True  # Try to enable MKL-DNN for better CPU performance
        )
        print("PaddleOCR initialized successfully.")
    except Exception as e:
        print(f"Error initializing PaddleOCR: {e}")
        raise
    
    # Process the image with PaddleOCR for initial text recognition
    print(f"Processing image with PaddleOCR: {image_path}")
    try:
        print("Running OCR...")
        ocr_result = ocr_engine.ocr(img_path=image_path, cls=True)
        print("PaddleOCR processing completed.")
    except Exception as e:
        print(f"Error during OCR processing: {e}")
        # If OCR fails, continue with an empty result
        ocr_result = None
    
    # Extract text from OCR results
    extracted_texts = []
    if ocr_result is not None:
        for idx, line in enumerate(ocr_result):
            if line is not None:
                for box_info in line:
                    if len(box_info) >= 2:  # Some versions return [points, [text, confidence]]
                        text_info = box_info[1]
                        if isinstance(text_info, list) and len(text_info) >= 1:
                            text = text_info[0]
                            confidence = text_info[1] if len(text_info) > 1 else None
                            extracted_texts.append((text, confidence))
                        elif isinstance(text_info, str):
                            extracted_texts.append((text_info, None))
    
    print(f"OCR detected {len(extracted_texts)} text regions")
    
    # Create visualization image
    try:
        # Create a simple image showing the recognized text and boxes
        image = Image.open(image_path).convert('RGB')
        # Draw the text detections
        if ocr_result is not None:
            draw = ImageDraw.Draw(image)
            for line in ocr_result:
                if line is not None:
                    for box_info in line:
                        if isinstance(box_info, list) and len(box_info) >= 1:
                            box = box_info[0]
                            if isinstance(box, list) and len(box) >= 4:
                                # Draw a rectangle around detected text
                                try:
                                    points = [(int(p[0]), int(p[1])) for p in box]
                                    draw.polygon(points, outline=(255, 0, 0))
                                except Exception as err:
                                    print(f"Error drawing box: {err}")
        image.save(output_path)
        print(f"Visualization saved to {output_path}")
    except Exception as e:
        print(f"Error creating visualization: {e}")
    
    # Create a comprehensive text summary file
    summary_path = os.path.splitext(output_path)[0] + "_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("DOCUMENT ANALYSIS SUMMARY\n")
        f.write("========================\n\n")
        
        # Document title from filename or OCR
        doc_title = os.path.basename(image_path)
        if extracted_texts:
            # Try to use the first recognized text as title
            doc_title = extracted_texts[0][0]
                
        f.write(f"Document Title: {doc_title}\n\n")
        
        # Write platform information
        f.write(f"System Platform: {platform.platform()}\n")
        f.write(f"Python Version: {platform.python_version()}\n\n")
        
        # Add OCR text
        if extracted_texts:
            f.write("OCR DETECTED TEXT:\n")
            for i, (text, confidence) in enumerate(extracted_texts[:20]):  # Limit to first 20
                conf_info = f" (Confidence: {confidence:.2f})" if confidence is not None else ""
                f.write(f"{i+1}. {text}{conf_info}\n")
    
    print(f"Analysis complete! Results saved to {output_dir}")
    print(f"Summary saved to {summary_path}")
    
    return output_path, summary_path, output_dir

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced document layout analysis")
    parser.add_argument("image_path", help="Path to the input document image")
    parser.add_argument("--output", "-o", default="result.jpg", 
                        help="Path to save the visualization result (default: result.jpg)")
    parser.add_argument("--font", "-f", 
                        default="/Users/phuong.nguyen24/Downloads/UNIVERSITY/Honour Project/OCR_Method/Code/fonts/simfang.ttf",
                        help="Path to the font file for visualization")
    
    args = parser.parse_args()
    
    analyze_layout(args.image_path, args.output, args.font) 