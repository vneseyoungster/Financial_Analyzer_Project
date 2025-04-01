
from financial_document_parser import FinancialDocumentParser
from paddleocr import PaddleOCR,draw_ocr

ocr = PaddleOCR(lang='en') # need to run only once to download and load model into memory
img_path = '/Users/phuong.nguyen24/Downloads/UNIVERSITY/Honour Project/OCR_Method/Code/imgs/13.png'
result = ocr.ocr(img_path, cls=False)
for idx in range(len(result)):
    res = result[idx]
    for line in res:
        print(line)

# draw result
from PIL import Image
result = result[0]
image = Image.open(img_path).convert('RGB')
boxes = [line[0] for line in result]
txts = [line[1][0] for line in result]
scores = [line[1][1] for line in result]
im_show = draw_ocr(image, boxes, txts, scores, font_path='/Users/phuong.nguyen24/Downloads/UNIVERSITY/Honour Project/OCR_Method/Code/fonts/simfang.ttf')
im_show = Image.fromarray(im_show)
im_show.save('result.jpg')

# Initialize the parser
parser = FinancialDocumentParser(lang='en')

# Process a financial document and get structured data
output_files, financial_structure = parser.process_document(img_path)

# The output_files dictionary contains paths to the generated files
text_file = output_files['text']    # Plain text summary
