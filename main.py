from doctr.io import DocumentFile
from doctr.models import ocr_predictor


model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)

# Path to the input image
image_path = "page_1.png"

# Load the image into a DocumentFile
single_img_doc = DocumentFile.from_images(image_path)

# Use the OCR model to process the document
result = model(single_img_doc)

# Show the result
result.show(single_img_doc)

# Export result to JSON
json_output = result.export()

# Save JSON to a file
output_json_file = os.path.join(tiled_folder, "output.json")
with open(output_json_file, "w") as json_file:
    json.dump(json_output, json_file, indent=2)

