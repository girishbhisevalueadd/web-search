
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
import base64
from pdf2image import convert_from_path

load_dotenv()

model_name = "gpt-5.5"  

# Download a sample scanned image or use your own
pdf_path = "image.pdf" # Change to your image path

# Extract first page as image from PDF
images = convert_from_path(pdf_path, dpi=600, first_page=1, last_page=1)

image = images[0]

# Save temporarily
img_path = "temp_page.png"
image.save(img_path)

# Convert image to base64
with open(img_path, "rb") as image_file:
    base64_image = base64.b64encode(image_file.read()).decode('utf-8')
    
llm = ChatOpenAI(
    model=model_name, 
    max_tokens=8000,
    temperature=0)

# Test extraction
response = llm.invoke([
    {"role": "user", "content": [
        {"type": "text", "text": "Extract all text from this PDF page:"},
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{base64_image}"}}
    ]}
])

print("Extracted Text:")
print(response.content)
print(f"{model_name} successfully extracted text from the PDF page!")