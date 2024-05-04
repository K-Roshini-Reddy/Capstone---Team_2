from flask import Flask, request, jsonify, render_template
from torchvision import transforms
from PIL import Image
import torch

app = Flask(__name__)

# this step is to load the trained model
model = torch.load("C:/Users/User/OneDrive/Documents/Spring Capstone 2024/Model Path/vit_model_path.pth")
model.eval()

# performs image preprocessing of the image uploaded
preprocess = transforms.Compose([
    transforms.Resize(150),
    transforms.CenterCrop(128),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

@app.route('/')
def index():
    return render_template('frontend updates.html')  # it will render the html page

@app.route('/process-image', methods=['POST'])  
def process_image():
    # Check for the image uploaded in the user interface
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    
    image_file = request.files['image']
    image = Image.open(image_file)
    
    # Preprocessing the image
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(input_batch)
        _, predicted = torch.max(output, 1)
    
    if predicted.item() == 0:
        result = 'Normal (No pneumonia detected)'
    else:
        result = 'Pneumonia detected'
    
    #result = 'Normal (No pneumonia detected)'
    
    # Return the result back to the frontend as respnse
    return jsonify({'result': result}), 200

if __name__ == '__main__':
    app.run(debug=True)
