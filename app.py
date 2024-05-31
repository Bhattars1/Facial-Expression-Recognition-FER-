# Importing necessary libraries
from flask import Flask, render_template, request
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Create a class of TinyVGG architecture
class TinyVGG(nn.Module):
    def __init__(self, num_classes=8):
        super(TinyVGG, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.classifier = nn.Sequential(
            nn.Linear(36864, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(128, num_classes))
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
# Load the weights and biases of the trained model
loaded_model = TinyVGG()
loaded_model.load_state_dict(torch.load('./model/FER_Model.pt'))

# Set the label and label index for prediction labels
labels_index={0:'Fear',
              1:'Disgust',
              2:'Surprised',
              3:'Contempt',
              4:'Angry',
              5:'Neutral',
              6:'Sad',
              7:'Happy'}

# Creating a Flask application instance
app = Flask(__name__)

# Defining a route for the root URL that responds to GET requests
@app.route('/', methods = ['GET'])

# Rendering the index.html template and returning it as the response
def FER():
    return render_template('index.html')

# Define the transformation and any preprocessing to ensure data consistency
transform = transforms.Compose([
    transforms.Resize((96,96)), 
    transforms.ToTensor()])

# Define another route for the root URL  that responds to POST requests
@app.route('/', methods=['POST'])

# Definining the predict function
def predict():
    imagefile = request.files['imagefile']
    image_path = './images/'+ imagefile.filename
    imagefile.save(image_path)

    # Doing some preprocessing with the image to make sure it is passed through model without error
    image = Image.open(image_path)
    image = transform(image)

    # Predicting the image that the user uploaded
    with torch.inference_mode():
        pred_logits = loaded_model(image.unsqueeze(dim=0))
        pred= torch.softmax(pred_logits, dim=1).argmax(dim=1)
        pred_label = labels_index[pred.item()]
    return render_template('index.html', prediction = pred_label)

# Run the Flask application when the script is executed
if __name__=='__main__':
    app.run(port=3000)