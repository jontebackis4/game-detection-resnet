import io
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

model = resnet18(weights=None)
class_names = [
    'brawlhalla', 'csgo', 'fortnite', 'leagueoflegends', 
    'overwatch', 'rainbowsix', 'rocketleague', 'valorant'
]
model.fc = torch.nn.Linear(model.fc.in_features, len(class_names))

model.load_state_dict(torch.load("fine_tuned_resnet18.pth"))
model.eval()

if torch.cuda.is_available():
    model.cuda()

def predict(image_bytes):
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    input_tensor = test_transform(image).unsqueeze(0)
    if torch.cuda.is_available():
        input_tensor = input_tensor.cuda()

    with torch.no_grad():
        output = model(input_tensor)
        _, prediction = torch.max(output.data, 1)

    return class_names[prediction.item()]

@app.route("/predict", methods=["POST"])
def predict_endpoint():
    if request.method == "POST":
        file = request.files.get("file")
        if file is None or file.filename == "":
            return jsonify({"error": "no file"}), 400

        try:
            img_bytes = file.read()
            class_name = predict(img_bytes)
            return jsonify({"class_name": class_name})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)