import argparse
import numpy as np
import tritonclient.http as httpclient
from PIL import Image
import json
import urllib.request

def load_imagenet_labels():
    # Helper function to get human-readable labels for ResNet50
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Warning: Could not download labels ({e}). Using indices instead.")
        return [f"Class {i}" for i in range(1000)]

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    
    # ResNet50 normalization (ImageNet standards: mean, std)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array / 255.0 - mean) / std
    
    # HWC to CHW format (e.g. 224x224x3 to 3x224x224)
    img_array = np.transpose(img_array, (2, 0, 1))
    
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def run_inference(image_path=None):
    print("Initializing Triton Client...")
    
    # Connect to the HTTP endpoint of Triton Server
    url = "localhost:8000"
    try:
        triton_client = httpclient.InferenceServerClient(url=url)
    except Exception as e:
        print("Client creation failed: " + str(e))
        return

    model_name = "resnet50"

    print(f"Checking if model '{model_name}' is ready...")
    if not triton_client.is_model_ready(model_name):
        print(f"Error: Model {model_name} is not ready on the server. Please check the server logs.")
        return
        
    if image_path:
        print(f"Loading and preprocessing image: {image_path}")
        input_data = preprocess_image(image_path)
    else:
        print("Generating trial data (random noise mimicking an image)...")
        # Generate dummy input data (Batch Size 1, 3 Channels, 224x224 Resolution)
        input_data = np.random.randn(1, 3, 224, 224).astype(np.float32)

    # Set up inputs and outputs matching config.pbtxt
    inputs = []
    inputs.append(httpclient.InferInput('input', [1, 3, 224, 224], "FP32"))
    inputs[0].set_data_from_numpy(input_data)

    outputs = []
    outputs.append(httpclient.InferRequestedOutput('output'))

    print("Sending inference request to Triton Server...")
    # Perform inference request
    results = triton_client.infer(model_name=model_name,
                                  inputs=inputs,
                                  outputs=outputs)

    # Process results
    out_data = results.as_numpy('output')
    print("Inference successful!")
    print(f"Server output shape: {out_data.shape}")
    
    # Softmax to get probabilities
    exp_out = np.exp(out_data - np.max(out_data, axis=1, keepdims=True))
    probs = exp_out / np.sum(exp_out, axis=1, keepdims=True)

    predicted_class = np.argmax(probs[0])
    confidence = probs[0][predicted_class] * 100
    
    labels = load_imagenet_labels()
    label = labels[predicted_class]
    
    print("\n========== INFERENCE RESULTS ==========")
    print(f"Predicted class index : {predicted_class}")
    print(f"Predicted item        : {label}")
    print(f"Confidence score      : {confidence:.2f}%")
    print("=======================================")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Triton Python Client for ResNet50")
    parser.add_argument("--image", type=str, help="Path to an input image", default=None)
    args = parser.parse_args()
    
    run_inference(args.image)
