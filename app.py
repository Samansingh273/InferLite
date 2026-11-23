import streamlit as st
import numpy as np
import tritonclient.http as httpclient
from PIL import Image
import json
import urllib.request

# Page config
st.set_page_config(page_title="InferLite - ResNet50 Classifier", page_icon="🚀", layout="centered")

@st.cache_data
def load_imagenet_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    try:
        with urllib.request.urlopen(url) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        return [f"Class {i}" for i in range(1000)]

def preprocess_image(img):
    img = img.convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array / 255.0 - mean) / std
    img_array = np.transpose(img_array, (2, 0, 1))
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def run_inference(img):
    url = "localhost:8000"
    try:
        triton_client = httpclient.InferenceServerClient(url=url)
    except Exception as e:
        return None, None, f"Could not connect to Triton Server: {e}"

    if not triton_client.is_model_ready("resnet50"):
        return None, None, "ResNet50 model is not ready on the server. Please check the server logs."
        
    input_data = preprocess_image(img)
    inputs = [httpclient.InferInput('input', [1, 3, 224, 224], "FP32")]
    inputs[0].set_data_from_numpy(input_data)
    outputs = [httpclient.InferRequestedOutput('output')]

    try:
        results = triton_client.infer(model_name="resnet50", inputs=inputs, outputs=outputs)
    except Exception as e:
        return None, None, f"Inference failed: {e}"

    out_data = results.as_numpy('output')
    exp_out = np.exp(out_data - np.max(out_data, axis=1, keepdims=True))
    probs = exp_out / np.sum(exp_out, axis=1, keepdims=True)
    
    # Get top 3 predictions
    top3_idx = np.argsort(probs[0])[-3:][::-1]
    top3_probs = probs[0][top3_idx] * 100
    
    return top3_idx, top3_probs, None

# --- UI Layout ---
st.title("🚀 InferLite Image Classifier")
st.markdown("Upload an image to classify it using the **ResNet50** model powered by **Triton Inference Server**.")

uploaded_file = st.file_uploader("Choose an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_container_width=True)
    
    # Trigger classification
    if st.button("🔍 Classify Image", type="primary", use_container_width=True):
        with st.spinner("Analyzing image with Triton Server..."):
            top3_idx, top3_probs, error = run_inference(img)
            
        if error:
            st.error(error)
        else:
            labels = load_imagenet_labels()
            st.success("Inference completed successfully!")
            
            # Show top prediction prominently
            st.subheader("Top Prediction")
            top_label = labels[top3_idx[0]].capitalize()
            top_conf = float(top3_probs[0])
            st.info(f"**{top_label}** — {top_conf:.2f}% Confidence")
            st.progress(top_conf / 100)
            
            # Show runner-up predictions
            st.markdown("### Other Probable Classes")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"1. **{labels[top3_idx[1]].capitalize()}** ({top3_probs[1]:.2f}%)")
            with col2:
                st.write(f"2. **{labels[top3_idx[2]].capitalize()}** ({top3_probs[2]:.2f}%)")
