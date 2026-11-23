# InferLite: Accelerated Neural Network Inference Using TensorRT & Triton Server

Welcome to **InferLite**. This project provides a full end-to-end implementation for optimizing a PyTorch model and serving it using NVIDIA's industry-standard Triton Inference Server with TensorRT for maximum acceleration.

## Prerequisites
- **NVIDIA GPU** installed on your system.
- **Docker & Docker Desktop** installed and configured for WSL2 (Windows Subsystem for Linux), since we are running NVIDIA containers on Windows.
- **Python 3.9+**

## Project Structure
- `export_model.py`: Script to download a standard ResNet50 model and export it to an interoperable ONNX format.
- `build_engine.ps1`: Powershell script that leverages NVIDIA's TensorRT Docker container to compile the ONNX model into a highly optimized TensorRT Engine (Plan file).
- `start_server.ps1`: Powershell script to deploy the Triton server Docker container.
- `client.py`: Python client application that sends an inference request over HTTP to the running Triton server.
- `model_repository/`: Directory structure expected by Triton to serve the model.

## Step-by-Step Execution Guide

### Step 1: Environment Setup
First, prepare your Python environment and install the required dependencies:
```powershell
# We recommend creating a virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Step 2: Export to ONNX
Export the PyTorch model to `resnet50.onnx`.
```powershell
python export_model.py
```
*Depending on your internet connection, this might take a moment to download the pre-trained weights from PyTorch hub.*

### Step 3: Build the TensorRT Engine
Compile the ONNX model. **This requires Docker to be running.**
```powershell
.\build_engine.ps1
```
*Note: The first time you run this, Docker will download the TensorRT container. The compilation process itself can take between 1-5 minutes depending on your GPU architecture.*

### Step 4: Start the Triton Inference Server
Once the engine is successfully built in `model_repository/resnet50/1/model.plan`, you can boot up the inference server:
```powershell
.\start_server.ps1
```
*Keep this terminal window open. Wait until the server logs indicate `Started HTTPService at 0.0.0.0:8000`.*

### Step 5: Run the Inference Client
Open a **new** terminal window (activate your python environment if needed) and run the client script to request predictions. You can run the client with a specific image:
```powershell
python client.py --image sample_dog.jpg
```

Or you can run it without any arguments to generate dummy random noise data:
```powershell
python client.py
```

The client will interact with the Triton Server and print the classification results and confidence scores!

### Step 6 (optional): Run the Web App
With the Triton server running, launch the Streamlit interface to upload an image and see the top-3 predictions:
```powershell
streamlit run app.py
```

## Notes
- **Current server configuration:** `model_repository/resnet50/config.pbtxt` serves the exported ONNX model (`model.onnx`) through Triton's ONNX Runtime backend, with dynamic batching enabled (`max_batch_size: 8`, preferred batch sizes 4 and 8). Copy `resnet50.onnx` to `model_repository/resnet50/1/model.onnx` before starting the server.
- **TensorRT path:** `build_engine.ps1` compiles the ONNX model into a TensorRT engine (`model.plan`) with FP16 precision and batch sizes 1–8. To serve the TensorRT engine instead, set `platform: "tensorrt_plan"` and `kind: KIND_GPU` in `config.pbtxt`.
- Large model files (`*.onnx`, `*.plan`) and the virtual environment are not included in this repository; regenerate them with the steps above.
