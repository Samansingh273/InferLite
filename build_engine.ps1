# ==============================================================================
# Script to build TensorRT engine from ONNX model
# ==============================================================================
$ErrorActionPreference = "Stop"

Write-Host "Starting TensorRT Engine Build..."

$ModelRepoPath = "$PSScriptRoot\model_repository\resnet50\1"
if (-not (Test-Path $ModelRepoPath)) {
    New-Item -ItemType Directory -Force -Path $ModelRepoPath | Out-Null
}

$OnnxFile = "$PSScriptRoot\resnet50.onnx"
if (-not (Test-Path $OnnxFile)) {
    Write-Host "Error: resnet50.onnx not found. Please run export_model.py first." -ForegroundColor Red
    exit 1
}

$EngineFile = "$ModelRepoPath\model.plan"

Write-Host "Using NVIDIA TensorRT docker container to build engine..."
Write-Host "This process may take a few minutes as it optimizes the neural network for your specific GPU architecture."

# We mount the current directory to /workspace in the container and use trtexec
docker run --rm --gpus all -v "$($PSScriptRoot):/workspace" nvcr.io/nvidia/tensorrt:23.10-py3 `
    trtexec --onnx=/workspace/resnet50.onnx `
            --saveEngine=/workspace/model_repository/resnet50/1/model.plan `
            --fp16 `
            --minShapes=input:1x3x224x224 `
            --optShapes=input:4x3x224x224 `
            --maxShapes=input:8x3x224x224

if ($LASTEXITCODE -eq 0) {
    Write-Host "Engine successfully built at: $EngineFile" -ForegroundColor Green
} else {
    Write-Host "Engine build failed with exit code $LASTEXITCODE" -ForegroundColor Red
}
