# ==============================================================================
# Script to start NVIDIA Triton Inference Server
# ==============================================================================

$ModelRepoPath = "$PSScriptRoot\model_repository"

Write-Host "Starting Triton Inference Server..." -ForegroundColor Cyan
Write-Host "Model Repository Path: $ModelRepoPath" 
Write-Host "Server will be available at: http://localhost:8000"
Write-Host "gRPC endpoint will be available at: localhost:8001"

# Run the Triton server docker container
docker run --gpus all --rm `
    -p 8000:8000 -p 8001:8001 -p 8002:8002 `
    -v "$($ModelRepoPath):/models" `
    nvcr.io/nvidia/tritonserver:23.10-py3 `
    tritonserver --model-repository=/models --strict-model-config=false

