import torch
import torchvision.models as models
import os

def export_onnx():
    print("Loading pre-trained ResNet50 model...")
    # Using ResNet50 as it's an industry standard for inference benchmarks
    model = models.resnet50(pretrained=True)
    model.eval()

    # Create dummy input that matches ResNet50 input dimensions
    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = "resnet50.onnx"

    print("Exporting model to ONNX format...")
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=13,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        # Dynamic batching allows us to send multiple images at once to Triton
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Success! Model exported to {os.path.abspath(onnx_path)}")

if __name__ == "__main__":
    export_onnx()
