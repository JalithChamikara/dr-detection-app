"""
Export best_dr_model.pth to ONNX format for Cloudflare Workers / Browser In-Client Inference.
Run:
    python export_onnx.py
Output:
    DrAppCF/public/model.onnx
"""
import os
import torch
import dr_utils as U

MODEL_PATH = os.environ.get('DR_MODEL', 'best_dr_model.pth')
OUTPUT_PATH = os.path.join('DrAppCF', 'public', 'model.onnx')

print(f"Loading checkpoint from: {MODEL_PATH}...")
model, ck = U.load_checkpoint(MODEL_PATH, device='cpu')
model.eval()

img_size = ck.get('img_size', 300)
print(f"Model architecture: {ck.get('arch')}, Input size: {img_size}x{img_size}")

dummy_input = torch.randn(1, 3, img_size, img_size, dtype=torch.float32)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
print(f"Exporting ONNX model to {OUTPUT_PATH}...")

torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_PATH,
    export_params=True,
    opset_version=14,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['logits'],
    dynamic_axes={
        'input': {0: 'batch_size'},
        'logits': {0: 'batch_size'}
    }
)

size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
print(f" Successfully exported {OUTPUT_PATH} ({size_mb:.2f} MB)")
print("You can now deploy with: npx wrangler deploy (inside DrAppCF folder)")
