"""
export_onnx.py
--------------
Converts the trained BoardEvaluator PyTorch model into ONNX format so it can
be imported into Unity via Unity Sentis.

Usage:
    python export_onnx.py
"""

import os
import sys

# Ensure stdout / stderr use UTF-8 on Windows so PyTorch's log messages
# (which may contain Unicode characters) don't cause encoding errors.
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import torch
from NeuralNet import BoardEvaluator
from ConfigLoader import load_config

# ---------------------------------------------------------------------------
# 1. Load configuration
# ---------------------------------------------------------------------------
config = load_config()
checkpoint_path = config['paths']['model_checkpoint']   # e.g. "ai_brain.pth"
output_path = "ai_brain.onnx"

# ---------------------------------------------------------------------------
# 2. Load model weights
# ---------------------------------------------------------------------------
print(f"[INFO] Loading model weights from '{checkpoint_path}' ...")

try:
    model = BoardEvaluator()
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
except FileNotFoundError:
    print(f"[ERROR] Checkpoint file '{checkpoint_path}' not found.")
    print("[ERROR] Please train the model first (Train_Model.py) before exporting.")
    raise SystemExit(1)

print("[INFO] Model weights loaded successfully.")

# ---------------------------------------------------------------------------
# 3. Switch to evaluation mode
# ---------------------------------------------------------------------------
model.eval()
print("[INFO] Model set to evaluation mode.")

# ---------------------------------------------------------------------------
# 4. Create a representative dummy input  (batch=1, flattened 7x7 board = 49)
# ---------------------------------------------------------------------------
dummy_input = torch.randn(1, 49)
print(f"[INFO] Dummy input tensor created with shape: {list(dummy_input.shape)}")

# ---------------------------------------------------------------------------
# 5. Export to ONNX
# ---------------------------------------------------------------------------
print(f"[INFO] Exporting model to '{output_path}' (opset 12) ...")

torch.onnx.export(
    model,                          # model to export
    dummy_input,                    # representative input
    output_path,                    # destination file
    export_params=True,             # store trained weights inside the file
    opset_version=12,               # ONNX opset — Unity Sentis supports 12+
    do_constant_folding=True,       # fold constants for a smaller/faster graph
    input_names=['board_state'],    # name visible in Unity Sentis
    output_names=['score'],         # name visible in Unity Sentis
    dynamic_axes={
        'board_state': {0: 'batch_size'},   # allow variable batch dimension
        'score':       {0: 'batch_size'},
    },
    dynamo=False,                   # use the stable TorchScript-based exporter
                                    # (required for opset 12 + Unity Sentis)
)

print(f"[SUCCESS] ONNX model exported successfully to '{output_path}'.")
print("[INFO] You can now import 'ai_brain.onnx' into your Unity project via Unity Sentis.")
