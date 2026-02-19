"""
Export YOLO11n to ONNX format for ARM deployment via NCNN.
Run this on your PC to create the ONNX file.
"""

from ultralytics import YOLO
import os

def export_for_arm():
    print("=" * 60)
    print("YOLO11n Export for ARM Deployment")
    print("=" * 60)
    
    # Load the model
    print("\n[1] Loading YOLO11n model...")
    model = YOLO("yolo11n.pt")
    print("    ✓ Model loaded")
    
    # Export to ONNX (416x416 for ARM efficiency)
    print("\n[2] Exporting to ONNX format (416x416)...")
    export_path = model.export(
        format="onnx",
        imgsz=416,           # Smaller size for ARM
        simplify=True,       # Optimize graph
        opset=11,            # NCNN compatible opset
        dynamic=False,       # Fixed input size for ARM
    )
    print(f"    ✓ Exported to: {export_path}")
    
    # Verify the export
    print("\n[3] Verifying ONNX model...")
    import onnx
    onnx_model = onnx.load(export_path)
    onnx.checker.check_model(onnx_model)
    print("    ✓ ONNX model is valid")
    
    # Get model info
    print("\n[4] Model Information:")
    file_size = os.path.getsize(export_path) / (1024 * 1024)
    print(f"    File: {export_path}")
    print(f"    Size: {file_size:.2f} MB")
    print(f"    Input: 1x3x416x416 (BCHW)")
    
    # Get input/output names (needed for NCNN)
    print("\n[5] Layer Names (for NCNN):")
    for inp in onnx_model.graph.input:
        print(f"    Input: {inp.name}")
    for out in onnx_model.graph.output:
        print(f"    Output: {out.name}")
    
    print("\n" + "=" * 60)
    print("✅ Export Complete!")
    print("=" * 60)
    print("\nNext Steps:")
    print("  1. Install onnx2ncnn from https://github.com/Tencent/ncnn/releases")
    print("  2. Run: onnx2ncnn yolo11n.onnx yolo11n.param yolo11n.bin")
    print("  3. Copy .param and .bin files to Zedboard")
    
    return export_path


def test_onnx_inference():
    """Test that the ONNX model works correctly"""
    print("\n\n" + "=" * 60)
    print("Testing ONNX Inference")
    print("=" * 60)
    
    import onnxruntime as ort
    import numpy as np
    import cv2
    
    # Load ONNX model
    session = ort.InferenceSession("yolo11n.onnx")
    
    # Get input details
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    print(f"Input: {input_name}, Shape: {input_shape}")
    
    # Load and preprocess test image
    img = cv2.imread("bus.jpg")
    if img is None:
        print("No test image found, using random data")
        input_data = np.random.rand(1, 3, 416, 416).astype(np.float32)
    else:
        img_resized = cv2.resize(img, (416, 416))
        input_data = img_resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        input_data = np.expand_dims(input_data, axis=0)
    
    # Run inference
    import time
    start = time.time()
    outputs = session.run(None, {input_name: input_data})
    elapsed = time.time() - start
    
    print(f"Inference time: {elapsed*1000:.1f} ms")
    print(f"Output shapes: {[o.shape for o in outputs]}")
    print("✅ ONNX inference successful!")


if __name__ == "__main__":
    export_for_arm()
    test_onnx_inference()
