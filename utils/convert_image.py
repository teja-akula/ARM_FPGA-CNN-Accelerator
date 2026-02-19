#!/usr/bin/env python3
"""Convert image to C header - 64x64 for better detection quality"""
from PIL import Image
import numpy as np
import sys

def convert(image_path, output_path, size=64):
    img = Image.open(image_path).convert('RGB')
    orig_w, orig_h = img.size
    img = img.resize((size, size), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)
    
    name = image_path.replace('\\','/').split('/')[-1]
    
    with open(output_path, 'w') as f:
        f.write(f"/* Auto-generated from {name} ({orig_w}x{orig_h} -> {size}x{size}) */\n")
        f.write("#ifndef TEST_IMAGE_H\n#define TEST_IMAGE_H\n\n")
        f.write(f"#define IMG_WIDTH  {size}\n")
        f.write(f"#define IMG_HEIGHT {size}\n")
        f.write("#define IMG_CHANNELS 3\n\n")
        f.write(f"static const unsigned char TEST_IMAGE[{size*size*3}] = {{\n")
        
        flat = arr.flatten()
        for i in range(0, len(flat), 24):
            chunk = flat[i:i+24]
            f.write("    " + ",".join(f"{v:3d}" for v in chunk))
            if i + 24 < len(flat):
                f.write(",")
            f.write("\n")
        
        f.write("};\n\n#endif\n")
    
    print(f"Generated {output_path} ({name} {orig_w}x{orig_h} -> {size}x{size})")

if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else r"c:\Users\teja1\Desktop\Trail1\bus.jpg"
    out = sys.argv[2] if len(sys.argv) > 2 else r"c:\Users\teja1\Desktop\Trail1\cnn_accel_vitis\cnn_yolo_app\src\test_image.h"
    sz = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    convert(img, out, sz)
