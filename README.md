# FPGA-Accelerated CNN for Real-Time Object Detection

[![Platform](https://img.shields.io/badge/Platform-Zedboard%20Zynq--7000-blue)]()
[![Tool](https://img.shields.io/badge/Tool-Vivado%20%2B%20Vitis%202024.1-green)]()
[![Language](https://img.shields.io/badge/HLS-C%2FC%2B%2B-orange)]()

Hardware/software co-design implementation of a CNN-based object detection system on the Xilinx Zynq-7000 SoC (Zedboard). Demonstrates **2.07× speedup** by offloading convolutional layers to an HLS-generated FPGA accelerator, compared to ARM-only software execution.

---

## Performance Results

| Metric | ARM-Only (Cortex-A9) | FPGA Accelerated (PL) | Improvement |
|--------|---------------------|----------------------|-------------|
| **Latency** | 39,665 ms | 19,195 ms | **2.07×** |
| **Throughput** | 0.025 FPS | 0.052 FPS | **2.07×** |
| **Power** | — | — | — |
| **Efficiency** | 0.014 FPS/W | 0.021 FPS/W | **1.5×** |

### Per-Layer Breakdown

| Layer | Description | ARM (ms) | FPGA (ms) | Speedup |
|-------|-------------|----------|-----------|---------|
| L0 | Conv 3→16, 224×224 + BN + Pool | 3,049 | 1,574 | 1.9× |
| L1 | Conv 16→32, 112×112 + BN + Pool | 7,668 | 3,585 | 2.1× |
| L2 | Conv 32→64, 56×56 + BN + Pool | 7,556 | 3,519 | 2.1× |
| L3 | Conv 64→128, 28×28 + BN + Pool | 7,410 | 3,488 | 2.1× |
| L4 | Conv 128→256, 14×14 + BN + Pool | 7,164 | 3,469 | 2.1× |
| L5 | Conv 256→512, 7×7 + BN | 6,723 | 3,475 | 1.9× |
| L6 | Conv 512→24, 7×7 (1×1) | 95 | 72 | 1.3× |

---

## System Architecture

```
┌──────────────────── Zynq-7000 SoC ─────────────────────┐
│                                                        │
│  ┌──── PS (Processing System) ────┐                    │
│  │  ARM Cortex-A9 @ 667 MHz       │                    │
│  │  • Image preprocessing         │    512MB DDR3      │
│  │  • Layer sequencing (driver)   │◄──►(Shared)        │
│  │  • Post-processing (NMS)       │                    │
│  └──────────┬─────────────────────┘                    │
│             │ AXI Interconnect                         │
│  ┌──────────▼──── PL (FPGA Fabric) ──┐                 │
│  │  HLS CNN Accelerator @ 100 MHz    │                 │
│  │  • Conv2D Engine (3×3, 1×1)       │                 │
│  │  • 8 parallel MAC units (DSP48E1) │                 │
│  │  • BatchNorm + LeakyReLU          │                 │
│  │  • MaxPool 2×2                    │                 │
│  │  • Tiled processing via BRAM      │                 │
│  │  Resources: 220 DSP | 280KB BRAM  │                 │
│  └───────────────────────────────────┘                 │
└────────────────────────────────────────────────────────┘
```

### Two Hardware Designs

| Design | Block Diagram | Purpose |
|--------|--------------|---------|
| **ARM-Only** | PS → DDR + UART | Software baseline (no PL usage) |
| **FPGA Accelerated** | PS → AXI-Lite → CNN Accel IP → AXI HP0 → DDR | HW-accelerated inference |

---

## CNN Model: YOLO Lite (7-Layer)

Simplified YOLO-based architecture designed for Zynq-7000 resource constraints:

| Layer | Operation | Input | Output | Parameters |
|-------|-----------|-------|--------|------------|
| L0 | Conv(3→16, 3×3) + BN + ReLU + Pool | 224×224×3 | 112×112×16 | 448 |
| L1 | Conv(16→32, 3×3) + BN + ReLU + Pool | 112×112×16 | 56×56×32 | 4,640 |
| L2 | Conv(32→64, 3×3) + BN + ReLU + Pool | 56×56×32 | 28×28×64 | 18,496 |
| L3 | Conv(64→128, 3×3) + BN + ReLU + Pool | 28×28×64 | 14×14×128 | 73,856 |
| L4 | Conv(128→256, 3×3) + BN + ReLU + Pool | 14×14×128 | 7×7×256 | 295,168 |
| L5 | Conv(256→512, 3×3) + BN + ReLU | 7×7×256 | 7×7×512 | 1,180,160 |
| L6 | Conv(512→24, 1×1) + Bias | 7×7×512 | 7×7×24 | 12,312 |
| **Total** | | | | **~1.58M** |

Output: 7×7 grid × 3 anchors × (5 + 3 classes) → NMS → final detections

---

## Project Structure

```
├── hw/                              # Hardware designs
│   ├── arm_only/                    # Block Design 1: PS-only (baseline)
│   │   └── create_design.tcl        #   Vivado TCL: Zynq PS + DDR + UART
│   ├── fpga_accelerated/            # Block Design 2: PS + PL accelerator
│   │   └── create_design.tcl        #   Vivado TCL: PS + AXI + CNN Accel IP
│   └── hls/                         # HLS CNN Accelerator IP
│       ├── src/
│       │   ├── cnn_accel.cpp        #   Top-level accelerator
│       │   ├── cnn_accel.h          #   Data types (Q8.8 fixed-point)
│       │   ├── conv_layer.cpp       #   Convolution engine (8 MACs)
│       │   ├── activation.cpp       #   BatchNorm + LeakyReLU
│       │   └── pooling.cpp          #   MaxPool 2×2
│       ├── tb/
│       │   └── tb_cnn_accel.cpp     #   HLS testbench
│       ├── script.tcl               #   HLS synthesis script
│       └── export_ip.tcl            #   IP export script
│
├── sw/                              # Software (ARM firmware)
│   ├── arm_only/                    # Step 1: ARM-only inference
│   │   └── main.c                   #   All CNN layers on ARM Cortex-A9
│   ├── fpga_accelerated/            # Step 2: FPGA-accelerated inference
│   │   └── main.c                   #   Conv on PL, pre/post on ARM
│   └── common/                      # Shared source files
│       ├── cnn_driver.c/h           #   HLS accelerator AXI driver
│       ├── yolo_layers.h            #   ARM software conv2d, batchnorm, maxpool
│       ├── yolo_postprocess.c/h     #   YOLO decode + NMS
│       ├── image_preprocess.c/h     #   Image loading & bilinear resize
│       ├── tiny_yolo_weights.h      #   CNN weights (~1.58M params)
│       ├── test_image.h             #   Embedded test image (64×64)
│       └── lscript.ld               #   Linker script
│
├── utils/                           # Utility scripts
│   ├── convert_image.py             #   Image → C header converter
│   └── export_for_arm.py            #   Model export for ARM
│
├── README.md
└── .gitignore
```

---

## Key Optimizations

| Technique | Description | Impact |
|-----------|-------------|--------|
| **8-way MAC parallelism** | 8 DSP48E1 slices compute 8 output channels simultaneously | ~8× compute throughput |
| **Fixed-point Q8.8** | 16-bit arithmetic: 1 DSP per MAC vs 3-5 for float32 | 2× memory savings |
| **Tiled processing** | Feature maps divided into BRAM-sized tiles | Handles any size within 280KB |
| **Fused operations** | Conv+BN+ReLU+Pool in single pipeline pass | 4× fewer DDR accesses |
| **AXI burst transfers** | 256-word DDR bursts amortize memory latency | Reduced memory bottleneck |
| **HLS pipelining** | Fully pipelined datapath with II=1 | 1 result per cycle |
| **Ping-pong DDR buffers** | Alternate input/output addresses between layers | Zero-copy layer chaining |

---

## Build & Deploy

### 1. Build HLS IP
```bash
cd hw/hls
vivado_hls -f script.tcl      # Synthesize
vivado_hls -f export_ip.tcl    # Export IP
```

### 2. Build Hardware (Vivado)
```bash
# ARM-only baseline
cd hw/arm_only
vivado -mode batch -source create_design.tcl

# FPGA-accelerated
cd hw/fpga_accelerated
vivado -mode batch -source create_design.tcl
```

### 3. Build & Deploy Software (Vitis)
1. Open Vitis 2024.1
2. Create platform from `.xsa` file
3. Copy desired `sw/arm_only/main.c` or `sw/fpga_accelerated/main.c` to `src/`
4. Include all files from `sw/common/`
5. Build (Ctrl+B) → Deploy to Zedboard via JTAG

### 4. Monitor Results
```bash
python utils/uart_bridge.py COM7
```

---

## Hardware Requirements

- **Zedboard** (Xilinx Zynq-7000 XC7Z020-CLG484)
- USB-UART cable (micro-USB)

## Software Requirements

- Xilinx Vivado 2024.1
- Xilinx Vitis 2024.1
- Python 3.8+ (`pyserial` for UART monitoring)

---

## Resource Utilization (Zynq XC7Z020)

| Resource | Used | Available | Utilization |
|----------|------|-----------|-------------|
| DSP48E1 | ~200 | 220 | ~91% |
| BRAM (36Kb) | ~95 | 140 | ~68% |
| LUT | ~35,000 | 53,200 | ~66% |
| FF | ~28,000 | 106,400 | ~26% |

---

## Author

**Teja Akula** — [GitHub](https://github.com/teja-akula)
