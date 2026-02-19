/*******************************************************************************
 * CNN Accelerator - Vitis HLS Implementation
 * Header file with types, constants, and interface definitions
 * 
 * Target: Zedboard (Zynq XC7Z020)
 * Reference: ZynqNet architecture
 ******************************************************************************/

#ifndef CNN_ACCEL_H
#define CNN_ACCEL_H

#include <ap_fixed.h>
#include <ap_int.h>
#include <hls_stream.h>

/*******************************************************************************
 * Fixed-Point Type Definitions
 ******************************************************************************/
// Q8.8 format: 8 integer bits, 8 fractional bits (16-bit total)
typedef ap_fixed<16, 8, AP_RND, AP_SAT> data_t;      // Activations
typedef ap_fixed<16, 8, AP_RND, AP_SAT> weight_t;    // Weights
typedef ap_fixed<32, 16, AP_RND, AP_SAT> acc_t;      // Accumulator (extra precision)

// Unsigned versions for indices
typedef ap_uint<8> channel_t;
typedef ap_uint<8> dim_t;

/*******************************************************************************
 * Network Configuration Constants
 ******************************************************************************/
// Maximum supported dimensions (for buffer sizing)
#define MAX_INPUT_SIZE      224
#define MAX_CHANNELS        512
#define MAX_KERNEL_SIZE     3
#define MAX_OUTPUT_CHANNELS 512

// YOLO Lite specific
#define INPUT_SIZE          224
#define NUM_CLASSES         3
#define NUM_ANCHORS         3

/*******************************************************************************
 * Hardware Design Parameters
 ******************************************************************************/
// Parallelism factors (adjust based on resource constraints)
#define PARALLEL_OUT_CH     8    // Process 8 output channels in parallel
#define PARALLEL_IN_CH      8    // Process 8 input channels in parallel
#define BURST_LENGTH        64   // AXI burst length

// Line buffer depth (for 3x3 convolution)
#define LINE_BUFFER_SIZE    (MAX_INPUT_SIZE * 2)  // 2 full lines

/*******************************************************************************
 * Layer Configuration Structure
 ******************************************************************************/
struct LayerConfig {
    ap_uint<8> in_channels;
    ap_uint<8> out_channels;
    ap_uint<8> in_height;
    ap_uint<8> in_width;
    ap_uint<8> kernel_size;    // 1 or 3
    ap_uint<8> stride;         // 1 or 2
    ap_uint<8> padding;        // 0 or 1
    ap_uint<1> use_relu;       // Apply LeakyReLU
    ap_uint<1> use_maxpool;    // Apply 2x2 maxpool
    ap_uint<1> use_batchnorm;  // Apply batch normalization
};

/*******************************************************************************
 * AXI Stream Data Types
 ******************************************************************************/
// 64-bit wide data bus (4x 16-bit values packed)
typedef ap_uint<64> axi_data_t;

// Stream packet with last signal for AXI-Stream
struct axis_packet {
    axi_data_t data;
    ap_uint<1> last;
};

/*******************************************************************************
 * Function Prototypes
 ******************************************************************************/

// Top-level accelerator function (exported as IP)
void cnn_accelerator_top(
    // AXI-Lite control interface
    ap_uint<32> *control,
    ap_uint<32> *status,
    
    // AXI Memory-Mapped interfaces for DDR access
    data_t *input_fm,         // Input feature map in DDR
    data_t *output_fm,        // Output feature map in DDR
    weight_t *weights,        // Convolution weights in DDR
    weight_t *bn_params,      // BatchNorm parameters in DDR
    
    // Layer configuration
    LayerConfig config
);

// Convolution core
void conv2d_hw(
    data_t *input,
    data_t *output,
    weight_t *weights,
    int in_ch, int in_h, int in_w,
    int out_ch, int kernel, int stride, int pad
);

// Batch normalization + LeakyReLU (fused)
// Uses pre-computed: scale = gamma / sqrt(var + eps)
//                    shift = beta - mean * scale
void batchnorm_relu_hw(
    data_t *data,
    weight_t *scale,    // Pre-computed: gamma / sqrt(var + eps)
    weight_t *shift,    // Pre-computed: beta - mean * scale
    int channels, int height, int width
);

// Max pooling 2x2
void maxpool2d_hw(
    data_t *input,
    data_t *output,
    int channels, int in_h, int in_w
);

// Activation functions
data_t leaky_relu_hw(data_t x);
data_t sigmoid_hw(data_t x);

/*******************************************************************************
 * Utility Functions
 ******************************************************************************/

// Fixed-point conversion
inline data_t float_to_fixed(float f) {
    return data_t(f);
}

inline float fixed_to_float(data_t d) {
    return d.to_float();
}

// Pack/unpack for AXI bus
inline axi_data_t pack_4x16(data_t d0, data_t d1, data_t d2, data_t d3) {
    axi_data_t packed;
    packed.range(15, 0) = d0.range(15, 0);
    packed.range(31, 16) = d1.range(15, 0);
    packed.range(47, 32) = d2.range(15, 0);
    packed.range(63, 48) = d3.range(15, 0);
    return packed;
}

inline void unpack_4x16(axi_data_t packed, data_t &d0, data_t &d1, data_t &d2, data_t &d3) {
    ap_uint<16> tmp0 = packed.range(15, 0);
    ap_uint<16> tmp1 = packed.range(31, 16);
    ap_uint<16> tmp2 = packed.range(47, 32);
    ap_uint<16> tmp3 = packed.range(63, 48);
    d0.range(15, 0) = tmp0;
    d1.range(15, 0) = tmp1;
    d2.range(15, 0) = tmp2;
    d3.range(15, 0) = tmp3;
}

#endif // CNN_ACCEL_H
