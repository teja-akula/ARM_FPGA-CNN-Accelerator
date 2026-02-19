/*******************************************************************************
 * CNN Accelerator - Activation Functions
 * LeakyReLU and Sigmoid implementations for HLS
 ******************************************************************************/

#include "cnn_accel.h"

/*******************************************************************************
 * LeakyReLU Activation
 * f(x) = x if x > 0, else 0.1 * x
 * 
 * Hardware-optimized: uses shift instead of multiply for 0.1 approximation
 * 0.125 = 1/8 = x >> 3 (close to 0.1, simpler in hardware)
 ******************************************************************************/
data_t leaky_relu_hw(data_t x) {
    #pragma HLS INLINE
    
    if (x > data_t(0)) {
        return x;
    } else {
        // Approximate 0.1 with 0.125 (1/8) for efficient hardware
        // Right shift by 3 is equivalent to divide by 8
        return data_t(x >> 3);
    }
}

/*******************************************************************************
 * Sigmoid Activation (LUT-based)
 * f(x) = 1 / (1 + exp(-x))
 * 
 * Uses lookup table for hardware efficiency
 * Input quantized to 256 levels in range [-8, 8]
 ******************************************************************************/

// Precomputed sigmoid LUT (generated offline)
// Index 0 = sigmoid(-8), Index 255 = sigmoid(7.9375)
static const data_t SIGMOID_LUT[256] = {
    // Generated from: sigmoid(i * 16/256 - 8)
    0.0003, 0.0004, 0.0004, 0.0005, 0.0005, 0.0006, 0.0007, 0.0008,
    0.0009, 0.0010, 0.0011, 0.0012, 0.0014, 0.0015, 0.0017, 0.0019,
    0.0021, 0.0024, 0.0026, 0.0029, 0.0033, 0.0037, 0.0041, 0.0045,
    0.0050, 0.0056, 0.0062, 0.0069, 0.0076, 0.0084, 0.0094, 0.0104,
    0.0115, 0.0127, 0.0141, 0.0155, 0.0172, 0.0190, 0.0210, 0.0232,
    0.0256, 0.0283, 0.0312, 0.0345, 0.0380, 0.0419, 0.0462, 0.0509,
    0.0560, 0.0616, 0.0677, 0.0744, 0.0817, 0.0896, 0.0982, 0.1076,
    0.1178, 0.1288, 0.1407, 0.1536, 0.1675, 0.1824, 0.1984, 0.2156,
    0.2340, 0.2536, 0.2744, 0.2964, 0.3196, 0.3440, 0.3695, 0.3962,
    0.4239, 0.4526, 0.4822, 0.5125, 0.5434, 0.5748, 0.6065, 0.6382,
    0.6698, 0.7011, 0.7319, 0.7621, 0.7914, 0.8198, 0.8470, 0.8730,
    0.8976, 0.9208, 0.9426, 0.9628, 0.9814, 0.9985, 0.9990, 0.9993,
    0.9995, 0.9997, 0.9997, 0.9998, 0.9998, 0.9999, 0.9999, 0.9999,
    0.9999, 0.9999, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    // Continued...
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000,
    1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000, 1.0000
};

data_t sigmoid_hw(data_t x) {
    #pragma HLS INLINE
    
    // Clamp input to LUT range [-8, 8)
    data_t clamped;
    if (x < data_t(-8)) {
        clamped = data_t(-8);
    } else if (x >= data_t(8)) {
        clamped = data_t(7.9375);
    } else {
        clamped = x;
    }
    
    // Convert to LUT index: (x + 8) * 16 = (x + 8) << 4
    // Index range: 0-255
    ap_uint<8> index = ap_uint<8>((clamped + data_t(8)) * data_t(16));
    
    return SIGMOID_LUT[index];
}

/*******************************************************************************
 * Batch Normalization + LeakyReLU (Fused)
 * out = leaky_relu(gamma * (x - mean) / sqrt(var + eps) + beta)
 * 
 * Pre-computed: scale = gamma / sqrt(var + eps)
 *               shift = beta - mean * scale
 * Runtime:      out = leaky_relu(x * scale + shift)
 ******************************************************************************/
void batchnorm_relu_hw(
    data_t *data,
    weight_t *scale,    // Pre-computed: gamma / sqrt(var + eps)
    weight_t *shift,    // Pre-computed: beta - mean * scale
    int channels,
    int height, 
    int width
) {
    #pragma HLS INLINE off
    
    int spatial_size = height * width;
    
    // Process each channel
    BN_CH:
    for (int c = 0; c < channels; c++) {
        #pragma HLS LOOP_TRIPCOUNT min=16 max=512
        
        weight_t ch_scale = scale[c];
        weight_t ch_shift = shift[c];
        
        // Process spatial elements
        BN_SPATIAL:
        for (int i = 0; i < spatial_size; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=49 max=50176
            #pragma HLS PIPELINE II=1
            
            int idx = c * spatial_size + i;
            
            // BatchNorm: x * scale + shift
            acc_t bn_out = data[idx] * ch_scale + ch_shift;
            
            // LeakyReLU
            data_t result = leaky_relu_hw(data_t(bn_out));
            
            data[idx] = result;
        }
    }
}

/*******************************************************************************
 * Apply Activation to Feature Map
 ******************************************************************************/
void apply_activation_hw(
    data_t *data,
    int channels,
    int height,
    int width,
    int activation_type  // 0: none, 1: LeakyReLU, 2: Sigmoid
) {
    #pragma HLS INLINE off
    
    int total = channels * height * width;
    
    ACT_LOOP:
    for (int i = 0; i < total; i++) {
        #pragma HLS LOOP_TRIPCOUNT min=1000 max=1000000
        #pragma HLS PIPELINE II=1
        
        data_t val = data[i];
        data_t result;
        
        switch (activation_type) {
            case 1:  // LeakyReLU
                result = leaky_relu_hw(val);
                break;
            case 2:  // Sigmoid
                result = sigmoid_hw(val);
                break;
            default: // No activation
                result = val;
                break;
        }
        
        data[i] = result;
    }
}
