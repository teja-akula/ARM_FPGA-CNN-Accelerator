/*******************************************************************************
 * CNN Accelerator - Convolution Layer
 * Optimized 3x3 and 1x1 convolution for HLS
 * 
 * Key optimizations:
 * - Line buffer for data reuse
 * - Parallel MAC array using DSP48
 * - Loop tiling for large feature maps
 ******************************************************************************/

#include "cnn_accel.h"

/*******************************************************************************
 * Line Buffer for 3x3 Convolution
 * Stores 3 rows to enable sliding window operation
 ******************************************************************************/
void line_buffer_3x3(
    data_t pixel_in,
    data_t window[3][3],
    data_t line_buf[2][MAX_INPUT_SIZE],
    int col,
    int width
) {
    #pragma HLS INLINE off
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    
    // Shift window horizontally
    SHIFT_WINDOW:
    for (int i = 0; i < 3; i++) {
        #pragma HLS UNROLL
        window[i][0] = window[i][1];
        window[i][1] = window[i][2];
    }
    
    // Load new column from line buffers
    window[0][2] = line_buf[0][col];
    window[1][2] = line_buf[1][col];
    window[2][2] = pixel_in;
    
    // Shift line buffers
    line_buf[0][col] = line_buf[1][col];
    line_buf[1][col] = pixel_in;
}

/*******************************************************************************
 * Multiply-Accumulate Unit
 * Single MAC operation optimized for DSP48
 ******************************************************************************/
acc_t mac_unit(data_t activation, weight_t weight, acc_t acc) {
    #pragma HLS INLINE
    
    acc_t product = activation * weight;
    return acc + product;
}

/*******************************************************************************
 * 3x3 Convolution Kernel
 * Computes one output pixel using 3x3 window
 ******************************************************************************/
acc_t conv_3x3_kernel(
    data_t window[3][3],
    weight_t kernel[3][3]
) {
    #pragma HLS INLINE
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0
    #pragma HLS ARRAY_PARTITION variable=kernel complete dim=0
    
    acc_t sum = 0;
    
    CONV_KH:
    for (int kh = 0; kh < 3; kh++) {
        #pragma HLS UNROLL
        CONV_KW:
        for (int kw = 0; kw < 3; kw++) {
            #pragma HLS UNROLL
            sum = mac_unit(window[kh][kw], kernel[kh][kw], sum);
        }
    }
    
    return sum;
}

/*******************************************************************************
 * 1x1 Convolution (Pointwise)
 * Used for channel reduction/expansion
 ******************************************************************************/
void conv1x1_channel(
    data_t input[MAX_CHANNELS],
    weight_t weights[MAX_CHANNELS],
    acc_t &output,
    int in_channels
) {
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1
    
    acc_t sum = 0;
    
    CONV1X1_IC:
    for (int ic = 0; ic < in_channels; ic++) {
        #pragma HLS LOOP_TRIPCOUNT min=3 max=512
        #pragma HLS PIPELINE II=1
        sum = mac_unit(input[ic], weights[ic], sum);
    }
    
    output = sum;
}

/*******************************************************************************
 * 2D Convolution - Main Implementation
 * Supports 1x1 and 3x3 kernels with padding and stride
 ******************************************************************************/
void conv2d_hw(
    data_t *input,
    data_t *output,
    weight_t *weights,
    int in_ch, int in_h, int in_w,
    int out_ch, int kernel, int stride, int pad
) {
    #pragma HLS INLINE off
    
    // Calculate output dimensions
    int out_h = (in_h + 2 * pad - kernel) / stride + 1;
    int out_w = (in_w + 2 * pad - kernel) / stride + 1;
    
    // Local buffers
    static data_t line_buf[PARALLEL_IN_CH][2][MAX_INPUT_SIZE];
    static data_t window[PARALLEL_IN_CH][3][3];
    static weight_t weight_buf[PARALLEL_OUT_CH][PARALLEL_IN_CH][3][3];
    static acc_t acc_buf[PARALLEL_OUT_CH];
    
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=1
    #pragma HLS ARRAY_PARTITION variable=line_buf complete dim=2
    #pragma HLS ARRAY_PARTITION variable=window complete dim=0
    #pragma HLS ARRAY_PARTITION variable=weight_buf complete dim=0
    #pragma HLS ARRAY_PARTITION variable=acc_buf complete dim=1
    
    // Process output channels in tiles
    OC_TILE:
    for (int oc_tile = 0; oc_tile < out_ch; oc_tile += PARALLEL_OUT_CH) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=64
        
        int oc_limit = (oc_tile + PARALLEL_OUT_CH > out_ch) ? out_ch : oc_tile + PARALLEL_OUT_CH;
        
        // Process input channels in tiles
        IC_TILE:
        for (int ic_tile = 0; ic_tile < in_ch; ic_tile += PARALLEL_IN_CH) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=64
            
            int ic_limit = (ic_tile + PARALLEL_IN_CH > in_ch) ? in_ch : ic_tile + PARALLEL_IN_CH;
            
            // Load weights for this tile
            LOAD_WEIGHTS:
            for (int oc = 0; oc < PARALLEL_OUT_CH && (oc_tile + oc) < out_ch; oc++) {
                for (int ic = 0; ic < PARALLEL_IN_CH && (ic_tile + ic) < in_ch; ic++) {
                    #pragma HLS PIPELINE II=1
                    for (int kh = 0; kh < kernel; kh++) {
                        for (int kw = 0; kw < kernel; kw++) {
                            int w_idx = (oc_tile + oc) * in_ch * kernel * kernel +
                                       (ic_tile + ic) * kernel * kernel +
                                       kh * kernel + kw;
                            weight_buf[oc][ic][kh][kw] = weights[w_idx];
                        }
                    }
                }
            }
            
            // Process spatial dimensions
            ROW_LOOP:
            for (int oh = 0; oh < out_h; oh++) {
                #pragma HLS LOOP_TRIPCOUNT min=7 max=224
                
                COL_LOOP:
                for (int ow = 0; ow < out_w; ow++) {
                    #pragma HLS LOOP_TRIPCOUNT min=7 max=224
                    #pragma HLS PIPELINE II=1
                    
                    // Initialize accumulators (first input channel tile only)
                    if (ic_tile == 0) {
                        INIT_ACC:
                        for (int oc = 0; oc < PARALLEL_OUT_CH; oc++) {
                            #pragma HLS UNROLL
                            acc_buf[oc] = 0;
                        }
                    }
                    
                    // Compute for each input channel in tile
                    IC_COMPUTE:
                    for (int ic = 0; ic < PARALLEL_IN_CH && (ic_tile + ic) < in_ch; ic++) {
                        #pragma HLS UNROLL
                        
                        // Load 3x3 window (with padding handling)
                        data_t local_window[3][3];
                        #pragma HLS ARRAY_PARTITION variable=local_window complete dim=0
                        
                        LOAD_WINDOW:
                        for (int kh = 0; kh < kernel; kh++) {
                            #pragma HLS UNROLL
                            for (int kw = 0; kw < kernel; kw++) {
                                #pragma HLS UNROLL
                                int ih = oh * stride + kh - pad;
                                int iw = ow * stride + kw - pad;
                                
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int in_idx = (ic_tile + ic) * in_h * in_w + ih * in_w + iw;
                                    local_window[kh][kw] = input[in_idx];
                                } else {
                                    local_window[kh][kw] = 0;  // Zero padding
                                }
                            }
                        }
                        
                        // Accumulate across output channels
                        OC_MAC:
                        for (int oc = 0; oc < PARALLEL_OUT_CH && (oc_tile + oc) < out_ch; oc++) {
                            #pragma HLS UNROLL
                            acc_t partial = conv_3x3_kernel(local_window, weight_buf[oc][ic]);
                            acc_buf[oc] += partial;
                        }
                    }
                    
                    // Write output (last input channel tile only)
                    if (ic_tile + PARALLEL_IN_CH >= in_ch) {
                        WRITE_OUTPUT:
                        for (int oc = 0; oc < PARALLEL_OUT_CH && (oc_tile + oc) < out_ch; oc++) {
                            #pragma HLS UNROLL
                            int out_idx = (oc_tile + oc) * out_h * out_w + oh * out_w + ow;
                            output[out_idx] = data_t(acc_buf[oc]);
                        }
                    }
                }
            }
        }
    }
}
