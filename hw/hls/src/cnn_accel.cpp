/*******************************************************************************
 * CNN Accelerator - Top Level (Optimized for BRAM)
 * Vitis HLS IP with AXI interfaces
 * 
 * This version uses DDR for feature map storage with small on-chip tile buffers
 * to fit within the Zedboard's limited BRAM resources.
 * 
 * Memory Strategy:
 * - Feature maps: Stored in DDR, processed tile-by-tile
 * - Weights: Streamed from DDR per layer
 * - On-chip: Only tile buffers (small) and line buffers for conv
 ******************************************************************************/

#include "cnn_accel.h"

// Optimized tile sizes to fit in BRAM
// Each tile: TILE_H x TILE_W x TILE_CH = 14 x 14 x 32 = 6272 elements = 12.5 KB
#define TILE_H      14
#define TILE_W      14
#define TILE_CH     32

// Small on-chip buffers (total ~100 KB BRAM)
#define INPUT_TILE_SIZE   (TILE_CH * (TILE_H + 2) * (TILE_W + 2))  // With padding
#define OUTPUT_TILE_SIZE  (TILE_CH * TILE_H * TILE_W)
#define WEIGHT_BUF_SIZE   (TILE_CH * TILE_CH * 9)  // 3x3 kernel

/*******************************************************************************
 * Top-Level Accelerator Function (Optimized)
 * Processes one layer at a time with DDR-based feature maps
 ******************************************************************************/
void cnn_accelerator_top(
    // Control/Status (memory-mapped)
    volatile ap_uint<32> *control,
    volatile ap_uint<32> *status,
    
    // Feature maps in DDR (memory-mapped)
    data_t *input_fm,
    data_t *output_fm,
    
    // Weights in DDR (memory-mapped)
    weight_t *weights,
    weight_t *bn_scale,
    weight_t *bn_shift,
    
    // Layer configuration
    int layer_type,       // 0: conv+bn+relu, 1: conv+bn+relu+pool, 2: conv only
    int in_channels,
    int out_channels,
    int in_height,
    int in_width,
    int kernel_size,
    int stride,
    int padding
) {
    // AXI Interface Pragmas
    #pragma HLS INTERFACE s_axilite port=return bundle=control
    #pragma HLS INTERFACE s_axilite port=control bundle=control
    #pragma HLS INTERFACE s_axilite port=status bundle=control
    #pragma HLS INTERFACE s_axilite port=layer_type bundle=control
    #pragma HLS INTERFACE s_axilite port=in_channels bundle=control
    #pragma HLS INTERFACE s_axilite port=out_channels bundle=control
    #pragma HLS INTERFACE s_axilite port=in_height bundle=control
    #pragma HLS INTERFACE s_axilite port=in_width bundle=control
    #pragma HLS INTERFACE s_axilite port=kernel_size bundle=control
    #pragma HLS INTERFACE s_axilite port=stride bundle=control
    #pragma HLS INTERFACE s_axilite port=padding bundle=control
    
    // AXI Master interfaces for DDR access with reasonable depths
    #pragma HLS INTERFACE m_axi port=input_fm offset=slave bundle=gmem0 depth=150528 max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=output_fm offset=slave bundle=gmem1 depth=150528 max_write_burst_length=64
    #pragma HLS INTERFACE m_axi port=weights offset=slave bundle=gmem2 depth=18432 max_read_burst_length=64
    #pragma HLS INTERFACE m_axi port=bn_scale offset=slave bundle=gmem3 depth=512
    #pragma HLS INTERFACE m_axi port=bn_shift offset=slave bundle=gmem3 depth=512
    
    // Small on-chip tile buffers (fits in BRAM)
    static data_t input_tile[INPUT_TILE_SIZE];
    static data_t output_tile[OUTPUT_TILE_SIZE];
    static weight_t weight_tile[WEIGHT_BUF_SIZE];
    static acc_t acc_tile[OUTPUT_TILE_SIZE];
    
    #pragma HLS BIND_STORAGE variable=input_tile type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=output_tile type=ram_2p impl=bram
    #pragma HLS BIND_STORAGE variable=weight_tile type=ram_1p impl=bram
    #pragma HLS BIND_STORAGE variable=acc_tile type=ram_2p impl=bram
    
    // Calculate output dimensions
    int out_height = (in_height + 2 * padding - kernel_size) / stride + 1;
    int out_width = (in_width + 2 * padding - kernel_size) / stride + 1;
    
    // Signal processing start
    *status = 1;  // Running
    
    // Process output channels in tiles
    int oc_tiles = (out_channels + TILE_CH - 1) / TILE_CH;
    int ic_tiles = (in_channels + TILE_CH - 1) / TILE_CH;
    int oh_tiles = (out_height + TILE_H - 1) / TILE_H;
    int ow_tiles = (out_width + TILE_W - 1) / TILE_W;
    
    // Main processing loop - tile by tile
    OC_TILE_LOOP:
    for (int oc_t = 0; oc_t < oc_tiles; oc_t++) {
        #pragma HLS LOOP_TRIPCOUNT min=1 max=16
        
        int oc_start = oc_t * TILE_CH;
        int oc_end = (oc_start + TILE_CH > out_channels) ? out_channels : oc_start + TILE_CH;
        int oc_count = oc_end - oc_start;
        
        // Process spatial tiles
        OH_TILE_LOOP:
        for (int oh_t = 0; oh_t < oh_tiles; oh_t++) {
            #pragma HLS LOOP_TRIPCOUNT min=1 max=16
            
            int oh_start = oh_t * TILE_H;
            int oh_end = (oh_start + TILE_H > out_height) ? out_height : oh_start + TILE_H;
            
            OW_TILE_LOOP:
            for (int ow_t = 0; ow_t < ow_tiles; ow_t++) {
                #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                
                int ow_start = ow_t * TILE_W;
                int ow_end = (ow_start + TILE_W > out_width) ? out_width : ow_start + TILE_W;
                
                // Initialize accumulators for this tile
                int tile_size = oc_count * (oh_end - oh_start) * (ow_end - ow_start);
                INIT_ACC:
                for (int i = 0; i < tile_size; i++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=6272
                    acc_tile[i] = 0;
                }
                
                // Accumulate over input channel tiles
                IC_TILE_LOOP:
                for (int ic_t = 0; ic_t < ic_tiles; ic_t++) {
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=16
                    
                    int ic_start = ic_t * TILE_CH;
                    int ic_end = (ic_start + TILE_CH > in_channels) ? in_channels : ic_start + TILE_CH;
                    int ic_count = ic_end - ic_start;
                    
                    // Load input tile from DDR (with halo for convolution)
                    int ih_start = oh_start * stride - padding;
                    int ih_end = (oh_end - 1) * stride + kernel_size - padding;
                    int iw_start = ow_start * stride - padding;
                    int iw_end = (ow_end - 1) * stride + kernel_size - padding;
                    
                    // Load input tile
                    int tile_idx = 0;
                    LOAD_INPUT:
                    for (int ic = ic_start; ic < ic_end; ic++) {
                        for (int ih = ih_start; ih < ih_end; ih++) {
                            for (int iw = iw_start; iw < iw_end; iw++) {
                                #pragma HLS PIPELINE II=1
                                #pragma HLS LOOP_TRIPCOUNT min=1 max=8192
                                
                                if (ih >= 0 && ih < in_height && iw >= 0 && iw < in_width) {
                                    int ddr_idx = ic * in_height * in_width + ih * in_width + iw;
                                    input_tile[tile_idx] = input_fm[ddr_idx];
                                } else {
                                    input_tile[tile_idx] = 0;  // Zero padding
                                }
                                tile_idx++;
                            }
                        }
                    }
                    
                    // Load weights for this tile pair
                    LOAD_WEIGHTS:
                    for (int oc = oc_start; oc < oc_end; oc++) {
                        for (int ic = ic_start; ic < ic_end; ic++) {
                            for (int kh = 0; kh < kernel_size; kh++) {
                                for (int kw = 0; kw < kernel_size; kw++) {
                                    #pragma HLS PIPELINE II=1
                                    #pragma HLS LOOP_TRIPCOUNT min=1 max=9216
                                    
                                    int w_ddr_idx = oc * in_channels * kernel_size * kernel_size +
                                                   ic * kernel_size * kernel_size +
                                                   kh * kernel_size + kw;
                                    int w_tile_idx = (oc - oc_start) * ic_count * kernel_size * kernel_size +
                                                    (ic - ic_start) * kernel_size * kernel_size +
                                                    kh * kernel_size + kw;
                                    weight_tile[w_tile_idx] = weights[w_ddr_idx];
                                }
                            }
                        }
                    }
                    
                    // Compute convolution for this tile
                    int tile_ih = ih_end - ih_start;
                    int tile_iw = iw_end - iw_start;
                    
                    COMPUTE_TILE:
                    for (int oc = 0; oc < oc_count; oc++) {
                        for (int oh = 0; oh < (oh_end - oh_start); oh++) {
                            for (int ow = 0; ow < (ow_end - ow_start); ow++) {
                                #pragma HLS PIPELINE II=1
                                #pragma HLS LOOP_TRIPCOUNT min=1 max=6272
                                
                                acc_t sum = acc_tile[oc * (oh_end - oh_start) * (ow_end - ow_start) + 
                                                    oh * (ow_end - ow_start) + ow];
                                
                                // Convolve over input channels and kernel
                                for (int ic = 0; ic < ic_count; ic++) {
                                    #pragma HLS UNROLL factor=4
                                    for (int kh = 0; kh < kernel_size; kh++) {
                                        for (int kw = 0; kw < kernel_size; kw++) {
                                            int ih_local = oh * stride + kh;
                                            int iw_local = ow * stride + kw;
                                            
                                            int in_idx = ic * tile_ih * tile_iw + ih_local * tile_iw + iw_local;
                                            int w_idx = oc * ic_count * kernel_size * kernel_size +
                                                       ic * kernel_size * kernel_size +
                                                       kh * kernel_size + kw;
                                            
                                            sum += input_tile[in_idx] * weight_tile[w_idx];
                                        }
                                    }
                                }
                                
                                acc_tile[oc * (oh_end - oh_start) * (ow_end - ow_start) + 
                                        oh * (ow_end - ow_start) + ow] = sum;
                            }
                        }
                    }
                }
                
                // Apply BatchNorm + LeakyReLU and write output tile to DDR
                WRITE_OUTPUT:
                for (int oc = 0; oc < oc_count; oc++) {
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=32
                    
                    weight_t scale_val = (layer_type != 2) ? bn_scale[oc_start + oc] : weight_t(1);
                    weight_t shift_val = (layer_type != 2) ? bn_shift[oc_start + oc] : weight_t(0);
                    
                    for (int oh = 0; oh < (oh_end - oh_start); oh++) {
                        for (int ow = 0; ow < (ow_end - ow_start); ow++) {
                            #pragma HLS PIPELINE II=1
                            #pragma HLS LOOP_TRIPCOUNT min=1 max=196
                            
                            int tile_idx = oc * (oh_end - oh_start) * (ow_end - ow_start) + 
                                          oh * (ow_end - ow_start) + ow;
                            
                            // BatchNorm: out = acc * scale + shift
                            acc_t bn_out = acc_tile[tile_idx] * scale_val + shift_val;
                            
                            // LeakyReLU (if not output layer)
                            data_t result;
                            if (layer_type != 2) {
                                if (bn_out > acc_t(0)) {
                                    result = data_t(bn_out);
                                } else {
                                    result = data_t(bn_out >> 3);  // Approx 0.1x
                                }
                            } else {
                                result = data_t(bn_out);
                            }
                            
                            // Write to DDR
                            int out_idx = (oc_start + oc) * out_height * out_width +
                                         (oh_start + oh) * out_width +
                                         (ow_start + ow);
                            output_fm[out_idx] = result;
                        }
                    }
                }
            }
        }
    }
    
    // Handle max pooling as separate pass if needed
    if (layer_type == 1) {
        int pool_height = out_height / 2;
        int pool_width = out_width / 2;
        
        POOL_LOOP:
        for (int c = 0; c < out_channels; c++) {
            #pragma HLS LOOP_TRIPCOUNT min=16 max=512
            for (int ph = 0; ph < pool_height; ph++) {
                for (int pw = 0; pw < pool_width; pw++) {
                    #pragma HLS PIPELINE II=1
                    #pragma HLS LOOP_TRIPCOUNT min=1 max=12544
                    
                    // Read 2x2 window
                    int base_idx = c * out_height * out_width + (ph * 2) * out_width + (pw * 2);
                    data_t v00 = output_fm[base_idx];
                    data_t v01 = output_fm[base_idx + 1];
                    data_t v10 = output_fm[base_idx + out_width];
                    data_t v11 = output_fm[base_idx + out_width + 1];
                    
                    // Max
                    data_t max01 = (v00 > v01) ? v00 : v01;
                    data_t max23 = (v10 > v11) ? v10 : v11;
                    data_t max_val = (max01 > max23) ? max01 : max23;
                    
                    // Write back (in-place to different region)
                    int pool_idx = c * pool_height * pool_width + ph * pool_width + pw;
                    output_fm[pool_idx] = max_val;
                }
            }
        }
    }
    
    // Signal completion
    *status = 0;  // Done
}
