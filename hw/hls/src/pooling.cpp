/*******************************************************************************
 * CNN Accelerator - Pooling Layer
 * Max Pooling 2x2 with stride 2
 ******************************************************************************/

#include "cnn_accel.h"

/*******************************************************************************
 * Max of 4 Values
 * Comparator tree for efficient 2x2 max operation
 ******************************************************************************/
data_t max4_hw(data_t a, data_t b, data_t c, data_t d) {
    #pragma HLS INLINE
    
    // Two-level comparator tree
    data_t max_ab = (a > b) ? a : b;
    data_t max_cd = (c > d) ? c : d;
    return (max_ab > max_cd) ? max_ab : max_cd;
}

/*******************************************************************************
 * Max Pooling 2x2 Stride 2
 * Reduces spatial dimensions by half
 ******************************************************************************/
void maxpool2d_hw(
    data_t *input,
    data_t *output,
    int channels,
    int in_h,
    int in_w
) {
    #pragma HLS INLINE off
    
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    
    // Process each channel
    POOL_CH:
    for (int c = 0; c < channels; c++) {
        #pragma HLS LOOP_TRIPCOUNT min=16 max=512
        
        // Process output spatial positions
        POOL_OH:
        for (int oh = 0; oh < out_h; oh++) {
            #pragma HLS LOOP_TRIPCOUNT min=3 max=112
            
            POOL_OW:
            for (int ow = 0; ow < out_w; ow++) {
                #pragma HLS LOOP_TRIPCOUNT min=3 max=112
                #pragma HLS PIPELINE II=1
                
                // 2x2 input window
                int ih = oh * 2;
                int iw = ow * 2;
                
                // Load 4 input values
                int base_idx = c * in_h * in_w;
                data_t v00 = input[base_idx + ih * in_w + iw];
                data_t v01 = input[base_idx + ih * in_w + iw + 1];
                data_t v10 = input[base_idx + (ih + 1) * in_w + iw];
                data_t v11 = input[base_idx + (ih + 1) * in_w + iw + 1];
                
                // Compute max
                data_t max_val = max4_hw(v00, v01, v10, v11);
                
                // Write output
                int out_idx = c * out_h * out_w + oh * out_w + ow;
                output[out_idx] = max_val;
            }
        }
    }
}

/*******************************************************************************
 * Global Average Pooling
 * Reduces each channel to a single value (average of all spatial elements)
 * Used in some CNN architectures for classification head
 ******************************************************************************/
void global_avgpool_hw(
    data_t *input,
    data_t *output,
    int channels,
    int height,
    int width
) {
    #pragma HLS INLINE off
    
    int spatial_size = height * width;
    data_t scale = data_t(1.0 / spatial_size);
    
    GAP_CH:
    for (int c = 0; c < channels; c++) {
        #pragma HLS LOOP_TRIPCOUNT min=16 max=512
        
        acc_t sum = 0;
        
        GAP_SPATIAL:
        for (int i = 0; i < spatial_size; i++) {
            #pragma HLS LOOP_TRIPCOUNT min=49 max=50176
            #pragma HLS PIPELINE II=1
            
            int idx = c * spatial_size + i;
            sum += input[idx];
        }
        
        // Average
        output[c] = data_t(sum * scale);
    }
}
