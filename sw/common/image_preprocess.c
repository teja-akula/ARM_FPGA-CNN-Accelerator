/*******************************************************************************
 * Image Preprocessing Implementation
 * For ARM Cortex-A9 on Zedboard
 ******************************************************************************/

#include "image_preprocess.h"
#include <stdlib.h>
#include <string.h>

/*******************************************************************************
 * Load image from memory
 ******************************************************************************/
void image_load_from_memory(const uint8_t *src, int width, int height, Image *img) {
    img->width = width;
    img->height = height;
    img->channels = 3;
    img->data = (uint8_t *)src;  // Point to source data (no copy)
}

/*******************************************************************************
 * Bilinear interpolation resize
 * Simple CPU implementation for ARM
 ******************************************************************************/
void image_resize_bilinear(const Image *src, Image *dst, int dst_width, int dst_height) {
    float x_ratio = (float)(src->width - 1) / (dst_width - 1);
    float y_ratio = (float)(src->height - 1) / (dst_height - 1);
    
    for (int y = 0; y < dst_height; y++) {
        for (int x = 0; x < dst_width; x++) {
            float gx = x * x_ratio;
            float gy = y * y_ratio;
            
            int gxi = (int)gx;
            int gyi = (int)gy;
            
            float dx = gx - gxi;
            float dy = gy - gyi;
            
            // Clamp to image bounds
            int gxi1 = (gxi + 1 < src->width) ? gxi + 1 : gxi;
            int gyi1 = (gyi + 1 < src->height) ? gyi + 1 : gyi;
            
            for (int c = 0; c < 3; c++) {
                // Get 4 neighboring pixels
                float p00 = src->data[(gyi * src->width + gxi) * 3 + c];
                float p10 = src->data[(gyi * src->width + gxi1) * 3 + c];
                float p01 = src->data[(gyi1 * src->width + gxi) * 3 + c];
                float p11 = src->data[(gyi1 * src->width + gxi1) * 3 + c];
                
                // Bilinear interpolation
                float value = p00 * (1 - dx) * (1 - dy) +
                             p10 * dx * (1 - dy) +
                             p01 * (1 - dx) * dy +
                             p11 * dx * dy;
                
                dst->data[(y * dst_width + x) * 3 + c] = (uint8_t)(value + 0.5f);
            }
        }
    }
    
    dst->width = dst_width;
    dst->height = dst_height;
    dst->channels = 3;
}

/*******************************************************************************
 * Convert to fixed-point with normalization
 * Output layout: CHW (channel-first) for CNN
 ******************************************************************************/
void image_to_fixed_point(const Image *img, fixed16_t *output) {
    int hw = img->height * img->width;
    
    for (int y = 0; y < img->height; y++) {
        for (int x = 0; x < img->width; x++) {
            int pixel_idx = y * img->width + x;
            
            for (int c = 0; c < 3; c++) {
                // Get pixel value (0-255)
                uint8_t pixel = img->data[pixel_idx * 3 + c];
                
                // Normalize to [0, 1] and convert to Q8.8
                // pixel / 255.0 * 256 = pixel * 256 / 255 ≈ pixel
                // For simplicity, we store normalized value directly
                float normalized = (float)pixel / 255.0f;
                
                // Output in CHW format
                output[c * hw + pixel_idx] = FLOAT_TO_FIXED(normalized);
            }
        }
    }
}

/*******************************************************************************
 * Full preprocessing pipeline
 ******************************************************************************/
void preprocess_image(const uint8_t *src_data, int src_width, int src_height,
                      fixed16_t *output) {
    // Allocate temporary buffer for resized image
    static uint8_t resize_buffer[CNN_INPUT_WIDTH * CNN_INPUT_HEIGHT * 3];
    
    Image src_img, resized_img;
    
    // Load source image
    image_load_from_memory(src_data, src_width, src_height, &src_img);
    
    // Setup resized image
    resized_img.data = resize_buffer;
    resized_img.width = CNN_INPUT_WIDTH;
    resized_img.height = CNN_INPUT_HEIGHT;
    resized_img.channels = 3;
    
    // Resize if needed
    if (src_width != CNN_INPUT_WIDTH || src_height != CNN_INPUT_HEIGHT) {
        image_resize_bilinear(&src_img, &resized_img, CNN_INPUT_WIDTH, CNN_INPUT_HEIGHT);
    } else {
        // Just copy if already correct size
        memcpy(resize_buffer, src_data, CNN_INPUT_WIDTH * CNN_INPUT_HEIGHT * 3);
    }
    
    // Convert to fixed-point
    image_to_fixed_point(&resized_img, output);
}

/*******************************************************************************
 * Create test pattern (gradient + checkerboard)
 * Useful for testing without actual image
 ******************************************************************************/
void create_test_pattern(fixed16_t *output) {
    int hw = CNN_INPUT_HEIGHT * CNN_INPUT_WIDTH;
    
    for (int y = 0; y < CNN_INPUT_HEIGHT; y++) {
        for (int x = 0; x < CNN_INPUT_WIDTH; x++) {
            int idx = y * CNN_INPUT_WIDTH + x;
            
            // Red channel: horizontal gradient
            float r = (float)x / CNN_INPUT_WIDTH;
            
            // Green channel: vertical gradient
            float g = (float)y / CNN_INPUT_HEIGHT;
            
            // Blue channel: checkerboard
            float b = ((x / 32 + y / 32) % 2) ? 1.0f : 0.0f;
            
            // Store in CHW format
            output[0 * hw + idx] = FLOAT_TO_FIXED(r);  // R
            output[1 * hw + idx] = FLOAT_TO_FIXED(g);  // G
            output[2 * hw + idx] = FLOAT_TO_FIXED(b);  // B
        }
    }
}
