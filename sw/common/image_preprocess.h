/*******************************************************************************
 * Image Preprocessing for CNN Accelerator
 * Runs on ARM Cortex-A9
 * 
 * Functions:
 * - Load image from memory/file
 * - Resize to 224x224
 * - Normalize to [-1, 1] or [0, 1]
 * - Convert to Q8.8 fixed-point
 ******************************************************************************/

#ifndef IMAGE_PREPROCESS_H
#define IMAGE_PREPROCESS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Image dimensions
#define CNN_INPUT_WIDTH   224
#define CNN_INPUT_HEIGHT  224
#define CNN_INPUT_CHANNELS 3

// Fixed-point Q8.8 format
typedef int16_t fixed16_t;
#define FIXED_SHIFT 8
#define FLOAT_TO_FIXED(x) ((fixed16_t)((x) * (1 << FIXED_SHIFT)))
#define FIXED_TO_FLOAT(x) ((float)(x) / (1 << FIXED_SHIFT))

// Image structure
typedef struct {
    uint8_t *data;      // RGB888 format
    int width;
    int height;
    int channels;
} Image;

/*******************************************************************************
 * Load raw RGB image from memory
 * @param src: Source RGB888 data
 * @param width: Image width
 * @param height: Image height
 * @param img: Output image structure
 ******************************************************************************/
void image_load_from_memory(const uint8_t *src, int width, int height, Image *img);

/*******************************************************************************
 * Resize image using bilinear interpolation
 * @param src: Source image
 * @param dst: Destination image (must be pre-allocated)
 * @param dst_width: Target width
 * @param dst_height: Target height
 ******************************************************************************/
void image_resize_bilinear(const Image *src, Image *dst, int dst_width, int dst_height);

/*******************************************************************************
 * Convert image to Q8.8 fixed-point and normalize
 * Normalization: pixel / 255.0 (range [0, 1])
 * 
 * @param img: Input RGB image
 * @param output: Output fixed-point buffer (must be pre-allocated)
 *                Size: channels * height * width * sizeof(fixed16_t)
 *                Layout: CHW (channel-first, as expected by CNN)
 ******************************************************************************/
void image_to_fixed_point(const Image *img, fixed16_t *output);

/*******************************************************************************
 * Full preprocessing pipeline
 * Resize + Normalize + Convert to fixed-point
 * 
 * @param src_data: Source RGB888 image data
 * @param src_width: Source width
 * @param src_height: Source height
 * @param output: Output buffer for CNN input (pre-allocated)
 *                Size: 3 * 224 * 224 * sizeof(fixed16_t) = 301,056 bytes
 ******************************************************************************/
void preprocess_image(const uint8_t *src_data, int src_width, int src_height, 
                      fixed16_t *output);

/*******************************************************************************
 * Create test pattern image (for testing without camera)
 * @param output: Output buffer for CNN input
 ******************************************************************************/
void create_test_pattern(fixed16_t *output);

#ifdef __cplusplus
}
#endif

#endif // IMAGE_PREPROCESS_H
