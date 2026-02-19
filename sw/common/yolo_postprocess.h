/*******************************************************************************
 * YOLO Post-Processing for CNN Accelerator
 * Runs on ARM Cortex-A9
 * 
 * Functions:
 * - Decode YOLO output grid to bounding boxes
 * - Apply confidence threshold
 * - Non-Maximum Suppression (NMS)
 * - Output detection results
 ******************************************************************************/

#ifndef YOLO_POSTPROCESS_H
#define YOLO_POSTPROCESS_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// YOLO Lite output dimensions
#define YOLO_GRID_H      7
#define YOLO_GRID_W      7
#define YOLO_NUM_ANCHORS 5
#define YOLO_NUM_CLASSES 20  // Pascal VOC classes

// Output channels: 5 * (5 + 20) = 125
// Per anchor: tx, ty, tw, th, objectness, class_probs[20]
#define YOLO_OUTPUT_CHANNELS 125

// Detection thresholds
#define CONFIDENCE_THRESHOLD 0.3f
#define NMS_THRESHOLD        0.45f
#define MAX_DETECTIONS       100

// Fixed-point format (same as HLS)
typedef int16_t fixed16_t;
#define FIXED_SHIFT 8
#define FIXED_TO_FLOAT(x) ((float)(x) / (1 << FIXED_SHIFT))

// Bounding box structure
typedef struct {
    float x;        // Center x (0-1, relative to image)
    float y;        // Center y
    float w;        // Width
    float h;        // Height
    float confidence;
    int class_id;
    float class_prob;
} Detection;

// Detection results
typedef struct {
    Detection detections[MAX_DETECTIONS];
    int count;
} DetectionResult;

// Anchor box sizes (pre-defined for YOLO Lite)
extern const float YOLO_ANCHORS[YOLO_NUM_ANCHORS][2];

// Class names (Pascal VOC)
extern const char *YOLO_CLASS_NAMES[YOLO_NUM_CLASSES];

/*******************************************************************************
 * Decode YOLO output to bounding boxes
 * @param output: Raw CNN output (fixed-point, CHW format)
 *                Size: 125 x 7 x 7 = 6125 elements
 * @param result: Output detection results
 * @param conf_threshold: Confidence threshold (e.g., 0.3)
 ******************************************************************************/
void yolo_decode(const fixed16_t *output, DetectionResult *result, float conf_threshold);

/*******************************************************************************
 * Apply Non-Maximum Suppression
 * @param result: Detection results (modified in place)
 * @param nms_threshold: IoU threshold for suppression (e.g., 0.45)
 ******************************************************************************/
void yolo_nms(DetectionResult *result, float nms_threshold);

/*******************************************************************************
 * Full post-processing pipeline
 * Decode + NMS
 * @param output: Raw CNN output
 * @param result: Output detection results
 ******************************************************************************/
void yolo_postprocess(const fixed16_t *output, DetectionResult *result);

/*******************************************************************************
 * Print detection results (for debugging)
 * @param result: Detection results
 ******************************************************************************/
void yolo_print_detections(const DetectionResult *result);

/*******************************************************************************
 * Convert detection to screen coordinates
 * @param det: Detection (relative coords 0-1)
 * @param img_width: Input image width
 * @param img_height: Input image height
 * @param x1, y1, x2, y2: Output bounding box (pixel coords)
 ******************************************************************************/
void detection_to_pixels(const Detection *det, int img_width, int img_height,
                        int *x1, int *y1, int *x2, int *y2);

#ifdef __cplusplus
}
#endif

#endif // YOLO_POSTPROCESS_H
