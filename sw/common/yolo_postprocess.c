/*******************************************************************************
 * YOLO Post-Processing Implementation
 * For ARM Cortex-A9 on Zedboard
 ******************************************************************************/

#include "yolo_postprocess.h"
#include <stdio.h>
#include <string.h>

/*******************************************************************************
 * Custom fast exponential (no libm dependency)
 ******************************************************************************/
static inline float fast_expf(float x) {
    if (x > 20.0f) return 485165195.4f;
    if (x < -20.0f) return 0.0f;
    /* Taylor series approximation */
    float result = 1.0f;
    float term = 1.0f;
    for (int i = 1; i <= 12; i++) {
        term *= x / (float)i;
        result += term;
    }
    return result;
}

/*******************************************************************************
 * YOLO Lite anchor boxes (width, height) relative to grid cell
 * These are typical values - adjust based on your trained model
 ******************************************************************************/
const float YOLO_ANCHORS[YOLO_NUM_ANCHORS][2] = {
    {1.08f, 1.19f},
    {3.42f, 4.41f},
    {6.63f, 11.38f},
    {9.42f, 5.11f},
    {16.62f, 10.52f}
};

/*******************************************************************************
 * Pascal VOC class names
 ******************************************************************************/
const char *YOLO_CLASS_NAMES[YOLO_NUM_CLASSES] = {
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
};

/*******************************************************************************
 * Sigmoid function
 ******************************************************************************/
static inline float sigmoid(float x) {
    return 1.0f / (1.0f + fast_expf(-x));
}

/*******************************************************************************
 * Softmax for class probabilities
 ******************************************************************************/
static void softmax(const float *input, float *output, int n) {
    float max_val = input[0];
    for (int i = 1; i < n; i++) {
        if (input[i] > max_val) max_val = input[i];
    }
    
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
        output[i] = fast_expf(input[i] - max_val);
        sum += output[i];
    }
    
    for (int i = 0; i < n; i++) {
        output[i] /= sum;
    }
}

/*******************************************************************************
 * Calculate IoU (Intersection over Union)
 ******************************************************************************/
static float calculate_iou(const Detection *a, const Detection *b) {
    // Convert center coords to corners
    float a_x1 = a->x - a->w / 2;
    float a_y1 = a->y - a->h / 2;
    float a_x2 = a->x + a->w / 2;
    float a_y2 = a->y + a->h / 2;
    
    float b_x1 = b->x - b->w / 2;
    float b_y1 = b->y - b->h / 2;
    float b_x2 = b->x + b->w / 2;
    float b_y2 = b->y + b->h / 2;
    
    // Intersection
    float inter_x1 = (a_x1 > b_x1) ? a_x1 : b_x1;
    float inter_y1 = (a_y1 > b_y1) ? a_y1 : b_y1;
    float inter_x2 = (a_x2 < b_x2) ? a_x2 : b_x2;
    float inter_y2 = (a_y2 < b_y2) ? a_y2 : b_y2;
    
    float inter_w = (inter_x2 - inter_x1 > 0) ? (inter_x2 - inter_x1) : 0;
    float inter_h = (inter_y2 - inter_y1 > 0) ? (inter_y2 - inter_y1) : 0;
    float inter_area = inter_w * inter_h;
    
    // Union
    float a_area = a->w * a->h;
    float b_area = b->w * b->h;
    float union_area = a_area + b_area - inter_area;
    
    if (union_area <= 0) return 0;
    return inter_area / union_area;
}

/*******************************************************************************
 * Decode YOLO output
 ******************************************************************************/
void yolo_decode(const fixed16_t *output, DetectionResult *result, float conf_threshold) {
    result->count = 0;
    
    int grid_size = YOLO_GRID_H * YOLO_GRID_W;
    int anchor_stride = 5 + YOLO_NUM_CLASSES;  // tx, ty, tw, th, obj, classes
    
    for (int cy = 0; cy < YOLO_GRID_H; cy++) {
        for (int cx = 0; cx < YOLO_GRID_W; cx++) {
            for (int a = 0; a < YOLO_NUM_ANCHORS; a++) {
                // Calculate base index for this anchor
                int base_ch = a * anchor_stride;
                int spatial_idx = cy * YOLO_GRID_W + cx;
                
                // Extract raw values (CHW format)
                float tx = FIXED_TO_FLOAT(output[(base_ch + 0) * grid_size + spatial_idx]);
                float ty = FIXED_TO_FLOAT(output[(base_ch + 1) * grid_size + spatial_idx]);
                float tw = FIXED_TO_FLOAT(output[(base_ch + 2) * grid_size + spatial_idx]);
                float th = FIXED_TO_FLOAT(output[(base_ch + 3) * grid_size + spatial_idx]);
                float obj = FIXED_TO_FLOAT(output[(base_ch + 4) * grid_size + spatial_idx]);
                
                // Apply sigmoid to objectness
                float objectness = sigmoid(obj);
                
                // Skip low confidence
                if (objectness < conf_threshold) continue;
                
                // Extract class probabilities
                float class_probs_raw[YOLO_NUM_CLASSES];
                float class_probs[YOLO_NUM_CLASSES];
                for (int c = 0; c < YOLO_NUM_CLASSES; c++) {
                    class_probs_raw[c] = FIXED_TO_FLOAT(output[(base_ch + 5 + c) * grid_size + spatial_idx]);
                }
                softmax(class_probs_raw, class_probs, YOLO_NUM_CLASSES);
                
                // Find best class
                int best_class = 0;
                float best_prob = class_probs[0];
                for (int c = 1; c < YOLO_NUM_CLASSES; c++) {
                    if (class_probs[c] > best_prob) {
                        best_prob = class_probs[c];
                        best_class = c;
                    }
                }
                
                // Final confidence = objectness * class_prob
                float confidence = objectness * best_prob;
                if (confidence < conf_threshold) continue;
                
                // Decode bounding box
                // x, y relative to image (0-1)
                float bx = (sigmoid(tx) + cx) / YOLO_GRID_W;
                float by = (sigmoid(ty) + cy) / YOLO_GRID_H;
                float bw = (fast_expf(tw) * YOLO_ANCHORS[a][0]) / YOLO_GRID_W;
                float bh = (fast_expf(th) * YOLO_ANCHORS[a][1]) / YOLO_GRID_H;
                
                // Add detection
                if (result->count < MAX_DETECTIONS) {
                    Detection *det = &result->detections[result->count];
                    det->x = bx;
                    det->y = by;
                    det->w = bw;
                    det->h = bh;
                    det->confidence = confidence;
                    det->class_id = best_class;
                    det->class_prob = best_prob;
                    result->count++;
                }
            }
        }
    }
}

/*******************************************************************************
 * Non-Maximum Suppression
 ******************************************************************************/
void yolo_nms(DetectionResult *result, float nms_threshold) {
    // Sort by confidence (simple bubble sort for small arrays)
    for (int i = 0; i < result->count - 1; i++) {
        for (int j = i + 1; j < result->count; j++) {
            if (result->detections[j].confidence > result->detections[i].confidence) {
                Detection temp = result->detections[i];
                result->detections[i] = result->detections[j];
                result->detections[j] = temp;
            }
        }
    }
    
    // Mark suppressed detections
    int keep[MAX_DETECTIONS] = {0};
    memset(keep, 1, result->count * sizeof(int));
    
    for (int i = 0; i < result->count; i++) {
        if (!keep[i]) continue;
        
        for (int j = i + 1; j < result->count; j++) {
            if (!keep[j]) continue;
            
            // Only suppress same-class detections
            if (result->detections[i].class_id != result->detections[j].class_id) continue;
            
            float iou = calculate_iou(&result->detections[i], &result->detections[j]);
            if (iou > nms_threshold) {
                keep[j] = 0;  // Suppress
            }
        }
    }
    
    // Compact array
    int new_count = 0;
    for (int i = 0; i < result->count; i++) {
        if (keep[i]) {
            if (new_count != i) {
                result->detections[new_count] = result->detections[i];
            }
            new_count++;
        }
    }
    result->count = new_count;
}

/*******************************************************************************
 * Full post-processing pipeline
 ******************************************************************************/
void yolo_postprocess(const fixed16_t *output, DetectionResult *result) {
    // Decode YOLO output
    yolo_decode(output, result, CONFIDENCE_THRESHOLD);
    
    // Apply NMS
    yolo_nms(result, NMS_THRESHOLD);
}

/*******************************************************************************
 * Print detections (for debugging)
 ******************************************************************************/
void yolo_print_detections(const DetectionResult *result) {
    printf("\n=== Detection Results (%d objects) ===\n", result->count);
    
    for (int i = 0; i < result->count; i++) {
        const Detection *det = &result->detections[i];
        printf("[%d] %s: %.1f%% @ (%.3f, %.3f, %.3f, %.3f)\n",
               i,
               YOLO_CLASS_NAMES[det->class_id],
               det->confidence * 100,
               det->x, det->y, det->w, det->h);
    }
    
    printf("=====================================\n");
}

/*******************************************************************************
 * Convert to pixel coordinates
 ******************************************************************************/
void detection_to_pixels(const Detection *det, int img_width, int img_height,
                        int *x1, int *y1, int *x2, int *y2) {
    float half_w = det->w / 2;
    float half_h = det->h / 2;
    
    *x1 = (int)((det->x - half_w) * img_width);
    *y1 = (int)((det->y - half_h) * img_height);
    *x2 = (int)((det->x + half_w) * img_width);
    *y2 = (int)((det->y + half_h) * img_height);
    
    // Clamp to image bounds
    if (*x1 < 0) *x1 = 0;
    if (*y1 < 0) *y1 = 0;
    if (*x2 >= img_width) *x2 = img_width - 1;
    if (*y2 >= img_height) *y2 = img_height - 1;
}
