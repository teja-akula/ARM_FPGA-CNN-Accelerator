/*******************************************************************************
 * Tiny YOLO Layers - CNN Operations for Bare-Metal ARM
 ******************************************************************************/

#ifndef YOLO_LAYERS_H
#define YOLO_LAYERS_H

/* Bare-metal math implementations (no libm required) */

/* Check for NaN */
static inline int is_nan(float x) {
    return x != x;  /* NaN is the only value that doesn't equal itself */
}

/* Fast approximate square root using Newton-Raphson */
static inline float fast_sqrtf(float x) {
    if (x <= 0.0f) return 0.0001f;  /* Avoid division by zero */
    float guess = x * 0.5f;
    if (guess < 0.0001f) guess = 0.0001f;
    int i;
    for (i = 0; i < 5; i++) {
        guess = 0.5f * (guess + x / guess);
    }
    return guess;
}

/* Robust exp approximation using Taylor series */
static inline float fast_expf(float x) {
    /* Clamp to avoid overflow - be more conservative */
    if (x > 20.0f) return 485165195.0f;  /* e^20 approx */
    if (x < -20.0f) return 0.0f;
    
    /* Taylor series: e^x = 1 + x + x^2/2! + x^3/3! + ... */
    float result = 1.0f;
    float term = 1.0f;
    int i;
    for (i = 1; i <= 12; i++) {
        term *= x / (float)i;
        result += term;
    }
    
    /* Safety check */
    if (is_nan(result) || result < 0.0f) return 0.0001f;
    return result;
}

/*******************************************************************************
 * Activation Functions
 ******************************************************************************/
static inline float leaky_relu(float x) {
    if (is_nan(x)) return 0.0f;
    return x > 0.0f ? x : 0.1f * x;
}

static inline float sigmoid(float x) {
    if (is_nan(x)) return 0.5f;
    if (x > 20.0f) return 1.0f;
    if (x < -20.0f) return 0.0f;
    return 1.0f / (1.0f + fast_expf(-x));
}

/*******************************************************************************
 * Convolution 2D with padding
 * Input:  in_data  [in_ch][in_h][in_w]
 * Output: out_data [out_ch][out_h][out_w]
 * Kernel: weights  [out_ch][in_ch][k][k]
 ******************************************************************************/
void conv2d(
    const float* in_data, 
    float* out_data,
    const float* weights,
    int in_ch, int in_h, int in_w,
    int out_ch, int kernel, int stride, int pad
) {
    int out_h = (in_h + 2*pad - kernel) / stride + 1;
    int out_w = (in_w + 2*pad - kernel) / stride + 1;
    
    int oc, oh, ow, ic, kh, kw;
    
    for (oc = 0; oc < out_ch; oc++) {
        for (oh = 0; oh < out_h; oh++) {
            for (ow = 0; ow < out_w; ow++) {
                float sum = 0.0f;
                
                for (ic = 0; ic < in_ch; ic++) {
                    for (kh = 0; kh < kernel; kh++) {
                        for (kw = 0; kw < kernel; kw++) {
                            int ih = oh * stride + kh - pad;
                            int iw = ow * stride + kw - pad;
                            
                            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                int in_idx = ic * in_h * in_w + ih * in_w + iw;
                                int w_idx = oc * in_ch * kernel * kernel + 
                                           ic * kernel * kernel + 
                                           kh * kernel + kw;
                                sum += in_data[in_idx] * weights[w_idx];
                            }
                        }
                    }
                }
                
                out_data[oc * out_h * out_w + oh * out_w + ow] = sum;
            }
        }
    }
}

/*******************************************************************************
 * Batch Normalization + LeakyReLU (fused for efficiency)
 * out = leaky_relu(gamma * (in - mean) / sqrt(var + eps) + beta)
 ******************************************************************************/
void batchnorm_leaky(
    float* data,
    const float* gamma,
    const float* beta,
    const float* mean,
    const float* var,
    int channels, int h, int w
) {
    int c, i;
    float eps = 1e-5f;
    
    for (c = 0; c < channels; c++) {
        float scale = gamma[c] / fast_sqrtf(var[c] + eps);
        float shift = beta[c] - mean[c] * scale;
        
        for (i = 0; i < h * w; i++) {
            int idx = c * h * w + i;
            data[idx] = leaky_relu(data[idx] * scale + shift);
        }
    }
}

/*******************************************************************************
 * Add bias + activation (for layers without BN)
 ******************************************************************************/
void add_bias(
    float* data,
    const float* bias,
    int channels, int h, int w,
    int use_activation
) {
    int c, i;
    
    for (c = 0; c < channels; c++) {
        for (i = 0; i < h * w; i++) {
            int idx = c * h * w + i;
            data[idx] += bias[c];
        }
    }
}

/*******************************************************************************
 * Max Pooling 2x2 stride 2
 ******************************************************************************/
void maxpool2d(
    const float* in_data,
    float* out_data,
    int channels, int in_h, int in_w
) {
    int out_h = in_h / 2;
    int out_w = in_w / 2;
    int c, oh, ow;
    
    for (c = 0; c < channels; c++) {
        for (oh = 0; oh < out_h; oh++) {
            for (ow = 0; ow < out_w; ow++) {
                int ih = oh * 2;
                int iw = ow * 2;
                
                float max_val = in_data[c * in_h * in_w + ih * in_w + iw];
                
                if (ih + 1 < in_h) {
                    float v = in_data[c * in_h * in_w + (ih+1) * in_w + iw];
                    if (v > max_val) max_val = v;
                }
                if (iw + 1 < in_w) {
                    float v = in_data[c * in_h * in_w + ih * in_w + (iw+1)];
                    if (v > max_val) max_val = v;
                }
                if (ih + 1 < in_h && iw + 1 < in_w) {
                    float v = in_data[c * in_h * in_w + (ih+1) * in_w + (iw+1)];
                    if (v > max_val) max_val = v;
                }
                
                out_data[c * out_h * out_w + oh * out_w + ow] = max_val;
            }
        }
    }
}

/*******************************************************************************
 * Detection structure
 ******************************************************************************/
typedef struct {
    float x, y, w, h;
    float confidence;
    int class_id;
} YoloDetection;

/*******************************************************************************
 * Decode YOLO output to detections
 ******************************************************************************/
int decode_yolo_output(
    const float* output,
    int grid_h, int grid_w,
    int num_anchors, int num_classes,
    const float* anchors,
    int input_size,
    float conf_thresh,
    YoloDetection* detections,
    int max_detections
) {
    int num_dets = 0;
    int gh, gw, a, c;
    
    /* Output format: [anchors][5+classes][grid_h][grid_w] */
    int stride = grid_h * grid_w;
    
    for (gh = 0; gh < grid_h; gh++) {
        for (gw = 0; gw < grid_w; gw++) {
            for (a = 0; a < num_anchors; a++) {
                int base = a * (5 + num_classes) * stride + gh * grid_w + gw;
                
                /* Get box coordinates */
                float tx = output[base + 0 * stride];
                float ty = output[base + 1 * stride];
                float tw = output[base + 2 * stride];
                float th = output[base + 3 * stride];
                
                /* Skip if any raw value is NaN */
                if (is_nan(tx) || is_nan(ty) || is_nan(tw) || is_nan(th)) continue;
                
                /* Clamp tw/th to prevent exp overflow */
                if (tw > 10.0f) tw = 10.0f;
                if (tw < -10.0f) tw = -10.0f;
                if (th > 10.0f) th = 10.0f;
                if (th < -10.0f) th = -10.0f;
                
                float obj_conf = sigmoid(output[base + 4 * stride]);
                
                if (obj_conf < conf_thresh) continue;
                
                /* Find best class */
                int best_class = 0;
                float best_prob = 0;
                for (c = 0; c < num_classes; c++) {
                    float prob = sigmoid(output[base + (5 + c) * stride]);
                    if (prob > best_prob) {
                        best_prob = prob;
                        best_class = c;
                    }
                }
                
                float confidence = obj_conf * best_prob;
                if (confidence < conf_thresh) continue;
                if (is_nan(confidence)) continue;
                
                /* Decode box to image coordinates */
                float box_x = (sigmoid(tx) + gw) / grid_w * input_size;
                float box_y = (sigmoid(ty) + gh) / grid_h * input_size;
                float box_w = fast_expf(tw) * anchors[a * 2];
                float box_h = fast_expf(th) * anchors[a * 2 + 1];
                
                /* Skip invalid boxes */
                if (is_nan(box_x) || is_nan(box_y) || is_nan(box_w) || is_nan(box_h)) continue;
                if (box_w <= 0 || box_h <= 0 || box_w > input_size * 2 || box_h > input_size * 2) continue;
                
                /* Convert to corner format */
                if (num_dets < max_detections) {
                    detections[num_dets].x = box_x - box_w / 2;
                    detections[num_dets].y = box_y - box_h / 2;
                    detections[num_dets].w = box_w;
                    detections[num_dets].h = box_h;
                    detections[num_dets].confidence = confidence;
                    detections[num_dets].class_id = best_class;
                    num_dets++;
                }
            }
        }
    }
    
    return num_dets;
}

/*******************************************************************************
 * Non-Maximum Suppression
 ******************************************************************************/
float iou(YoloDetection* a, YoloDetection* b) {
    float x1 = (a->x > b->x) ? a->x : b->x;
    float y1 = (a->y > b->y) ? a->y : b->y;
    float x2 = (a->x + a->w < b->x + b->w) ? a->x + a->w : b->x + b->w;
    float y2 = (a->y + a->h < b->y + b->h) ? a->y + a->h : b->y + b->h;
    
    float inter_w = (x2 - x1 > 0) ? x2 - x1 : 0;
    float inter_h = (y2 - y1 > 0) ? y2 - y1 : 0;
    float inter = inter_w * inter_h;
    
    float area_a = a->w * a->h;
    float area_b = b->w * b->h;
    
    return inter / (area_a + area_b - inter + 1e-6f);
}

int nms(YoloDetection* dets, int num_dets, float nms_thresh) {
    int i, j;
    int keep[100];
    int num_keep = 0;
    
    /* Simple bubble sort by confidence */
    for (i = 0; i < num_dets - 1; i++) {
        for (j = i + 1; j < num_dets; j++) {
            if (dets[j].confidence > dets[i].confidence) {
                YoloDetection tmp = dets[i];
                dets[i] = dets[j];
                dets[j] = tmp;
            }
        }
    }
    
    /* NMS */
    for (i = 0; i < num_dets; i++) {
        int suppress = 0;
        for (j = 0; j < num_keep; j++) {
            if (iou(&dets[keep[j]], &dets[i]) > nms_thresh) {
                suppress = 1;
                break;
            }
        }
        if (!suppress && num_keep < 100) {
            keep[num_keep++] = i;
        }
    }
    
    /* Compact array */
    for (i = 0; i < num_keep; i++) {
        dets[i] = dets[keep[i]];
    }
    
    return num_keep;
}

#endif /* YOLO_LAYERS_H */
