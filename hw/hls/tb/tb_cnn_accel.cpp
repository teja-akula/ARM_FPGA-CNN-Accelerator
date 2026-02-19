/*******************************************************************************
 * CNN Accelerator Testbench
 * C++ testbench for Vitis HLS simulation
 ******************************************************************************/

#include <iostream>
#include <cmath>
#include <cstdlib>
#include "cnn_accel.h"

// Reference software implementations (from existing yolo_layers.h logic)
void conv2d_ref(float* in, float* out, float* w, int ic, int ih, int iw, int oc, int k, int s, int p);
void leaky_relu_ref(float* data, int size);
void maxpool2d_ref(float* in, float* out, int c, int h, int w);

/*******************************************************************************
 * Test Utilities
 ******************************************************************************/
void init_random(data_t* arr, int size, float scale = 1.0f) {
    for (int i = 0; i < size; i++) {
        arr[i] = data_t((rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale);
    }
}

void init_random_float(float* arr, int size, float scale = 1.0f) {
    for (int i = 0; i < size; i++) {
        arr[i] = (rand() / (float)RAND_MAX - 0.5f) * 2.0f * scale;
    }
}

bool compare_results(data_t* hw, float* ref, int size, float tol = 0.1f) {
    int errors = 0;
    float max_err = 0;
    
    for (int i = 0; i < size; i++) {
        float hw_val = hw[i].to_float();
        float diff = fabs(hw_val - ref[i]);
        if (diff > max_err) max_err = diff;
        if (diff > tol) {
            errors++;
            if (errors < 10) {
                std::cout << "  Mismatch at " << i << ": HW=" << hw_val 
                          << " REF=" << ref[i] << " diff=" << diff << std::endl;
            }
        }
    }
    
    std::cout << "  Max error: " << max_err << ", Errors: " << errors 
              << "/" << size << std::endl;
    return errors == 0;
}

/*******************************************************************************
 * Reference Implementations
 ******************************************************************************/
void conv2d_ref(float* in, float* out, float* w, 
                int ic, int ih, int iw, int oc, int k, int s, int p) {
    int oh = (ih + 2*p - k) / s + 1;
    int ow = (iw + 2*p - k) / s + 1;
    
    for (int oco = 0; oco < oc; oco++) {
        for (int oho = 0; oho < oh; oho++) {
            for (int owo = 0; owo < ow; owo++) {
                float sum = 0;
                for (int ico = 0; ico < ic; ico++) {
                    for (int kh = 0; kh < k; kh++) {
                        for (int kw = 0; kw < k; kw++) {
                            int ihi = oho * s + kh - p;
                            int iwi = owo * s + kw - p;
                            if (ihi >= 0 && ihi < ih && iwi >= 0 && iwi < iw) {
                                int in_idx = ico * ih * iw + ihi * iw + iwi;
                                int w_idx = oco * ic * k * k + ico * k * k + kh * k + kw;
                                sum += in[in_idx] * w[w_idx];
                            }
                        }
                    }
                }
                out[oco * oh * ow + oho * ow + owo] = sum;
            }
        }
    }
}

void leaky_relu_ref(float* data, int size) {
    for (int i = 0; i < size; i++) {
        data[i] = data[i] > 0 ? data[i] : data[i] * 0.125f;  // Match HW approximation
    }
}

void maxpool2d_ref(float* in, float* out, int c, int h, int w) {
    int oh = h / 2;
    int ow = w / 2;
    
    for (int ch = 0; ch < c; ch++) {
        for (int ohi = 0; ohi < oh; ohi++) {
            for (int owi = 0; owi < ow; owi++) {
                int ih = ohi * 2;
                int iw_base = owi * 2;
                int base = ch * h * w;
                
                float v00 = in[base + ih * w + iw_base];
                float v01 = in[base + ih * w + iw_base + 1];
                float v10 = in[base + (ih+1) * w + iw_base];
                float v11 = in[base + (ih+1) * w + iw_base + 1];
                
                float max_val = v00;
                if (v01 > max_val) max_val = v01;
                if (v10 > max_val) max_val = v10;
                if (v11 > max_val) max_val = v11;
                
                out[ch * oh * ow + ohi * ow + owi] = max_val;
            }
        }
    }
}

/*******************************************************************************
 * Test Cases
 ******************************************************************************/
int test_leaky_relu() {
    std::cout << "\n=== Test LeakyReLU ===" << std::endl;
    
    const int size = 100;
    data_t hw_data[size];
    float ref_data[size];
    
    // Initialize with random values
    srand(42);
    for (int i = 0; i < size; i++) {
        float val = (rand() / (float)RAND_MAX - 0.5f) * 4.0f;
        hw_data[i] = data_t(val);
        ref_data[i] = val;
    }
    
    // HW computation
    for (int i = 0; i < size; i++) {
        hw_data[i] = leaky_relu_hw(hw_data[i]);
    }
    
    // Reference
    leaky_relu_ref(ref_data, size);
    
    return compare_results(hw_data, ref_data, size, 0.01f) ? 0 : 1;
}

int test_maxpool() {
    std::cout << "\n=== Test MaxPool 2x2 ===" << std::endl;
    
    const int C = 16, H = 8, W = 8;
    const int OUT_H = H/2, OUT_W = W/2;
    
    data_t hw_input[C * H * W];
    data_t hw_output[C * OUT_H * OUT_W];
    float ref_input[C * H * W];
    float ref_output[C * OUT_H * OUT_W];
    
    srand(123);
    for (int i = 0; i < C * H * W; i++) {
        float val = (rand() / (float)RAND_MAX) * 2.0f;
        hw_input[i] = data_t(val);
        ref_input[i] = val;
    }
    
    // HW
    maxpool2d_hw(hw_input, hw_output, C, H, W);
    
    // Reference
    maxpool2d_ref(ref_input, ref_output, C, H, W);
    
    return compare_results(hw_output, ref_output, C * OUT_H * OUT_W, 0.01f) ? 0 : 1;
}

int test_conv2d() {
    std::cout << "\n=== Test Conv2D 3x3 ===" << std::endl;
    
    const int IC = 3, OC = 16, H = 8, W = 8, K = 3, S = 1, P = 1;
    const int OUT_H = (H + 2*P - K) / S + 1;
    const int OUT_W = (W + 2*P - K) / S + 1;
    
    // Allocate memory
    data_t* hw_input = new data_t[IC * H * W];
    data_t* hw_output = new data_t[OC * OUT_H * OUT_W];
    weight_t* hw_weights = new weight_t[OC * IC * K * K];
    
    float* ref_input = new float[IC * H * W];
    float* ref_output = new float[OC * OUT_H * OUT_W];
    float* ref_weights = new float[OC * IC * K * K];
    
    // Initialize
    srand(456);
    for (int i = 0; i < IC * H * W; i++) {
        float val = (rand() / (float)RAND_MAX - 0.5f) * 2.0f;
        hw_input[i] = data_t(val);
        ref_input[i] = val;
    }
    for (int i = 0; i < OC * IC * K * K; i++) {
        float val = (rand() / (float)RAND_MAX - 0.5f) * 0.5f;
        hw_weights[i] = weight_t(val);
        ref_weights[i] = val;
    }
    
    std::cout << "  Input: " << IC << "x" << H << "x" << W << std::endl;
    std::cout << "  Output: " << OC << "x" << OUT_H << "x" << OUT_W << std::endl;
    
    // HW
    conv2d_hw(hw_input, hw_output, hw_weights, IC, H, W, OC, K, S, P);
    
    // Reference
    conv2d_ref(ref_input, ref_output, ref_weights, IC, H, W, OC, K, S, P);
    
    bool pass = compare_results(hw_output, ref_output, OC * OUT_H * OUT_W, 0.5f);
    
    delete[] hw_input;
    delete[] hw_output;
    delete[] hw_weights;
    delete[] ref_input;
    delete[] ref_output;
    delete[] ref_weights;
    
    return pass ? 0 : 1;
}

/*******************************************************************************
 * Main Testbench
 ******************************************************************************/
int main() {
    int errors = 0;
    
    std::cout << "=======================================" << std::endl;
    std::cout << "  CNN Accelerator HLS Testbench" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    errors += test_leaky_relu();
    errors += test_maxpool();
    errors += test_conv2d();
    
    std::cout << "\n=======================================" << std::endl;
    if (errors == 0) {
        std::cout << "  ALL TESTS PASSED!" << std::endl;
    } else {
        std::cout << "  TESTS FAILED: " << errors << " errors" << std::endl;
    }
    std::cout << "=======================================" << std::endl;
    
    return errors;
}
