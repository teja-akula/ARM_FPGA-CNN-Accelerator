/*******************************************************************************
 * STEP 1: ARM-ONLY CNN Inference (Software Baseline)
 * All convolution runs on ARM Cortex-A9 processor
 * Deploy this first to measure ARM-only performance
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "xil_printf.h"
#include "xil_cache.h"
#include "xil_io.h"

#include "cnn_driver.h"
#include "image_preprocess.h"
#include "tiny_yolo_weights.h"
#include "yolo_layers.h"
#include "test_image.h"

/*******************************************************************************
 * ARM Global Timer @ 333 MHz (CPU_FREQ / 2)
 ******************************************************************************/
#define GLOBAL_TMR_BASE  0xF8F00200U

static unsigned int read_timer_lo(void) {
    return Xil_In32(GLOBAL_TMR_BASE);
}

static void enable_timer(void) {
    unsigned int ctrl = Xil_In32(GLOBAL_TMR_BASE + 0x08);
    Xil_Out32(GLOBAL_TMR_BASE + 0x08, ctrl | 1U);
}

/* Simple ms timer using 32-bit low counter (wraps at ~12.8 sec)
   For longer measurements, we accumulate small chunks */
static unsigned int t_start_lo;

static void timer_start(void) {
    t_start_lo = read_timer_lo();
}

/* Returns elapsed milliseconds (handles single wrap) */
static int timer_elapsed_ms(void) {
    unsigned int t_end = read_timer_lo();
    unsigned int diff = t_end - t_start_lo; /* unsigned wrap is fine */
    return (int)(diff / 333333U);  /* 333333 ticks per ms at 333 MHz */
}

/*******************************************************************************
 * Helpers
 ******************************************************************************/
static float* alloc_buf(int size, const char* name) {
    float* p = (float*)malloc(size * sizeof(float));
    if (!p) {
        xil_printf("FATAL: malloc %s failed\r\n", name);
        while(1);
    }
    return p;
}

static void load_image(float* out) {
    int ch, y, x;
    for (ch = 0; ch < 3; ch++) {
        for (y = 0; y < INPUT_SIZE; y++) {
            for (x = 0; x < INPUT_SIZE; x++) {
                int sy = y * IMG_HEIGHT / INPUT_SIZE;
                int sx = x * IMG_WIDTH / INPUT_SIZE;
                if (sy >= IMG_HEIGHT) sy = IMG_HEIGHT - 1;
                if (sx >= IMG_WIDTH) sx = IMG_WIDTH - 1;
                out[ch * INPUT_SIZE * INPUT_SIZE + y * INPUT_SIZE + x] =
                    TEST_IMAGE[(sy * IMG_WIDTH + sx) * 3 + ch] / 255.0f;
            }
        }
    }
}

/* Output buffer */
static float output_buffer[NUM_ANCHORS * (5 + NUM_CLASSES) * 7 * 7];

/*******************************************************************************
 * Main - ARM Only Inference
 ******************************************************************************/
int main(void) {
    int h, w, layer_ms, total_ms = 0;
    float *input, *a, *b, *t;

    Xil_DCacheFlush();
    enable_timer();

    xil_printf("\r\n\r\n");
    xil_printf("==============================================\r\n");
    xil_printf("  STEP 1: ARM-Only CNN Inference\r\n");
    xil_printf("  Zedboard Zynq-7020 (Cortex-A9 @ 667 MHz)\r\n");
    xil_printf("  All computation on ARM processor\r\n");
    xil_printf("  Architecture: 3->16->32->64->128->256->512->24\r\n");
    xil_printf("==============================================\r\n\r\n");

    cnn_accel_init();

    /* Load image */
    xil_printf("[1] Loading image (%dx%d -> %dx%d)...\r\n",
               IMG_WIDTH, IMG_HEIGHT, INPUT_SIZE, INPUT_SIZE);
    input = alloc_buf(3 * INPUT_SIZE * INPUT_SIZE, "input");
    load_image(input);
    xil_printf("    Done.\r\n\r\n");

    xil_printf("[2] Running ARM-only inference (all layers on CPU)...\r\n\r\n");

    h = INPUT_SIZE; w = INPUT_SIZE;
    a = alloc_buf(16 * 224 * 224, "buf_a");
    b = alloc_buf(16 * 224 * 224, "buf_b");

    /* Layer 0 */
    timer_start();
    conv2d(input, a, CONV0_W, 3, h, w, 16, 3, 1, 1);
    batchnorm_leaky(a, BN0_GAMMA, BN0_BETA, BN0_MEAN, BN0_VAR, 16, h, w);
    maxpool2d(a, b, 16, h, w); h /= 2; w /= 2;
    layer_ms = timer_elapsed_ms();
    total_ms += layer_ms;
    t = a; a = b; b = t;
    xil_printf("    L0: Conv  3->16  224x224 + BN + Pool -> 112x112  %d ms\r\n", layer_ms);

    /* Layer 1 */
    timer_start();
    conv2d(a, b, CONV1_W, 16, h, w, 32, 3, 1, 1);
    batchnorm_leaky(b, BN1_GAMMA, BN1_BETA, BN1_MEAN, BN1_VAR, 32, h, w);
    t = a; a = b; b = t;
    maxpool2d(a, b, 32, h, w); h /= 2; w /= 2;
    layer_ms = timer_elapsed_ms();
    total_ms += layer_ms;
    t = a; a = b; b = t;
    xil_printf("    L1: Conv 16->32  112x112 + BN + Pool ->  56x56  %d ms\r\n", layer_ms);

    /* Layer 2 */
    timer_start();
    conv2d(a, b, CONV2_W, 32, h, w, 64, 3, 1, 1);
    batchnorm_leaky(b, BN2_GAMMA, BN2_BETA, BN2_MEAN, BN2_VAR, 64, h, w);
    t = a; a = b; b = t;
    maxpool2d(a, b, 64, h, w); h /= 2; w /= 2;
    layer_ms = timer_elapsed_ms();
    total_ms += layer_ms;
    t = a; a = b; b = t;
    xil_printf("    L2: Conv 32->64   56x56  + BN + Pool ->  28x28  %d ms\r\n", layer_ms);

    /* Layer 3 */
    timer_start();
    conv2d(a, b, CONV3_W, 64, h, w, 128, 3, 1, 1);
    batchnorm_leaky(b, BN3_GAMMA, BN3_BETA, BN3_MEAN, BN3_VAR, 128, h, w);
    t = a; a = b; b = t;
    maxpool2d(a, b, 128, h, w); h /= 2; w /= 2;
    layer_ms = timer_elapsed_ms();
    total_ms += layer_ms;
    t = a; a = b; b = t;
    xil_printf("    L3: Conv 64->128  28x28  + BN + Pool ->  14x14  %d ms\r\n", layer_ms);

    /* Layer 4 */
    timer_start();
    conv2d(a, b, CONV4_W, 128, h, w, 256, 3, 1, 1);
    batchnorm_leaky(b, BN4_GAMMA, BN4_BETA, BN4_MEAN, BN4_VAR, 256, h, w);
    t = a; a = b; b = t;
    maxpool2d(a, b, 256, h, w); h /= 2; w /= 2;
    layer_ms = timer_elapsed_ms();
    total_ms += layer_ms;
    t = a; a = b; b = t;
    xil_printf("    L4: Conv 128->256 14x14  + BN + Pool ->   7x7   %d ms\r\n", layer_ms);

    /* Layer 5 */
    timer_start();
    conv2d(a, b, CONV5_W, 256, h, w, 512, 3, 1, 1);
    batchnorm_leaky(b, BN5_GAMMA, BN5_BETA, BN5_MEAN, BN5_VAR, 512, h, w);
    layer_ms = timer_elapsed_ms();
    total_ms += layer_ms;
    t = a; a = b; b = t;
    xil_printf("    L5: Conv 256->512  7x7   + BN (no pool)          %d ms\r\n", layer_ms);

    /* Layer 6 */
    timer_start();
    conv2d(a, output_buffer, CONV6_W, 512, h, w,
           NUM_ANCHORS * (5 + NUM_CLASSES), 1, 1, 0);
    add_bias(output_buffer, CONV6_B, NUM_ANCHORS * (5 + NUM_CLASSES), h, w, 0);
    layer_ms = timer_elapsed_ms();
    total_ms += layer_ms;
    xil_printf("    L6: Conv 512->24   7x7   (1x1 output)            %d ms\r\n", layer_ms);

    free(a);
    free(b);
    free(input);

    xil_printf("\r\n");
    xil_printf("==============================================\r\n");
    xil_printf("  ARM-Only Total Time: %d ms (%d seconds)\r\n", total_ms, total_ms / 1000);
    xil_printf("  Processor: ARM Cortex-A9 @ 667 MHz\r\n");
    xil_printf("  FPGA PL: Not used\r\n");
    xil_printf("==============================================\r\n");
    xil_printf("\r\nDeploy STEP 2 (FPGA-accelerated) to compare.\r\n");

    while(1);
    return 0;
}
