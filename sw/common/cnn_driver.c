/*******************************************************************************
 * CNN Accelerator ARM Driver Implementation
 * For Zedboard Zynq-7020 + CNN HLS Accelerator
 ******************************************************************************/

#include "cnn_driver.h"
#include "xil_io.h"
#include <stdio.h>

/*******************************************************************************
 * Memory-Mapped Register Access
 ******************************************************************************/
static inline void write_reg(uint32_t base, uint32_t offset, uint32_t value) {
    Xil_Out32(base + offset, value);
}

static inline uint32_t read_reg(uint32_t base, uint32_t offset) {
    return Xil_In32(base + offset);
}

/*******************************************************************************
 * Driver Implementation
 ******************************************************************************/

int cnn_accel_init(void) {
    // Check if accelerator is present by reading AP_CTRL
    uint32_t ctrl = read_reg(CNN_ACCEL_CONTROL_BASE, REG_AP_CTRL);
    
    // Should be idle initially
    if (ctrl & AP_IDLE) {
        printf("[CNN] Accelerator initialized, status: IDLE\n");
        return 0;
    }
    
    printf("[CNN] Warning: Accelerator not idle (ctrl=0x%08X)\n", ctrl);
    return -1;
}

int cnn_accel_is_ready(void) {
    uint32_t ctrl = read_reg(CNN_ACCEL_CONTROL_BASE, REG_AP_CTRL);
    return (ctrl & AP_IDLE) ? 1 : 0;
}

int cnn_accel_is_done(void) {
    uint32_t ctrl = read_reg(CNN_ACCEL_CONTROL_BASE, REG_AP_CTRL);
    return (ctrl & AP_DONE) ? 1 : 0;
}

void cnn_accel_wait_done(void) {
    while (!cnn_accel_is_done()) {
        // Busy wait - could add timeout or sleep
    }
    // Clear done bit by reading it
    read_reg(CNN_ACCEL_CONTROL_BASE, REG_AP_CTRL);
}

void cnn_accel_configure_layer(const LayerConfig *cfg) {
    write_reg(CNN_ACCEL_CONTROL_BASE, REG_LAYER_TYPE, cfg->layer_type);
    write_reg(CNN_ACCEL_CONTROL_BASE, REG_IN_CHANNELS, cfg->in_channels);
    write_reg(CNN_ACCEL_CONTROL_BASE, REG_OUT_CHANNELS, cfg->out_channels);
    write_reg(CNN_ACCEL_CONTROL_BASE, REG_IN_HEIGHT, cfg->in_height);
    write_reg(CNN_ACCEL_CONTROL_BASE, REG_IN_WIDTH, cfg->in_width);
    write_reg(CNN_ACCEL_CONTROL_BASE, REG_KERNEL_SIZE, cfg->kernel_size);
    write_reg(CNN_ACCEL_CONTROL_BASE, REG_STRIDE, cfg->stride);
    write_reg(CNN_ACCEL_CONTROL_BASE, REG_PADDING, cfg->padding);
}

void cnn_accel_set_addresses(
    uint32_t input_addr,
    uint32_t output_addr,
    uint32_t weights_addr,
    uint32_t bn_scale_addr,
    uint32_t bn_shift_addr
) {
    // 64-bit addresses (upper 32 bits = 0 for Zynq)
    write_reg(CNN_ACCEL_CONTROL_R_BASE, REG_INPUT_FM_LO, input_addr);
    write_reg(CNN_ACCEL_CONTROL_R_BASE, REG_INPUT_FM_HI, 0);
    
    write_reg(CNN_ACCEL_CONTROL_R_BASE, REG_OUTPUT_FM_LO, output_addr);
    write_reg(CNN_ACCEL_CONTROL_R_BASE, REG_OUTPUT_FM_HI, 0);
    
    write_reg(CNN_ACCEL_CONTROL_R_BASE, REG_WEIGHTS_LO, weights_addr);
    write_reg(CNN_ACCEL_CONTROL_R_BASE, REG_WEIGHTS_HI, 0);
    
    write_reg(CNN_ACCEL_CONTROL_R_BASE, REG_BN_SCALE_LO, bn_scale_addr);
    write_reg(CNN_ACCEL_CONTROL_R_BASE, REG_BN_SCALE_HI, 0);
    
    write_reg(CNN_ACCEL_CONTROL_R_BASE, REG_BN_SHIFT_LO, bn_shift_addr);
    write_reg(CNN_ACCEL_CONTROL_R_BASE, REG_BN_SHIFT_HI, 0);
}

void cnn_accel_start(void) {
    write_reg(CNN_ACCEL_CONTROL_BASE, REG_AP_CTRL, AP_START);
}

void cnn_accel_run_layer(
    const LayerConfig *cfg,
    uint32_t input_addr,
    uint32_t output_addr,
    uint32_t weights_addr,
    uint32_t bn_scale_addr,
    uint32_t bn_shift_addr
) {
    // Wait for ready
    while (!cnn_accel_is_ready()) {
        // Busy wait
    }
    
    // Configure layer
    cnn_accel_configure_layer(cfg);
    
    // Set addresses
    cnn_accel_set_addresses(input_addr, output_addr, weights_addr, bn_scale_addr, bn_shift_addr);
    
    // Start
    cnn_accel_start();
    
    // Wait for completion
    cnn_accel_wait_done();
}

uint32_t cnn_accel_get_status(void) {
    return read_reg(CNN_ACCEL_CONTROL_BASE, REG_STATUS);
}

/*******************************************************************************
 * YOLO Lite Network Runner
 * Layer structure: 7 conv layers with pooling
 ******************************************************************************/
 
// YOLO Lite layer configurations
static const LayerConfig yolo_lite_layers[] = {
    // Layer 0: Conv 224x224x3 -> 112x112x16 (conv+bn+relu+pool)
    {1, 3, 16, 224, 224, 3, 1, 1},
    
    // Layer 1: Conv 112x112x16 -> 56x56x32 (conv+bn+relu+pool)
    {1, 16, 32, 112, 112, 3, 1, 1},
    
    // Layer 2: Conv 56x56x32 -> 28x28x64 (conv+bn+relu+pool)
    {1, 32, 64, 56, 56, 3, 1, 1},
    
    // Layer 3: Conv 28x28x64 -> 14x14x128 (conv+bn+relu+pool)
    {1, 64, 128, 28, 28, 3, 1, 1},
    
    // Layer 4: Conv 14x14x128 -> 14x14x256 (conv+bn+relu, no pool)
    {0, 128, 256, 14, 14, 3, 1, 1},
    
    // Layer 5: Conv 14x14x256 -> 7x7x512 (conv+bn+relu+pool)
    {1, 256, 512, 14, 14, 3, 1, 1},
    
    // Layer 6: Conv 7x7x512 -> 7x7x125 (output, conv only)
    {2, 512, 125, 7, 7, 1, 1, 0}
};

#define NUM_YOLO_LAYERS (sizeof(yolo_lite_layers) / sizeof(yolo_lite_layers[0]))

// Weight offsets for each layer (pre-calculated based on layer sizes)
static const uint32_t weight_offsets[] = {
    0,                    // Layer 0: 3*16*3*3 = 432
    432 * 2,              // Layer 1: 16*32*3*3 = 4608
    (432 + 4608) * 2,     // Layer 2: 32*64*3*3 = 18432
    // ... continue for all layers
};

void cnn_accel_run_yolo_lite(uint32_t image_addr, uint32_t output_addr) {
    printf("[CNN] Running YOLO Lite inference...\n");
    
    uint32_t current_input = image_addr;
    uint32_t current_output = DDR_OUTPUT_FM_ADDR;
    uint32_t weight_ptr = DDR_WEIGHTS_ADDR;
    uint32_t bn_scale_ptr = DDR_BN_SCALE_ADDR;
    uint32_t bn_shift_ptr = DDR_BN_SHIFT_ADDR;
    
    for (int i = 0; i < NUM_YOLO_LAYERS; i++) {
        printf("[CNN] Layer %d: %dx%dx%d -> out_ch=%d\n", 
               i, yolo_lite_layers[i].in_height, yolo_lite_layers[i].in_width,
               yolo_lite_layers[i].in_channels, yolo_lite_layers[i].out_channels);
        
        // Calculate weight size for this layer
        int weight_size = yolo_lite_layers[i].in_channels * 
                         yolo_lite_layers[i].out_channels *
                         yolo_lite_layers[i].kernel_size *
                         yolo_lite_layers[i].kernel_size * 2;  // 16-bit = 2 bytes
        
        int bn_size = yolo_lite_layers[i].out_channels * 2;  // 16-bit = 2 bytes
        
        // Run layer
        cnn_accel_run_layer(
            &yolo_lite_layers[i],
            current_input,
            current_output,
            weight_ptr,
            bn_scale_ptr,
            bn_shift_ptr
        );
        
        // Update pointers for next layer
        weight_ptr += weight_size;
        bn_scale_ptr += bn_size;
        bn_shift_ptr += bn_size;
        
        // Swap input/output buffers
        uint32_t temp = current_input;
        current_input = current_output;
        current_output = temp;
    }
    
    // Copy final output to requested address
    // (memcpy or DMA from current_input to output_addr)
    
    printf("[CNN] YOLO Lite inference complete!\n");
}
