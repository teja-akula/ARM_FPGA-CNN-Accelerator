/*******************************************************************************
 * CNN Accelerator ARM Driver
 * For Zedboard Zynq-7020 + CNN HLS Accelerator
 * 
 * Based on synthesized register map from Vitis HLS
 ******************************************************************************/

#ifndef CNN_DRIVER_H
#define CNN_DRIVER_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*******************************************************************************
 * Base Addresses (from Vivado address assignment)
 ******************************************************************************/
#define CNN_ACCEL_CONTROL_BASE    0x40000000  // s_axi_control (from Vivado)
#define CNN_ACCEL_CONTROL_R_BASE  0x40010000  // s_axi_control_r (DDR addresses)

/*******************************************************************************
 * DDR Memory Layout for CNN Accelerator
 ******************************************************************************/
#define DDR_INPUT_FM_ADDR      0x10000000  // Input image/feature maps
#define DDR_OUTPUT_FM_ADDR     0x14000000  // Output feature maps
#define DDR_WEIGHTS_ADDR       0x18000000  // Network weights
#define DDR_BN_SCALE_ADDR      0x1C000000  // BatchNorm scale
#define DDR_BN_SHIFT_ADDR      0x1C010000  // BatchNorm shift

/*******************************************************************************
 * Register Map - s_axi_control (Layer Configuration)
 * Offsets from HLS synthesis report
 ******************************************************************************/
#define REG_AP_CTRL         0x00  // Control: [0]=start, [1]=done, [2]=idle, [3]=ready
#define REG_GIE             0x04  // Global Interrupt Enable
#define REG_IP_IER          0x08  // IP Interrupt Enable
#define REG_IP_ISR          0x0C  // IP Interrupt Status
#define REG_CONTROL         0x10  // Control signal (unused)
#define REG_STATUS          0x18  // Status output
#define REG_STATUS_CTRL     0x1C  // Status control (ap_vld)
#define REG_LAYER_TYPE      0x28  // Layer type (0=conv+bn+relu, 1=+pool, 2=conv only)
#define REG_IN_CHANNELS     0x30  // Input channels
#define REG_OUT_CHANNELS    0x38  // Output channels
#define REG_IN_HEIGHT       0x40  // Input height
#define REG_IN_WIDTH        0x48  // Input width
#define REG_KERNEL_SIZE     0x50  // Kernel size (1 or 3)
#define REG_STRIDE          0x58  // Stride
#define REG_PADDING         0x60  // Padding

/*******************************************************************************
 * Register Map - s_axi_control_r (DDR Addresses, 64-bit)
 ******************************************************************************/
#define REG_INPUT_FM_LO     0x10  // Input FM address [31:0]
#define REG_INPUT_FM_HI     0x14  // Input FM address [63:32]
#define REG_OUTPUT_FM_LO    0x1C  // Output FM address [31:0]
#define REG_OUTPUT_FM_HI    0x20  // Output FM address [63:32]
#define REG_WEIGHTS_LO      0x28  // Weights address [31:0]
#define REG_WEIGHTS_HI      0x2C  // Weights address [63:32]
#define REG_BN_SCALE_LO     0x34  // BN scale address [31:0]
#define REG_BN_SCALE_HI     0x38  // BN scale address [63:32]
#define REG_BN_SHIFT_LO     0x40  // BN shift address [31:0]
#define REG_BN_SHIFT_HI     0x44  // BN shift address [63:32]

/*******************************************************************************
 * AP_CTRL Bit Definitions
 ******************************************************************************/
#define AP_START    (1 << 0)
#define AP_DONE     (1 << 1)
#define AP_IDLE     (1 << 2)
#define AP_READY    (1 << 3)
#define AP_AUTO_RESTART  (1 << 7)

/*******************************************************************************
 * Layer Configuration Structure
 ******************************************************************************/
typedef struct {
    int layer_type;      // 0: conv+bn+relu, 1: conv+bn+relu+pool, 2: conv only
    int in_channels;
    int out_channels;
    int in_height;
    int in_width;
    int kernel_size;     // 1 or 3
    int stride;
    int padding;
} LayerConfig;

/*******************************************************************************
 * Driver Functions
 ******************************************************************************/

// Initialize accelerator (returns 0 on success)
int cnn_accel_init(void);

// Check if accelerator is ready
int cnn_accel_is_ready(void);

// Check if accelerator is done
int cnn_accel_is_done(void);

// Wait for accelerator to complete
void cnn_accel_wait_done(void);

// Configure layer parameters
void cnn_accel_configure_layer(const LayerConfig *cfg);

// Set DDR addresses
void cnn_accel_set_addresses(
    uint32_t input_addr,
    uint32_t output_addr,
    uint32_t weights_addr,
    uint32_t bn_scale_addr,
    uint32_t bn_shift_addr
);

// Start accelerator
void cnn_accel_start(void);

// Run a single layer (configure + start + wait)
void cnn_accel_run_layer(
    const LayerConfig *cfg,
    uint32_t input_addr,
    uint32_t output_addr,
    uint32_t weights_addr,
    uint32_t bn_scale_addr,
    uint32_t bn_shift_addr
);

// Get status
uint32_t cnn_accel_get_status(void);

#ifdef __cplusplus
}
#endif

#endif // CNN_DRIVER_H
