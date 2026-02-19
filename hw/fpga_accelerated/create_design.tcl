################################################################################
# Block Design 2: FPGA Accelerated (PS + PL CNN Accelerator)
# Zynq PS with DDR3 + UART + HLS CNN Accelerator via AXI
# Used for FPGA PL accelerated CNN inference
################################################################################

# Create project
create_project fpga_accel_design ./fpga_accel_design -part xc7z020clg484-1
set_property board_part digilentinc.com:zedboard:part0:1.0 [current_project]

# Add HLS IP repository
set_property ip_repo_paths ../hls/ip_export [current_project]
update_ip_catalog

# Create block design
create_bd_design "fpga_accel"

# Add Zynq PS
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" } [get_bd_cells ps7]

# Configure PS: Enable AXI GP + HP ports, UART, clocks
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {1} \
    CONFIG.PCW_USE_S_AXI_HP0 {1} \
    CONFIG.PCW_UART1_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
] [get_bd_cells ps7]

################################################################################
# Add CNN Accelerator IP (HLS-generated)
################################################################################
create_bd_cell -type ip -vlnv xilinx.com:hls:cnn_accel:1.0 cnn_accel_0

################################################################################
# AXI Interconnect for Control (AXI-Lite: PS GP0 → CNN Accel s_axi_control)
################################################################################
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_ctrl
set_property CONFIG.NUM_MI {1} [get_bd_cells axi_ctrl]

# PS M_AXI_GP0 → AXI Interconnect → CNN Accel (control registers)
connect_bd_intf_net [get_bd_intf_pins ps7/M_AXI_GP0] \
                    [get_bd_intf_pins axi_ctrl/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_ctrl/M00_AXI] \
                    [get_bd_intf_pins cnn_accel_0/s_axi_control]

################################################################################
# AXI Interconnect for Data (AXI Master: CNN Accel m_axi_data → PS HP0 → DDR)
################################################################################
create_bd_cell -type ip -vlnv xilinx.com:ip:axi_interconnect:2.1 axi_data
set_property CONFIG.NUM_SI {1} [get_bd_cells axi_data]
set_property CONFIG.NUM_MI {1} [get_bd_cells axi_data]

# CNN Accel m_axi_data → AXI Interconnect → PS S_AXI_HP0 (DDR access)
connect_bd_intf_net [get_bd_intf_pins cnn_accel_0/m_axi_data] \
                    [get_bd_intf_pins axi_data/S00_AXI]
connect_bd_intf_net [get_bd_intf_pins axi_data/M00_AXI] \
                    [get_bd_intf_pins ps7/S_AXI_HP0]

################################################################################
# Clock and Reset connections
################################################################################
# PL clock from PS (100 MHz)
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] \
               [get_bd_pins cnn_accel_0/ap_clk] \
               [get_bd_pins axi_ctrl/ACLK] \
               [get_bd_pins axi_ctrl/S00_ACLK] \
               [get_bd_pins axi_ctrl/M00_ACLK] \
               [get_bd_pins axi_data/ACLK] \
               [get_bd_pins axi_data/S00_ACLK] \
               [get_bd_pins axi_data/M00_ACLK] \
               [get_bd_pins ps7/M_AXI_GP0_ACLK] \
               [get_bd_pins ps7/S_AXI_HP0_ACLK]

# Reset
create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset:5.0 rst_ps7
connect_bd_net [get_bd_pins ps7/FCLK_CLK0] \
               [get_bd_pins rst_ps7/slowest_sync_clk]
connect_bd_net [get_bd_pins ps7/FCLK_RESET0_N] \
               [get_bd_pins rst_ps7/ext_reset_in]

connect_bd_net [get_bd_pins rst_ps7/peripheral_aresetn] \
               [get_bd_pins cnn_accel_0/ap_rst_n] \
               [get_bd_pins axi_ctrl/ARESETN] \
               [get_bd_pins axi_ctrl/S00_ARESETN] \
               [get_bd_pins axi_ctrl/M00_ARESETN] \
               [get_bd_pins axi_data/ARESETN] \
               [get_bd_pins axi_data/S00_ARESETN] \
               [get_bd_pins axi_data/M00_ARESETN]

################################################################################
# Address mapping
################################################################################
# CNN Accel control registers: 0x43C00000
assign_bd_address [get_bd_addr_segs cnn_accel_0/s_axi_control/reg0]

# CNN Accel DDR access: full 512MB range
assign_bd_address [get_bd_addr_segs ps7/S_AXI_HP0/HP0_DDR_LOWOCM]

################################################################################
# Validate, synthesize, implement
################################################################################
validate_bd_design
save_bd_design

make_wrapper -files [get_files fpga_accel.bd] -top
add_files -norecurse fpga_accel_design/fpga_accel_design.gen/sources_1/bd/fpga_accel/hdl/fpga_accel_wrapper.v

launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# Export hardware
write_hw_platform -fixed -include_bit \
    -file fpga_accel_design/fpga_accel.xsa

puts "FPGA Accelerated block design complete."
puts "XSA exported: fpga_accel.xsa"
puts ""
puts "Block Design Summary:"
puts "  PS (ARM):  Cortex-A9 @ 667 MHz"
puts "  PL (CNN):  HLS Accelerator @ 100 MHz"
puts "  AXI-Lite:  PS GP0 -> CNN Accel (control)"
puts "  AXI HP0:   CNN Accel -> DDR (data)"
