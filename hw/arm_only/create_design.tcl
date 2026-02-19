################################################################################
# Block Design 1: ARM-Only (PS Only)
# Zynq PS with DDR3 + UART - No PL accelerator
# Used for ARM-only CNN inference baseline
################################################################################

# Create project
create_project arm_only_design ./arm_only_design -part xc7z020clg484-1
set_property board_part digilentinc.com:zedboard:part0:1.0 [current_project]

# Create block design
create_bd_design "arm_only"

# Add Zynq PS
create_bd_cell -type ip -vlnv xilinx.com:ip:processing_system7:5.5 ps7
apply_bd_automation -rule xilinx.com:bd_rule:processing_system7 \
    -config {make_external "FIXED_IO, DDR" apply_board_preset "1" } [get_bd_cells ps7]

# Configure PS: Enable UART, set clocks
set_property -dict [list \
    CONFIG.PCW_USE_M_AXI_GP0 {0} \
    CONFIG.PCW_UART1_PERIPHERAL_ENABLE {1} \
    CONFIG.PCW_FPGA0_PERIPHERAL_FREQMHZ {100} \
] [get_bd_cells ps7]

# Validate and save
validate_bd_design
save_bd_design

# Generate wrapper
make_wrapper -files [get_files arm_only.bd] -top
add_files -norecurse arm_only_design/arm_only_design.gen/sources_1/bd/arm_only/hdl/arm_only_wrapper.v

# Synthesize and implement
launch_runs synth_1 -jobs 4
wait_on_run synth_1
launch_runs impl_1 -to_step write_bitstream -jobs 4
wait_on_run impl_1

# Export hardware (XSA) for Vitis
write_hw_platform -fixed -include_bit \
    -file arm_only_design/arm_only.xsa

puts "ARM-Only block design complete."
puts "XSA exported: arm_only.xsa"
