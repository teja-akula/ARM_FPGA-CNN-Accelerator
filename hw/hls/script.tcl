# ==============================================================================
# Vitis HLS Build Script - CNN Accelerator
# ==============================================================================
# Usage:
#   vitis_hls -f script.tcl              # Synthesis only
#   vitis_hls -f script.tcl -tclargs csim    # C Simulation
#   vitis_hls -f script.tcl -tclargs cosim   # RTL Co-Simulation
#   vitis_hls -f script.tcl -tclargs export  # Export IP
# ==============================================================================

# Project configuration
set project_name "cnn_accelerator"
set solution_name "solution1"
set top_function "cnn_accelerator_top"
set part_name "xc7z020clg484-1"
set clock_period 10

# Get script directory
set script_dir [file dirname [info script]]
set src_dir "$script_dir/src"
set tb_dir "$script_dir/tb"

# Parse command line argument
set run_mode "synth"
if {$argc > 0} {
    set run_mode [lindex $argv 0]
}

puts "====================================="
puts "CNN Accelerator HLS Build"
puts "====================================="
puts "Mode: $run_mode"
puts "Part: $part_name"
puts "Clock: ${clock_period}ns (100MHz)"
puts "====================================="

# Create or open project
open_project -reset $project_name

# Add source files
add_files "$src_dir/cnn_accel.h"
add_files "$src_dir/cnn_accel.cpp"
add_files "$src_dir/conv_layer.cpp"
add_files "$src_dir/activation.cpp"
add_files "$src_dir/pooling.cpp"

# Add testbench
add_files -tb "$tb_dir/tb_cnn_accel.cpp"

# Set top function
set_top $top_function

# Create/open solution
open_solution -reset -flow_target vivado $solution_name

# Set target FPGA
set_part $part_name

# Set clock
create_clock -period $clock_period -name default

# Configure solution
config_compile -pipeline_loops 64
config_interface -m_axi_alignment_byte_size 64
config_interface -m_axi_max_widen_bitwidth 512

# Run based on mode
switch $run_mode {
    "csim" {
        puts "\n>>> Running C Simulation..."
        csim_design
    }
    "synth" {
        puts "\n>>> Running C Synthesis..."
        csynth_design
    }
    "cosim" {
        puts "\n>>> Running C Synthesis..."
        csynth_design
        puts "\n>>> Running RTL Co-Simulation..."
        cosim_design -rtl verilog
    }
    "export" {
        puts "\n>>> Running C Synthesis..."
        csynth_design
        puts "\n>>> Exporting IP..."
        export_design -rtl verilog -format ip_catalog -output "$script_dir/ip_output"
    }
    "all" {
        puts "\n>>> Running Full Flow..."
        csim_design
        csynth_design
        cosim_design -rtl verilog
        export_design -rtl verilog -format ip_catalog -output "$script_dir/ip_output"
    }
    default {
        puts "\n>>> Running C Synthesis (default)..."
        csynth_design
    }
}

puts "\n====================================="
puts "Build Complete!"
puts "====================================="

exit
