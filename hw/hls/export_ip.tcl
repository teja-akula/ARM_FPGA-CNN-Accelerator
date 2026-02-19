# Quick export script - run on existing synthesis
open_project cnn_accelerator
open_solution solution1
export_design -rtl verilog -format ip_catalog -output "./ip_output"
exit
