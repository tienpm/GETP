#!/bin/bash

# Get application name and flags from command line arguments
PROGRAM_NAME=$1
shift
PROGRAM_FLAGS=$@
M = 2048
N = 2048
K = 2048

# Get current time for file names
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Profiling directories
NSYS_PROFILE_DIR=".profile_$TIMESTAMP"
NCU_PROFILE_DIR=".profile_$TIMESTAMP"

# Create profiling directories
mkdir -p "$NSYS_PROFILE_DIR" "$NCU_PROFILE_DIR"

# Profile with Nsys
echo "Profiling with Nsys..."
nsys profile --stats=true -o "$NSYS_PROFILE_DIR/profile_$TIMESTAMP.nsys-rep" $PROGRAM_NAME $PROGRAM_FLAGS $M $N $K

# Profile with Ncu
echo "Profiling with Ncu..."
ncu --set full -o "$NCU_PROFILE_DIR/profile_$TIMESTAMP.ncu-rep" $PROGRAM_NAME $PROGRAM_FLAGS $M $N $K

echo "Profiling complete! Reports generated in:"
echo "- Nsys: $NSYS_PROFILE_DIR/profile_$TIMESTAMP.nsys-rep"
echo "- Ncu: $NCU_PROFILE_DIR/profile_$TIMESTAMP.ncu-rep"

# Open profiling tools (optional)
# Replace these commands with your preferred launchers
# nsys open "$NSYS_PROFILE_DIR/profile_$TIMESTAMP.nsys"
# nsight compute "$NCU_PROFILE_DIR/profile_$TIMESTAMP.nsight-cuprof"
