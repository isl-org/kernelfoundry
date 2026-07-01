#!/bin/bash

# Script to retry unitrace until we get a valid metrics file
# Usage: retry_unitrace.sh <unitrace_cmd> <output_dir> <output_prefix> <group> <timeline_flag> <target_cmd>

set -e

UNITRACE_CMD="$1"
OUTPUT_DIR="$2" 
OUTPUT_PREFIX="$3"
GROUP="$4"
TIMELINE_FLAG="$5"
SESSION_FLAG=$6
TARGET_CMD="$7"

MAX_RETRIES=20
MIN_METRICS_LINES=10  # Require at least 10 lines for a valid metrics file

echo "Starting unitrace retry script..."
echo "Output directory: $OUTPUT_DIR"
echo "Max retries: $MAX_RETRIES"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

for attempt in $(seq 1 $MAX_RETRIES); do
    echo "=============================="
    echo "ATTEMPT $attempt of $MAX_RETRIES"
    echo "=============================="
    
    # Clean up any existing trace files from previous attempts
    rm -f "$OUTPUT_DIR"/trace.metrics.* 2>/dev/null || true
    rm -f "$OUTPUT_DIR"/trace.* 2>/dev/null || true
    rm -f "$OUTPUT_DIR"/*.json 2>/dev/null || true
    
    # Run the unitrace command
    echo "Running: PTI_ENABLE_COLLECTION=0 $UNITRACE_CMD --opencl $TIMELINE_FLAG $SESSION_FLAG --group $GROUP --metric-sampling --output-dir-path $OUTPUT_DIR -o $OUTPUT_PREFIX $TARGET_CMD"
    
    # Capture output to extract metrics file path
    output=$(PTI_ENABLE_COLLECTION=0 $UNITRACE_CMD --opencl $TIMELINE_FLAG $SESSION_FLAG --group $GROUP --metric-sampling --output-dir-path "$OUTPUT_DIR" -o "$OUTPUT_PREFIX" $TARGET_CMD 2>&1) || {
        echo "Command failed on attempt $attempt, retrying..."
        continue
    }
    
    echo "Command output:"
    echo "$output"
    
    # Look for metrics files
    metrics_files=($(find "$OUTPUT_DIR" -name "trace.metrics.*" 2>/dev/null))
    
    if [ ${#metrics_files[@]} -eq 0 ]; then
        echo "No metrics file found on attempt $attempt"
        continue
    fi
    
    # Check the most recent metrics file
    metrics_file="${metrics_files[-1]}"  # Get the last (most recent) file
    
    if [ ! -f "$metrics_file" ]; then
        echo "Metrics file does not exist: $metrics_file"
        continue
    fi
    
    line_count=$(wc -l < "$metrics_file" 2>/dev/null || echo "0")
    echo "Metrics file: $metrics_file"
    echo "Number of lines in metrics file: $line_count"
    
    if [ "$line_count" -ge "$MIN_METRICS_LINES" ]; then
        echo ""  
        echo "SUCCESS! Found valid metrics file with $line_count lines"
        echo "Metrics file: $metrics_file"
        
        # Clean up any other invalid trace files, but keep the good ones
        for file in "$OUTPUT_DIR"/trace.metrics.*; do
            if [ "$file" != "$metrics_file" ]; then
                file_lines=$(wc -l < "$file" 2>/dev/null || echo "0")
                if [ "$file_lines" -lt "$MIN_METRICS_LINES" ]; then
                    echo "Removing invalid trace file: $file (only $file_lines lines)"
                    rm -f "$file"
                fi
            fi
        done
        
        echo "Unitrace completed successfully on attempt $attempt"
        echo "UNITRACE_DONE"
        exit 0
    else
        echo "Metrics file has only $line_count lines (need >= $MIN_METRICS_LINES), retrying..."
    fi
done

echo ""
echo "ERROR: Failed to get valid metrics file after $MAX_RETRIES attempts"
echo "UNITRACE_DONE"
exit 1
