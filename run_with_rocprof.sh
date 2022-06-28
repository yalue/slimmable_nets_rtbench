#!/bin/bash
# This script generates a filtered trace with rocprof, deleting the other crap
# rocprof spews out.
#
# Usage: bash run_with_rocprof.sh <JSON filename w/o extension> [args to rtbenchmark.py]

OUTPUT_NAME=$1
shift

rocprof --hip-trace -o "$OUTPUT_NAME.csv" python rtbenchmark.py --insert_trace_marks --output_file "$OUTPUT_NAME.jobs.json" "$@"
ruby filter_traces.rb "$OUTPUT_NAME.json" "$OUTPUT_NAME.jobs.json" "results/$OUTPUT_NAME.json"
rm $OUTPUT_NAME.*

