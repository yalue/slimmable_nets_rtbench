# This script generates a filtered trace with rocprof, deleting the other crap
# rocprof spews out.

rocprof --hip-trace -o rocprof_trace.csv python rtbenchmark.py --job_count 10
ruby filter_traces.rb rocprof_trace.json rocprof_trace_filtered.json
rm rocprof_trace.*

