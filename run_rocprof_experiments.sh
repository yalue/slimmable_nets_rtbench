#!/bin/bash
# This script just runs a series of experiments to collect data.

./run_with_rocprof.sh full_width --job_count 10 --width_mult 1.0 --use_data_blobs
./run_with_rocprof.sh 50_width --job_count 10 --width_mult 0.50 --use_data_blobs
./run_with_rocprof.sh 25_width --job_count 10 --width_mult 0.25 --use_data_blobs
./run_with_rocprof.sh 8_batch --job_count 10 --batch_size 8 --use_data_blobs
./run_with_rocprof.sh 32_batch --job_count 10 --batch_size 32 --use_data_blobs
./run_with_rocprof.sh 25_width_8_batch --job_count 10 --batch_size 8 --width_mult 0.25 --use_data_blobs
./run_with_rocprof.sh 25_width_32_batch --job_count 10 --batch_size 32 --width_mult 0.25 --use_data_blobs

