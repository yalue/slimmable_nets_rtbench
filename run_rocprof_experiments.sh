#!/bin/bash
# This script just runs a series of experiments to collect data.

./run_with_rocprof.sh full_width --width_mult 1.0 --use_data_blobs
./run_with_rocprof.sh 50_width --width_mult 0.50 --use_data_blobs
./run_with_rocprof.sh 25_width --width_mult 0.25 --use_data_blobs

./run_with_rocprof.sh 64_batch --batch_size 64 --use_data_blobs
./run_with_rocprof.sh 128_batch --batch_size 128 --use_data_blobs

./run_with_rocprof.sh 25_width_64_batch --batch_size 64 --width_mult 0.25 --use_data_blobs
./run_with_rocprof.sh 25_width_128_batch --batch_size 128 --width_mult 0.25 --use_data_blobs

