#!/bin/bash

# Run setup.sh to prepare the sift dataset
# Only run this file manually in case something went wrong during setup.

set -e

rm -rf t10k-images* train-images* mnist.dat
