#!/bin/bash

#Run to prepate mnist.dat dataset

set -e

#Flag to exist on any non-zero return values.

bash ./clean.sh
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
gunzip train-images-idx3-ubyte.gz
gunzip t10k-images-idx3-ubyte.gz
python3 parse.py
