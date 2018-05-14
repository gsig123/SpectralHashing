#!/bin/bash

#Run this file to prepare the sift dataset into sift.dat

set -e
#Exit immediately if a pipeline (see Pipelines), which may consist of a single simple command (see Simple Commands), a list (see Lists), or a compound command (see Compound Commands) returns a non-zero status.

#Read about dataset here http://corpus-texmex.irisa.fr/#matlab

bash ./clean.sh
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz
tar -xf sift.tar.gz
python3 ./parse.py
