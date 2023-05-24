#!/bin/bash

# set the proxy environment variables
export http_proxy=http://localhost:7890
export https_proxy=http://localhost:7890

# Extract the clash binary
gunzip clash-linux-amd64-v1.15.1.gz

# Run clash in the background
./clash-linux-amd64-v1.15.1 