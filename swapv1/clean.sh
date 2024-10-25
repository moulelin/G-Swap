#!/bin/bash
rm -rf ./__pycache__
rm -rf ./dist
rm -rf ./build
rm -rf ./swap_kernel_linear_cpp.egg-info

cd /home1/mllin/anaconda3/lib/python3.9/site-packages
rm -rf swap*
cd -
