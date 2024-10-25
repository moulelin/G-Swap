#!/bin/bash
rm -rf ./__pycache__
rm -rf ./dist
rm -rf ./build
rm -rf ./swap.egg-info

cd /home/lml/anaconda3/lib/python3.9/site-packages
rm -rf swapv-*
cd -
