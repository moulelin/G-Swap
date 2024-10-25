#!/bin/bash
python setup.py install --record files.txt
cd ..
python demo.py
cd -


