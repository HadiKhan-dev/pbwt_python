#!/bin/bash
echo "Bash version ${BASH_VERSION}..."
for i in {0..100}
do
python pbwt_xgboost.py
done
