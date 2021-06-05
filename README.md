# Stereo Matching using SGM based on python

## Introduction

This project is trying to implementing stereo matching by python based on the middleburry dataset. The whole process contains the following steps.

+ Census Transform
+ cost volume Calculation by hamming distance
+ cost aggregation
+ WTA to calculate the disparity
+ consistency check to exclude bad points
+ map the disparity to 0~255 and display

## Environment

+ Python 3.7
+ NumPy
+ OpenCV

## File structure

+ **dataset**: some standard images contained in middleburry dataset
+ **SGM_fun.py**: all the source codes