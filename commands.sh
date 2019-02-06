#!/bin/bash
python -B main.py -datadir /data/milatmp1/sharmasa/KITTI/dataset/ -gradClip 45. -imageWidth 448 -imageHeight 192 -outputParameterization default -expID tmp
# -loadModel ./models/flownets_EPE1.951.pth.tar