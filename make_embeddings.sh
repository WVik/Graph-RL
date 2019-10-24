#!/bin/bash

#python ./node2vec/graph/graphgennerd.py
echo graph_$1_dir_nerd

../nerd/NERD -train ../nerd/graph_$1_dir_nerd -output1 hub.txt -output2 auth.txt -binary 0 -size $2 -walkSize 5 -negative 5 -samples 10 -rho 0.025 -threads 20 -joint 1 -inputvertex 0
mv ../nerd/hub.txt source.txt
mv ../nerd/auth.txt dest.txt
