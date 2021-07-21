# [run] bash ./2_download_data.sh



#!/bin/bash
# For each Mesh model of Shape-Net-Core download 1 point-cloud with 2048 points
# sampled uniformly at random (around 1.4GB).

# wget for windows
# Download latest .exe file https://eternallybored.org/misc/wget/
# Copy the file into 'c:/windows/system32/'

wget -O shape_net_core_uniform_samples_2048.zip https://www.dropbox.com/s/vmsdrae6x5xws1v/shape_net_core_uniform_samples_2048.zip?dl=0

