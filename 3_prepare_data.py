# [run] python 3_prepare_data.py

import shutil
import zipfile
import os

zip_file = zipfile.ZipFile('./shape_net_core_uniform_samples_2048.zip')
zip_file.extractall()

try:
    os.remove('./shape_net_core_uniform_samples_2048.zip')
except OSError as e:
    print("Error: %s - %s." % (e.filename, e.strerror))

os.mkdir('./data')

shutil.move('./shape_net_core_uniform_samples_2048', './data')
