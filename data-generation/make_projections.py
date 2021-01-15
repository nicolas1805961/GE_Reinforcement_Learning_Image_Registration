#!/usr/bin/python3

from __future__ import division

import numpy as np
from os import mkdir
from os.path import join, isdir
from imageio import get_writer

import argparse
import glob
import matplotlib.pyplot as plt
import os
import pydicom
import sys

import astra


# Parse arguments
parser = argparse.ArgumentParser(description='Generate projections from DICOM files.')
parser.add_argument(
    'folders', metavar='folder', type=str, nargs='+', help='Folders storing DICOM files.'
)
args = parser.parse_args()

# Load the DICOM files into dictionary "files"
directories_names = args.folders

directories = {}
for d in directories_names:
    directories[d] = []
    for filename in os.listdir(d):
        if filename.endswith(".dcm"):
            print("Loading: {}".format(filename), end='\r', flush=True)
            directories[d].append(pydicom.dcmread(d + filename))
    print('')

# Skip files with no SliceLocation (eg scout views)
slices = {}
print("\nLoading slices...")
for d in directories:
    skipcount = 0
    print(d + "...")
    slices[d] = []
    for f in directories[d]:
        if hasattr(f, 'ImagePositionPatient'):
            slices[d].append(f)
        else:
            skipcount += 1
    print("Skipped, no ImagePositionPatient for", d, ": {}".format(skipcount))

# Ensure they are in the correct order
for d in slices:
   slices[d] = sorted(slices[d], key=lambda s: -s.ImagePositionPatient[2])

# Configuration
distance_source_origin = 300  # [mm]
distance_origin_detector = 200  # [mm]
detector_pixel_size = 1.05  # [mm] base : 1.05
num_of_projections = 180
angles = np.linspace(0, 2 * np.pi, num=num_of_projections, endpoint=False)
output_dir = 'dataset_'

# Create Numpy arrays
img3d       = {}
img_shape   = {}
print("\nCreating 3D arrays...")
for d in slices:
    img_shape[d] = list(slices[d][0].pixel_array.shape)
    img_shape[d].append(len(slices[d]))
    img3d[d] = np.zeros(img_shape[d], np.uint8)

# Fill 3D array with the images from the files
print("\nFilling 3D arrays...")
for d in slices:
    for i, s in enumerate(slices[d]):
        print(f"{d} : {i}th slice", end='\r', flush=True)
        img2d = s.pixel_array
        img3d[d][:, :, i] = img2d
    print('')

# Transpose to get different rotation viewpoint
for key in img3d:
    img3d[key] = img3d[key].T

for key in img3d:
    print(f"CURRENT DIR : {key}")
    print(f"Data shape : {img3d[key].shape}")
    # Configure geometry
    slice_nb      = img3d[key].shape[0]
    detector_rows = img3d[key].shape[1]
    detector_cols = img3d[key].shape[2]
    vol_geom = astra.creators.create_vol_geom(
            detector_rows,
            detector_cols,
            slice_nb,
    )
    # Link data to geometry
    phantom_id = astra.data3d.create('-vol', vol_geom, data=img3d[key])

    # Create projections. With increasing angles, the projection are such that the
    # object is rotated clockwise. Slice zero is at the top of the object. The
    # projection from angle zero looks upwards from the bottom of the slice.
    proj_geom = \
      astra.create_proj_geom('cone', 1, 1, detector_rows, detector_cols, angles,
                             (distance_source_origin + distance_origin_detector) /
                             detector_pixel_size, 0)
    projections_id, projections = \
      astra.creators.create_sino3d_gpu(phantom_id, proj_geom, vol_geom)
    projections /= np.max(projections)

    # Apply Poisson noise.
    projections = np.random.poisson(projections * 10000) / 10000
    projections[projections > 1.1] = 1.1
    projections /= 1.1

    # Save projections.
    tmp_output_dir = output_dir + key
    if not isdir(tmp_output_dir):
        mkdir(tmp_output_dir)
    projections = np.round(projections * 65535).astype(np.uint8)

    for i in range(num_of_projections):
        projection = projections[:, i, :]
        with get_writer(join(tmp_output_dir, 'proj%04d.tif' %i)) as writer:
            writer.append_data(projection, {'compress': 9})

    # Cleanup.
    astra.data3d.delete(projections_id)
    astra.data3d.delete(phantom_id)
