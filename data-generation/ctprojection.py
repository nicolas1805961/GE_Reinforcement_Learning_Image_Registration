#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 16:33:04 2019

This file simulates fluoroscopic images from projections of a CT volume.
1 / input geometry (projection matrices and rigid tranform of the input CT
volume) is extracted from Vision application.
    The file performs conversion of the geometry from Dicom and Vision
    conventions to Pyramids/Astra, to reach the equivalent point of view
2 / This file also applies the relevant steps of the image chain to produce
    fluoroscopic like images (rescale, noise, DRM, contrast)

@author: 100042937
"""

import argparse
import astra
import numpy as np
import pydicom
import time
import os

#%%

def load_from_dicom_folder(address_case):
    """
    Load a volume and the voxel sizes from a folder containing the dicom files
    of the reformatted slices.\n
    Parameters
    ----------
    address_case : str
        Full path to the folder
    Returns
    ----------
    vol : numpy.array
        the loaded volume
    l : list of floats
        the voxel size in in mm in x,y,z
    2 : list of floats
        position of the center of first voxel of the volume
    """

    files = os.listdir(address_case)
    imgList = []
    for f in files:
        try:
            im = pydicom.read_file(address_case+"/"+f)
            imgList.append((im.ImagePositionPatient[2], im))
        except pydicom.errors.InvalidDicomError:
            pass

    # Sort slices in ascending order of im.ImagePositionPatient[2]
    imgList = sorted(imgList)

    # Create 3D volume in this order
    vol = np.array([imdcm.pixel_array for _, imdcm in imgList], dtype = im.pixel_array.dtype)
    vol[vol==-2000]=0

    # Compute interslice distance
    # Do not use SpacingBetweenSlices dicom tag that is seldom present,
    # neither SliceThickness that may be different from slice spacing
    slicePositions = np.array([pos for pos, _ in imgList])
    spacingBetweenSlices = slicePositions[1:] - slicePositions[:-1]
    if len(np.unique(np.round(spacingBetweenSlices,2))) > 1:
        print('WARNING interslice distance is not constant : ', np.unique(spacingBetweenSlices))
    slice_ct = np.median(spacingBetweenSlices)

    # Use first image to get volume imageOrigin - ie the one that has the lowest z position
    im0 = imgList[0][1]
    pix_ct = list(map(float, im0.PixelSpacing))
    vox_sizes = pix_ct + [slice_ct]
    imageOrigin = im0.ImagePositionPatient
    print("Volume shape is {}".format(vol.shape))
    print("Voxel size is {}".format(vox_sizes))
    print("Volume origin is {}".format(imageOrigin))
    # ratio_ct=slice_ct/pix_ct
    return vol,vox_sizes, imageOrigin


from scipy.ndimage import zoom
def make_isotropic(
        vol,vox_sizes,order_of_interpolation=1, vox_size_iso = None
):
    """
    Wrap the scipy.nidimage.zoom interpolation to make ct volumes isotropic in
    the z dimension.
    Although not clearly documented, resize considers the center
    of the first and last voxel of a volume.
    This means that origin of the volume (center of the fist voxel) remains
    unchanged.

    Parameters
    ----------
    vol : numpy.array
        the volume to turn iostropic
    vox_sizes: list of float
        the voxel size in x,y,z
    vox_size_iso (optional):
        final voxel size after resize
        byt default, final isotropic voxel size will be vox_sizes[0]

    Returns
    ----------
    vol_iso : numpy.array
        the interpolated volume
    """

    if vox_size_iso == None:
        ratio_ct = np.array(vox_sizes)[::-1]/vox_sizes[0]
    else:
        ratio_ct = np.array(vox_sizes)[::-1]/vox_size_iso
    return zoom(vol,ratio_ct,order=order_of_interpolation)


def computeCamToXrayRotWithPCangles(Pdeg, Cdeg, Ldeg=0):
    """
        returns the rotation from Camera CS (ie rotated with P, C) toXray CS.
        Cra and Lao are positive. L positive moves gantry to patient left.
        Cam CS has the same orientation as Xray CS. X is optical axis but
        pointing downwards,
               Y oriented with the lines of detector to the right
               Z oriented with the columns of the detector to patient feet
    """
    sC = np.sin(np.radians(-Cdeg)) #minus because C positive is cra
    cC = np.cos(np.radians(-Cdeg))
    sP = np.sin(np.radians(-Pdeg)) #minus because P positive is lao
    cP = np.cos(np.radians(-Pdeg))
    sL = np.sin(np.radians(Ldeg))
    cL = np.cos(np.radians(Ldeg))
    RP = np.array([[cP, -sP, 0], [sP, cP, 0],[0,0,1]])
    RC = np.array([[cC, 0, sC],[0,1,0], [-sC, 0, cC]])
    RL = np.array([[1, 0, 0],[0,cL,-sL], [0, sL, cL]])
    return np.linalg.multi_dot((RL,RP,RC));


def computeCameramWithPCangles(Pdeg, Cdeg, SOD, SID, pixSize, Ldeg=0):
    """
        transform system paramters into projection matrix
        world is Xray reference frame
           Cra and Lao are positive. L positive moves gantry to patient left.
    """
    camToXray = computeCamToXrayRotWithPCangles(Pdeg, Cdeg, Ldeg)
    FS = SOD*camToXray[:,0]
    kext = np.linalg.inv(camToXray).dot(np.vstack((np.identity(3), -FS )).T)
    kswap = np.zeros((3,3))
    kswap[2,0] = kswap[0,1] = kswap[1,2] = 1
    kint = np.identity(3)
    kint[0,0] = kint[1,1] = SID/pixSize
    kint[2,2] = -1
    return kint.dot(kswap).dot(kext)


def decomposeCameraFSdetector2(tdir, pixSize):
    """
    turn a projection matrix obtained from rvFeldkamp into a camera vector
    expected by astra.\n

    Parameters
    ----------
    tdir : 3x4 numpy.array
        projection matrix
    pixSize : float

    Returns
    ----------
    opticalCenter : 3x1 np.array

    detOrigin : 3x1 np.array

    U : 3x1 np.array
        the vector from detector pixel (0,0) to (0,1)
    V : 3x1 np.array
        the vector from detector pixel (0,0) to (1,0)
    """
    tdir = tdir / np.linalg.norm(tdir[2,:2])
    tdir3 = tdir[:,:3]
    b = -tdir[:,3]
    invtdir3 = np.linalg.inv(tdir3)
    opticalCenter = invtdir3.dot(b);
    normFactor = pixSize/np.mean(np.linalg.norm(invtdir3[:,:2], axis = 0))
    detOrigin = opticalCenter + (invtdir3[:,2]*normFactor)
    U = invtdir3[:,0] * normFactor
    V = invtdir3[:,1] * normFactor
    return(opticalCenter, detOrigin, U, V)


def generate_camera_vector(
        pos,
        vox_sizes_mm,
        shape_input,
        proj_mat_Xray_mm,
        input_img_size,
        output_img_size
):
    """
    transforms the input projection matrices expressed in Xray CS
    into a camera vector expected by astra.\n

    Parameters
    ----------
    pos : list of floats
        Coordinates in voxels (indz,indy,indx) of the point of the volume to
        project to put at the center of the FOV
    vox_sizes_mm : list of floats
        Size of the volume voxels in mm (order is (x,y,z))
    shape_input : tuple of ints
        Size of the volume to project (order is (z,y,x))
    proj_mat_Xray_mm : list of [3,4] arrays
        list of projection matrix
    input_img_size : int
        size in pixel of input image
    output_img_size : int
        size in pixel of output image allows image resize
    Returns
    ----------
    v : list of 12x1 np.array
        list of camera vectors corresponding
    """
    Mdet_center = np.eye(3)
    Mdet_center[:2,2]=-(input_img_size/2)
    Mdet_resize =np.diag([output_img_size/input_img_size]*2+[1])

    Swap_xray_to_ct=np.zeros([4,4])
    Swap_xray_to_ct[3,3] = 1
    Swap_xray_to_ct[2,2] = -1
    Swap_xray_to_ct[0,1] = 1
    Swap_xray_to_ct[1,0] = 1

    Mscale = np.diag(list(vox_sizes_mm)+[1.0])
    # place in t the coordinates of center in voxels - reverse order because
    # pos is returned in z, y, x
    t = -(np.array(pos)[::-1]-np.array(shape_input)[::-1]/2)
    Mtranslation = np.eye(4)
    Mtranslation[0:3,3]=t

    proj_mat_ct = [
            np.linalg.multi_dot((
                Mdet_resize,
                Mdet_center,
                p,
                Swap_xray_to_ct,
                Mscale,
                Mtranslation)) for p in proj_mat_Xray_mm
    ]
    v = np.vstack([
        np.concatenate(decomposeCameraFSdetector2(p, 1.0)) for p in proj_mat_ct
    ])
    return v


import matplotlib.pyplot as plt
''' interactive display of image series
    seq is a Nframe*nrows*ncolumns array
    capture mouse wheel, keyboqrd up,down,pageup,pagedown to page through
    images/slices of input array
'''
def displaySequence(seq, vmin = None, vmax = None):
    global curr_frame
    curr_frame = 0#current frame
    if vmin == None:
        vmin = seq.min()
    vmin = float(vmin)
    if vmax == None:
        vmax = seq.max()
    vmax = float(vmax)

    def key_event(e):#response to keyboard input
#        print('you pressed', e.key, e.xdata, e.ydata)
        global curr_frame
        #if arrows on the keyboard are pressed, change curr_frame
        if e.key == "up":
            curr_frame = curr_frame + 1
        elif e.key == "down":
            curr_frame = curr_frame - 1
        elif e.key == "pageup":
            curr_frame = curr_frame + 5
        elif e.key == "pagedown":
            curr_frame = curr_frame - 5
        else:
            return
        #handle end of the series
        curr_frame = curr_frame % seq.shape[0]
        #refresh axis
        ax.cla()#clear axis
        ax.imshow(seq[curr_frame], cmap='gray', vmin = vmin, vmax = vmax)
        ax.set_title("frame {0}".format(int(curr_frame)))
        fig.canvas.draw()#refresh

    def on_scroll(e):#response to mouse scroll input
        global curr_frame
        # update the frame with respect to the scroll input
        curr_frame = curr_frame - int(e.step)
        # handle end of series
        curr_frame = curr_frame % seq.shape[0]
        #refresh axis
        ax.cla()  # clear axis
        ax.imshow(seq[curr_frame], cmap='gray', vmin = vmin, vmax = vmax)
        ax.set_title("frame {0}".format(int(curr_frame)))
        fig.canvas.draw()  # refresh

    fig = plt.figure()  # Create figure
    # Connect the keyboard listener function key_event to the figure
    fig.canvas.mpl_connect('key_press_event', key_event)
    # connect the scroll listener function on_scroll to the figure
    fig.canvas.mpl_connect('scroll_event', on_scroll)
    ax = fig.add_subplot(111)  # Add axis to figure
    ax.imshow(seq[curr_frame], cmap='gray', vmin = vmin, vmax = vmax)
    ax.set_title("frame {0}".format(int(curr_frame)))
    plt.show()  # Show figure


def parse_args():
    parser = argparse.ArgumentParser(description='Generate projections from DICOM files.')
    parser.add_argument("folder", type=str, help="Location of folder storing your DICOM files.")
    return parser.parse_args()


def main():
    # Parse arguments
    path = parse_args().folder

    # Display axial
    vol, vox_sizes_mm, imageOrigin = load_from_dicom_folder(path)
    projViewer = displaySequence(vol)

    # Next : create and display projections

    # Select center of the projection, in voxels with order Z (slice), Y, X
    pos = (58, 350, 256)

    # Make the volume isotropic
    t1 = time.time()
    ct_isotropic = make_isotropic(vol,vox_sizes_mm,order_of_interpolation=1)
    pos_iso = [pos[0]*vox_sizes_mm[2]/vox_sizes_mm[0], pos[1], pos[2]]
    vox_size_iso_mm = vox_sizes_mm[0]
    t2 = time.time()
    print("Shape of interpolated CT :", ct_isotropic.shape)
    print(f"Time for interpolation : {t2-t1:.3f}s")

    # Make the volume isotropic
    LaoRaolist, CraCaulist = [
            x.flatten() for x in np.meshgrid(np.arange(-90,91,15),
            np.arange(-30,31,15))
    ]
    Llist = np.zeros(len(CraCaulist))
    SIDlist = np.random.randint(1000,1195,len(CraCaulist))
    FOVlist = [400]*len(CraCaulist)
    imgSize = 1000
    SOD = 820

    # tdir is expressed in Xray coordinates system, in mm to the right. in image
    # domain, unit is pixel and origin located at image center.
    tdirList = [
            computeCameramWithPCangles(
                Pdeg,
                Cdeg,
                SOD,
                SID,
                FOV/imgSize,
                Ldeg
            ) for ((
                Ldeg,
                Pdeg,
                Cdeg,
                SID,
                FOV
            )) in zip(Llist, LaoRaolist, CraCaulist,SIDlist,FOVlist)
    ]

    # Generate the geometry needed by astra
    v = generate_camera_vector(
            pos_iso, [vox_size_iso_mm] * 3,
            ct_isotropic.shape,tdirList,
            1,
            1
    )

    # Astra part, declare the volume and projection geometries, generate the spin
    # For 3D volume geometry, parameter order: rows, colums, slices (y, x, z)
    # NB : astra works with size 1 isotropic voxels
    nz,ny,nx = ct_isotropic.shape
    vol_geom_ctiso = astra.create_vol_geom(ny, nx, nz)
    proj_geom_ctiso = astra.create_proj_geom('cone_vec',imgSize,imgSize,v)
    t1 = time.time()
    proj_id_ctiso, proj_data_ctiso = astra.create_sino3d_gpu(
            ct_isotropic, proj_geom_ctiso, vol_geom_ctiso
    )
    # proj_data_ctiso*=vox_sizes_mm[0]/vox_cbct_mm
    t2 = time.time()
    print(f"time for generating {proj_data_ctiso.shape[1]} views = {t1-t1:.2f}s")

    projImg = proj_data_ctiso.transpose(1,0,2)
    I0 = 1
    mu = 1.837e-2 # (mm-1) for water at 80kV
    projImg1 = I0*np.exp((-mu*vox_size_iso_mm/1000.0)*projImg)

    # Rescaling . average of the center part of image at 512
    avgList = np.average(projImg1[:,250:-250,250:-250], axis=(1,2))
    projImg2 = np.asarray([proj*(512/avg) for proj, avg in zip(projImg1, avgList)])
    # projImg2 = np.max(projImg2,axis=(1,2),keepdims=True) - projImg2

    projViewer = displaySequence(np.log(projImg2))


if __name__ == "__main__":
    main()
