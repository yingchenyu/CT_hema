#!/usr/bin/python
# -*- coding: utf-8 -*-

import nibabel as nib
import numpy as np
import dicom
import os
import matplotlib.pyplot as plt
import scipy.ndimage
from preprocess_utils2 import load_scan, get_pixels_hu
from functools import reduce
from collections import Counter
import cv2
import imutils

def resample_nii(images, header, new_spacing=[5,0.8,0.8]):
    #4-D images, resample all channels
    # Determine current pixel spacing
    re_images = []
    spacing = np.array([header['pixdim'][3]] + header['pixdim'][1:3].tolist(), dtype=np.float32)
    shape = images.shape[:-1]
    resize_factor = spacing / new_spacing
    new_real_shape = shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / shape
    new_spacing = spacing / real_resize_factor
    for i in range(images.shape[-1]):
        re_images.append(scipy.ndimage.interpolation.zoom(images[:,:,:,i], real_resize_factor, mode='nearest'))
    re_images = np.asarray(re_images, dtype=np.float32)
    re_images = np.transpose(re_images,[1,2,3,0])
    return re_images


def resample_dicom(images, Slice, new_spacing=[5,0.8,0.8]):
    #4-D images, resample all channels
    # Determine current pixel spacing
    re_images = []
    spacing = np.array([Slice.SliceThickness] + Slice.PixelSpacing, dtype=np.float32)
    shape = images.shape[:-1]
    resize_factor = spacing / new_spacing
    new_real_shape = shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / shape
    new_spacing = spacing / real_resize_factor
    for i in range(images.shape[-1]):
        re_images.append(scipy.ndimage.interpolation.zoom(images[:,:,:,i], real_resize_factor, mode='nearest'))
    re_images = np.asarray(re_images, dtype=np.float32)
    re_images = np.transpose(re_images,[1,2,3,0])
    return re_images

def match_hw(img1):
    #Inputs are 3-D image, (Depth, Height, Width)
    #Make image square
    if img1.shape[2] != img1.shape[1]:
        shape_diff = img1.shape[2] - img1.shape[1]
        img1 = img1[:,:,shape_diff//2:-shape_diff//2]
    return img1
def match_size(img1, img2):
    #Match two 4-D images (Depth, Height, Width, Channel)
    diff = abs(img2.shape[2] - img1.shape[2])
    if (img2.shape[1] > img1.shape[1]):
        img2 = img2[:,diff//2:-diff//2,diff//2:-diff//2,:]
        # if diff%2 !=0:
        #     img2 = img2[:,0:-1,0:-1,:]
    elif (img2.shape[1] < img1.shape[1]):
        img1 = img1[:,diff//2:-diff//2,diff//2:-diff//2,:]
        # if diff%2 !=0:
        #     img1 = img1[:,0:-1,0:-1,:]
    return img1, img2

def find_index_around(find_list, find_value):
    #Find index in list if nearly equal
    Threshold = 1
    for i in range(len(find_list)):
        if abs(find_list[i] - find_value) < Threshold:
            return i

def centralize(image):
    output = np.full(image.shape,-1024)
    mean = [0,0]
    count_off = 0
    
    for i in range(image.shape[0]):
        thresh = cv2.threshold(image[i,:,:,0], -100, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.convertScaleAbs(thresh)
        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        #Sort by area, only consider the largest object
        cnts.sort(key = lambda x: cv2.contourArea(x), reverse=True)
        if cnts:
            #Calculate Center Position
            M = cv2.moments(cnts[0])
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            #Group and apply offset for those slices whose center are nearby
            if mean == [0,0]:
                mean = [cX,cY]
                count_off += 1
            elif abs(mean[0]-cX) < 20 and abs(mean[1]-cY) < 20:
                mean[0] = round((count_off*mean[0] + cX)/(count_off+1))
                mean[1] = round((count_off*mean[1] + cY)/(count_off+1))
                count_off += 1
            else:
                #cY->Height, cX->Width
                shift = (256-mean[1],256-mean[0])  
                output[i-count_off:i] = np.roll(image[i-count_off:i], shift, axis=(1,2))
                mean = [cX,cY]
                count_off = 1
        else:
            count_off += 1

    shift = (256-mean[1],256-mean[0])        
    output[-count_off:] = np.roll(image[-count_off:], shift, axis=(1,2))

    return output

def rotateZ(image):
    output = image
    count_angle = 0
    mean_angle = 0.0
    
    for i in range(image.shape[0]):
        image = image.astype(np.int16)
        thresh = cv2.threshold(image[i,:,:,0], -100, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.convertScaleAbs(thresh)
        # find contours in the thresholded image
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]
        #Sort by area, only consider the largest object
        cnts.sort(key = lambda x: cv2.contourArea(x), reverse=True)
        if cnts:
            #Calculate angle of main axis, despite last 5 images
            _,(MA,ma),angle = cv2.fitEllipse(cnts[0])
            if angle > 90:
                angle = angle-180
            #Adjust orientation
            if mean_angle == 0.0 and ma-MA > 30.0:
                mean_angle = angle
                count_angle += 1
            #Only consider large contour, direction of small countour is not accurate
            elif abs(mean_angle-angle) <30 and ma-MA > 30.0:
                mean_angle = (count_angle*mean_angle + angle) /(count_angle+1)
                count_angle += 1
            #Exclue last 5 images
            elif ma-MA <= 25.0:
                count_angle += 1
            else:
                #Guardband random rotation due to wrong calculated direction
                if count_angle > 5:
                    print('apply mean angle',mean_angle)
                    output[i-count_angle:i] = scipy.ndimage.rotate(image[i-count_angle:i], mean_angle, reshape=False, 
                                                                   axes=(1,2), mode='nearest')
                mean_angle = angle
                count_angle = 1
        else:
            count_angle += 1
    if count_angle > 5:
        print('apply mean angle',mean_angle)
        output[-count_angle:] = scipy.ndimage.rotate(image[-count_angle:], mean_angle, reshape=False, 
                                                     axes=(1,2), mode='nearest')

    return output

def load_data_dicom(path):
    slices = load_scan(path)
    #Store thickness and position info
    thickness = Counter([s.SliceThickness for s in slices])
    z_pos = [s.ImagePositionPatient[2] for s in slices]
    print(thickness)
    print(z_pos)
    #Convert slices into image matrix
    image = get_pixels_hu(slices)
    image = match_hw(image)
    #Fill zeros for mask channel
    label = np.zeros_like(image)
    output = np.stack([image,label], axis=-1)
    #Load labelling file
    label_nii = [os.path.join(path, p) for p in os.listdir(path) if '.nii' in p and 'label' in p]
    label_nii.sort()
    if not label_nii: 
        print("No label data detected...")
        return np.array([])
    print(label_nii)
    for label_path in label_nii:
        label_file = nib.load(label_path)
        label = match_hw(label_file.get_data())
        #Rotate 90 as NIFTI uses RAI but dicom uses RPI orientation
        label = np.transpose(np.rot90(label),[2,0,1])
        label_pos = label_file.header['qoffset_z']
        print(label_pos)
        label_index = find_index_around(z_pos, label_pos)
        print('Insert in to postition: ', label_index)
        output[label_index:label_index+label.shape[0],:,:,1] = label

    print('before resampling: ', output.shape)
#     plot_img(normalize(output[:,:,:,0]))
#     plot_img(output[:,:,:,1])
    #Centralize images (grouped)
    output = centralize(output)
    #Rotate images (grouped)
    output = rotateZ(output)
    #Resampling
    #Assume only two thickness
    if len(thickness.keys())>2:
        print('More than two thickness')
    elif len(thickness.keys())==2:
        for t, t_count in thickness.items():
            if t == slices[0].SliceThickness:
                part1 = resample_dicom(output[0:t_count],slices[0])
            elif t == slices[-1].SliceThickness:
                part2 = resample_dicom(output[-t_count-1:-1],slices[-1])
        part1, part2 = match_size(part1, part2)
        output = np.concatenate([part1,part2],axis=0)
    else:
        output = resample_dicom(output,slices[0])
        
#     plot_img(normalize(output[:,:,:,0]))
#     plot_img(output[:,:,:,1])
    
    print(output.shape)
    return output.astype(np.float32)   

def load_data_nii(path):
    #Backup for load_data_dicom, may not be able to handle all situation
    output = np.array([])
    images = []
    labels = []
    nii = [os.path.join(path, p) for p in os.listdir(path) if '.gz' in p]
    nii.sort(reverse=True)
    for n in nii:
        #Seperate label and img data
        ct = nib.load(n)
        if 'label' in n:
            labels.append(ct)
        else:
            if len(ct.get_data().shape) == 3 and ct.get_data().shape[2] > 2:
                images.append(ct)
    if not labels:
        print("No label data detected...")
        return output
    
    images.sort(key=lambda x: x.header['qoffset_z'])
    labels.sort(key=lambda x: x.header['qoffset_z'])
    #Remove image nii which have same location with previous one
    threshold = 10
    images = reduce(lambda x, y: x + [y] if len(x) == 0 or y.header['qoffset_z'] > x[-1].header['qoffset_z'] + threshold else x, images, [])

    label_id = 0
    for i, image in enumerate(images):
        img, img_hdr = image.get_data(), image.header
        if len(labels) > label_id:
            label, label_hdr = labels[label_id].get_data(), labels[label_id].header
        
        img = np.transpose(img,[2,0,1])
        label = np.transpose(label,[2,0,1])
        if abs(img_hdr['qoffset_z'] - label_hdr['qoffset_z']) < 3 and img.shape == label.shape:
            img, label = match_hw(img), match_hw(label)
            output_add = np.stack([img,label], axis=-1)
            label_id += 1
        else:
            img = match_hw(img)
            output_add = np.stack([img,np.zeros_like(img)], axis=-1)
        if i == 0:
            output = output_add
        else:
            output, output_add = match_size(output, output_add)
            output = np.concatenate([output, output_add], axis = 0)
    if label_id == 0:
        print('somehting wrong')

    #Rotate 90 degree to convert orientation to RPI
    output = np.rot90(output, axes=(1, 2))
    print('Before resampling', output.shape)
    #Centralize images (grouped)
    output = centralize(output)
    #Rotate images (grouped)
    output = rotateZ(output)
    
    #Resampling
    start_id = 0
    for i, image in enumerate(images):
        img, img_hdr = image.get_data(), image.header
        part = resample_nii(output[start_id:start_id+img.shape[2]], img_hdr)
        start_id = img.shape[2]
        if i == 0:
            output_rs = part
        else:
            output_rs, part = match_size(output_rs, part)
            output_rs = np.concatenate([output_rs,part],axis=0)
    print(output_rs.shape)
#     plot_img(normalize(output_rs[:,:,:,0]))
#     plot_img(output_rs[:,:,:,1])
    return output_rs.astype(np.float32)

