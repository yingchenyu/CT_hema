import nibabel as nib
import glob
import numpy as np
import pandas as pd
import dicom
import os
import matplotlib.pyplot as plt
import scipy.ndimage
import shutil

def resample(image, header, new_spacing=[0.8,0.8,9]):
    # Determine current pixel spacing
    spacing = np.array(header['pixdim'][1:4], dtype=np.float32)
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image

#Get list of patients under input folder:
def get_patients(path):
    patients = os.listdir(path)
    for p in patients:
        if p.startswith('.'):
            patients.remove(p)
    patients.sort()
    return patients

# get all the patient folders under the root directory
def get_folders(path):
    patients = [os.path.join(path, p) for p in os.listdir(path) if '.' not in p]
    folders = []
    for patient in patients:
        for s in os.listdir(patient):
            if '.' not in s:
                ps = os.path.join(patient, s)
                folders.append(ps)
    return folders

# Load the scans in given folder path
def load_scan(path):
    series_count = {}
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path) if '.' not in s]
    for s in slices:
        if s.SeriesNumber not in series_count.keys():
            series_count[s.SeriesNumber] = 1
        else:
            series_count[s.SeriesNumber] += 1
    for s in slices:
        if series_count[s.SeriesNumber] < 2:
            slices.remove(s)
    return slices


# check if slices are nameless
def nameless(slices):
    for s in slices:
        if(hasattr(s, 'SeriesNumber')):
            if(hasattr(s, 'SeriesDescription')):
                return False
    return True

def get_baselines(path):
    #Get baseline folders with excluding patients needed for relabelling
    folders = get_folders(path)
    baselines = []
    folders.sort()
    for f in folders:
        if 'baseline' in f:
            baselines.append(f)
    #         if 'fixed' in f:
    #             try:
    #                 baselines.remove(f.replace('_fixed',''))
    #                 baselines.remove(f)
    #             except:
    #                 baselines.remove(f)
    # for f in folders:
    #     if '24hNCCT_fixed' in f:
    #         try:
    #             baselines.remove(f.replace('24hNCCT_fixed','baselineNCCT'))
    #         except:
    #             pass
    return baselines

def load_data(path):
    output = np.array([])
    img = {}
    label = {}
    hdr_img = {}
    hdr_label = {}
    img_used = []
    nii = [os.path.join(path, p) for p in os.listdir(path) if '.gz' in p]
    nii.sort()
    for n in nii:
        #Seperate label and img data
        ct = nib.load(n)
        if 'label' in n:
            label[n] = ct.get_data()
            hdr_label[n] = ct.header
        else:
            if len(ct.get_data().shape) ==3:
                img[n] = ct.get_data()
                hdr_img[n] = ct.header
    if not label:
        print("No label data detected...")
        return output
    
    for label_name, label_header in hdr_label.items():
        for img_name, img_header in hdr_img.items():
            if img_name not in img_used and round(label_header['pixdim'][3],1) == round(img_header['pixdim'][3],1) \
            and img[img_name].shape == label[label_name].shape:
                #To make sure label and img same size, both using img's spacing
                img_resample = resample(img[img_name], img_header)
                label_resample = resample(label[label_name], img_header)
#                 print(img_resample.shape, label_resample.shape)
                #Crop width to match img's height
                output_img1 = np.stack([img_resample, label_resample],axis=-1)
                if output_img1.shape[1] != output_img1.shape[0]:
                    shape_diff = output_img1.shape[1] - output_img1.shape[0]
                    output_img1 = output_img1[:,shape_diff//2:-shape_diff//2,:,:]

                #Special case for Help-gs-078
                if '20150524193107_HeadTTYY_2' in img_name:
                    output_img1 = output_img1[:,:,0:(output_img1.shape[2]+1)//2,:]

                if output.size == 0:
                    output = output_img1
                #To make sure 5mm data is above 10mm
                else:
                    if (output.shape[0] > output_img1.shape[0]):
                        output = crop_data(output, output.shape[0] - output_img1.shape[0])
                    elif (output.shape[0] < output_img1.shape[0]):
                        output_img1 = crop_data(output_img1, output_img1.shape[0] - output.shape[0])
#                     print(output.shape[0] , output_img1.shape[0])
                    if img_header['pixdim'][3] > thickness:
                        output = np.concatenate([output, output_img1], axis = 2)
                    elif img_header['pixdim'][3] < thickness:
                        output = np.concatenate([output_img1, output], axis = 2)
                    elif img_header['pixdim'][3] == thickness and output_img1.shape[2]!= output.shape[2]:
                        output = np.concatenate([output, output_img1], axis = 2)
                    #Special Case for HELP-GS223
                    elif '20170803122409' in img_name:
                        output = np.concatenate([output, output_img1], axis = 2)
                    
                thickness = img_header['pixdim'][3]
                img_used.append(img_name)
                print(img_name)
                break
    #Add data without label, and make sure 5mm data is above 10mm
    if output.size !=0:
        for img_name, img_header in hdr_img.items():
            if img[img_name].shape[2] > 2 and img_name not in img_used:
                img_resample = resample(img[img_name], img_header)
                seg = np.zeros_like(img_resample) 
                output_add = np.stack([img_resample, seg],axis=-1)
                #Crop to match the shape
                if output_add.shape[1] != output_add.shape[0]:
                    shape_diff = output_add.shape[1] - output_add.shape[0]
                    output_add = output_add[:,shape_diff//2:-shape_diff//2,:,:]
                if (output.shape[0] > output_add.shape[0]):
                    output = crop_data(output, output.shape[0] - output_add.shape[0])
                elif (output.shape[0] < output_add.shape[0]):
                    output_add = crop_data(output_add, output_add.shape[0] - output.shape[0])
#                 print(output.shape, output_add.shape)
                if img_header['pixdim'][3] < thickness:
                    output = np.concatenate([output_add, output], axis=2)
                    break
                elif img_header['pixdim'][3] > thickness:
                    output = np.concatenate([output, output_add], axis=2)
                    break
                elif img_header['pixdim'][3] == thickness and output_add.shape[2]!= output_img1.shape[2]:
                    output = np.concatenate([output, output_add], axis = 2)
                    break
    return output.astype(np.int16)


'''
load all baseline data under the path
return a list of 3d images and corresponding masks
also return a list of patients
'''
def load_all_data(path):
    dataset = []
    patients = []
    for p in glob.glob(path):
        baselines += get_baselines(p)
    #Remove HELP-GS017&026
    try:
        baselines.remove('./CT_filtered_318/HELP-GS/HELP-GS-017/baselineNCCT')
        baselines.remove('./CT_filtered_318/HELP-GS/HELP-GS-026/baselineNCCT')
        baselines.remove('./CT_filtered/HELP-GS/HELP-GS-036/baselineNCCT_fixed')
        baselines.remove('./CT_filtered/HELP-GS/HELP-GS-222/baselineNCCT')
    except:
        pass
    print(len(baselines))
    for base in baselines:
        print(base)
        try:
            data = load_data(base)
            if data.size >0:
                dataset.append(data)
                patients.append(base.split('/')[-2])
        except:
            print('Unable to load data, patient: ', base)

    return dataset, patients

def crop_data(data, diff):
    crop_size = diff//2
    crop = data[crop_size:-crop_size,crop_size:-crop_size,:,:]
    if diff%2 !=0:
        crop = crop[0:-1,0:-1,:,:]
    return crop


# resize the dataset to 18 slices
def resize_data(dataset, patients):
#Fix Size of data
#Remove redundant data in HELP-GS-78
    min_dim = 180
    for i,data in enumerate(dataset):
        if patients[i] == 'HELP-GS-078':
            dataset[i] = np.concatenate([data[:,:,0:8,:],data[:,:,-9:-1,:]],axis=2)
        if data.shape[0] < min_dim:
            min_dim = data.shape[0]
        print(patients[i],dataset[i].shape)
    print(min_dim)
    #Pad or Crop to [min_dim,min_dim,18,*]
    for i, data in enumerate(dataset):
        data[data<-1024] = -1024
        if data.shape[2] > 18:
            diff = data.shape[2] - 18
            dataset[i] = data[:,:,:-diff,:]
        elif data.shape[2] < 18:
            diff = 18 - data.shape[2]
            pad = data[:,:,0:diff,:].copy()
            pad[:,:,:,:] = 0
            dataset[i] = np.concatenate([data,pad] ,axis=2)

    for i, data in enumerate(dataset):
        if data.shape[0] > min_dim:
            dataset[i] = crop_data(data, data.shape[0] - min_dim)

    for data in dataset:
        print(data.shape)

# pad the images or masks to the dismension of 180*180
def pad_data(dataset, image = True):
    if image: constant_value = -1024
    else: constant_value = 0
    padded_data = []
    for data in dataset:
        original_size = data.shape[0]
        if original_size == 180:
            padded_data.append(data)
        elif original_size != 180:
            if original_size % 2 == 0:
                pad_size = (180 - original_size)//2
                pad_config = ((pad_size,pad_size), (pad_size,pad_size), (0,0))
                data = np.pad(data, pad_config, 'constant', constant_values=constant_value)
                padded_data.append(data)
            else:
                pad1 = (180 - original_size)//2
                pad2 = pad1 + 1
                pad_config = ((pad1, pad2), (pad1, pad2), (0,0))
                data = np.pad(data, pad_config, 'constant', constant_values=constant_value)
                padded_data.append(data)
    return padded_data

#pad both the images and masks in the dataset
def pad_dataset(dataset):

    images = [data[:,:,:,0] for data in dataset]
    masks = [data[:,:,:,1] for data in dataset]
    padded_images = pad_data(images, image=True)
    padded_masks = pad_data(masks, image=False)
    a = np.array(padded_images)
    b = np.array(padded_masks)
    padded_data = np.stack((a,b), axis=4)

    return padded_data

# get the boundary positions of the masks
def get_boundary(masks):
    X, Y, Z = masks.shape
    left = X - 1
    right = 0
    up = Y - 1
    down = 0
    top = Z - 1
    bottom = 0
    for i in range(X):
        for j in range(Y):
            for k in range(Z):
                pixel = masks[i, j, k]
                if pixel == 1:
                    left = min(left, j)
                    right = max(right, j)
                    up = min(up, i)
                    down = max(down, i)
                    top = min(top, k)
                    bottom = max(bottom, k)

    return up, down, left, right, top, bottom

'''
input a dataset of both masks and iamges
return a numpy array of cropped images
'''

def crop_actual_size(dataset, margin=1):
    cropped_images = []
    for i, data in enumerate(dataset):
        print(i, 'th')
        images = data[:,:,:,0]
        masks = data[:,:,:,1]
        u, d, l, r, t, b = get_boundary(masks)
        print(u, d, l, r, t, b)
        print(masks.shape)
        if(u - margin < 0):
            u = margin
        if(d + margin > masks.shape[0] - 1):
            d = masks.shape[0] - margin - 1
        if(l - margin < 0):
            l = margin
        if(r + margin > masks.shape[1] - 1):
            r = masks.shape[1] - margin - 1
        if(u - margin < 0):
            t = margin
        if(b + margin > masks.shape[2] - 1):
            b = masks.shape[2] - margin - 1

        print(u-margin, d+margin, l-margin, r+margin, t-margin, b+margin)
        cropped_image = images[u-margin:d+margin, l-margin:r+margin, t-margin:b+margin]
        cropped_images.append(cropped_image)

    # return np.array(cropped_images)
    return cropped_images


def get_labels(excel_path, patients):
    if 'GS' in patients[0]:
        measure = pd.read_excel(excel_path, 0)
    else:
        measure = pd.read_excel(excel_path, 1)
    label = []
    for p in patients:
        # print(p)
        if 'GS' in patients[0]:
            p_sel = measure[measure['GSnum'] == p]
        else:
            p_sel = measure[measure['NUM'] == int(p.split('-')[-1])]
        print(p)
        v_fuzhen = p_sel['复诊体积'].values[0]
        v_shouzhen = p_sel['首诊体积'].values[0]
        #Check whether xuezhong increase
        if(pd.isnull(v_shouzhen) or pd.isnull(v_fuzhen)):
            label.append(None)
        elif (v_fuzhen - v_shouzhen) >= 6.0 or (v_fuzhen/v_shouzhen) > 1.33:
            label.append(1)
        else:
            label.append(0)
    return label

def calculate_pad(x, mx):
    if(x == mx):
        return 0, 0
    elif((mx - x)%2==0):
        pad = (mx - x)//2
        return pad, pad
    else:
        pad1 = (mx - x)//2
        pad2 = pad1 + 1
        return pad1, pad2

def pad_cropped(dataset):
    mx, my, mz = 0, 0, 0
    for data in dataset:
        x, y, z = data.shape
        mx = max(mx, x)
        my = max(my, y)
        mz = max(mz, z)
    print(mx, my, mz)
    mx = max(mx, my)
    for i, data in enumerate(dataset):
        x, y, z = data.shape
        pad1, pad2 = calculate_pad(x, mx)
        pad3, pad4 = calculate_pad(y, mx)
        pad5, pad6 = calculate_pad(z, mz)
        padded = np.pad(data,((pad1,pad2), (pad3,pad4), (pad5,pad6)) , 'constant', constant_values=-0.25)
        dataset[i] = padded
    return dataset

def process_cropeed_data(cropped, label):
    processed = []
    for i, c in enumerate(cropped):
        if(label[i]==1):
            l = np.array([0,1])
            processed.append([np.transpose(c, (2, 0, 1)), l])

        elif(label[i]==0):
            l = np.array([1,0])
            processed.append([np.transpose(c, (2, 0, 1)), l])

        else:
            pass

    return processed

def process_padded_data(dataset, label):
    processed = []
    for i, c in enumerate(dataset):
        if(label[i]==1):
            l = np.array([0,1])
            processed.append([np.transpose(c, (2, 0, 1, 3)), l])

        elif(label[i]==0):
            l = np.array([1,0])
            processed.append([np.transpose(c, (2, 0, 1, 3)), l])

        else:
            pass

    return processed

def normalize(image, MIN_BOUND = -50.0, MAX_BOUND = 100.0):
    #PIXEL MEAN of the whole dataset is around 0.26
    #Zero Centering
    PIXEL_MEAN = 0.25
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image.astype(np.float32) - PIXEL_MEAN

def normalize_dataset(dataset):
    for i, data in enumerate(dataset):
        image = data[:,:,:,0]
        image = normalize(image)
        dataset[i][:,:,:,0] = image
    return dataset

def prepare_original_data(path, excel_path, filename='original_data.npy'):
    dataset, patients = load_all_data(path)
    resized = resize_data(dataset, patients)
    padded = pad_dataset(resized)
    labels = get_labels(excel_path, patients)
    processed = process_padded_data(padded, labels)
    np.save(filename, processed)

def prepare_cropped_data(path, excel_path, filename='cropped_data.npy'):
    dataset, patients = load_all_data(path)
    normalize_dataset(dataset)
    cropped = crop_images(dataset)
    labels = get_labels(excel_path, patients)
    processed = process_cropeed_data(cropped, labels)
    np.save(filename, processed)

if __name__ == '__main__':
    path = './CT_filtered_318/combined/'
    excel_path = './CT_measure_318.xlsx'
    filename = './combined1.npy'

    dataset, patients = load_all_data(path)
    normalize_dataset(dataset)
    cropped = crop_actual_size(dataset)
    padded_crop = pad_cropped(cropped)
    labels = get_all_labels(excel_path, patients)
    processed = process_cropeed_data(padded_crop, labels)
    np.save(filename, processed)

