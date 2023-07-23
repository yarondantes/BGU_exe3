import glob
import shutil
import os
import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian
import cv2


def split_data(train_path, test_path):
    print('splitting the data')
    test_ratio = 0.5
    images_classes_dict = {}
    num_of_rec = []
    classes = []
    for f_class in os.listdir(train_path):
        class_path = os.path.join(train_path, f_class)
        files = [entry for entry in os.listdir(class_path) if entry.startswith('image')]
        num_of_records = len(files)   
        num_of_files_to_be_moved = int(np.round(num_of_records * test_ratio))

        files_to_be_moved = np.random.choice(files, size=num_of_files_to_be_moved, replace=False)

        for f in files_to_be_moved:    
            src_path = os.path.join(class_path, f)
            dst_path = os.path.join(test_path, f_class)
            if not os.path.exists(dst_path):
                os.mkdir(dst_path)
            num_of_files_in_dst = len([entry for entry in os.listdir(dst_path)])
            if num_of_files_in_dst < num_of_files_to_be_moved:
                shutil.move(src_path, os.path.join(dst_path),f)


def reset_data(train_path, test_path):
    print('re-setting the data')
    for f_class in os.listdir(test_path):
        class_path = os.path.join(test_path, f_class)
        files = [entry for entry in os.listdir(class_path)]
        
        for f in files:    
            src_path = os.path.join(class_path, f)
            dst_path = os.path.join(train_path, f_class, f)

            shutil.move(src_path, os.path.join(dst_path),f)

def cut_out(img):
    num_of_cutoff_cubes = np.random.randint(2,10)
    for _ in range(num_of_cutoff_cubes):
        witdh = 70
        hight = 70
        x_start = np.random.randint(0,img.shape[0]-witdh)
        y_start = np.random.randint(0,img.shape[1]-hight)
        img[x_start:x_start+witdh,y_start:y_start+hight] = 0
    return img 

def rotate_image(img):
    angle = np.random.randint(30,90)
    return rotate(img, angle=angle, mode = 'wrap', preserve_range= True).astype('uint8')


def tranlation(img):
    percent_of_x = np.random.randint(10, 30)
    percent_of_y = np.random.randint(10, 30)
    num_of_pixs_x = int(img.shape[0] * percent_of_x / 100)
    num_of_pixs_y = int(img.shape[1] * percent_of_y / 100)
    transform = AffineTransform(translation=(num_of_pixs_x,num_of_pixs_y))
    return warp(img,transform,mode='wrap',preserve_range=True).astype('uint8')

def save_img_with_prefix(file_name, prefix, path, img):
    file_name = prefix + '_' + file_name
    cv2.imwrite(os.path.join(path,file_name), img)


def augment_images(train_path):
    print('augmenting the data')
    for class_name in os.listdir(train_path):
        class_path = os.path.join(train_path,class_name)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):

                # load original train image
                cv_img = cv2.imread(os.path.join(class_path, file_name))

                # rotate image
                rotated = rotate_image(cv_img)
                save_img_with_prefix(file_name, 'rotated', class_path, rotated)

                # translation 
                wrapShift = tranlation(cv_img)
                save_img_with_prefix(file_name, 'wrapShift', class_path, wrapShift)

                # flip image left right 
                flipLR = np.fliplr(cv_img)
                save_img_with_prefix(file_name, 'flipLR', class_path, flipLR)

                # flip image upside down 
                flipUD = np.flipud(cv_img)
                save_img_with_prefix(file_name, 'flipUD', class_path, flipUD)

                # add random noise to the image
                sigma=0.155
                noisyRandom = random_noise(cv_img,var=sigma**2,)
                noisyRandom = (noisyRandom *255 ).astype('uint8')
                save_img_with_prefix(file_name, 'gaussian_noise', class_path, noisyRandom)

                # cutout
                cutout_img = cut_out(cv_img)
                save_img_with_prefix(file_name, 'cut_out', class_path, cutout_img)


def delete_all_augmented(train_path):
    print(f'deleting all augmented files')
    for class_name in os.listdir(train_path):
        class_path = os.path.join(train_path,class_name)
        if os.path.isdir(class_path):
            for file_name in os.listdir(class_path):
                if not file_name.startswith('image'):
                    image_path = os.path.join(class_path,file_name)
                    os.remove(image_path)
    