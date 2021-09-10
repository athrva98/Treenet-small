# Helper Functions for Images and Batches
import numpy as np
import cv2
import tensorflow as tf
import glob
import os

input_image_paths = []
output_image_paths = []

in_path = r'D:\Dataset_impressionism\laplacian'
out_path = r'D:\Dataset_impressionism\real'

""" Helper Functions """
def sort_by_path_names(path):
    number = int(eval(path.split(os.path.sep)[-1].split('.jpg')[0]))
    return number  

def normalize_image(path, compute_laplacian = False): # We dont resize the image because we have a FConvNN
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (128,128), cv2.INTER_CUBIC)
    if compute_laplacian == True:
        img = cv2.Laplacian(img, cv2.CV_8UC1, ksize=5)
    img = img / 255.
    img = np.expand_dims(img, axis = 0)
    img_tensor = tf.convert_to_tensor(img, dtype = tf.float32)
    return img_tensor
def denormalize(img):
    img = (img.numpy() * 255).astype(np.uint8)
    return img
def initialize_paths():
    input_image_paths = sorted(glob.glob(in_path + '\*.jpg'), key = sort_by_path_names)
    output_image_paths = sorted(glob.glob(out_path + '\*.jpg'), key = sort_by_path_names)
    return input_image_paths, output_image_paths





