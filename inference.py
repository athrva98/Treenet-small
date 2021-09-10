# Inference
import numpy as np
import tensorflow as tf
from helper import *

MODEL_PATH = r'D:\treenet\Model'



def infer(path):
	img = normalize_image(path, compute_laplacian = True)
	model = tf.keras.models.load_model(MODEL_PATH)
	img_out = model(img, training = True)[0]
	img_out = denormalize(img_out)
	return img_out
    
list_of_files = glob.glob(MODEL_PATH + os.path.sep + '*') # * means all if need specific format then *.csv
latest_file = max(list_of_files, key=os.path.getctime)
print("Getting : ",latest_file)

MODEL_PATH = latest_file

img_path = r'D:\treenet\Sample\1_in.jpg'
output_path = r'D:\treenet\Sample\out.jpg'
output_img = infer(img_path)
cv2.imwrite(output_path, output_img)
