import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import RMSprop
from helper import *
import glob
import os

MODEL_PATH = r'D:\treenet\Model'
SAMPLE_PATH = r'D:\treenet\Sample'


class treenet:
    def __init__(self):
        self.input_shape = (128, 128, 3)
        self._create_model()
        

    @staticmethod
    def _downsample(filters, size, batch_norm = True):
        initializer = tf.random_normal_initializer(0, 0.02);
        
        layers = Sequential()
        layers.add(Conv2D(filters=filters,
                            kernel_size=size,
                            strides=2,
                            padding='same',
                            kernel_initializer=initializer,
                            use_bias=False))
        if batch_norm:
            layers.add(BatchNormalization())
        
        layers.add(LeakyReLU())
        
        return layers

    @staticmethod
    def _upsample(filters, size, add_dropout = True):
        initializer = tf.random_normal_initializer(0, 0.02)
        
        layers = Sequential()
        layers.add(Conv2DTranspose(filters=filters,
                                    kernel_size=size,
                                    strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))
        
        layers.add(BatchNormalization())
        
        if add_dropout:
            layers.add(Dropout(rate = 0.5))
        layers.add(LeakyReLU())
        
        return layers
    def get_model(self):
        return self.model
        
    def _create_model(self):
        inputs = Input(shape = self.input_shape)
        x = self.tree_block(inputs)
        x = self.tree_block(x)
        self.model = Model(inputs = inputs, outputs = x)
        
        
    def tree_block(self, inputs):
        # Here we shall define four pathways.
        # First shall have a downsampling followed by an upsampling
        # second shall have a upsampling followed by a downsampling
        # third shall have processing steps that retain the original size of the image
        # Fourth shall have the original image being lightly processed
        # We shall then concatenate the above featuremaps and produce a segmentation map.
        x = inputs
        y = inputs
        z = inputs
        orig  = inputs
        for i, filters in enumerate([64,128]):
            up_block = self._upsample(filters, 4, add_dropout = (i+1)*(0.075))
            y = up_block(y)
            
        for filters in [128,64]:
            down_block = self._downsample(filters, 4, batch_norm = True)
            y = down_block(y)
            
        for filters in [64,128]:
            down_block = self._downsample(filters, 4, batch_norm = True)
            x = down_block(x)

        for i, filters in enumerate([128,64]):
            up_block = self._upsample(filters, 4, add_dropout = (i+1)*(0.075)) 
            x = up_block(x)
        
        for filters in [32,64,128,64,32,16,3]:
            z = Conv2D(filters=filters,
                            kernel_size=4,
                            strides=1,
                            padding='same',
                            use_bias=False)(z)
            z = BatchNormalization()(z)
            z = LeakyReLU()(z)
            
        for filters in [32,16,3]:
            x = Conv2D(filters=filters,
                            kernel_size=4,
                            strides=1,
                            padding='same',
                            use_bias=False)(x)
            y = Conv2D(filters=filters,
                            kernel_size=4,
                            strides=1,
                            padding='same',
                            use_bias=False)(y)
        x = BatchNormalization()(x)
        x = LeakyReLU()(x)
        
        y = BatchNormalization()(y)
        y = LeakyReLU()(y)
        
        x = Add()([x,y,z,orig])
        
        return x
        
EPOCHS = 3
gen_net = treenet()
model_ = gen_net.get_model()
model_.summary()
#loss_partial_1 = tf.keras.losses.CosineSimilarity(axis = 1) # range (-1,1)
loss_partial_2 = tf.keras.losses.MeanAbsoluteError()
loss_partial_3 = tf.keras.losses.MeanSquaredError()
model_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)
i_paths, o_paths = initialize_paths() # Get all the file paths from memory.

@tf.function
def loss_function(pred, real):
    #s1 = loss_partial_1(pred, real)
    s2 = loss_partial_2(pred, real)
    s3 = loss_partial_3(pred, real)
    synthesis_loss = tf.math.pow(tf.add(tf.math.maximum(s2,s3), 1), 7)
    return synthesis_loss
    
    
@tf.function
def train_step(input_image, output_image):
    with tf.GradientTape() as gtape:
        y_bar = model_(input_image, training = True)
        
        synthesis_loss = loss_function(y_bar, output_image)
    
    model_gradients = gtape.gradient(synthesis_loss, model_.trainable_variables)
    model_optimizer.apply_gradients(zip(model_gradients, model_.trainable_variables))
    
    return synthesis_loss



def train(generate_model_on_path = False):
    try:
        list_of_files = glob.glob(MODEL_PATH + os.path.sep + '*') # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        print("Getting : ",latest_file)
        model_.load_weights(latest_file)
    except:
        print("Could Not Load Saved Model")
        
    if generate_model_on_path:
        tf.keras.models.save_model(model_, MODEL_PATH + os.path.sep + 'immpressionism_art.h5')
        import sys
        sys.exit(0)
        
    for i in range(EPOCHS):
        DISP_LOSS = 0
        for j in range(len(o_paths)):
            img_in = normalize_image(i_paths[j])
            img_out = normalize_image(o_paths[j])
            DLOSS = train_step(img_in, img_out)
            DISP_LOSS += DLOSS.numpy()
        print(f'Epoch {i}, loss {DISP_LOSS}')
        if i % 2 == 0:
            model_.save_weights(MODEL_PATH + os.path.sep + f'{i}_{DISP_LOSS}_MODEL.h5')
            indices = np.array([1,5,55])
            for index in indices:
                pred = model_(normalize_image(i_paths[index]), training = True)[0]
                pred_img = denormalize(pred)
                comparison_image = cv2.cvtColor(cv2.imread(o_paths[index]), cv2.COLOR_BGR2RGB)
                cv2.imwrite(SAMPLE_PATH + os.path.sep + f'p_{index}.png', pred_img)
                cv2.imwrite(SAMPLE_PATH + os.path.sep + f'r_{index}.png', cv2.resize(comparison_image, (128,128), cv2.INTER_CUBIC))
    






