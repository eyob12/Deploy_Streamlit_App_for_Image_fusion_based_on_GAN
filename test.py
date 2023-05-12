# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import scipy.misc
import time
import os
import glob
import cv2
import PIL
from PIL import Image, ImageOps
from tensorflow.python.keras import backend as K
import streamlit as st
import json

#reader = tf.train.NewCheckpointReader("./checkpoint_20/CGAN_120/CGAN.model-9")
os.environ['CUDA_VISIBLE_DEVICES']='3'
# os.environ['CUDA_VISIBLE_DEVICES']='0'
log_device_placement=True
allow_soft_placement=True
import tensorflow as tf
tf.compat.v1.ConfigProto(log_device_placement=True,allow_soft_placement=True)
config = tf.compat.v1.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.99
config.gpu_options.allow_growth = True
K.set_session(tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(),config=config))


def singleScaleRetinex(img, variance):
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), variance))
    return retinex


def SSR(img, variance):
    img = np.float64(img) + 1.0
    img_retinex = singleScaleRetinex(img, variance)
    for i in range(img_retinex.shape[2]):
        unique, count = np.unique(np.int32(img_retinex[:, :, i] * 100), return_counts=True)
        for u, c in zip(unique, count):
            if u == 0:
                zero_count = c
                break
        low_val = unique[0] / 100.0
        high_val = unique[-1] / 100.0
        for u, c in zip(unique, count):
            if u < 0 and c < zero_count * 0.1:
                low_val = u / 100.0
            if u > 0 and c < zero_count * 0.1:
                high_val = u / 100.0
                break
        img_retinex[:, :, i] = np.maximum(np.minimum(img_retinex[:, :, i], high_val), low_val)

        img_retinex[:, :, i] = (img_retinex[:, :, i] - np.min(img_retinex[:, :, i])) / \
                               (np.max(img_retinex[:, :, i]) - np.min(img_retinex[:, :, i])) \
                               * 255
    img_retinex = np.uint8(img_retinex)
    return img_retinex


def imread(path, is_grayscale=True):
  """
  Read image using its path.
  Default value is gray-scale, and image is read by YCbCr format as the paper said.
  """
  if is_grayscale:
      
    return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
  else:
    return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def imsave(image, path):
  return scipy.misc.imsave(path, image)


def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.tif"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
    # data.sort(key=lambda x:int(x[len(data_dir)+1:-4]))

    return data

def lrelu(x, leak=0.2):
    return tf.maximum(x, leak * x)

def fusion_model(img):
    with tf.variable_scope('fusion_model'):
        with tf.variable_scope('layer1'):
            weights=tf.get_variable("w1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/w1')))
            bias=tf.get_variable("b1",initializer=tf.constant(reader.get_tensor('fusion_model/layer1/b1')))
            conv1_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1_ir = lrelu(conv1_ir)
        with tf.variable_scope('layer2'):
            weights=tf.get_variable("w2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/w2')))
            bias=tf.get_variable("b2",initializer=tf.constant(reader.get_tensor('fusion_model/layer2/b2')))
            vivi = tf.concat([images_vi, images_vi], axis=-1)
            conv2_add =tf.concat([vivi,conv1_ir],axis=-1)
            # conv2_add =conv1_ir  #without add
            # conv2_add = tf.concat([vivi, conv1_ir], axis=-1)
            conv2_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_add, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_ir = lrelu(conv2_ir)
        with tf.variable_scope('layer3'):
            weights=tf.get_variable("w3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/w3')))
            bias=tf.get_variable("b3",initializer=tf.constant(reader.get_tensor('fusion_model/layer3/b3')))
            conv3_add = tf.concat([conv2_add, conv2_ir], axis=-1)
            # conv3_add = tf.concat([vivi, conv2_ir], axis=-1)
            conv3_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_add, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_ir = lrelu(conv3_ir)
        with tf.variable_scope('layer4'):
            weights=tf.get_variable("w4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/w4')))
            bias=tf.get_variable("b4",initializer=tf.constant(reader.get_tensor('fusion_model/layer4/b4')))
            conv4_add = tf.concat([conv3_add, conv3_ir], axis=-1)
            # conv4_add = tf.concat([vivi, conv3_ir], axis=-1)
            conv4_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv4_add, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_ir = lrelu(conv4_ir)
        with tf.variable_scope('layer5'):
            weights=tf.get_variable("w5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/w5')))
            bias=tf.get_variable("b5",initializer=tf.constant(reader.get_tensor('fusion_model/layer5/b5')))
            conv5_add = tf.concat([conv4_add, conv4_ir], axis=-1)
            # conv5_add = tf.concat([vivi, conv4_ir], axis=-1)
            conv5_ir= tf.nn.conv2d(conv5_add, weights, strides=[1,1,1,1], padding='VALID') + bias
            conv5_ir=tf.nn.tanh(conv5_ir)
    return conv5_ir

def encoder_ir(img):
          
    with tf.compat.v1.variable_scope('encoder_ir'):
        with tf.compat.v1.variable_scope('layer1'):
            weights=tf.Variable("w1",initializer=tf.constant(reader.get_tensor('encoder_ir/layer1/w1')))
            bias=tf.Variable("b1",initializer=tf.constant(reader.get_tensor('encoder_ir/layer1/b1')))
            conv1_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1_ir = lrelu(conv1_ir)
          
        with tf.compat.v1.variable_scope('layer2'):
            weights=tf.Variable("w2",initializer=tf.constant(reader.get_tensor('encoder_ir/layer2/w2')))
            
            bias=tf.Variable("b2",[128],initializer=tf.constant_initializer(0.0))
            
            vivi = tf.concat([images_vi, images_vi], axis=-1)
            ref  = tf.concat([images_vi,images_vi],axis=-1)
            
            #conv2_vi=tf.concat([vivi,conv1_ir],axis=-1)
            #conv2_ref=tf.concat([ref,conv1_ir],axis=-1)
            
            #conv2_add =tf.concat([conv2_vi,conv2_ref],axis=-1)
            
            
            #vivi = tf.concat([self.images_vi, self.images_vi], axis=-1)
            conv2_add =tf.concat([vivi,conv1_ir],axis=-1)
            # conv2_add = conv1_ir
            # conv2_add = conv1_ir
            conv2_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_add, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_ir = lrelu(conv2_ir)

        with tf.compat.v1.variable_scope('layer3'):
            # weights=tf.Variable("w3",[3,3,130,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=tf.tf.Variable("w3",initializer=tf.constant(reader.get_tensor('encoder_ir/layer3/w3')))
            
            bias=tf.Variable("b3",initializer=tf.constant(reader.get_tensor('encoder_ir/layer3/b3')))
            conv3_add = tf.concat([conv2_add, conv2_ir], axis=-1)
            conv3_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_add, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_ir = lrelu(conv3_ir)

        with tf.compat.v1.variable_scope('layer4'):
            # weights=tf.Variable("w4",[3,3,66,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=tf.get_variable("w4",initializer=tf.constant(reader.get_tensor('encoder_ir/layer4/w4')))
            
            bias=tf.Variable("b4",initializer=tf.constant(reader.get_tensor('encoder_ir/layer4/b4')))
            conv4_add = tf.concat([conv3_add, conv3_ir], axis=-1)
            conv4_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv4_add, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_ir = lrelu(conv4_ir)

            
        return conv4_ir      
    '''
def encoder_ir(img):
    with tf.compat.v1.variable_scope('encoder_ir'):
        with tf.compat.v1.variable_scope('layer1'):
            weights = tf.Variable(tf.constant(reader.get_tensor('encoder_ir/layer1/w1')), name='w1')
            bias = tf.Variable(tf.constant(reader.get_tensor('encoder_ir/layer1/b1')), name='b1')
            conv1_ir = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='SAME', use_bias=False)(img)
            

            conv1_ir = tf.keras.layers.BatchNormalization(scale=True, center=True)(conv1_ir)
            conv1_ir = tf.keras.layers.LeakyReLU(alpha=0.2)(conv1_ir)

        with tf.compat.v1.variable_scope('layer2'):
            weights = tf.Variable(tf.constant(reader.get_tensor('encoder_ir/layer2/w2')), name='w2')
            bias = tf.Variable(tf.constant(reader.get_tensor('encoder_ir/layer2/b2')), name='b2')
            vivi = tf.concat([images_vi, images_vi], axis=-1)
            ref = tf.concat([images_vi, images_vi], axis=-1)
            conv2_add = tf.concat([vivi, conv1_ir], axis=-1)
            conv2_ir = tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), strides=(1,1), padding='SAME', use_bias=False)(conv2_add)
            conv2_ir = tf.keras.layers.BatchNormalization(scale=True, center=True)(conv2_ir)
            conv2_ir = tf.keras.layers.LeakyReLU(alpha=0.2)(conv2_ir)

        with tf.compat.v1.variable_scope('layer3'):
            weights = tf.Variable(tf.constant(reader.get_tensor('encoder_ir/layer3/w3')), name='w3')
            bias = tf.Variable(tf.constant(reader.get_tensor('encoder_ir/layer3/b3')), name='b3')
            conv3_add = tf.concat([conv2_add, conv2_ir], axis=-1)
            conv3_ir = tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='SAME', use_bias=False)(conv3_add)
            conv3_ir = tf.keras.layers.BatchNormalization(scale=True, center=True)(conv3_ir)
            conv3_ir = tf.keras.layers.LeakyReLU(alpha=0.2)(conv3_ir)

        with tf.compat.v1.variable_scope('layer4'):
            weights = tf.Variable(tf.constant(reader.get_tensor('encoder_ir/layer4/w4')), name='w4')
            bias = tf.Variable(tf.constant(reader.get_tensor('encoder_ir/layer4/b4')), name='b4')
            conv4_add = tf.concat([conv3_add, conv3_ir], axis=-1)
            conv4_ir = tf.keras.layers.Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='SAME', use_bias=False)(conv4_add)
            conv4_ir = tf.keras.layers.BatchNormalization(scale=True, center=True)(conv4_ir)
            conv4_ir = tf.keras.layers.LeakyReLU(alpha=0.2)(conv4_ir)

        return conv4_ir
    '''
def encoder_vi(img):
    with tf.compat.v1.variable_scope('encoder_vi'):
        with tf.compat.v1.variable_scope('layer1'):
            weights = tf.Variable(reader.get_tensor('encoder_vi/layer1/w1'), name="w1")
            bias = tf.Variable(reader.get_tensor('encoder_vi/layer1/b1'), name="b1")
            conv1_ir = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False, name="conv1_ir")(img)
            conv1_ir = tf.keras.layers.BatchNormalization(name="bn1")(conv1_ir)
            conv1_ir = tf.keras.layers.LeakyReLU(name="relu1")(conv1_ir)
          
        with tf.variable_scope('layer2'):
            weights = tf.Variable(reader.get_tensor('encoder_vi/layer2/w2'), name="w2")
            bias = tf.Variable(reader.get_tensor('encoder_vi/layer2/b2'), name="b2")
            vivi = tf.concat([images_vi, images_vi], axis=-1)
            ref = tf.concat([images_vi, images_vi], axis=-1)
            conv2_add = tf.concat([vivi, conv1_ir], axis=-1)
            conv2_ir = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, padding='same', use_bias=False, name="conv2_ir")(conv2_add)
            conv2_ir = tf.keras.layers.BatchNormalization(name="bn2")(conv2_ir)
            conv2_ir = tf.keras.layers.LeakyReLU(name="relu2")(conv2_ir)

        with tf.compat.v1.variable_scope('layer3'):
            weights = tf.Variable(reader.get_tensor('encoder_vi/layer3/w3'), name="w3")
            bias = tf.Variable(reader.get_tensor('encoder_vi/layer3/b3'), name="b3")
            conv3_add = tf.concat([conv2_add, conv2_ir], axis=-1)
            conv3_ir = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False, name="conv3_ir")(conv3_add)
            conv3_ir = tf.keras.layers.BatchNormalization(name="bn3")(conv3_ir)
            conv3_ir = tf.keras.layers.LeakyReLU(name="relu3")(conv3_ir)

        with tf.compat.v1.variable_scope('layer4'):
            weights = tf.Variable(reader.get_tensor('encoder_vi/layer4/w4'), name="w4")
            bias = tf.Variable(reader.get_tensor('encoder_vi/layer4/b4'), name="b4")
            conv4_add = tf.concat([conv3_add, conv3_ir], axis=-1)
            conv4_ir = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=1, padding='same', use_bias=False, name="conv4_ir")(conv4_add)
            conv4_ir = tf.keras.layers.BatchNormalization(name="bn4")(conv4_ir)
            conv4_ir = tf.keras.layers.LeakyReLU(name="relu4")(conv4_ir)

        return conv4_ir

'''
def encoder_vi(img):
  with tf.variable_scope('encoder_vi'):
        with tf.variable_scope('layer1'):
            weights=tf.get_variable("w1",initializer=tf.constant(reader.get_tensor('encoder_vi/layer1/w1')))
            bias=tf.get_variable("b1",initializer=tf.constant(reader.get_tensor('encoder_vi/layer1/b1')))
            conv1_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1_ir = lrelu(conv1_ir)
          
        with tf.variable_scope('layer2'):
            weights=tf.get_variable("w2",initializer=tf.constant(reader.get_tensor('encoder_vi/layer2/w2')))
            
            bias=tf.get_variable("b2",[128],initializer=tf.constant_initializer(0.0))
            
            vivi = tf.concat([images_vi, images_vi], axis=-1)
            ref  = tf.concat([images_vi,images_vi],axis=-1)
            
            #conv2_vi=tf.concat([vivi,conv1_ir],axis=-1)
            #conv2_ref=tf.concat([ref,conv1_ir],axis=-1)
            
            #conv2_add =tf.concat([conv2_vi,conv2_ref],axis=-1)
            
            
            #vivi = tf.concat([self.images_vi, self.images_vi], axis=-1)
            conv2_add =tf.concat([vivi,conv1_ir],axis=-1)
            # conv2_add = conv1_ir
            # conv2_add = conv1_ir
            conv2_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2_add, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2_ir = lrelu(conv2_ir)

        with tf.variable_scope('layer3'):
            # weights=tf.get_variable("w3",[3,3,130,64],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=tf.get_variable("w3",initializer=tf.constant(reader.get_tensor('encoder_vi/layer3/w3')))
            
            bias=tf.get_variable("b3",initializer=tf.constant(reader.get_tensor('encoder_vi/layer3/b3')))
            conv3_add = tf.concat([conv2_add, conv2_ir], axis=-1)
            conv3_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv3_add, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3_ir = lrelu(conv3_ir)

        with tf.variable_scope('layer4'):
            # weights=tf.get_variable("w4",[3,3,66,32],initializer=tf.truncated_normal_initializer(stddev=1e-3))
            weights=tf.get_variable("w4",initializer=tf.constant(reader.get_tensor('encoder_vi/layer4/w4')))
            
            bias=tf.get_variable("b4",initializer=tf.constant(reader.get_tensor('encoder_vi/layer4/b4')))
            conv4_add = tf.concat([conv3_add, conv3_ir], axis=-1)
            conv4_ir= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv4_add, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv4_ir = lrelu(conv4_ir)

            
        return conv4_ir 
        '''
def decoder(img):
    #Flag1 = tf.concat([ir,vi],axis=-1)
    with tf.compat.v1.variable_scope('decoder'):
        with tf.compat.v1.variable_scope('Layer1'):
            weights=tf.compat.v1.get_variable("W1",initializer=tf.constant(reader.get_tensor('decoder/Layer1/W1')))
            #weights=weights_spectral_norm(weights)
            bias=tf.compat.v1.get_variable("B1",initializer=tf.constant(reader.get_tensor('decoder/Layer1/B1')))
            conv1= tf.compat.v1.layers.batch_normalization(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)

        with tf.compat.v1.variable_scope('Layer2'):
            weights=tf.compat.v1.get_variable("W2",initializer=tf.constant(reader.get_tensor('decoder/Layer2/W2')))
            #weights=weights_spectral_norm(weights)
            bias=tf.compat.v1.get_variable("B2",initializer=tf.constant(reader.get_tensor('decoder/Layer2/B2')))
            conv2= tf.compat.v1.layers.batch_normalization(tf.nn.conv2d(conv1, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)

        with tf.compat.v1.variable_scope('Layer3'):
            weights=tf.compat.v1.get_variable("W3",initializer=tf.constant(reader.get_tensor('decoder/Layer3/W3')))
            #weights=weights_spectral_norm(weights)
            bias=tf.compat.v1.get_variable("B3",initializer=tf.constant(reader.get_tensor('decoder/Layer3/B3')))
            conv3= tf.compat.v1.layers.batch_normalization(tf.nn.conv2d(conv2, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)

        with tf.compat.v1.variable_scope('Layer4'):
            weights=tf.compat.v1.get_variable("W4",initializer=tf.constant(reader.get_tensor('decoder/Layer4/W4')))
            #weights=weights_spectral_norm(weights)
            bias=tf.compat.v1.get_variable("B4",initializer=tf.constant(reader.get_tensor('decoder/Layer4/B4')))
            conv4= tf.nn.conv2d(conv3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4 = tf.nn.tanh(conv4)
    return conv4

'''      
def decoder(img):
    #Flag1 = tf.concat([ir,vi],axis=-1)
    with tf.variable_scope('decoder'):
        with tf.variable_scope('Layer1'):
            weights=tf.get_variable("W1",initializer=tf.constant(reader.get_tensor('decoder/Layer1/W1')))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("B1",initializer=tf.constant(reader.get_tensor('decoder/Layer1/B1')))
            conv1= tf.contrib.layers.batch_norm(tf.nn.conv2d(img, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv1 = lrelu(conv1)

        with tf.variable_scope('Layer2'):
            weights=tf.get_variable("W2",initializer=tf.constant(reader.get_tensor('decoder/Layer2/W2')))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("B2",initializer=tf.constant(reader.get_tensor('decoder/Layer2/B2')))
            conv2= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv1, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv2 = lrelu(conv2)

        with tf.variable_scope('Layer3'):
            weights=tf.get_variable("W3",initializer=tf.constant(reader.get_tensor('decoder/Layer3/W3')))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("B3",initializer=tf.constant(reader.get_tensor('decoder/Layer3/B3')))
            conv3= tf.contrib.layers.batch_norm(tf.nn.conv2d(conv2, weights, strides=[1,1,1,1], padding='SAME') + bias, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)
            conv3 = lrelu(conv3)

        with tf.variable_scope('Layer4'):
            weights=tf.get_variable("W4",initializer=tf.constant(reader.get_tensor('decoder/Layer4/W4')))
            #weights=weights_spectral_norm(weights)
            bias=tf.get_variable("B4",initializer=tf.constant(reader.get_tensor('decoder/Layer4/B4')))
            conv4= tf.nn.conv2d(conv3, weights, strides=[1,1,1,1], padding='SAME') + bias
            conv4 = tf.nn.tanh(conv4)
    return conv4
'''

def input_setup(index):
    padding=0
    sub_ir_sequence = []
    sub_vi_sequence = []
    sub_ref_sequence = []
    input_ir=cv2.resize((imread(data_ir[index])-127.5)/127.5, (456, 456))
    input_ir=np.lib.pad(input_ir,((padding,padding),(padding,padding)),'edge')


    w,h=input_ir.shape
    input_ir=input_ir.reshape([w,h,1])
    input_vi=cv2.resize((imread(data_vi[index])-127.5)/127.5,(456,456))
    input_vi=np.lib.pad(input_vi,((padding,padding),(padding,padding)),'edge')

    w,h=input_vi.shape
    input_vi=input_vi.reshape([w,h,1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir= np.asarray(sub_ir_sequence)
    train_data_vi= np.asarray(sub_vi_sequence)
    
    
    input_ref=cv2.resize((imread(data_ref[index])-127.5)/127.5,(456,456))  
    input_ref=np.lib.pad(input_ref,((padding,padding),(padding,padding)),'edge')
    w,h=input_ref.shape
    input_ref=input_ref.reshape([w,h,1])
    sub_ref_sequence.append(input_ref)
    train_data_ref=np.asarray(sub_ref_sequence)
    
    return train_data_ir,train_data_vi,train_data_ref

@st.cache_data()
def load_s3_file_structure(path: str = 'output.json') -> dict:  # str = 'src/all_image_files.json'
    """Retrieves JSON document outining the S3 file structure"""
    with open(path, 'r') as f:
        return json.load(f)
    
all_image_files = load_s3_file_structure()
index_of_image = sorted(list(all_image_files.keys()))

dataset_type = st.sidebar.selectbox(
            "select the image", index_of_image)


org_vis = all_image_files[dataset_type]['org_vis']
enh_vis = all_image_files[dataset_type]['enh_vis']
org_ir = all_image_files[dataset_type]['org_ir']
enh_ir = all_image_files[dataset_type]['enh_ir']
ref_img = all_image_files[dataset_type]['ref_img']


num_epoch=28
path = '_100_onlyadd_THREE22'
#reader = tf.train.load_checkpoint('./checkpoint_20/ENH_CGAN'+path+'/CGAN.model-'+ str(num_epoch))

checkpoint_path = './checkpoint_20/ENH_CGAN'+path+'/CGAN.model-'+ str(num_epoch)
reader = tf.train.load_checkpoint(checkpoint_path)
var_names = reader.get_variable_to_shape_map().keys()
# retrieve the variable values
var_values = [reader.get_tensor(var_name) for var_name in var_names]


with tf.name_scope('IR_input'):

    #images_ir = tf.placeholder(tf.float32, [1,None,None,None], name='images_ir')
    images_ir = tf.keras.Input(shape=(None,None,None), batch_size=1, name='images_ir')
    
with tf.name_scope('VI_input'):
    
        #images_vi = tf.placeholder(tf.float32,  [1,None,None,None], name='images_vi')
        images_vi = tf.keras.Input(shape=(None,None,None), batch_size=1, name='images_vi')
        
with tf.name_scope('input'):
    #self.resize_ir=tf.image.resize_images(self.images_ir, (self.image_size, self.image_size), method=2)
    input_image_ir=tf.concat([images_ir, images_ir,images_vi],axis=-1)
    input_image_vi=tf.concat([images_vi, images_vi,images_ir],axis=-1)
#self.pred=tf.clip_by_value(tf.sign(self.pred_ir-self.pred_vi),0,1)
with tf.name_scope('ref_input'):
  #images_ref=referance_images(images_ir, images_vi)
    #images_ref=tf.placeholder(tf.float32, [1,None,None,None], name='images_ref')
    images_ref = tf.keras.Input(shape=(None,None,None), batch_size=1, name='images_ref')


with tf.name_scope('fusion'): 
    #self.fusion_image=self.fusion_model(self.input_image)
    
    encoder_image_ir=encoder_ir(input_image_ir)
    encoder_image_vi=encoder_vi(input_image_vi)
    fusion_image_fff=tf.concat([encoder_image_ir,encoder_image_vi],axis=-1)
    fusion_image=decoder(fusion_image_fff)


with tf.compat.v1.Session() as sess:
    init_op=tf.global_variables_initializer()
    sess.run(init_op)
    #data_ir=prepare_data('Enh_IR')
    #data_vi=prepare_data('Enh_VIS')
    #data_ref=prepare_data('referance_TNO')
    data_ir=enh_ir
    data_vi=enh_vis
    data_ref=ref_img
    # data_ir = prepare_data('road/ir')
    # data_vi = prepare_data('road/vi')

    data_vi = Image.open(enh_vis)
    data_vi = data_vi.resize((456, 456))
   
    data_ir = Image.open(enh_ir)
    data_ir = data_ir.resize((456, 456))

    data_vi = np.asarray(data_vi)
    data_ir = np.asarray(data_ir)

    data_ref = Image.open(ref_img)

    data_ref = np.asarray(data_ref)

    org_vis = Image.open(org_vis)
    org_ir = Image.open(org_ir)

    #data_ir = data_ir.resize((456, 456))
    #data_vi = data_vi.resize((456, 456))
    #data_ref = data_ref.resize((456, 456))


    org_vis = org_vis.resize((456, 456))
    org_ir = org_ir.resize((456, 456))
    st.title("GAN Model Testing  Section")
    'Done by Eyob: Infrared and Visible Image Fusion based on the deep learning algorithm called GAN '
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(org_vis, caption='Visible image', use_column_width='auto')
    with col2:
        st.image(org_ir, caption='Infrared Image')


    padding=0
    sub_ir_sequence = []
    sub_vi_sequence = []
    sub_ref_sequence = []
    #input_ir=cv2.resize(data_ir, (456, 456))
    #input_ir=np.lib.pad(data_ir,((padding,padding),(padding,padding)),'edge')


    #w,h=data_ir.size
    input_ir=np.expand_dims(data_ir, axis=-1)
    #input_ir=data_ir.reshape((w,h,1))
    #input_vi=cv2.resize(((data_vi)-127.5)/127.5,(456,456))
    #input_vi=np.lib.pad(data_vi,((padding,padding),(padding,padding)),'edge')

    #w,h=data_vi.size
    input_vi = np.expand_dims(data_vi, axis=-1)
    #input_vi=data_vi.reshape((w,h,1))
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir= np.asarray(sub_ir_sequence)
    train_data_vi= np.asarray(sub_vi_sequence)
    
    
    #input_ref=cv2.resize(((data_ref)-127.5)/127.5,(456,456))
    #input_ref=np.lib.pad(data_ref,((padding,padding),(padding,padding)),'edge')
    #w,h=data_ref.size
    input_ref = np.expand_dims(data_ref, axis=-1)
    #input_ref=data_ref.reshape((w,h,1))
    sub_ref_sequence.append(input_ref)
    train_data_ref=np.asarray(sub_ref_sequence)
    

    result =sess.run(fusion_image,feed_dict={images_ir: train_data_ir,images_vi: train_data_vi, images_ref: train_data_vi})
    result=result*127.5+127.5
    result = result.squeeze()
    variance = 300

    result = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    
    result = np.asarray(result)
    result = SSR(result, variance)
    with col3:
        st.image(result, caption='Fused Image')
