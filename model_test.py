from typing import List
import numpy as np
import tensorflow as tf
import gym
import os
import datetime

from statistics import mean
from gym import wrappers

from tensorflow.keras import Model
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import SGD

input_shape = (64, 64, 1)
feature_extractor_input = Input(shape=input_shape)
x = Conv2D(
    filters=32, kernel_size=8, strides=4
)(feature_extractor_input)

x = Conv2D(filters=64, kernel_size=4, strides=2)(x)
x = Conv2D(filters=64, kernel_size=3, strides=1)(x)
x = Flatten()(x)
x = Dense(units=512)(x)
x = Dense(units=256, name="phi_state")(x)

reward_regression = Dense(
    units=1, 
    name="reward_regression"
)(x)

feature_branch = Model(
    feature_extractor_input, 
    reward_regression, 
    name="feature_branch"
)
feature_branch.summary()

# Deconv
phi_layer = feature_branch.get_layer("phi_state")
deconv_input = Input(shape=(phi_layer.output_shape))

x = Reshape(target_shape=[16, 16, 1])(deconv_input)
x = Conv2DTranspose(
    filters=512, kernel_size=4, strides=1, padding="same", activation="relu"
)(x)
x = Conv2DTranspose(
    filters=256, kernel_size=4, strides=2, padding="same", activation="relu"
)(x)
x = Conv2DTranspose(
    filters=128, kernel_size=4, strides=2, padding="same", activation="relu"
)(x)
x = Conv2DTranspose(
    filters=64, kernel_size=4, strides=2, padding="same", activation="relu"
)(x)

reconstruction = Conv2D(
    filters=1, kernel_size=4, strides=2, padding="same"
)(x)

deconv_decoder = Model(
    deconv_input, 
    reconstruction, 
    name="deconv_decoder"
)
deconv_decoder.summary()


img_input = Input(shape=(64, 64, 1))
reward = feature_branch(img_input)
deconv_img = deconv_decoder(phi_layer.output)
print(type(deconv_img))