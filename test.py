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




input=Input(shape=(100,))
x1=Dense(50, activation='relu')(input)
x2=Dense(50, activation='relu')(x1)
x3=Dense(50, activation='relu')(x2)
aux_model1 = Model(inputs=input, outputs=x3)

x3_input= Input(shape=x3.shape.as_list()[1:])
x4=Dense(50, activation='relu')(x3_input)
output=Dense(10, activation='softmax')(x4)
aux_model2 = Model(inputs=x3_input, outputs=output)

x3 = aux_model1(input)
output = aux_model2(x3)
model1 = Model(inputs=input, outputs=output)
model1.compile(optimizer='rmsprop', loss='cross_entropy')

for layer in aux_model2.layers:
    layer.trainable=False
model2 = Model(inputs=input, outputs=output)

model2.compile(optimizer='rmsprop', loss='cross_entropy')
print()