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


class DSRModel:
    def __init__(
        self, 
        num_states: int, 
        num_actions: int, 
        learning_rate: float,
    ) -> None:
        super(DSRModel, self).__init__()
        self.num_states = num_states
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.num_units = 512
        self.num_outputs_regression = 1

    def get_feature_extractor(self, input_shape: tuple) -> Model:
        feature_extractor_input = Input(shape=input_shape)
        x = Conv2D(
            filters=32, kernel_size=[8, 8], strides=[2, 2]
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

        model = Model(
            feature_extractor_input, 
            reward_regression, 
            name="feature_branch"
        )
        model.summary()
        
        tf.keras.utils.plot_model(model, to_file="reward_regression.png", show_shapes=True)

        return model

    def get_deconv(self, input_shape: tuple) -> Model:
        
        deconv_input = Input(shape=input_shape)
        
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

        model = Model(
            deconv_input, 
            reconstruction, 
            name="deconv_decoder"
        )
        model.summary()

        tf.keras.utils.plot_model(model, to_file="deconv_decoder.png", show_shapes=True)

        return model

    def add_sr_block(self, input_layer: Input, action: int):
        x = Dense(
            units=512,
        )(input_layer)
        x = Dense(units=256)(x)
        x = Dense(units=256)(x)
        x = Dense(units=1, name=f'm_state_a{action}')(x)
        return x

    def get_successor(self, input_shape: tuple) -> Model:
        output_layers = []

        successor_input = Input(shape=input_shape)
        x = Dense(256)(successor_input)

        for a in range(self.num_actions):
            output_layers.append(self.add_sr_block(x, action=a))

        model = Model(successor_input, output_layers, name="successor_branch")
        model.summary()

        tf.keras.utils.plot_model(model, to_file="successor_branch.png", show_shapes=True)

        return model

    def create_DSR_model(self, shape: tuple) -> Model:
        state_img_input = Input(shape=shape)
        extractor = self.get_feature_extractor(input_shape=shape)
        phi_state_layer = extractor.get_layer("phi_state")
        phi_state_output_shape = phi_state_layer.output_shape

        # Feature Extractor + Reward Regressor
        self.reward_regression = extractor(state_img_input)

        # State Reconstruction
        deconv_decoder = self.get_deconv(input_shape=phi_state_output_shape)
        self.deconv_decoder = deconv_decoder(phi_state_layer.output)

        # SR Branch
        successor_branch = self.get_successor(input_shape=phi_state_output_shape)
        self.successor_branch = successor_branch(phi_state_layer.output)


        


class DQN:
    def __init__(
        self,
        num_states,
        num_actions,
        hidden_units,
        gamma,
        max_experiences,
        min_experiences,
        batch_size,
        lr,
    ):
        self.num_actions = num_actions
        self.batch_size = batch_size
        self.optimizer = tf.optimizers.Adam(lr)
        self.gamma = gamma
        self.model = DSRModel(num_states, hidden_units, num_actions)
        self.experience = {'s': [], 'a': [], 'r': [], 's2': [], 'done': []}
        self.max_experiences = max_experiences
        self.min_experiences = min_experiences

    def predict(self, inputs):
        return self.model(np.atleast_2d(inputs.astype('float32')))

    def train(self, TargetNet):
        if len(self.experience['s']) < self.min_experiences:
            return 0
        ids = np.random.randint(
            low=0, high=len(self.experience['s']), size=self.batch_size
        )
        states = np.asarray([self.experience['s'][i] for i in ids])
        actions = np.asarray([self.experience['a'][i] for i in ids])
        rewards = np.asarray([self.experience['r'][i] for i in ids])
        states_next = np.asarray([self.experience['s2'][i] for i in ids])
        dones = np.asarray([self.experience['done'][i] for i in ids])
        value_next = np.max(TargetNet.predict(states_next), axis=1)
        actual_values = np.where(
            dones, rewards, rewards + self.gamma * value_next
        )

        with tf.GradientTape() as tape:
            selected_action_values = tf.math.reduce_sum(
                self.predict(states) * tf.one_hot(actions, self.num_actions),
                axis=1,
            )
            loss = tf.math.reduce_mean(
                tf.square(actual_values - selected_action_values)
            )
        variables = self.model.trainable_variables
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        return loss

    def get_action(self, states, epsilon):
        if np.random.random() < epsilon:
            return np.random.choice(self.num_actions)
        else:
            return np.argmax(self.predict(np.atleast_2d(states))[0])

    def add_experience(self, exp):
        if len(self.experience['s']) >= self.max_experiences:
            for key in self.experience.keys():
                self.experience[key].pop(0)
        for key, value in exp.items():
            self.experience[key].append(value)

    def copy_weights(self, TrainNet):
        variables1 = self.model.trainable_variables
        variables2 = TrainNet.model.trainable_variables
        for v1, v2 in zip(variables1, variables2):
            v1.assign(v2.numpy())


def play_game(env, TrainNet, TargetNet, epsilon, copy_step):
    rewards = 0
    iter = 0
    done = False
    observations = env.reset()
    losses = list()
    while not done:
        action = TrainNet.get_action(observations, epsilon)
        prev_observations = observations
        observations, reward, done, _ = env.step(action)
        rewards += reward
        if done:
            reward = -200
            env.reset()

        exp = {
            's': prev_observations,
            'a': action,
            'r': reward,
            's2': observations,
            'done': done,
        }
        TrainNet.add_experience(exp)
        loss = TrainNet.train(TargetNet)
        if isinstance(loss, int):
            losses.append(loss)
        else:
            losses.append(loss.numpy())
        iter += 1
        if iter % copy_step == 0:
            TargetNet.copy_weights(TrainNet)
    return rewards, mean(losses)


def make_video(env, TrainNet):
    env = wrappers.Monitor(env, os.path.join(os.getcwd(), "videos"), force=True)
    rewards = 0
    steps = 0
    done = False
    observation = env.reset()
    while not done:
        env.render()
        action = TrainNet.get_action(observation, 0)
        observation, reward, done, _ = env.step(action)
        steps += 1
        rewards += reward
    print("Testing steps: {} rewards {}: ".format(steps, rewards))


def main():
    num_states = 10
    num_actions = 8
    learning_rate = 1e-3
    img_shape = (64, 64, 1)
    m = DSRModel(num_states, num_actions, learning_rate)
    m.create_DSR_model(shape=img_shape)


if __name__ == '__main__':
    main()
