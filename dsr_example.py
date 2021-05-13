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

    def get_feature_extractor(self) -> Sequential:
        model = Sequential(name="feature_extractor")
        model.add(
            Conv2D(
                filters=8, kernel_size=[3, 3], strides=[2, 2],
                padding='SAME', input_shape=(10, 10, 1))
        )
        model.add(
            Conv2D(filters=16, kernel_size=[3, 3], strides=[2, 2])
        )
        model.add(Flatten())
        model.add(Dense(units=self.num_units, name="phi_state"))
        model.add(Dense(units=self.num_outputs_regression, name="reward_regression"))
        return model

    def get_deconv(self, input_layer: Layer) -> Sequential:
        model = Sequential(name="deconv_decoder")
        model.add(InputLayer(input_shape=input_layer.output_shape))
        model.add(Reshape(target_shape=(8, 8, 4)))
        model.add(
            Conv2DTranspose(
                filters=3, kernel_size=[3, 3], strides=[2, 2]
            )
        )
        model.add(
            Conv2DTranspose(
                filters=3, kernel_size=[3, 3], strides=[2, 2]
            )
        )
        model.add(Activation("tanh"))
        return model

    def add_sr_layer(self, input_layer: Layer):
        x = Dense(
            units=self.num_units,
        )(input_layer)
        x = Dense(units=256)(x)
        x = Dense(units=self.num_units)(x)
        return x

    def get_successor(self, input_layer) -> Sequential:
        model = Sequential()

        mu_list: List[Dense] = []
        self.q: List[Dense] = []

        for i in range(self.num_actions):
            mu_list.append(self.add_sr_layer(input_layer))

            self.q.append(
                Dense(
                    units=1
                )(mu_list[i])
            )

        stacked = Concatenate(axis=-1)([mu_list])
        model.add(stacked)
        # model.add(Reshape(target_shape=()))

        return model


    def create_feature_model(self, shape: tuple) -> Model:
        extractor = self.get_feature_extractor()
        extractor.summary()
        phi_state_layer = extractor.get_layer("phi_state")

        deconv = self.get_deconv(input_layer=phi_state_layer)
        deconv.summary()

        successor = self.get_successor(input_layer=phi_state_layer)
        successor.summary()

        # model = Model(inputs=[extractor.layers[0]], outputs=[extractor.layers[-1]])


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
    m = DSRModel(num_states, num_actions, learning_rate)
    m.create_feature_model(shape=(10, 10, 1))


if __name__ == '__main__':
    main()
