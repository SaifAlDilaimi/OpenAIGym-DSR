import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from collections import deque
from gym import wrappers
from gym_minigrid.wrappers import *
from tensorflow.keras import callbacks
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import clone_model
from tensorflow.keras.layers import Input, InputLayer
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Conv2DTranspose
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Lambda
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import ReLU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.initializers import HeUniform
from tensorflow.keras.initializers import GlorotUniform
from tensorflow.keras.initializers import GlorotNormal
from tensorflow.keras.regularizers import L2

class DSRModel:
    def __init__(
        self,
        input_shape: tuple,
        num_actions: int
    ) -> None:
        super(DSRModel, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.weight_initializer = GlorotNormal()
        self.weight_regularizer = L2()
        self.nb_phi_units = 256

        self.create_model()

    def get_feature_extractor(self) -> Model:
        feature_extractor_input = Input(shape=self.input_shape, name="Feature_Input")

        x = Conv2D(
            filters=32, kernel_size=8, strides=2, 
            kernel_initializer=self.weight_initializer
        )(feature_extractor_input)
        x = ReLU()(x)
        x = Conv2D(filters=64, kernel_size=4, strides=2, 
            kernel_initializer=self.weight_initializer
        )(x)
        x = ReLU()(x)
        x = Conv2D(filters=64, kernel_size=3, strides=1, 
            kernel_initializer=self.weight_initializer
        )(x)
        x = ReLU()(x)

        x = Flatten()(x)
        phi_state = Dense(units=self.nb_phi_units, activation='tanh', 
                    kernel_initializer=self.weight_initializer, 
                    name="phi_state"
        )(x)

        model = Model(
            feature_extractor_input, 
            phi_state, 
            name="feature_branch"
        )

        return model

    def get_reward_model(self) -> Model:
        feature_extractor_output = Input(shape=(self.nb_phi_units,))
        
        reward_regression = Dense(
            units=1,
            kernel_initializer=self.weight_initializer,
            name="reward_regressor"
        )(feature_extractor_output)

        model = Model(
            feature_extractor_output, 
            reward_regression, 
            name="model_r_regression"
        )


        return model

    def get_deconv(self) -> Model:
        deconv_input = Input(shape=(self.nb_phi_units,), name="Deconv_Input")
        
        x = Reshape(target_shape=[16, 16, 1])(deconv_input)
        x = Conv2DTranspose(
            filters=512, kernel_size=4, strides=1,
            kernel_initializer=self.weight_initializer, padding="same"
        )(x)
        x = ReLU()(x)
        x = Conv2DTranspose(
            filters=256, kernel_size=4, strides=2,
            kernel_initializer=self.weight_initializer, padding="same"
        )(x)
        x = ReLU()(x)
        x = Conv2DTranspose(
            filters=128, kernel_size=4, strides=2,
            kernel_initializer=self.weight_initializer, padding="same"
        )(x)
        x = ReLU()(x)

        reconstruction = Conv2DTranspose(
            filters=3, kernel_size=4, strides=1,
            kernel_initializer=self.weight_initializer, padding="same",
            activation="tanh", name="state_reconstruction"
        )(x)

        model = Model(
            deconv_input, 
            reconstruction, 
            name="model_decoder"
        )

        return model

    def add_sr_block(self, input_layer: Input, action: int):
        x = Dense(units=self.nb_phi_units,
            kernel_initializer=self.weight_initializer
        )(input_layer)
        x = ReLU()(x)
        x = Dense(units=self.nb_phi_units//2,
            kernel_initializer=self.weight_initializer)(x)
        x = ReLU()(x)
        x = Dense(units=self.nb_phi_units, kernel_initializer=self.weight_initializer, 
            name=f'm_state_a{action}'
        )(x)
        return x

    def get_successor(self) -> Model:
        output_layers = []

        successor_input = Input(shape=(self.nb_phi_units,), name="SR_Input")
        x_1_stop_grad = Lambda(lambda x1: tf.stop_gradient(x1))(successor_input)

        for a in range(self.num_actions):
            output_layers.append(self.add_sr_block(x_1_stop_grad, action=a))

        model = Model(successor_input, output_layers, name="successor_branch")

        return model

    def create_model(self) -> Model:
        state_img_input = Input(shape=self.input_shape, name="Pipeline_start")

        # Feature Extractor 
        feature_branch_model = self.get_feature_extractor()
        feature_extractor_output = feature_branch_model(state_img_input)

        # Feature Extractor + Reward Regressor
        reward_regressor_model = self.get_reward_model()
        reward_regressor_output = reward_regressor_model(feature_extractor_output)

        # State Reconstruction
        deconv_decoder_model = self.get_deconv()
        deconv_decoder_output = deconv_decoder_model(feature_extractor_output)

        # SR Branch
        successor_branch_model = self.get_successor()
        successor_branch_output = successor_branch_model(feature_extractor_output)

        outputs = [reward_regressor_output, deconv_decoder_output, *successor_branch_output]

        dsr = Model(
            inputs=[state_img_input],
            outputs=outputs,
            name="DSR"
        )

        return dsr

class DQN:
    def __init__(
        self,
        action_space: int,
        observation_shape: tuple,
        memory_capacity: int = 10000, 
        nb_episodes: int = 100, 
        nb_steps: int = 50,
        optimizer: RMSprop = None,
        batch_size=32,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.3,
        epsilon_decay: float = 0.995,
        nb_episodes_warmup: int = 100,
        update_epsilon_episode: int = 1,
        target_model_update: float = 1000,
        tau: float = 5e-3,
        verbose=False
    ) -> float:
        self.nb_actions = action_space
        self.observation_shape = observation_shape

        # prepare the memory for the RL agent
        self.memory  = deque(maxlen=memory_capacity)
        self.nonzero_memory  = deque(maxlen=memory_capacity)

        # Setup the Params
        self.nb_episodes = nb_episodes
        self.nb_steps = nb_steps
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.nb_episodes_warmup = nb_episodes_warmup
        self.target_model_update = target_model_update
        self.update_epsilon_episode = update_epsilon_episode
        self.tau = tau
        self.optimizer = optimizer
        self.steps_since_last_update = 0
        self.verbose = verbose
        # define the starting index of the SF output layers
        self.sf_output_start_index = 2

        self.build_model()

    def build_model(
        self,
    ) -> None:
        # build target model
        dsr = DSRModel(
            input_shape=self.observation_shape,
            num_actions=self.nb_actions
        )
        self.model_target = dsr.create_model()

        def l2_norm(y_true, y_pred):
            return tf.sqrt(tf.reduce_sum(tf.subtract(y_true, y_pred) ** 2))

        loss_fn = {
            'model_r_regression': mean_squared_error,
            'model_decoder': l2_norm,
            'successor_branch': mean_squared_error
        }

        for a in range(1, self.nb_actions):
            loss_fn[f'successor_branch_{a}'] = mean_squared_error

        self.model_target.compile(optimizer=self.optimizer, loss=loss_fn, metrics=['accuracy'])
        # build online model by cloning the target model
        self.model_online = clone_model(self.model_target)
        self.model_online.compile(optimizer=self.optimizer, loss=loss_fn, metrics=['accuracy'])

    def update_target_model_hard(self):
        for l_sr, l_tg in zip(self.model_online.layers, self.model_target.layers):
            wk0 = l_sr.get_weights()
            l_tg.set_weights(wk0)
        self.steps_since_last_update = 0

    def update_target_model_soft(self):
        for l_sr, l_tg in zip(self.model_online.layers, self.model_target.layers):
            if type(l_sr) != InputLayer:
                wk0 = l_sr.get_weights()
                wk1 = l_tg.get_weights()
                new_weights = []
                for i in range(len(wk0)):
                    wk1_ = self.tau * wk0[i] + (1.0 - self.tau) * wk1[i]
                    new_weights.append(wk1_)
                l_tg.set_weights(new_weights)

    def calculate_a_prime(
        self,
        state1_batch: np.ndarray
    ) -> int:
        # Get only the outputs from model responsible for successor features
        successor_features_state1 = self.model_online.predict_on_batch(state1_batch)[self.sf_output_start_index:]
        w = self.calculate_w()

        Q_s = np.matmul(successor_features_state1, w)
        #a_prime = np.argmax(Q_s, axis=0)
        # Get max action column-wise based on a 2D Matrix (due train on batch)
        a_prime = np.where(Q_s == np.amax(Q_s, axis=0))

        return a_prime

    def calculate_w(
        self,
        use_target: bool = False
    ) -> np.ndarray:
        model = self.model_online
        if use_target:
            model = self.model_target
        feature_branch = model.get_layer("feature_branch")
        phi_state_weights = feature_branch.get_layer("phi_state").get_weights()
        w = phi_state_weights[0][1] # 0 = weights (not bais), 1 = output weights
        return w

    def calculate_phi(
        self,
        state_batch: np.ndarray,
        use_target: bool = False
    ) -> np.ndarray:
        model = self.model_online
        if use_target:
            model = self.model_target
        feature_branch = model.get_layer("feature_branch")
        phi_input = feature_branch.inputs[0]
        phi_output = feature_branch.get_layer("phi_state").output
        
        phi_layer_model = Model(inputs=phi_input, outputs=phi_output)
        phi_states = phi_layer_model(state_batch)

        return phi_states

    def calculate_target_r(
        self,
        phi_states: np.ndarray,
        use_target: bool = False
    ) -> np.ndarray:
        w = self.calculate_w(use_target=use_target)
        target_r_batch = np.matmul(phi_states, w)
        return target_r_batch

    def remember(self, state, action, reward, new_state, done):
        if reward == 10:
            self.nonzero_memory.append([state, action, reward, new_state, done])
        
        self.memory.append([state, action, reward, new_state, done])

    def sample_batch(
        self
    ) -> list:
        memory = self.memory

        if np.random.random() <= 0.3:
            memory = self.nonzero_memory

        sample_indicies = np.random.choice(len(memory), self.batch_size)
        samples = [memory[idx] for idx in sample_indicies]

        return samples

    def memory_ready(self) -> bool:
        if (
            len(self.memory) < self.batch_size * 4 
            or len(self.nonzero_memory) < self.batch_size * 4
        ): 
            return False
        else:
            return True

    def select_action(
        self,
        state: np.ndarray,
        epsilon: float = None
    ) -> int:
        if epsilon == None:
            epsilon = self.epsilon

        if np.random.random() < self.epsilon:
            action = np.random.choice(self.nb_actions)
            return action

        batch = np.array([state])
        successor_features = self.model_online.predict_on_batch(batch)[self.sf_output_start_index:]
        w = self.calculate_w()
        q_values = np.matmul(successor_features, w)
        action = np.argmax(q_values)
        
        # if self.verbose:
        #     print('-------------------------------------------------------')
        #     print(f'Min/Max w: {np.min(w)}/{np.max(w)}')
        #     print(f'Q-Values: {q_values}')
        #     print(f'Selecting action based on successor features: {action}')
        #     print('-------------------------------------------------------')

        return action

    def replay(self):
        '''
        This function replays experiences to update the Q-function.
        
        | **Args**
        | replayBatchSize:              The number of random that will be replayed.
        '''
        if not self.memory_ready():
            return

        experiences = self.sample_batch()

        # Start by extracting the necessary parameters (we use a vectorized implementation).
        state0_batch = []
        reward_batch = []
        action_batch = []
        terminal1_batch = []
        state1_batch = []
        for e in experiences:
            state, action, reward, new_state, done = e
            state0_batch.append(state)
            state1_batch.append(new_state)
            reward_batch.append(reward)
            action_batch.append(action)
            terminal1_batch.append(1. if done else 0.)

        state0_batch = np.array(state0_batch)
        state1_batch = np.array(state1_batch)
        reward_batch = np.array(reward_batch)
        
        phi_states0 = self.calculate_phi(state0_batch)

        # Get only the successor features from the predictions
        target_sf_state1_batch = self.model_target.predict_on_batch(state1_batch)[self.sf_output_start_index:]
        target_sf_state1_batch = np.array(target_sf_state1_batch)
        action_indicies, batch_indicies = self.calculate_a_prime(state1_batch)
        # Update only the SF based on the actions & batches selected by a'
        target_sf_state1_batch[action_indicies][batch_indicies] *= self.gamma
        target_sf_state1_batch[action_indicies][batch_indicies] += phi_states0

        target_r_batch = self.calculate_target_r(phi_states0)

        # If the experiences contain terminal states set the reward
        terminal_indicies = np.argwhere(terminal1_batch == 1)
        target_r_batch[terminal_indicies] = reward_batch[terminal_indicies]

        losses = self.model_online.train_on_batch(
            x=state0_batch, 
            y=[target_r_batch, state0_batch, *target_sf_state1_batch]
        )

        if self.target_model_update < 1.:
            self.update_target_model_soft()
        elif self.steps_since_last_update % self.target_model_update == 0:
            print(f'Updating target model...')
            self.update_target_model_hard()
        
        return losses

def resize_img(img: np.array) -> np.array:
    # dsize
    dsize = (64, 64)

    # resize image
    output = cv2.resize(img, dsize, interpolation = cv2.INTER_AREA)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    output = np.reshape(output, newshape=(64, 64, 1))        
    output = output.astype('float32')
    return output

def collect_nonzero_samples(env, dqn_agent: DQN):
    trial_len = 255
    while not dqn_agent.memory_ready():
        cur_state = env.reset()
        cur_state = resize_img(cur_state)
        for _ in range(trial_len):
            action = dqn_agent.select_action(cur_state)
            new_state, reward, done, _ = env.step(action)
            new_state = resize_img(new_state)

            print(f'Collecting Non-Zero Memory {len(dqn_agent.nonzero_memory)}/{dqn_agent.batch_size*2}', end="\r")
            
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            cur_state = new_state
            if done:
                break

def main():
    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = RGBImgPartialObsWrapper(env) # Get pixel observations
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    gamma = 0.99
    num_actions = 3
    img_shape = (64, 64, 1)

    # DSR Hyperparameter
    nb_episodes = 2000
    nb_steps = 50
    learning_rate = 25e-4
    batch_size = 16
    momentum = 0.95
    min_epsilon = 0.3
    nb_episodes_warmup = 1
    target_model_update = 2000
    update_epsilon_episode = 5
    optimizer = RMSprop(learning_rate=learning_rate, momentum=momentum)

    dqn_agent = DQN(
        action_space=num_actions,
        observation_shape=img_shape,
        nb_episodes=nb_episodes,
        nb_steps=nb_steps,
        nb_episodes_warmup=nb_episodes_warmup,
        batch_size=batch_size,
        epsilon_min=min_epsilon,
        gamma=gamma,
        target_model_update=target_model_update,
        update_epsilon_episode=update_epsilon_episode,
        optimizer=optimizer,
        verbose=True
    )
    
    collect_nonzero_samples(env, dqn_agent)
    reward_per_episode = []
    for episode in range(nb_episodes):
        print(f'Episode {episode}/{nb_episodes}')
        episodic_reward = 0
        cur_state = env.reset()
        cur_state = resize_img(cur_state)
        for step in range(nb_steps):
            action = dqn_agent.select_action(cur_state)
            new_state, reward, done, _ = env.step(action)
            new_state = resize_img(new_state)
            
            episodic_reward += reward
            print(f'Step {step}/{nb_steps}; Agent Pos: {env.agent_pos}; Action: {action}, Reward: {reward}')
            
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()

            cur_state = new_state
            if done:
                print(f'\n Found Goal! Reward: {reward};')
                break
        print(f'Episodic Reward: {episodic_reward}')
        reward_per_episode.append(episodic_reward)
    print(f'Average Reward over {nb_episodes}: {np.mean(reward_per_episode)}')

if __name__ == '__main__':
    main()
