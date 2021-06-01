import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from collections import deque
from gym import wrappers
from gym_minigrid.wrappers import *
from tensorflow.keras import Model
from tensorflow.keras.layers import Input
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


class DSRModel:
    def __init__(
        self,
        input_shape: tuple,
        num_actions: int,
        learning_rate: float,
        momentum: float
    ) -> None:
        super(DSRModel, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.momentum = momentum

        self.create_DSR_model()

    def get_feature_extractor(self) -> Model:
        feature_extractor_input = Input(shape=self.input_shape, name="Feature_Input")
        x = Conv2D(
            filters=32, kernel_size=8, strides=2
        )(feature_extractor_input)
        x = ReLU()(x)
        
        x = Conv2D(filters=64, kernel_size=4, strides=2)(x)
        x = ReLU()(x)
        x = Conv2D(filters=64, kernel_size=3, strides=1)(x)
        x = ReLU()(x)
        x = Flatten()(x)
        x = Dense(units=512)(x)
        x = ReLU()(x)
        phi_state = Dense(units=256, activation='relu', name="phi_state")(x)

        model = Model(
            feature_extractor_input, 
            phi_state, 
            name="feature_branch"
        )

        return model

    def get_reward_model(self) -> Model:
        feature_extractor_output = Input(shape=(256,))
        
        reward_regression = Dense(
            units=1, 
            name="reward_regressor"
        )(feature_extractor_output)

        model = Model(
            feature_extractor_output, 
            reward_regression, 
            name="model_r_regression"
        )

        return model

    def get_deconv(self) -> Model:
        deconv_input = Input(shape=(256,), name="Deconv_Input")
        
        x = Reshape(target_shape=[16, 16, 1])(deconv_input)
        x = Conv2DTranspose(
            filters=512, kernel_size=4, strides=1, padding="same"
        )(x)
        x = ReLU()(x)
        x = Conv2DTranspose(
            filters=256, kernel_size=4, strides=2, padding="same"
        )(x)
        x = ReLU()(x)
        x = Conv2DTranspose(
            filters=128, kernel_size=4, strides=2, padding="same"
        )(x)
        x = ReLU()(x)
        x = Conv2DTranspose(
            filters=64, kernel_size=4, strides=1, padding="same"
        )(x)
        x = ReLU()(x)

        reconstruction = Conv2DTranspose(
            filters=1, kernel_size=4, strides=1,
            padding="same", name="state_reconstruction", activation="tanh"
        )(x)

        model = Model(
            deconv_input, 
            reconstruction, 
            name="model_decoder"
        )

        return model

    def add_sr_block(self, input_layer: Input, action: int):
        x = Dense(units=256)(input_layer)
        x = ReLU()(x)
        x = Dense(units=512)(x)
        x = ReLU()(x)
        x = Dense(units=256, name=f'm_state_a{action}')(x)
        return x

    def get_successor(self) -> Model:
        output_layers = []

        successor_input = Input(shape=(256,), name="SR_Input")
        x = Dense(512)(successor_input)
        x = Activation('tanh')(x)
        x_1_stop_grad = Lambda(lambda x: tf.stop_gradient(x))(x)

        for a in range(self.num_actions):
            output_layers.append(self.add_sr_block(x_1_stop_grad, action=a))

        model = Model(successor_input, output_layers, name="successor_branch")

        return model

    def create_DSR_model(self) -> Model:
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

        reward_deconv = Model(
            inputs=[state_img_input],
            outputs=[reward_regressor_output, deconv_decoder_output],
            name="DSR_reward_deconv"
        )
        reward_deconv.compile(
            loss={
                'model_r_regression': mean_squared_error,
                'model_decoder': mean_squared_error
            },
            optimizer=RMSprop(
                learning_rate=self.learning_rate,
                momentum=self.momentum
            )
        )

        # SR Branch
        successor_branch_model = self.get_successor()
        successor_branch_output = successor_branch_model(feature_extractor_output)

        phi_successor = Model(
            inputs=[state_img_input],
            outputs=[successor_branch_output],
            name="DSR_phi_successor"
        )

        phi_successor.compile(
            loss=mean_squared_error,
            optimizer=RMSprop(
                learning_rate=self.learning_rate,
                momentum=self.momentum
            )
        )

        tf.keras.utils.plot_model(
            reward_deconv,
            to_file="model_reward_deconv.png",
            show_shapes=True,
            expand_nested=True)
        tf.keras.utils.plot_model(
            phi_successor, 
            to_file="model_sr.png", 
            show_shapes=True,
            expand_nested=True)

        self.reward_deconv = reward_deconv
        self.sr = phi_successor

class DQN:
    def __init__(
        self,
        action_space: int,
        observation_shape: tuple,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.1,
        epsilon_decay: float = 0.995,
        batch_size: int = 32,
        lr: float = 2.5e-4,
        momentum: float = 0.95,
        memory_size: int = 1000000
    ) -> float:
        self.action_space = action_space
        self.observation_shape = observation_shape
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.lr = lr
        self.momentum = momentum
        self.memory  = deque(maxlen=memory_size)
        self.nonzero_memory  = deque(maxlen=int(5e4))
        self.tau = .125

        dsr = DSRModel(
            input_shape=observation_shape, 
            num_actions=action_space, 
            learning_rate=lr,
            momentum=momentum
        )
        dsr_target = DSRModel(
            input_shape=observation_shape, 
            num_actions=action_space, 
            learning_rate=lr,
            momentum=momentum
        )
        self.model_reward_deconv = dsr.reward_deconv
        self.model_sr = dsr.sr
        self.target_model_reward_deconv = dsr_target.reward_deconv
        self.target_model_sr = dsr_target.sr


    def calculate_a_prime(
        self,
        new_state: np.ndarray
    ) -> int:
        batch = np.array([new_state])

        successor_features_next_state = self.model_sr.predict_on_batch(batch)
        w = self.calculate_w()

        Q_s = np.matmul(successor_features_next_state, w)[0]
        a_prime = np.argmax(Q_s)

        return a_prime

    def calculate_w(
        self
    ) -> np.ndarray:
        feature_branch = self.model_reward_deconv.get_layer("feature_branch")
        phi_state_weights = feature_branch.get_layer("phi_state").get_weights()
        w = phi_state_weights[0][1] # 0 = weights (not bais), 1 = output weights
        return w

    def calculate_phi(
        self,
        state: np.ndarray
    ) -> np.ndarray:
        feature_branch = self.model_reward_deconv.get_layer("feature_branch")
        phi_input = feature_branch.input
        phi_output = feature_branch.get_layer("phi_state").output
        
        phi_layer_model = Model(inputs=phi_input, outputs=phi_output)
        phi_state = phi_layer_model(np.array([state])).numpy()

        return phi_state

    def calculate_target_r(
        self,
        phi_state: np.ndarray
    ) -> np.ndarray:
        w = self.calculate_w()
        target_r = np.matmul(phi_state, w)
        return target_r

    def act(self, state):
        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)

        batch = np.array([state])

        successor_features = self.model_sr.predict_on_batch(batch)[0]
        w = self.calculate_w()

        Q_s = np.matmul(successor_features, w)
        print("Qs: ", Q_s)
        a = np.argmax(Q_s)

        return a

    def remember(self, state, action, reward, new_state, done):
        if reward == 10:
            self.nonzero_memory.append([state, action, reward, new_state, done])
        else:
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
            len(self.memory) < self.batch_size * 2 
            or len(self.nonzero_memory) < self.batch_size * 2
        ): 
            return False
        else:
            return True

    def replay(self):
        if not self.memory_ready():
            return

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        samples = self.sample_batch()

        for sample in samples:
            state, action, reward, new_state, done = sample
            batch_state = np.array([state])
            batch_new_state = np.array([new_state])

            phi_state = self.calculate_phi(state)
            target_sf_next_state = self.target_model_sr.predict_on_batch(batch_new_state)
            a_prime = self.calculate_a_prime(new_state)
            target_sf_next_state[0][a_prime] *= self.gamma
            target_sf_next_state[0][a_prime] += phi_state

            target_r = self.calculate_target_r(phi_state)

            if done:
                target_r = reward

            self.model_sr.train_on_batch(batch_state, target_sf_next_state)
            self.model_reward_deconv.train_on_batch(
                x=batch_state, 
                y={
                    'model_r_regression': np.array([target_r]),
                    'model_decoder': batch_state
                }
            )

    def target_train(self):
        for l_sr, l_tg in zip(self.model_sr.layers, self.target_model_sr.layers):
            wk0 = l_sr.get_weights()
            l_tg.set_weights(wk0)

        for l_sr, l_tg in zip(self.model_reward_deconv.layers, self.target_model_reward_deconv.layers):
            wk0 = l_sr.get_weights()
            l_tg.set_weights(wk0)
'''
    def target_train(self):
        weights_sr = self.model_sr.get_weights()
        target_weights_sr = self.target_model_sr.get_weights()
        for i in range(len(target_weights_sr)):
            target_weights_sr[i] = weights_sr[i] * self.tau + target_weights_sr[i] * (1 - self.tau)
        self.target_model_sr.set_weights(target_weights_sr)

        weights_r_decoder = self.model_reward_deconv.get_weights()
        target_weights_r_decoder = self.target_model_reward_deconv.get_weights()
        for i in range(len(target_weights_r_decoder)):
            target_weights_r_decoder[i] = weights_r_decoder[i] * self.tau + target_weights_r_decoder[i] * (1 - self.tau)
        self.target_model_reward_deconv.set_weights(target_weights_r_decoder)
'''

def resize_img(img: np.array) -> np.array:
    # dsize
    dsize = (64, 64)

    # resize image
    output = cv2.resize(img, dsize, interpolation = cv2.INTER_AREA)
    output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY)
    output = np.reshape(output, newshape=(64, 64, 1))
    return output

def collect_nonzero_samples(env, dqn_agent: DQN):
    trial_len = 255
    while not dqn_agent.memory_ready():
        cur_state = env.reset()
        cur_state = resize_img(cur_state)
        for _ in range(trial_len):
            action = dqn_agent.act(cur_state)
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
    batch_size = 32
    lr = 2.5e-4
    momentum = 0.95
    n_episodes  = 400
    n_steps = 255

    dqn_agent = DQN(
        action_space=num_actions,
        observation_shape=img_shape,
        gamma=gamma,
        batch_size=batch_size, 
        lr=lr,
        momentum=momentum
    )
    
    collect_nonzero_samples(env, dqn_agent)
    reward_per_episode = []
    for episode in range(n_episodes):
        print(f'Episode {episode}/{n_episodes}')
        episodic_reward = 0
        cur_state = env.reset()
        cur_state = resize_img(cur_state)
        for step in range(n_steps):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)
            new_state = resize_img(new_state)
            
            episodic_reward += reward
            print(f'Step {step}/{n_steps}; Agent Pos: {env.agent_pos}; Action: {action}, Reward: {reward}')
            
            dqn_agent.remember(cur_state, action, reward, new_state, done)

            dqn_agent.replay()       # internally iterates default (prediction) model
            dqn_agent.target_train() # iterates target model

            cur_state = new_state
            if done:
                print(f'\n Found Goal! Reward: {reward};')
                break
        print(f'Episodic Reward: {episodic_reward}')
        reward_per_episode.append(episodic_reward)
    print(f'Average Reward over {n_episodes}: {np.mean(reward_per_episode)}')

if __name__ == '__main__':
    main()
