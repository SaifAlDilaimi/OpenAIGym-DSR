import numpy as np
import tensorflow as tf
import gym
import gym_minigrid
import cv2

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
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import mean_squared_error


class DSRModel:
    def __init__(
        self,
        input_shape: tuple,
        num_actions: int,
        learning_rate: float
    ) -> None:
        super(DSRModel, self).__init__()
        self.input_shape = input_shape
        self.num_actions = num_actions
        self.learning_rate = learning_rate

        self.create_DSR_model()

    def get_feature_extractor(self) -> Model:
        feature_extractor_input = Input(shape=self.input_shape, name="Feature_Input")
        x = Conv2D(
            filters=32, kernel_size=[8, 8], strides=[2, 2]
        )(feature_extractor_input)
        
        x = Conv2D(filters=64, kernel_size=4, strides=2)(x)
        x = Conv2D(filters=64, kernel_size=3, strides=1)(x)
        x = Flatten()(x)
        x = Dense(units=512)(x)
        phi_state = Dense(units=256, name="phi_state")(x)

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
            filters=512, kernel_size=4, strides=1, 
            padding="same", activation="relu"
        )(x)
        x = Conv2DTranspose(
            filters=256, kernel_size=4, strides=2,
            padding="same", activation="relu"
        )(x)
        x = Conv2DTranspose(
            filters=128, kernel_size=4, strides=2,
            padding="same", activation="relu"
        )(x)
        x = Conv2DTranspose(
            filters=64, kernel_size=4, strides=1,
            padding="same", activation="relu"
        )(x)

        reconstruction = Conv2DTranspose(
            filters=1, kernel_size=4, strides=1,
            padding="same", name="state_reconstruction"
        )(x)

        model = Model(
            deconv_input, 
            reconstruction, 
            name="model_decoder"
        )

        return model

    def add_sr_block(self, input_layer: Input, action: int):
        x = Dense(
            units=512,
        )(input_layer)
        x = Dense(units=256)(x)
        x = Dense(units=256)(x)
        x = Dense(units=1, name=f'm_state_a{action}')(x)
        return x

    def get_successor(self) -> Model:
        output_layers = []

        successor_input = Input(shape=(256,), name="SR_Input")
        x = Dense(256)(successor_input)

        for a in range(self.num_actions):
            output_layers.append(self.add_sr_block(x, action=a))

        model = Model(successor_input, output_layers, name="successor_branch")

        return model

    def create_DSR_model(self) -> Model:
        state_img_input = Input(shape=self.input_shape, name="Pipeline_start")

        # Feature Extractor 
        feature_extractor = self.get_feature_extractor()
        feature_extractor = feature_extractor(state_img_input)

        # Feature Extractor + Reward Regressor
        reward_regressor = self.get_reward_model()
        reward_regressor = reward_regressor(feature_extractor)

        # State Reconstruction
        deconv_decoder = self.get_deconv()
        deconv_decoder = deconv_decoder(feature_extractor)

        # SR Branch
        successor_branch = self.get_successor()
        successor_branch = successor_branch(feature_extractor)

        reward_deconv = Model(
            inputs=[state_img_input],
            outputs=[reward_regressor, deconv_decoder],
            name="DSR_reward_deconv"
        )
        reward_deconv.compile(
            loss={
                'model_r_regression': mean_squared_error,
                'model_decoder': mean_squared_error
            },
            optimizer=RMSprop(learning_rate=self.learning_rate)
        )

        phi_successor = Model(
            inputs=[state_img_input],
            outputs=[successor_branch],
            name="DSR_phi_successor"
        )

        # Freeze feature extractor as stated in paper
        layers = phi_successor.layers[2:6]
        for layer in layers:
            layer.trainable = False

        phi_successor.compile(
            loss=mean_squared_error,
            optimizer=RMSprop(learning_rate=self.learning_rate)
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
        memory_size: int = 1000000
    ) -> float:
        self.action_space = action_space
        self.observation_shape = observation_shape
        self.batch_size = batch_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.memory  = deque(maxlen=memory_size)
        self.tau = .125

        dsr = DSRModel(
            input_shape=observation_shape, 
            num_actions=action_space, 
            learning_rate=lr
        )
        dsr_target = DSRModel(
            input_shape=observation_shape, 
            num_actions=action_space, 
            learning_rate=lr
        )
        self.model_reward_deconv = dsr.reward_deconv
        self.model_sr = dsr.sr
        self.target_model_reward_deconv = dsr_target.reward_deconv
        self.target_model_sr = dsr_target.sr

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)

        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)

        batch = np.array([state])
        m_s_a = self.model_sr.predict(batch)
        a = np.argmax(m_s_a)

        return a

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        if len(self.memory) < self.batch_size: 
            return

        sample_indicies = np.random.choice(len(self.memory), self.batch_size)
        samples = [self.memory[idx] for idx in sample_indicies]

        for sample in samples:
            state, action, reward, new_state, done = sample
            batch_state = np.array([state])
            target = self.target_model_sr.predict(batch_state)

            if done:
                target[0][action] = reward
            else:
                batch_new_state = np.array([new_state])
                Q_future = max(self.target_model_sr.predict(batch_new_state)[0])
                target[0][action] = reward + Q_future * self.gamma

            self.model_sr.fit(batch_state, target, epochs=1, verbose=0)
            self.model_reward_deconv.fit(
                x=batch_state, 
                y={
                    'model_r_regression': np.array([reward]),
                    'model_decoder': batch_state
                },
                epochs=1,
                verbose=0
            )

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

def resize_img(img: np.array) -> np.array:
    # dsize
    dsize = (64, 64)

    # resize image
    output = cv2.resize(img, dsize, interpolation = cv2.INTER_AREA)
    return output


def main():
    env = gym.make('MiniGrid-Empty-8x8-v0')
    env = RGBImgPartialObsWrapper(env) # Get pixel observations
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    gamma = 0.99
    num_actions = env.action_space.n
    img_shape = (64, 64, 3)
    batch_size = 32
    lr = 1e-2
    trials  = 1000
    trial_len = 500

    dqn_agent = DQN(
        action_space=num_actions,
        observation_shape=img_shape,
        gamma=gamma,
        batch_size=batch_size, 
        lr=lr
    )
    
    steps = []
    for trial in range(trials):
        print(f'Trail {trial}/{trials}')
        cur_state = env.reset()
        cur_state = resize_img(cur_state)
        for step in range(trial_len):
            print(f'Step {step}/{trial_len}', end="\r")
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)
            new_state = resize_img(new_state)

            dqn_agent.remember(cur_state, action, reward, new_state, done)
            
            dqn_agent.replay()       # internally iterates default (prediction) model
            dqn_agent.target_train() # iterates target model

            cur_state = new_state
            if done:
                break
        if step >= 199:
            print("Failed to complete in trial {}".format(trial))
            if step % 10 == 0:
                dqn_agent.save_model("trial-{}.model".format(trial))
        else:
            print("Completed in {} trials".format(trial))
            dqn_agent.save_model("success.model")
            break



if __name__ == '__main__':
    main()
