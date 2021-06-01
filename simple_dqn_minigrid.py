import gym
import numpy as np
import random
import gym
import gym_minigrid

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
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import ReLU
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import mean_squared_error

from collections import deque

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

        self.model        = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        inputs = Input(shape=self.observation_shape)

        # Convolutions on the frames on the screen
        layer1 = Conv2D(32, 8, strides=4, activation="relu")(inputs)
        layer2 = Conv2D(64, 4, strides=2, activation="relu")(layer1)
        layer3 = Conv2D(64, 3, strides=1, activation="relu")(layer2)

        layer4 = Flatten()(layer3)

        layer5 = Dense(512, activation="relu")(layer4)
        action = Dense(self.action_space, activation="linear")(layer5)
        model = Model(inputs=inputs, outputs=action)
        model.compile(
            optimizer=RMSprop(learning_rate=self.lr, momentum=self.momentum),
            loss=mean_squared_error
        )

        return model

    def act(self, state):
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon_min, self.epsilon)
        
        batch = np.array([state])

        if np.random.random() < self.epsilon:
            return np.random.choice(self.action_space)
        Qs = self.model.predict(batch)[0]
        print("Qs: ", Qs)
        action = np.argmax(Qs)

        return action

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        batch_size = 32
        if len(self.memory) < batch_size: 
            return

        samples = random.sample(self.memory, batch_size)
        for sample in samples:
            state, action, reward, new_state, done = sample

            batch = np.array([state])
            batch_new_state = np.array([new_state])

            target = self.target_model.predict(batch)
            if done:
                target[0][action] = reward
            else:
                Q_future = max(self.target_model.predict(batch_new_state)[0])
                target[0][action] = reward + Q_future * self.gamma
            self.model.fit(batch, target, epochs=1, verbose=0)

    def target_train(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = weights[i] * self.tau + target_weights[i] * (1 - self.tau)
        self.target_model.set_weights(target_weights)

    def save_model(self, fn):
        self.model.save(fn)

def main():
    env = gym.make("MiniGrid-Empty-8x8-v0")
    env = RGBImgPartialObsWrapper(env) # Get pixel observations
    env = ImgObsWrapper(env) # Get rid of the 'mission' field
    gamma = 0.99
    num_actions = 3
    img_shape = (56, 56, 3)
    batch_size = 32
    lr = 2.5e-4
    momentum = 0.95
    trials  = 1000
    trial_len = 255

    dqn_agent = DQN(
        action_space=num_actions,
        observation_shape=img_shape,
        gamma=gamma,
        batch_size=batch_size, 
        lr=lr,
        momentum=momentum
    )

    trials  = 1000
    trial_len = 255
    steps = []
    for trial in range(trials):
        print(f'Trail {trial}/{trials}', end="\r")
        cur_state = env.reset()
        for step in range(trial_len):
            action = dqn_agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)

            print(f'Step {step}/{trial_len}; Agent Pos: {env.agent_pos}; Action: {action}, Reward: {reward}')

            # reward = reward if not done else -20
            new_state = new_state
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
            #dqn_agent.save_model("success.model")
            break

if __name__ == "__main__":
    main()