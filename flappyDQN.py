import random 
import numpy as np
import flappy_bird_gym
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import load_model, save_model, Sequential, Model
from tensorflow.keras.optimizers import RMSprop


#Neural Network For Agent
def NeuralNetwork(input_shape, output_shape):

    inp = Input (shape=input_shape)
    out = Dense (1024, activation = 'relu', kernel_initializer = 'he_uniform')(inp)
    out = Dense (512, activation = 'relu', kernel_initializer = 'he_uniform')(out)
    out = Dense (256, activation = 'relu', kernel_initializer = 'he_uniform')(out)
    out = Dense (128, activation = 'relu', kernel_initializer = 'he_uniform')(out)
    out = Dense (64, activation = 'relu', kernel_initializer = 'he_uniform')(out)
    out = Dense (32, activation = 'relu', kernel_initializer = 'he_uniform')(out)

    model = Model(inputs = inp, outputs = out)
    model.compile (loss = 'mse', optimizer = RMSprop (lr = 0.0001, rho = 0.95, epsilon = 0.01), metrics = ["accuracy"])
    model.summary()
    return model


#Brain of agent || Blueprint of Agent
class DQNAgent:
    def __init__(self):

        #Environment Variables
        self.env = flappy_bird_gym.make("FlappyBird-v0")
        self.episodes = 1000
        self.state_space = self.env.observation_space.shape[0]
        self.action_space = self.env.action_space.n
        self.memory = deque(maxlen = 2000)
        
        #Hyperparams
        self.gamma = 0.97
        self.epsilon = 1
        self.epsilon_decay = 0.99
        self.epsilon_min = 0.01
        self.batch_number = 64
        
        
        self.train_start = 1000
        self.jump_prob = 0.01  
        self.model = NeuralNetwork(input_shape=(self.state_space,), output_shape= self.action_space)

    def act(self, state):
        if np.random.random() > self.epsilon:
            return np.argmax(self.model.predict(state))

        return 1 if np.random.random() < self.jump_prob else 0 

    def learn(self):
        # make sure we have enough data
        if len(self.memory) < self.train_start:
            return


        # create minibatch        
        minibatch = random.sample(self.memory, min(len(self.memory), self.batch_number))


        # varriables to store minibatch info
        state = np.zeros((self.batch_number, self.state_space))
        next_state = np.zeros((self.batch_number, self.state_space))

        action, reward, done = [], [], []


        # Store data in variables
        for  i in range(self.batch_number):
            state[i] = minibatch[i][0]
            action.append(minibatch[i][1])
            reward.append(minibatch[i][2])
            next_state[i] = minibatch[i][3]
            done.append(minibatch[i][4])

        
        # predict y labels
        target = self.model.predict(state)
        target_next = self.model.predict(next_state)


        for i in range(self.batch_number):
            if done[i]: 
                target[i][action[i]] = reward[i]

            else:
                target[i][action[i]] = reward[i] + self.gamma * (np.argmax(target_next[i]))
        
        self.model.fit(state, target, batch_size = self.batch_number, verbose = 0)
        



    def train(self):
        for i in range (self.episodes):
            state =  self.env.reset()
            print(state)
            state = np.reshape(state, [1, self.state_space])
            done = False
            score = 0
            self.epsilon = self.epsilon * self.epsilon_decay if self.epsilon * self.epsilon_decay > self.epsilon_min else self.min_epsilon


            while not done:
                self.env.render()
                action = self.act(state)
                next_state, reward, done, info = self.env.step(action)


                #reshape next state
                next_state = np.reshape(next_state, [1, self.state_space])
                score += 1
                
                if done:
                    reward -= 100


                self.memory.append((state, action, reward, next_state, done))
                state = next_state


                if done:
                    print(f'Episode: {i}\nScore: {score}\nEpsilon: {self.epsilon}')
                    #save model
                    if score >= 1000:
                        self.model.save('flappybrain.h5')
                        return

                self.learn()







if __name__ == '__main__':
    agent = DQNAgent()
    agent.train()