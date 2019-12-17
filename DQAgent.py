import paho.mqtt.client as mqtt
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from collections import deque
import numpy as np
import random

MEMORY_LEN = 100
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
MEMORY_LEN = 1000
DISCOUNT_RATE = 0.95
BATCH_SIZE = 50
EPSILON_REDUCE = True

#The agent which interacts with the mazes
class DQAgent:
    #Constructor
    def __init__(self, env, embeddings ='./src.txt', num_states=100, dimension=10):
        self.action_size = 4
        self.state_size = num_states
        self.dimension = dimension
        self.embeddings = embeddings
        
        self.gamma = DISCOUNT_RATE  # discount rate
        self.num_actions = 4
        if EPSILON_REDUCE:
            self.epsilon = EPSILON  # exploration rate
            self.epsilon_min = EPSILON_MIN
            self.epsilon_decay = EPSILON_DECAY
        else:
            self.epsilon = EPSILON_MIN


        self.model = self._build_model()
        self.memory = deque(maxlen=MEMORY_LEN)
    
    
    #Build a ff sequential model
    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(64, input_shape=(self.dimension,), activation='tanh'))
        model.add(Dense(32, activation='tanh'))
        model.add(Dense(self.action_size))
        model.compile(loss='mse',
                      optimizer='adam')
        return model

    def remember(self, current_state, action, reward, next_state, game_over):
        self.memory.append((current_state, action, reward, next_state, game_over))

    
    def replay(self, batch_size):
        memory_size = len(self.memory)
        batch_size = min(memory_size, batch_size)
        minibatch = random.sample(self.memory, batch_size)
        inputs = np.zeros((batch_size, self.dimension))
        targets = np.zeros((batch_size, self.num_actions))
        i = 0
        
        #replace next_state in self.model.predict by the embedding vector from the file
        for state, action, next_state, reward, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * 1 #* np.amax(self.model.predict([(self.embeddings[str(next_state[0])])]))
            #print((self.embeddings[str(state[0])]).transpose().shape())

            arr = self.embeddings[str(state[0])]
            a2 = np.array([arr])
            #np.reshape(arr, (1,10,1))
            target_f = self.model.predict(a2)
            
            target_f[0][action] = target
            inputs[i] = state
            targets[i] = target_f
            i += 1

        # print("input = {}, target = {}".format(inputs[0],targets[0]))
        self.model.fit(inputs, targets, epochs=8,
                       batch_size=16, verbose=0)
        if EPSILON_REDUCE and (self.epsilon > self.epsilon_min):
            self.epsilon *= self.epsilon_decay
        return self.model.evaluate(inputs, targets, verbose=0)

    def predict(self, current_state):
        #print((self.embeddings[str(current_state)]).transpose().shape())
        a2 = np.array([self.embeddings[str(current_state[0])]])
        predict = (self.model.predict(a2))[0]

        #predict = (self.model.predict((self.embeddings[str(current_state[0])])))
        sort = np.argsort(predict)[-len(predict):]
        sort = np.flipud(sort)
        # action = None
        # for i in range(len(sort)):
        #     if sort[i] in valid_actions:
        #         action = sort[i]
        #         break
        # # print("***action = {}, valid_actions = {}".format(action, valid_actions))
        return sort[0]

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)
