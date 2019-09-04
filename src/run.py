import sys
import reversi
from tqdm import tqdm
import numpy as np
import copy
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation

class QFunction():

    def __init__(self):
        self.model = Sequential([
            Dense(128, activation='linear', input_shape=(64,)),
            Dense(128, activation='linear'),
            Dense(65, activation='linear')
        ])
        self.model.compile(optimizer='adagrad', loss='mse')
        print(self.model.summary())

    
    def fit(self,q_s, t):
        self.model.fit(q_s, t ,verbose=0)

def train_q_function(q_function, memory, 
                     batch_size=32, gamma=0.9, n_epoch=1):
    q_function_copy = copy.deepcopy(q_function)
    for e in range(n_epoch):
        perm = np.random.permutation(len(memory)) #memoryのデータは時系列データなのでデータ間に相関が出ないようにrandom samplingする
        for i in range(0, len(memory), batch_size):
            s, a, s_dash, r, e = memory.read(perm[i:i+batch_size])
            
            x = s.astype(np.float32)
            x_dash = s_dash.astype(np.float32)
            
            q_s = q_function.model.predict(x)
            q_s_dash = q_function_copy.model.predict(x_dash)

            max_q_s_a_dash = np.max(q_s_dash, axis=1)
            max_q_s_a_dash[e == 1] == 0
            t = q_s.copy()
            t[np.arange(len(t)), a] += r + gamma * \
                max_q_s_a_dash - t[np.arange(len(t)), a]
            
            q_function.model.fit(x, t) #学習        
            
class Memory(object):

    def __init__(self, size=128):
        self.size = size
        self.memory = np.empty((size, 131), dtype=np.float32)
        self.counter = 0

    def __len__(self):
        return min(self.size, self.counter)

    def read(self, ind):
        s = self.memory[ind, :64].astype(np.int32)
        a = self.memory[ind, 64].astype(np.int32)
        s_dash = self.memory[ind, 65:129].astype(np.int32)
        r = self.memory[ind, 129]
        e = self.memory[ind, 130]
        return s, a, s_dash, r, e

    def write(self, ind, s, a, s_dash, r, e):
        self.memory[ind, :64] = s
        self.memory[ind, 64] = a
        self.memory[ind, 65:129] = s_dash
        self.memory[ind, 129] = r
        self.memory[ind, 130] = e

    def append(self, s, a, s_dash, r, e):
        ind = self.counter % self.size
        self.write(ind, s, a, s_dash, r, e)
        self.counter += 1

q_function = QFunction()
memory = Memory(size=128)

CPU = reversi.player.RandomPlayer('ランダム')
ME = reversi.player.NNQPlayer('Q太郎', q_function, memory)


for episode in tqdm(range(10)):

    if np.random.random() > 0.5:
        B = CPU
        W = ME
    else:
        B = ME
        W = CPU

    game = reversi.Reversi(B,W)
    game.main_loop(episode=episode, print_game=False)
    train_q_function(q_function, memory)

game = reversi.Reversi(CPU,ME)
game.main_loop(print_game=True)

# 勝率の変化

wininig_Q = np.array(ME.record) == 1

plt.grid(True)
plt.ylim(0, 1)
plt.plot(np.cumsum(wininig_Q) / (np.arange(len(wininig_Q)) + 1))
plt.savefig("winning_plot.png")


