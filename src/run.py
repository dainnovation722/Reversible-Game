import sys
import reversi
from tqdm import tqdm
import numpy as np
import copy
import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 15
plt.tight_layout()
from keras.models import Sequential
from keras.layers import Dense, Activation, BatchNormalization
from time import time 
import os 
t1 = time()
total_episode = 10000 #訓練回数

class QFunction():

    def __init__(self,summary=False):
        self.model = Sequential([
            Dense(128, activation='linear', input_shape=(64,)),
            BatchNormalization(),
            Dense(128, activation='linear'),
            Dense(128, activation='linear'),
            Dense(65, activation='linear')
        ])
        self.model2 = Sequential([
            Dense(128, activation='linear', input_shape=(64,)),
            BatchNormalization(),
            Dense(128, activation='linear'),
            Dense(128, activation='linear'),
            Dense(65, activation='linear')
        ])
        
        self.model.compile(optimizer='adagrad', loss='mse')
        self.model2.compile(optimizer='adagrad', loss='mse')
        
        if summary:
            print(self.model.summary())

    def same_weights(self):
        self.model2.set_weights(self.model.get_weights())
        
        
def train_q_function(q_function, memory, target_q_function,
                     batch_size=32, gamma=0.9, n_epoch=1):
        
    for e in range(n_epoch): #1つの試合の経験値(memory)から学び取る回数
        perm = np.random.permutation(len(memory)) #memoryのデータは時系列データなのでデータ間に相関が出ないようにrandom samplingする
        for i in range(0, len(memory), batch_size):
            s, a, s_dash, r, e = memory.read(perm[i:i+batch_size])
            
            x = s.astype(np.float32)
            x_dash = s_dash.astype(np.float32)
            
            q_s = q_function.model.predict(x) #現状態sのq値候補(行動決定)
            q_s_dash = q_function.model.predict(x_dash) #未来状態s_dashのq値候補(価値決定の)
            max_q_s_a_dash_index = np.argmax(q_s_dash, axis=1).reshape(-1) #未来状態s_dashの行動をインデックスで決定
            max_q_s_a_dash = target_q_function.predict(x_dash)
            max_q_s_a_dash = np.array([max_q_s_a_dash[i,j] for i,j in enumerate(max_q_s_a_dash_index)]) #価値を評価し(←ここが異なる関数で計算！！)、上記で得られたインデックスに合うq値を計算
            
            max_q_s_a_dash[e == 1] == 0 #試合終了の状態があれば次の状態は存在しないので次の状態で得られる最大報酬は0
            t = q_s.copy()
            t[np.arange(len(t)), a] += r + gamma * \
                max_q_s_a_dash 
            
            q_function.model.fit(x, t, verbose=0) #学習        
       
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

q_function = QFunction(summary=True)
memory = Memory(size=128)

CPU = reversi.player.RandomPlayer('ランダム')
ME = reversi.player.NNQPlayer('Q太郎', q_function, memory)


for episode in tqdm(range(total_episode)):
    q_function.same_weights() #行動決定q_functionと価値計算target_q_functionのQnetworkを同じにする 
    target_q_function=q_function.model2

    sep = total_episode*0.2
    if np.random.random() > 0.5:
        B = CPU
        W = ME
    else:
        B = ME
        W = CPU

    game = reversi.Reversi(B,W)
    game.main_loop(episode=episode, print_game=False)
    train_q_function(q_function, memory, target_q_function)
    # if episode%sep==0:
    #     print("WinCounts ME:{} Enemy:{} Draw:{}, rate:{:.3f}".format(\
    #         ME.record.count(1),CPU.record.count(1),CPU.record.count(0),\
    #         sum(ME.record)/len(ME.record)))

# game = reversi.Reversi(CPU,ME)
# game.main_loop(print_game=True)

# 勝率の変化

winning_Q = np.array(ME.record)==1

if not os.path.exists("results"):
    os.mkdir("results")

plt.grid(True)
plt.ylim(0, 1)
plt.xlabel("epochs")
plt.ylabel("win rate")
plt.plot(np.cumsum(winning_Q) / (np.arange(len(winning_Q)) + 1))
plt.savefig("results/winning_plot_BNlayer.png")
Time = time() - t1
print("Execution Time : {:.3f} minutes".format(Time/60))

