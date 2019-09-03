import sys
import reversi
from tqdm import tqdm
import chainer
import chainer.functions as F
import chainer.links as L
import numpy as np
import copy
import matplotlib.pyplot as plt

class QFunction(chainer.Chain):

    def __init__(self):
        super(QFunction, self).__init__()
        with self.init_scope():
            self.l0 = L.Linear(64, 128)
            self.l1 = L.Linear(128, 128)
            self.l2 = L.Linear(128, 65)

    def __call__(self, s):
        h = F.elu(self.l0(s))
        h = F.elu(self.l1(h))
        qs = F.softmax(self.l2(h))
        return qs


def train_q_function(q_function, memory, optimizer,
                     batch_size=32, gamma=0.9, n_epoch=1):
    q_function_copy = copy.deepcopy(q_function)
    sum_loss = 0
    for e in range(n_epoch):
        perm = np.random.permutation(len(memory))
        for i in range(0, len(memory), batch_size):
            s, a, s_dash, r, e = memory.read(perm[i:i+batch_size])
            x = s.astype(np.float32)
            x_dash = s_dash.astype(np.float32)

            q_s = q_function(x)
            q_s_dash = q_function_copy(x_dash).data

            max_q_s_a_dash = np.max(q_s_dash, axis=1)
            max_q_s_a_dash[e == 1] == 0

            t = q_s.data.copy()
            t[np.arange(len(t)), a] += r + gamma * \
                max_q_s_a_dash - t[np.arange(len(t)), a]

            loss = F.mean_absolute_error(q_s, t)

            q_function.cleargrads() #パラメータの勾配を初期化
            loss.backward() #損失関数の値を元に、パラメータの更新量を計算
            optimizer.update() #パラメータを更新

            sum_loss += loss.data
    return sum_loss

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
ME = reversi.player.NNQPlayer('Q太郎', q_function, memory, eps=0.05)

optimizer = chainer.optimizers.Adam()
optimizer.setup(q_function)

for i in tqdm(range(100)):

    if np.random.random() > 0.5:
        B = CPU
        W = ME
    else:
        B = ME
        W = CPU

    game = reversi.Reversi(B,W)
    game.main_loop(print_game=False)
    loss = train_q_function(q_function, memory, optimizer)

game = reversi.Reversi(CPU,ME)
game.main_loop(print_game=True)

# 勝率の変化

wininig_Q = np.array(ME.record) == 1

plt.grid(True)
plt.ylim(0, 1)
plt.plot(np.cumsum(wininig_Q) / (np.arange(len(wininig_Q)) + 1))
plt.savefig("winning plot")


