import sys
import reversi
from tqdm import tqdm

player1 = reversi.player.RandomPlayer('ランダム1')
player2 = reversi.player.RandomPlayer('ランダム2')



game = reversi.Reversi(player1,player2)
game.main_loop(print_game=True)

