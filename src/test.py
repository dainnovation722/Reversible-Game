import reversi

p1 = reversi.player.RandomPlayer("1")
p2 = reversi.player.RandomPlayer("2")

game = reversi.Reversi(p1,p2)
game.main_loop(print_game=True)
