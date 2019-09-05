from reversi.board import ReversiBoard
from reversi.player import Player

class Reversi(object):

    """Game Master class of Reversi.

    Args:
        black(Player): Black player.
        white(Player): White player.
    """

    def __init__(self, black, white):
        if not (isinstance(black, Player) and
                isinstance(white, Player)):
            msg = 'Player must be a subclass of ``Player`` class'
            raise TypeError(msg)

        self.black = black #Playerクラスが格納
        self.white = white #Playerクラスが格納
        self.black.set_color(ReversiBoard.BLACK)
        self.white.set_color(ReversiBoard.WHITE)

        self.color2player = {
            ReversiBoard.BLACK: self.black,
            ReversiBoard.WHITE: self.white
        }

        self.board = ReversiBoard()

    @staticmethod
    def _format_board(board,
                      char_black='●', char_white='○', char_blank='-'):
        black = ReversiBoard.BLACK
        white = ReversiBoard.WHITE
        blank = ReversiBoard.BLANK

        _board = board.translate(
            str.maketrans(
                '{}{}{}'.format(blank, black, white),
                '{}{}{}'.format(char_blank, char_black, char_white)))
        return _board

    def main_loop(self, print_game=False):

        while True:
            state = self.board.get_status()
            board, last_hand, next_color, winner = state.split(',') 
            
            if winner != ReversiBoard.BLANK:
                break

            next_player = self.color2player[next_color]
            possible_hand = self.board.get_possible_hand(next_color)

            # action
            action = next_player.action(state, possible_hand)
            self.board.do_action(action)

            if print_game:
                print(self._format_board(board))
                print('player:{}, action:{}\n'.format(*action.split('_')))

        # finalize
        self.black.finalize(state, winner)
        self.white.finalize(state, winner)

        if print_game:
            n_black = board.count(ReversiBoard.BLACK)
            n_white = board.count(ReversiBoard.WHITE)

            print(self._format_board(board))
            print('black:{}, white:{}'.format(n_black, n_white))
            print('winner:{}'.format(winner))
