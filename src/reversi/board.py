class BitBoard(object):

    DIRECTIONS = (
        'up_left', 'right', 'up_right', 'left',
        'up', 'down_left', 'down', 'down_right'
    )

    DIRECTION_TO_SHIFT_WIDTH = {
        'up_left': -9,
        'up': -8,
        'up_right': -7,
        'left': -1,
        'right': 1,
        'down_left': 7,
        'down': 8,
        'down_right': 9
    }

    DIRECTION_TO_MASK_REVERSE = {
        'up_left':    0xfefefefefefefe00,
        'right':      0x7f7f7f7f7f7f7f7f,
        'up_right':   0x7f7f7f7f7f7f7f00,
        'left':       0xfefefefefefefefe,
        'up':         0xffffffffffffff00,
        'down_left':  0x00fefefefefefefe,
        'down':       0x00ffffffffffffff,
        'down_right': 0x007f7f7f7f7f7f7f
    }

    DIRECTION_TO_MASK_possible_hand = {
        'up_left':    0x007e7e7e7e7e7e00,
        'right':      0x7e7e7e7e7e7e7e7e,
        'up_right':   0x007e7e7e7e7e7e00,
        'left':       0x7e7e7e7e7e7e7e7e,
        'up':         0x00ffffffffffff00,
        'down_left':  0x007e7e7e7e7e7e00,
        'down':       0x00ffffffffffff00,
        'down_right': 0x007e7e7e7e7e7e00
    }

    def __init__(self, black='b', white='w'):
        self.BLACK = black
        self.WHITE = white
        self.init_board()

    def init_board(self):
        self.board = {
            self.BLACK: 0x0000000810000000,
            self.WHITE: 0x0000001008000000
        }

    def _opponent(self, player):
        if player == self.BLACK:
            return self.WHITE
        else:
            return self.BLACK

    def _move_searcher(self, searcher, shift_width, mask):
        if shift_width >= 0:
            searcher = (searcher >> shift_width) & mask
        else:
            searcher = (searcher << abs(shift_width)) & mask
        return searcher

    def _search_reversible_one_direction(self, b_player, b_opponent,
                                         put, shift_width, mask):
        searcher = self._move_searcher(put, shift_width, mask)
        pos_reverse = 0

        while (searcher != 0 and (searcher & b_opponent) != 0):
            pos_reverse = pos_reverse | searcher
            searcher = self._move_searcher(searcher, shift_width, mask)

        if (searcher & b_player) == 0:
            pos_reverse = 0

        return pos_reverse

    def do_reverse(self, player, put):
        opponent = self._opponent(player)

        b_player = self.board[player]
        b_opponent = self.board[opponent]

        pos_reverse = 0
        for direction in self.DIRECTIONS:
            shift_width = self.DIRECTION_TO_SHIFT_WIDTH[direction]
            mask = self.DIRECTION_TO_MASK_REVERSE[direction]

            _pos_reverse = self._search_reversible_one_direction(
                b_player, b_opponent, put, shift_width, mask)
            pos_reverse |= _pos_reverse

        # forbidden hand.
        if pos_reverse == 0:
            return True

        self.board[player] ^= put
        self.board[player] ^= pos_reverse
        self.board[opponent] ^= pos_reverse
        return False

    def _search_possible_hand_one_direction(self, b_player, b_opponent,
                                            blank, shift_width, mask):
        masked_b_opponent = b_opponent & mask
        searcher = self._move_searcher(
            b_player, shift_width, masked_b_opponent)

        for _ in range(5):
            searcher |= self._move_searcher(
                searcher, shift_width, masked_b_opponent)

        if shift_width > 0:
            return blank & (searcher >> shift_width)
        else:
            return blank & (searcher << abs(shift_width))

    def search_possible_hand(self, player):
        opponent = self._opponent(player)

        b_player = self.board[player]
        b_opponent = self.board[opponent]

        filler = 0xffffffffffffffff
        blank = filler ^ (b_player | b_opponent)

        pos_possible_hand = 0
        for direction in self.DIRECTIONS:
            pos_possible_hand |= self._search_possible_hand_one_direction(
                b_player, b_opponent, blank,
                self.DIRECTION_TO_SHIFT_WIDTH[direction],
                self.DIRECTION_TO_MASK_possible_hand[direction])

        return pos_possible_hand

    def count(self, player):
        n_player = bin(self.board[player]).count('1')
        return n_player

    def get_board(self, player):
        return self.board[player]


class ReversiBoard(object):

    BLACK = 'b'
    WHITE = 'w'
    BLANK = '-'

    DRAW = 'd'

    def __init__(self):
        self.bit_board = BitBoard(black=self.BLACK, white=self.WHITE)
        self.possible_hand = {}
        self.init_game()

    def init_game(self):
        self.bit_board.init_board()
        self._update_possible_hand()
        self.turn = self.BLACK
        self.last_hand = '{}_0'.format(self.BLANK)
        self.winner = self.BLANK
        self.game_is_over = False

    def _bit_board_to_str_board(self, black, white, blank):
        bit_black = self.bit_board.get_board(self.BLACK)
        bit_white = self.bit_board.get_board(self.WHITE)

        int_bit_black = int(bin(bit_black)[2:])
        int_bit_white = int(bin(bit_white)[2:])
        int_bit_board = int_bit_black + 2 * int_bit_white

        str_board = str(int_bit_board).zfill(64)
        str_board = str_board.translate(
            str.maketrans('012', '{}{}{}'.format(blank, black, white)))
        str_board = '\n'.join([str_board[i * 8: (i + 1) * 8]
                               for i in range(8)])
        return str_board

    @staticmethod
    def _bit_possible_hand_to_int_possible_hand(bit_possible_hand):
        bit_possible_hand = bin(bit_possible_hand)[2:].zfill(64)

        int_possible_hand = []
        for i, b in enumerate(bit_possible_hand):
            if b == '1':
                int_possible_hand.append(i + 1)

        if not int_possible_hand:
            int_possible_hand.append(0)
        return int_possible_hand

    def _search_possible_hand(self, player):
        bit_possible_hand = self.bit_board.search_possible_hand(player)
        possible_hand = self._bit_possible_hand_to_int_possible_hand(
            bit_possible_hand)
        return possible_hand

    def _update_possible_hand(self):
        self.possible_hand[self.BLACK] = self._search_possible_hand(self.BLACK)
        self.possible_hand[self.WHITE] = self._search_possible_hand(self.WHITE)

    def _opponent(self, player):
        if player == self.BLACK:
            return self.WHITE
        else:
            return self.BLACK

    def _judge(self):
        self.game_is_over = (self.possible_hand[self.BLACK][0] == 0 and
                             self.possible_hand[self.WHITE][0] == 0)

        if self.game_is_over:
            n_black = self.bit_board.count(self.BLACK)
            n_white = self.bit_board.count(self.WHITE)

            if n_black == n_white:
                self.winner = self.DRAW
            elif n_black > n_white:
                self.winner = self.BLACK
            else:
                self.winner = self.WHITE

    def _turn_player(self):
        self.turn = self._opponent(self.turn)

    def get_status(self):
        """Gets game status.

        Returns:
            str: A string representing the game status.
                The status is of the form:

                    '--------
                     --------
                     --------
                     ---wb---
                     ---bw---
                     --------
                     --------
                     --------,-_0,b,-'

                    - 'w': white stone.
                    - 'b': black stone.
                    - '-': blank space.

                The above example means that the board is in the initial state,
                the previous player's hand is None('-_0'), the next player is
                black('b'), and the winner has not been decided('-').
        """
        board = self._bit_board_to_str_board(
            black=self.BLACK, white=self.WHITE, blank=self.BLANK)
        last_hand = self.last_hand
        turn = self.turn
        winner = self.winner
        return ','.join([board, last_hand, turn, winner])

    def get_possible_hand(self, player):
        """Get a place to put the stone.

        Args:
            player: player must be ``self.BLACK`` or ``self.WHITE``.

        Returns:
            str: Places where ``player`` can put the stone (comma separated).
        """
        if player not in (self.BLACK, self.WHITE):
            msg = "``player`` must be '{}' or '{}'".format(
                self.BLACK, self.WHITE)
            raise ValueError(msg)

        return ','.join(list(map(str, self.possible_hand[player])))

    def do_action(self, action):
        """Puts the stone and reveres stones.

        Args:
            action (str): A string representing the next action.
                Each place on the board is represented as follows:

                    [[ 1  2  3  4  5  6  7  8]
                     [ 9 10 11 12 13 14 15 16]
                     [17 18 19 20 21 22 23 24]
                     [25 26 27 28 29 30 31 32]
                     [33 34 35 36 37 38 39 40]
                     [41 42 43 44 45 46 47 48]
                     [49 50 51 52 53 54 55 56]
                     [57 58 59 60 61 62 63 64]]

                * ``0`` means ``PASS``.

                The ``action`` is of the form:

                    'b_20'

                The above example means placing a black stone in the ``20``.
        """
        player, hand = action.split('_')
        hand = int(hand)

        if player not in (self.BLACK, self.WHITE):
            msg = "``player`` must be '{}' or '{}'".format(
                self.BLACK, self.WHITE)
            raise ValueError(msg)

        if hand < 0 or hand > 64:
            msg = '``hand`` must be in ``[0, ..., 64]``'
            raise ValueError(msg)

        if player != self.turn:
            return 'ERROR: Not turn player.'

        if hand not in self.possible_hand[player]:
            return 'ERROR: Forbidden hand.'

        if hand != 0:
            bit_put = 0x8000000000000000
            bit_put >>= (hand - 1)
            self.bit_board.do_reverse(player, bit_put)

            self._update_possible_hand()
            self._judge()

        self._turn_player()

        self.last_hand = action
        return 'DONE'
