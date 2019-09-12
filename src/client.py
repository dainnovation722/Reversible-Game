import time
import urllib.request
from reversi.board import ReversiBoard
from reversi.player import Player


class ReversiClient(object):
    """Game Client class of Reversi.

    Args:
        sender(Player): sender player.
        host(String): host url starting from `http`.
        port(String): port number.
    """
    def __init__(self, sender, host, port=None):
        if not(isinstance(sender, Player)):
            msg = 'Sender must be a subclass of ``Player`` class'
            raise TypeError(msg)
        
        self.base_url = host
        if port:
            self.base_url = self.base_url + ":" + port + "/"
        self.sender = sender
        self.sender.set_color(
            self.send_request("get_my_color", self.sender.name)
        )
    
    def main_loop(self):
        stop_cnt = 0
        while True:
            state = self.send_request("get_status")
            board, last_hand, next_color, winner = state.split(",")
            
            if winner != ReversiBoard.BLANK:
                break

            if next_color != self.sender.color:
                if stop_cnt >= 20:
                    break
                time.sleep(3)
                stop_cnt += 1
                continue

            possible_hand = self.send_request(
                "get_possible_hand", self.sender.color)
            action = self.sender.action(state, possible_hand)
            self.send_request("action", action)
            stop_cnt = 0
    
    def send_request(self, method=None, param=None):
        """Base method to connecting host.
        
        Args
            method(String): method name.
                - get_my_color: get sender player's color.
                - get_status: get status in host environment.
                - get_possible_hand: get possible hand for sender.
                - action: post action to host using `GET` method.
            param(String): parameter name.
                - name: sender's name.
                - color: sender's color.
                - action: sender's action.
        Returns
            result(String): returned string corresponding to method.
        """
        
        if method is None:
            msg = 'send_request() missing required argument: `method`'
            raise TypeError(msg)
        
        url = self.base_url + method
        if param:
            url = url + "/" + param
        
        request = urllib.request.Request(url)
        with urllib.request.urlopen(request) as response:
            result = response.read()
        
        return result.decode('utf-8')
