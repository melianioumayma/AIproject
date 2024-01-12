import os
import pickle
import logging
from rich import print
from rich.logging import RichHandler
import csv
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import sys
sys.path.append('cd C:\Users\melia\Desktop\TRAVAILM1NANTES\AI PROJET\source\source\classes')


# Hide Pygame welcome message
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "hide"
import pygame
import pandas as pd

from classes.logic import player2str
from classes.game import Game

FORMAT = "%(message)s"
logging.basicConfig(
    level="NOTSET",
    format=FORMAT,
    datefmt="[%X]",
    handlers=[RichHandler()]
)


class Tournament:
    def __init__(self, args: list):
        """
        Initialises a tournament with:
           * the size of the board,
           * the players strategies, eg., ("human", "random"),
           * the game counter,
           * the number of games to play.
        """
        self.args = args
        (self.BOARD_SIZE, self.STRAT, self.GAME_COUNT,
         self.N_GAMES, self.USE_UI) = args

        if self.USE_UI:
            pygame.init()
            pygame.display.set_caption("Polyline")

    def single_game(self, black_starts: bool=True) -> int:
        """
        Runs a single game between two opponents.

        @return   The number of the winner, either 1 or 2, for black
                  and white respectively.
        """

        game = Game(board_size=self.BOARD_SIZE,
                    black_starts=black_starts, 
                    strat=self.STRAT,
                    use_ui=self.USE_UI)
        game.print_game_info(
            [self.BOARD_SIZE, self.STRAT, self.GAME_COUNT]
         )
        while game.winner is None:
            game.play()

        print(f"{player2str[game.winner]} player ({self.STRAT[game.winner-1]}) wins!")

        return game.winner

    def championship(self):
        """
        Runs a number of games between the same two opponents.
        """
        scores = [0, 0]

        for _ in range(self.N_GAMES):
            self.GAME_COUNT = _

            # First half of the tournament started by one player.
            # Remaining half started by other player (see "no pie
            #  rule")
            winner = self.single_game(
                black_starts=self.GAME_COUNT < self.N_GAMES / 2
            )
            scores[winner-1] += 1

        log = logging.getLogger("rich")

        # TODO Design your own evaluation measure!
        # https://pyformat.info/
        # log.info("Design your own evaluation measure!")
        print(scores)
        print(self.STRAT[0], "|", self.STRAT[1], "| board :", self.BOARD_SIZE, "|", (scores [0] / (scores [0] + scores[1])) * 100, "% win")

        csv_filename = 'heatmap.csv'
        write_header = not os.path.exists(csv_filename) or os.stat(csv_filename).st_size == 0

        with open(csv_filename, 'a', newline='') as file:
            writer = csv.writer(file)
            
            if write_header:
                writer.writerow(['IA1', 'IA2', 'Board Size', '%win'])

            
            writer.writerow([self.STRAT[0], self.STRAT[1], self.BOARD_SIZE, (scores [0] / (scores [0] + scores[1])) * 100])

        self.heatmap()
        
        
    def heatmap(self):
        df = pd.read_csv("heatmap.csv")
        matrix = df.pivot_table(values="%win",
                                index = "IA1",
                                columns = "IA2")
        
        sns.heatmap(matrix)
        plt.show()
       