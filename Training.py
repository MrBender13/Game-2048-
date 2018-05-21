from GameData import GameData
import numpy as np
from NeuralNetwork import AI_2048
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, generations=1, drawing=False):
        self.generations = generations
        self.Mutations = np.linspace(-1, 1, 17)
        self.drawing = drawing
        if drawing:
            self.X = [i for i in range(generations)]
            self.Y = []

    def start(self):
        bots = []
        bots.append((AI_2048(name='bot_-1', file_name='bot_-1_195weights'), GameData()))
        bots.append((AI_2048(name='bot_-1', file_name='bot_7_804weights'), GameData()))
        bots.append((AI_2048(name='bot_-1', file_name='bot_7_862weights'), GameData()))
        bots.append((AI_2048(name='bot_-1', file_name='bot_8_490weights'), GameData()))
        for _ in range(4):
            bots.append((AI_2048(training=True,  name='bot_-1'), GameData()))

        for generation in range(self.generations):
            print('Generation №{}'.format(generation))
            average = 0
            for bot, data in bots:
                for _ in range(30):
                    self.bot_play(bot, data)

                average += bot.average_score

            average = round(average / 10)
            print('      average = {}'.format(average))
            if self.drawing:
                self.Y.append(average)
            bots = self.next_generation(bots, generation)

        bots.sort(key=lambda x: x[0].average_score, reverse=True)
        for bot, data in bots[:4]:
            bot.save()

        if self.drawing:
            plt.plot(self.X, self.Y)
            plt.axis([0, self.generations, 0, 10000])
            plt.grid()
            plt.show()

    @staticmethod
    def bot_play(bot, data):
        game_over = False
        while not game_over:
            predict = bot.predict(data)
            moves = predict.tolist()
            moves = list(enumerate(moves))
            moves.sort(key=lambda x: x[1], reverse=True)

            for move in moves:
                check = data.move(move[0])
                if check:
                    break
            data.crt_progress = 0

            data.rand_cell()
            game_over = data.check_GameOver()

        bot.crt_moves = data.moves
        bot.games_played += 1
        bot.progress = data.progress
        bot.average_score = round(((bot.games_played - 1) * bot.average_score + bot.progress) / bot.games_played)
        data.refresh()

    def next_generation(self, bots, generation):
        bots.sort(key=lambda x: x[0].average_score, reverse=True)

        bots = bots[:4]
        for bot, data in bots:
            print('  {} av_score = {}'.format(bot, bot.average_score))
        for i in range(3):
            bot = bots[0][0].mutation(self.Mutations)
            bot.change_name('bot_{}'.format(generation))
            bot.when_born = generation
            data = GameData()
            bots.append((bot, data))

        for i in range(2):
            bot = bots[1][0].mutation(self.Mutations)
            bot.change_name('bot_{}'.format(generation))
            bot.when_born = generation
            data = GameData()
            bots.append((bot, data))

        bot = bots[2][0].mutation(self.Mutations)
        bot.change_name('bot_{}'.format(generation))
        bot.when_born = generation
        data = GameData()
        bots.append((bot, data))
        return bots

Trainer(generations=300, drawing=True).start()
