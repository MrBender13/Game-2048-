import numpy as np
from random import randint


class AI_2048(object):
    when_born = -1
    games_played = 0
    average_score = 0
    progress = 0
    crt_moves = 0

    def __init__(self, training=False, Son=False, w16x16=None, w16x4=None, name='', file_name=''):
        super(AI_2048, self).__init__()
        self.name = name
        self.file_name = file_name

        if training:
            self.weights16x16 = 2 * np.random.random_sample((16, 16)) - 1
            self.weights16x4 = 2 * np.random.sample((16, 4)) - 1
        elif Son:
            self.weights16x16, self.weights16x4 = w16x16, w16x4
        else:
            self.read_csv()

    def change_name(self, name):
        self.name = name

    def predict(self, data):
        input_layer = np.array(data.get_number_field()).reshape((16))
        input_layer = np.ma.log2(input_layer)
        input_layer = input_layer.filled(0)
        input_layer = input_layer / np.max(input_layer)

        hiden_layer = input_layer.dot(self.weights16x16)
        hiden_layer = self.activation_function(hiden_layer)

        answer = hiden_layer.dot(self.weights16x4)
        answer = self.activation_function(answer)
        return answer

    def mutation(self, mut):
        w16x16 = self.weights16x16.copy()
        w16x4 = self.weights16x4.copy()
        for elem in np.nditer(w16x16, op_flags=['readwrite']):
            elem += mut[randint(0, np.size(mut)) - 1]

        for elem in np.nditer(w16x4, op_flags=['readwrite']):
            elem += mut[randint(0, np.size(mut)) - 1]

        return AI_2048(Son=True, w16x16=w16x16, w16x4=w16x4)

    def save(self):
        r = randint(0, 1000)
        np.savetxt("{}_{}weights16x16.csv".format(self.name, r), self.weights16x16, delimiter=",")
        np.savetxt("{}_{}weights16x4.csv".format(self.name, r), self.weights16x4, delimiter=",")

    @staticmethod
    def activation_function(x):
        return 1/(1 + np.exp(-x))

    def read_csv(self):
        self.weights16x16 = np.genfromtxt(self.file_name + '16x16.csv', delimiter=',')
        self.weights16x4 = np.genfromtxt(self.file_name + '16x4.csv', delimiter=',')

    def __repr__(self):
        return self.name
