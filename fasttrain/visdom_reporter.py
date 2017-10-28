import numpy as np
import visdom


class VisdomReporter:
    def __init__(self):
        self.__visdom = visdom.Visdom()
        self.__win = None
        self.__epoch = 0

    def __call__(self, acc):
        if not self.__win:
            self.__win = self.__visdom.line(np.array([0]), np.array([acc]))
        else:
            self.__epoch += 1
            self.__visdom.line(np.array([self.__epoch]), np.array([acc]), win=self.__win, update='append')
