import numpy as np
import visdom


class VisdomReporter:
    def __init__(self, model_name):
        self.__visdom = visdom.Visdom()
        self.__win = None
        self.__epoch = 0
        self.__model_name = model_name

    def __call__(self, acc):
        if not self.__win:
            self.__win = self.__visdom.line(np.array([acc]), np.array([0]), opts=dict(title='Accuracy of {}'.format(self.__model_name)))
        else:
            self.__epoch += 1
            self.__visdom.line(np.array([acc]), np.array([self.__epoch]), win=self.__win, update='append')
