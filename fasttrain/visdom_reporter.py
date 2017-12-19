"""
   Copyright 2017 JetBrains, s.r.o

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

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
