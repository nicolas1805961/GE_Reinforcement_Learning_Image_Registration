import os
import torch

import numpy as np
import pandas as pd
import tensorflow as tf

from enum import auto, Enum


def D(T_g, T):
    composed = T_g - T
    return torch.linalg.norm(composed.float(), ord=2)


class Action(Enum):
    ROTATE_CLOCKWISE     = (auto(), np.array([ 1, 0, 0]))
    RIGHT                = (auto(), np.array([ 0, 1, 0]))
    DOWN                 = (auto(), np.array([ 0, 0, 1]))
    ROTATE_ANTICLOCKWISE = (auto(), np.array([-1, 0, 0]))
    LEFT                 = (auto(), np.array([ 0,-1, 0]))
    UP                   = (auto(), np.array([ 0, 0,-1]))

    def __init__(self, value, transform, rotation=10, translation=1):
        self.transform = torch.from_numpy(transform)
        self.transform[0] *= rotation
        self.transform[1] *= translation
        self.transform[2] *= translation

    def __call__(self, T):
        return self.apply(T)

    def apply(self, T):
        return self.transform + T

    @classmethod
    def values(cls):
        return np.array(Action, dtype=cls)


class Q:

    def __init__(self, actions, T_g, gamma=0.9, epsilon=0.5, R=9):
        self.actions = actions
        self.T_g = T_g
        self.gamma = gamma
        self.epsilon = epsilon
        self.R = R

    def __call__(self, T_t, a_t):
        T_t_next = a_t.apply(T_t)
        reward = self.reward(T_t, T_t_next)

        if D(self.T_g, T_t_next) < self.epsilon:
            return reward + self.R

        a_next_star = self.optimal_action(T_t_next)
        return reward + self.gamma * self.__call__(T_t_next, a_next_star)

    def optimal_action(self, T_t):
        distances = [D(self.T_g, action.apply(T_t)) for action in self.actions]
        return self.actions[np.argmin(distances)]

    def reward(self, T_t, T_t_next):
        return D(self.T_g, T_t) - D(self.T_g, T_t_next)

    def values(self, T=None):
        if T is None:
            T = torch.zeros_like(self.T_g)

        values = torch.zeros(len(self.actions))
        for i, a in enumerate(self.actions):
            values[i] = self.__call__(T, a)

        return values


class Q_table:

    def __init__(self, actions, transformations=None):
        self.actions = actions
        self.actions_names = [action.name for action in actions]

        self.shape = (0, len(self.actions))
        self.table = torch.empty(self.shape)
        self.dataframe = pd.DataFrame(self.table, columns=self.actions_names)

        if transformations is not None:
            self.fit(transformations)

    def __getitem__(self, index):
        return self.table[index]

    def __str__(self):
        return self.table.__str__()

    def fit(self, transformations, verbose=1):
        progbar = tf.keras.utils.Progbar(len(transformations), verbose=verbose, interval=0.05, unit_name='transform')
        if verbose > 0:
            print('Q Table')

        if type(transformations) == np.ndarray:
            transformations = torch.from_numpy(transformations)

        self.shape = (len(transformations), len(self.actions))
        self.table = torch.zeros(self.shape)

        for i, transformation in enumerate(transformations):
            self.table[i] = Q(self.actions, transformation).values()
            progbar.update(i + 1)

        self.dataframe = pd.DataFrame(self.table, columns=self.dataframe.columns)

        return self

    def save(self, filepath='qvalues.pt', dirpath=None):
        if dirpath is not None:
            if not os.path.exists(dirpath):
                os.makedirs(dirpath)

            if not os.path.isdir(dirpath):
                raise NotADirectoryError

            filepath = os.path.join(dirpath, filepath)
        torch.save(self.table, filepath)
