import torch

import numpy as np

from deepqnet.deepqnet import DQN
from qvalues.qtable import Action
from utils import get_new_image, rotate_image


class RegistrationAgent:
    actions = Action.values()

    def __init__(self, depth, height, width, device):
        self.dqn = DQN(depth, height, width, outputs=len(self.actions), device=device)

    def fit(self, generator, epochs):
        self.dqn.fit(generator, epochs=epochs)

        return self

    """
    def _register(self, full_floating_image, floating_image, reference_image, n_iterations=64):
        floating_image = np.copy(floating_image)
        reference_image = np.copy(reference_image)
        working_image = floating_image

        T_t = np.zeros((2), dtype=np.int64)

        for iteration in range(n_iterations):
            new_top_left_corner = (TOP_LEFT_CORNER[0] + T_t[0], TOP_LEFT_CORNER[1] + T_t[1])
            working_image = full_floating_image[new_top_left_corner[0]:new_top_left_corner[0] + PATCH_SIZE,
                            new_top_left_corner[1]:new_top_left_corner[1] + PATCH_SIZE]
    """

    def register(self, generator, iterations=100):
        assert(generator.batch_size == 1)

        samples = len(generator.dataset)
        T_ts = np.zeros((samples, 3), dtype=np.int32)

        with torch.no_grad():

            for i, (reference_image, floating_image, full_image, center) in enumerate(generator):
                T_t = torch.zeros((3,), dtype=torch.int32)
                current_image = floating_image

                for iteration in range(iterations):
                    diff_image = reference_image - current_image
                    prediction = self.dqn.predict(diff_image)
                    action = self.actions[np.argmax(prediction, axis=1)].item()
                    T_t = action.apply(T_t)

                    inv_idx = torch.arange(1, -1, -1).long()
                    translation = torch.index_select(T_t[1:], 0, inv_idx).reshape(1, 2)
                    if torch.max(center + translation) + 75 == 511 or torch.min(center + translation) - 75 == 0:
                        break

                    print(center)
                    current_image = get_new_image(full_image, T_t, (center[0, 0].item(), center[0, 1].item()))

                T_ts[i] = T_t

        return T_ts

    def load(self, filepath):
        self.dqn.load(filepath=filepath)

    def save(self, filepath=None, dirpath=None):
        self.dqn.save(filepath, dirpath)
