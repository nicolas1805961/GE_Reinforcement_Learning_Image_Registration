import cv2
import torch

import numpy as np

from torch.utils.data import Dataset, DataLoader


from datasets.dataset import QNetDataset
from deepqnet.deepqnet import DQN

from qvalues.qtable import Action


def rotate_image(image, angle, image_center):
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    #result = cv2.warpAffine(image, rot_mat, image.shape[::-1], flags=cv2.INTER_LINEAR)
    result = cv2.warpAffine(image, rot_mat, image.shape[::-1])
    return result

def get_new_image(big_image_rotated, transform, center):
  big_image_before = rotate_image(big_image_rotated, transform[0], center)
  patch2_test = big_image_before[center[0] - 75 + transform[2]:center[0] + 75 + transform[2], center[1] - 75 + transform[1]:center[1] + 75 + transform[1]]
  return patch2_test


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
        T_ts = []

        with torch.no_grad():
            for x, y, q, first_image, big_image, center in generator:
                T_t = torch.zeros((3,), dtype=torch.int16)

                for iteration in range(iterations):
                    prediction = self.dqn.predict(x)

                    action = self.actions[np.argmax(prediction, axis=1)].item()
                    T_t = action.apply(T_t)

                    inv_idx = torch.arange(1, -1, -1).long()
                    translation = torch.index_select(T_t[1:], 0, inv_idx).reshape(1, 2)
                    if torch.max(center + translation) + 75 == 511 or torch.min(center + translation) - 75 == 0:
                        break
                    current_image = get_new_image(big_image, T_t, (center[0, 0].item(), center[0, 1].item()))
                    diff = first_image - current_image
                    x = diff.view(1, 1, 150, 150)

    def load(self, filepath):
        self.dqn.load(filepath=filepath)

    def save(self, filepath=None, dirpath=None):
        self.dqn.save(filepath, dirpath)
