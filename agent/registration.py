import torch

import ipywidgets as widgets
import numpy as np

from IPython.display import display
from ipywidgets import interact, interactive, fixed


from datasets.dataset import DQNDataLoader, RegisterDQNDataset
from deepqnet.deepqnet import DQN
from qvalues.qtable import Action
from utils import get_new_image, rotate_image, visualize_registration


class RegistrationAgent:
    actions = Action.values()

    def __init__(self, depth, height, width, device):
        self.visualize_registration = False
        self.dqn = DQN(depth, height, width, outputs=len(self.actions), device=device)

    def evaluate(self, generator):
        assert(generator.batch_size == 1)
        self.visualize_registration = False

        samples = len(generator.dataset)

        accuracy = 0.0
        rotation_distance = 0.0
        translation_distance = 0.0

        for i, (reference_image, floating_image, full_image, center, transformation) in enumerate(generator):
            reference_image = torch.squeeze(reference_image, dim=1)
            floating_image = torch.squeeze(floating_image, dim=1)
            full_image = torch.squeeze(full_image, dim=1)
            center = torch.squeeze(center, dim=1)

            register_dataset = RegisterDQNDataset(reference_image, floating_image, full_image, center)
            register_generator = DQNDataLoader(register_dataset, batch_size=1, shuffle=False)
            T_t = self.register(register_generator)

            transformation = torch.squeeze(transformation).numpy()
            T_t = np.squeeze(T_t)

            accuracy += np.array_equal(transformation, T_t)
            rotation_distance += np.abs(transformation[0] - T_t[0])
            translation_distance += np.abs(transformation[1:] - T_t[1:]).sum()

        accuracy /= samples
        mean_rotation_distance = rotation_distance / samples
        mean_translation_distance = translation_distance / samples

        return accuracy, rotation_distance, translation_distance, mean_rotation_distance, mean_translation_distance

    def fit(self, training_generator, validation_generator=None, epochs=1):
        self.dqn.fit(training_generator, validation_generator=validation_generator, epochs=epochs)

        return self

    def register(self, generator, iterations=64):
        assert(generator.batch_size == 1)

        samples = len(generator.dataset)
        T_ts = np.zeros((samples, 3), dtype=np.int32)

        with torch.no_grad():

            for i, (reference_image, floating_image, full_image, center) in enumerate(generator):
                T_t = torch.zeros((3,), dtype=torch.int32)
                visualization_transformations = [T_t.numpy()]

                full_image = torch.squeeze(full_image).numpy()
                current_image = floating_image

                for iteration in range(iterations):
                    diff_image = reference_image - current_image

                    prediction = self.dqn.predict(diff_image)
                    action = self.actions[torch.argmax(prediction, dim=1)]

                    T_t = action.apply(T_t)
                    visualization_transformations.append(T_t)

                    inv_idx = torch.arange(1, -1, -1).long()
                    translation = torch.index_select(T_t[1:], 0, inv_idx).reshape(1, 2)
                    if torch.max(center + translation) + 75 == 511 or torch.min(center + translation) - 75 == 0:
                        break

                    current_image = get_new_image(full_image, T_t, (center[0, 0].item(), center[0, 1].item()))

                T_ts[i] = T_t

                if self.visualize_registration:
                    visualization_reference_image = torch.squeeze(reference_image).numpy()
                    visualization_floating_images = [get_new_image(full_image, transformation, (center[0, 0].item(), center[0, 1].item())) for transformation in visualization_transformations]

                    max_step = len(visualization_floating_images) - 1
                    step = widgets.IntSlider(value=0, min=0, max=max_step, step=1, description='Registration step')
                    interact(
                        visualize_registration,
                        reference_image=fixed(visualization_reference_image),
                        floating_images=fixed(visualization_floating_images),
                        step=step
                    )

        return T_ts

    def visualize(self, generator, **kwargs):
        self.visualize_registration = True
        self.register(generator, **kwargs)
        self.visualize_registration = False

    def load(self, filepath):
        self.dqn.load(filepath=filepath)

    def save(self, **kwargs):
        self.dqn.save(**kwargs)
