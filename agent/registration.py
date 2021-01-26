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

    def __init__(self, size, big_size, device):
        self.visualize_registration = False
        self.dqn = DQN(1, size, size, outputs=len(self.actions), device=device)
        self.size = size
        self.big_size = big_size

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

            accuracy += np.linalg.norm(T_t - transformation) <= np.linalg.norm(transformation) / 4
            rotation_distance += np.abs(transformation[0] - T_t[0])
            translation_distance += np.abs(transformation[1:] - T_t[1:]).sum()

        accuracy /= samples
        mean_rotation_distance = rotation_distance / samples
        mean_translation_distance = translation_distance / samples

        return accuracy, rotation_distance, translation_distance, mean_rotation_distance, mean_translation_distance

    def fit(self, training_generator, validation_generator=None, epochs=1):
        self.dqn.fit(training_generator, validation_generator=validation_generator, epochs=epochs)

        return self

    def register(self, generator, iterations=200):
        assert(generator.batch_size == 1)

        samples = len(generator.dataset)
        T_ts = np.zeros((samples, 3), dtype=np.int32)

        with torch.no_grad():

            for i, (reference_image, floating_image, full_image, center) in enumerate(generator):
                T_t = torch.zeros((iterations, 3), dtype=torch.int32)
                q_values = torch.zeros((iterations, 6), dtype=torch.float32)

                full_image = torch.squeeze(full_image).numpy()
                current_image = floating_image

                for iteration in range(iterations - 1):
                    diff_image = reference_image - current_image

                    prediction = self.dqn.predict(diff_image)
                    action = self.actions[torch.argmax(prediction, dim=1)]

                    T_t[iteration + 1] = action.apply(T_t[iteration])
                    q_values[iteration + 1] = prediction

                    if torch.unique(T_t[:iteration + 2], dim=0).size(0) == torch.unique(T_t[:iteration + 1], dim=0).size(0):
                        if torch.sum(q_values[iteration + 1]) > torch.sum(q_values[iteration]):
                            T_t[iteration + 1, :] = torch.zeros((1, 3), dtype=torch.int16)
                        break

                    current_image = get_new_image(full_image, T_t[iteration + 1], (center[0, 0].item(), center[0, 1].item()), self.size, self.big_size)
                    
                    if current_image is None:
                        break
                    
                mask = T_t.eq(torch.zeros((iterations, 3))).all(dim=1)
                T_t = T_t[~mask]
                T_ts[i] = T_t[-1]

                if self.visualize_registration:
                    visualization_reference_image = torch.squeeze(reference_image).numpy()
                    visualization_floating_images = [get_new_image(full_image, transformation, (center[0, 0].item(), center[0, 1].item()), self.size, self.big_size) for transformation in T_t]

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
