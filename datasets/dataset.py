import numpy as np

from torch.utils.data import Dataset, DataLoader


class DQNDataset(Dataset):

    def __init__(self, reference_images, floating_images):
        assert(len(reference_images) == len(floating_images))

        self.reference_images = np.expand_dims(reference_images, axis=1)
        self.floating_images = np.expand_dims(floating_images, axis=1)

    def __len__(self):
        return len(self.reference_images)

    def __getitem__(self, key):
        return self.reference_images[key], self.floating_images[key]


class TrainDQNDataset(DQNDataset):

    def __init__(self, reference_images, floating_images, qvalues):
        super().__init__(reference_images, floating_images)
        assert(len(reference_images) == len(qvalues))

        self.qvalues = qvalues

    def __getitem__(self, key):
        return super().__getitem__(key) + (self.qvalues[key],)


class RegisterDQNDataset(DQNDataset):

    def __init__(self, reference_images, floating_images, full_images, centers):
        super().__init__(reference_images, floating_images)
        assert(len(reference_images) == len(full_images))
        assert(len(reference_images) == len(centers))

        self.full_images = full_images
        self.centers = centers

    def __getitem__(self, key):
        return super().__getitem__(key) + (self.full_images[key], self.centers[key],)


class TestDQNDataset(RegisterDQNDataset):

    def __init__(self, reference_images, floating_images, full_images, centers, transformations):
        super().__init__(reference_images, floating_images, full_images, centers)
        assert(len(reference_images) == len(transformations))

        self.transformations = transformations

    def __getitem__(self, key):
        return super().__getitem__(key) + (self.transformations[key],)


class DQNDataLoader(DataLoader):

    def __init__(self, dataset: DQNDataset, *args, **kwargs):
        super().__init__(dataset=dataset, *args, **kwargs)

