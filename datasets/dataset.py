from torch.utils.data import Dataset, DataLoader


class QNetDataset(Dataset):

    def __init__(self, X, Y, Q, first_images, big_images, centers):
        self.X = X
        self.Y = Y
        self.Q = Q
        self.first_images = first_images
        self.big_images = big_images
        self.centers = centers

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        X = self.X[index]
        Y = self.Y[index]
        Q = self.Q[index]
        first_image = self.first_images[index]
        big_image = self.big_images[index]
        center = self.centers[index]

        return X, Y, Q, first_image, big_image, center


class QNetDataLoader(DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(args, kwargs)

