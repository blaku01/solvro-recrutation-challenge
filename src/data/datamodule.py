import numpy as np
import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

from configs import TrainConfig


def augment_with_rotations(data, labels, rotation_angles):
    data = data.permute(0, 2, 1)

    augmented_data = [data]

    for angle in rotation_angles:
        rotation_matrix = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )

        rotated_data = torch.matmul(data, torch.from_numpy(rotation_matrix))
        augmented_data.append(rotated_data)
    concatenated_augmented_data = torch.cat(augmented_data)
    concatenated_augmented_data = concatenated_augmented_data.permute(0, 2, 1)
    return concatenated_augmented_data, np.concatenate(
        [labels for _ in range(len(augmented_data))]
    )


class AndiDataset(Dataset):
    def __init__(self, X, y, rotation_angles=None, transform=None):
        self.X = X.reshape(-1, X.shape[2], X.shape[1]).astype("double")
        self.y = y
        self.X = torch.tensor(self.X, dtype=torch.float64)

        if rotation_angles is not None:
            self.X, self.y = augment_with_rotations(self.X, self.y, rotation_angles)

        if transform is not None:
            self.X = transform(self.X)

        if self.y is not None:
            self.y = torch.tensor(self.y, dtype=torch.long)
            if len(self.y.shape) > 1:
                self.y = torch.argmax(self.y, dim=1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        if self.y is not None:
            return x, self.y[idx]
        return x, None


class AndiDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = TrainConfig().data_path,
        batch_size: int = 64,
        num_workers: int = 0,
        pin_memory: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

    def setup(self, stage=None):
        # Load and preprocess data here
        X_train = np.load(self.data_dir + "filtered_x_train.npy")
        y_train = np.load(self.data_dir + "filtered_y_train.npy")

        val_data = np.load(self.data_dir + "filtered_x_val.npy")
        val_labels = np.load(self.data_dir + "filtered_y_val.npy")

        test_data = np.load(self.data_dir + "X_test.npy")
        test_labels = None  # No y_test.npy available

        # Add Standard scaling transform to the train dataset
        train_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=self._get_mean(val_data), std=self._get_std(val_data)
                )
            ]
        )
        self.train_dataset = AndiDataset(
            X_train, y_train, transform=train_transform, rotation_angles=[7 * np.pi / 4]
        )
        self.val_dataset = AndiDataset(val_data, val_labels, transform=train_transform)
        self.test_dataset = AndiDataset(
            test_data, test_labels, transform=train_transform
        )

    def _get_mean(self, data):
        return np.mean(data)

    def _get_std(self, data):
        return np.std(data)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
        )


if __name__ == "__main__":
    _ = AndiDataModule()
