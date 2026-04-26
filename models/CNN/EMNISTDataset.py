import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class EMNISTDataset(Dataset):
    def __init__(
            self,
            csv_file: str,
            split: str = "train",
            seed: int = 42,             #seed for RNG
            train_ratio: float = 0.9
    ):
        
        #load CSV
        data = pd.read_csv(csv_file, header=None)

        #split dataset
        n = len(data)
        idx = np.arange(n)

        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

        train_end = int(train_ratio * n)
        train_idx = idx[:train_end]
        val_idx = idx[train_end:]

        if split =="train":
            split_idx = train_idx
        elif split in ("val", "validate", "validation"):
            split_idx = val_idx
        else:
            split_idx = np.arange(n)

        #process pixel data
        numeric_all = data.iloc[:, 1:].to_numpy(dtype=np.float32)

        #reshape to 28x28
        numeric_all = numeric_all.reshape(-1, 28, 28)   

        #Reorient data
        numeric_all = np.transpose(numeric_all, (0,2,1))

        #standardize to scale pixel vals between 0 and 1
        numeric_all = numeric_all / 255.0

        #extract labels
        y_all = data.iloc[:,0].to_numpy(dtype=np.int64) #64 is long

        #assign data to class vars
        self.features = numeric_all[split_idx]
        self.labels = y_all[split_idx]

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        #add channel for CNN (28x28) -> (1x28x28)
        features = torch.from_numpy(self.features[idx]).unsqueeze(0)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return features, label

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Loading datasets")

    train_dataset = EMNISTDataset('emnist-balanced-train.csv', split='train')
    val_dataset = EMNISTDataset('emnist-balanced-train.csv', split='validate')
    test_dataset = EMNISTDataset('emnist-balanced-test.csv', split='test')

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

