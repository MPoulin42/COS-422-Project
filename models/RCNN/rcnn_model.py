# Phl Lanes - RCNN Model for EMNIST Balanced Character Recognition
# -------------------------------------------------------
# Features:
#   - Patch-based image splitting (image divided into horizontal strips)
#   - CNN feature extraction per patch (multi-layer with batch normalization)
#   - LSTM sequential modeling across patches
#   - Mixed precision training (float16 via torch.amp)
#   - Cosine annealing learning rate scheduling
#   - Early stopping with best model checkpointing
#   - CSV logging of hyperparameters and results per run
#   - GPU accelerated (CUDA) with optimized DataLoader pipeline
# --------------------------------------------------------
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
import csv
import os
from datetime import datetime

# Changeable but not recommended
CONST_NUM_PATCHES = 7  # Number of vertical patches to split the image into

# --- Hyperparameters ---
FILTERS = 64  # Number of convolutional filters
HIDDEN_SIZE = 256  # Size of the LSTM hidden state
NUM_LAYERS = 2  # Number of LSTM layers
DROPOUT = 0.4  # Dropout rate for regularization
LR = 0.001  # Learning rate for the optimizer
EPOCHS = 50
NUM_CLASSES = 47

# --- Early Stopping ---
PATIENCE = 5  # Stop if no improvement after this many epochs

# --- Performance Parameters ---
BATCH_SIZE = 512  # Size of each training batch
NUM_WORKERS = 4  # Number of subprocesses for data loading
PREFETCH_FACTOR = 4  # Number of batches to prefetch

# -------------------
LOG_FILE = "results.csv"


def log_results(label, train_acc, test_acc, epochs_run):
    RESULTS_DIR = "models/RCNN/results"
    os.makedirs(RESULTS_DIR, exist_ok=True)

    file_path = os.path.join(RESULTS_DIR, LOG_FILE)
    file_exists = os.path.exists(file_path)
    with open(file_path, "a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(
                [
                    "timestamp",
                    "label",
                    "num_patches",
                    "filters",
                    "hidden_size",
                    "num_layers",
                    "dropout",
                    "batch_size",
                    "lr",
                    "epochs_run",
                    "patience",
                    "train_acc",
                    "test_acc",
                    "device",
                ]
            )

        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M"),
                label,
                CONST_NUM_PATCHES,
                FILTERS,
                HIDDEN_SIZE,
                NUM_LAYERS,
                DROPOUT,
                BATCH_SIZE,
                LR,
                epochs_run,
                PATIENCE,
                f"{train_acc:.2f}",
                f"{test_acc:.2f}",
                str(device),
            ]
        )


start_time = time.time()

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

root = os.path.join(os.path.dirname(__file__), "..", "..", "data")

train_dataset = datasets.EMNIST(
    root=root, split="balanced", train=True, download=False, transform=transform
)
test_dataset = datasets.EMNIST(
    root=root, split="balanced", train=False, download=False, transform=transform
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=PREFETCH_FACTOR,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=PREFETCH_FACTOR,
)


def get_cnn_out_size(cnn, num_patches):
    patch_h = 28 // num_patches
    dummy = torch.zeros(1, 1, patch_h, 28)
    return cnn(dummy).shape[1]


class RCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_patches = CONST_NUM_PATCHES
        patch_h = 28 // CONST_NUM_PATCHES

        self.cnn = nn.Sequential(
            nn.Conv2d(1, FILTERS, kernel_size=3, padding=1),
            nn.BatchNorm2d(FILTERS),
            nn.ReLU(),
            nn.Conv2d(FILTERS, FILTERS * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(FILTERS * 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(FILTERS * 2, FILTERS * 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(FILTERS * 4),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
        )
        cnn_out_size = get_cnn_out_size(self.cnn, CONST_NUM_PATCHES)
        self.rnn = nn.LSTM(
            input_size=cnn_out_size,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            dropout=DROPOUT if NUM_LAYERS > 1 else 0,
        )
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_SIZE, NUM_CLASSES)

    def forward(self, x):
        patch_h = 28 // self.num_patches
        patches = x.unfold(2, patch_h, patch_h)
        patches = patches.permute(0, 2, 1, 4, 3)
        cnn_out = torch.stack(
            [self.cnn(patches[:, i]) for i in range(self.num_patches)], dim=1
        )
        out, _ = self.rnn(cnn_out)
        return self.fc(self.dropout(out[:, -1]))


model = RCNN().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler("cuda")


def train(epochs=EPOCHS):
    best_acc = 0
    no_improve = 0
    final_acc = 0
    epochs_run = 0

    for epoch in range(epochs):
        model.train()
        total_loss, correct, total = 0, 0, 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda"):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)

        final_acc = 100 * correct / total
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        epochs_run = epoch + 1

        print(
            f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | Acc: {final_acc:.2f}% | LR: {current_lr:.6f}"
        )

        # Early stopping
        test_acc = evaluate(silent=True)
        if test_acc > best_acc:
            best_acc = test_acc
            no_improve = 0
            torch.save(model.state_dict(), "best_model.pt")
        else:
            no_improve += 1
            print(
                f"  No improvement ({no_improve}/{PATIENCE}) | Best test acc: {best_acc:.2f}%"
            )
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model before final evaluation
    model.load_state_dict(torch.load("best_model.pt"))
    return final_acc, epochs_run


def evaluate(silent=False):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    if not silent:
        print(f"Test Accuracy: {acc:.2f}%")
    return acc


if __name__ == "__main__":
    print("Device:", device)
    label = input("Label this run (press Enter to skip): ").strip()
    if not label:
        label = "unlabeled"
    train_acc, epochs_run = train(epochs=EPOCHS)
    test_acc = evaluate()
    log_results(label, train_acc, test_acc, epochs_run)
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    print(f"Results saved to {LOG_FILE}")
    train_loader._iterator = None
    test_loader._iterator = None
