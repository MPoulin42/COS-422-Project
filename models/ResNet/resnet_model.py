# Mathieu Poulin - ResNet18 Model for EMNIST Balanced Character Recognition
# -------------------------------------------------------
# Features:
#   - Custom ResNet18 adapted for 28x28 grayscale images (1 channel, 47 classes)
#   - Residual blocks with batch normalization
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

# --- Hyperparameters ---
LR = 0.001          # Learning rate for the optimizer
EPOCHS = 50         # Maximum number of training epochs
NUM_CLASSES = 47    # EMNIST Balanced has 47 character classes
DROPOUT = 0.4       # Dropout rate for regularization

# --- Early Stopping ---
PATIENCE = 5        # Stop if no improvement after this many epochs

# --- Performance Parameters ---
BATCH_SIZE = 128    # Size of each training batch
NUM_WORKERS = 4     # Number of subprocesses for data loading
PREFETCH_FACTOR = 4 # Number of batches to prefetch

# -------------------
LOG_FILE = "results.csv"


def log_results(label, train_acc, test_acc, epochs_run):
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
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


# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data
root = os.path.join(os.path.dirname(__file__), "..", "..", "data")

transform_train = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)
transform_test = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

train_dataset = datasets.EMNIST(
    root=root, split="balanced", train=True, download=False, transform=transform_train
)
test_dataset = datasets.EMNIST(
    root=root, split="balanced", train=False, download=False, transform=transform_test
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


class ResidualBlock(nn.Module):
    """Basic residual block with two 3x3 convolutions and a skip connection."""

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Projection shortcut when dimensions change
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet18 adapted for 28x28 grayscale EMNIST images.

    Modifications from the original ResNet18:
      - Input channels changed from 3 to 1 (grayscale).
      - Initial conv uses 3x3 kernel with stride 1 (instead of 7x7, stride 2)
        to preserve spatial resolution on the small 28x28 input.
      - Initial max-pooling layer removed for the same reason.
    """

    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()

        # Stem: 3x3 conv, no max-pool (adapted for 28x28 input)
        self.stem = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Four residual stages (two blocks each, matching ResNet18)
        self.layer1 = self._make_layer(64, 64, stride=1)
        self.layer2 = self._make_layer(64, 128, stride=2)
        self.layer3 = self._make_layer(128, 256, stride=2)
        self.layer4 = self._make_layer(256, 512, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, stride):
        return nn.Sequential(
            ResidualBlock(in_channels, out_channels, stride),
            ResidualBlock(out_channels, out_channels, stride=1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return self.fc(x)


_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "best_resnet_model.pt")

model = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
scaler = torch.amp.GradScaler(device.type)


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
            with torch.amp.autocast(device.type):
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
            f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(train_loader):.4f} | "
            f"Acc: {final_acc:.2f}% | LR: {current_lr:.6f}"
        )

        # Early stopping
        test_acc = evaluate(silent=True)
        if test_acc > best_acc:
            best_acc = test_acc
            no_improve = 0
            torch.save(model.state_dict(), _CHECKPOINT)
        else:
            no_improve += 1
            print(
                f"  No improvement ({no_improve}/{PATIENCE}) | Best test acc: {best_acc:.2f}%"
            )
            if no_improve >= PATIENCE:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model before final evaluation
    model.load_state_dict(torch.load(_CHECKPOINT, weights_only=True))
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
    start_time = time.time()
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
