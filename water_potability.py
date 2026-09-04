# install tqdm et torchmetrics
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import pandas as pd
from torchmetrics import Accuracy, Precision, Recall, F1Score
from tqdm import tqdm

import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 150

Myaccuracy = Accuracy(task="binary").to(device)
#précision = Precision(task="muticlassification").to(device)
# Charger les données csv sur Dataset

class WaterDataset(Dataset):
    def __init__(self, csv_path):
        super().__init__()
        # charger les données à partir du fichier CSV
        self.data = pd.read_csv(csv_path)
        # imputons par la médiane les valeurs manquantes
        self.data.fillna(self.data.median(), inplace=True)
        #print(self.data.isna().sum())

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        feature = torch.tensor(self.data.iloc[idx, :-1].values, dtype=torch.float32)
        target = torch.tensor(self.data.iloc[idx, -1], dtype=torch.float32)
        return feature, target


# Definir le model

class WaterModel(nn.Module):
    def __init__(self):
        super().__init__()
        # on définit nos couches linéaires
        self.fc1 = nn.Linear(9, 64)
        self.fc2 = nn.Linear(64, 32)
        self.output_layer = nn.Linear(32, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.output_layer(x)
        return x   

def train_one_epoch(model, dataloader, criterion, optimizer):
    model.to(device)
    model.train()
    running_loss = 0.0
    Myaccuracy.reset()
    for features, targets in dataloader:
        features, targets = features.to(device), targets.to(device)

        optimizer.zero_grad()

        outputs = model(features)

        probs = torch.sigmoid(outputs.squeeze())

        loss = criterion(outputs.squeeze(), targets)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * features.size(0)
        #running_loss = running_loss + loss.item() * features.size(0)

        Myaccuracy.update(probs, targets)

    loss = running_loss / len(dataloader.dataset)
    accuracy = Myaccuracy.compute()
    return loss, accuracy


def evaluate(model, dataloader, criterion):
    model.to(device)
    model.eval()
    running_loss = 0.0
    Myaccuracy.reset()
    with torch.no_grad():
        for features, targets in dataloader:
            features, targets = features.to(device), targets.to(device)

            outputs = model(features)

            probs = torch.sigmoid(outputs.squeeze())

            loss = criterion(outputs.squeeze(), targets)

            running_loss += loss.item() * features.size(0)

            Myaccuracy.update(probs, targets)

    eval_loss = running_loss / len(dataloader.dataset)
    eval_accuracy = Myaccuracy.compute()
    return eval_loss, eval_accuracy



def main():
    # testons de charger les données
    # afficher le device utilisé
    print(f"Using device: {device}")
    dataset = WaterDataset("data/water_potability4.csv")
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    # definir model, criterion et optimizer
    model = WaterModel()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    training_losses = []
    training_accuracies = []
    test_losses = []    
    test_accuracies = []

    for epoch in tqdm(range(num_epochs), desc="Training"):
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss, test_accuracy = evaluate(model, test_loader, criterion)
        training_losses.append(train_loss)
        training_accuracies.append(train_accuracy.item())
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy.item())
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], "
                  f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, "
                  f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    # faire un graphique des pertes et des précisions
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(training_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss over Epochs")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.show()


if __name__ == "__main__":
    main()
