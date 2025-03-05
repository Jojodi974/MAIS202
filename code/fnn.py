import data_loader
import joblib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder


#Open the pre-processed data using built-in function
file_path = "../data/S18/pre_prep/pre_prep_battlesStaging_12272020_WL_tagged.csv"
data_loader(file_path)
df_prep = joblib.load(f'data/joblib/{file_path}')

##ChatGPT implementation of FNN

# ----------------------
# 1. Dataset Preparation
# ----------------------
class CardDataset(Dataset):
    def __init__(self, df, card_columns, numerical_columns):
        self.card_columns = card_columns
        self.numerical_columns = numerical_columns

        # Convert card ID columns using ordinal encoding
        encoder = OrdinalEncoder(dtype=np.int64)
        self.card_data = encoder.fit_transform(df[card_columns])
        self.num_cards = int(self.card_data.max()) + 1  # Total unique cards

        # Normalize numerical columns
        scaler = StandardScaler()
        self.numerical_data = scaler.fit_transform(df[numerical_columns])

        # Target variable
        self.y = df['winner.value'].values.astype(np.float32)

        # Convert to tensors
        self.card_data = torch.tensor(self.card_data, dtype=torch.long)
        self.numerical_data = torch.tensor(self.numerical_data, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32).unsqueeze(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.card_data[idx], self.numerical_data[idx], self.y[idx]


# ----------------------
# 2. FNN Model Definition
# ----------------------
class FNNClassifier(nn.Module):
    def __init__(self, num_cards, embed_dim, num_numerical, hidden_dim):
        super(FNNClassifier, self).__init__()

        # Embedding layer for card IDs
        self.embedding = nn.Embedding(num_cards, embed_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(embed_dim * 16 + num_numerical, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Binary classification output

        self.dropout = nn.Dropout(0.3)

    def forward(self, card_ids, numerical_data):
        embedded = self.embedding(card_ids)  # Shape: (batch, 16, embed_dim)
        embedded = embedded.view(embedded.size(0), -1)  # Flatten embeddings

        x = torch.cat([embedded, numerical_data], dim=1)  # Concatenate with numerical features
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return torch.sigmoid(x)


# ----------------------
# 3. Training & Evaluation
# ----------------------

def train_model(model, train_loader, val_loader, epochs=10, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct, total = 0, 0

        for card_data, num_data, labels in train_loader:
            card_data, num_data, labels = card_data.to(device), num_data.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(card_data, num_data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        print(f'Epoch {epoch + 1}: Loss = {train_loss / len(train_loader):.4f}, Accuracy = {train_acc:.4f}')

    return model


# ----------------------
# 4. Data Loading & Execution
# ----------------------

def main():
    df = df_prep  # Load your dataset

    card_columns = [
        'player1.card1.id', 'player1.card2.id', 'player1.card3.id', 'player1.card4.id',
        'player1.card5.id', 'player1.card6.id', 'player1.card7.id', 'player1.card8.id',
        'player2.card1.id', 'player2.card2.id', 'player2.card3.id', 'player2.card4.id',
        'player2.card5.id', 'player2.card6.id', 'player2.card7.id', 'player2.card8.id'
    ]
    numerical_columns = ['player1.totalcard.level', 'player1.elixir.average', 'player2.totalcard.level',
                         'player2.elixir.average']

    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    train_dataset = CardDataset(train_df, card_columns, numerical_columns)
    val_dataset = CardDataset(val_df, card_columns, numerical_columns)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = FNNClassifier(num_cards=train_dataset.num_cards, embed_dim=10, num_numerical=len(numerical_columns),
                          hidden_dim=128)
    trained_model = train_model(model, train_loader, val_loader, epochs=10, lr=0.001)


if __name__ == "__main__":
    main()