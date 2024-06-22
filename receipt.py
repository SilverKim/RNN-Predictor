import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from my_rnn import CustomRNN

def read_data(filepath):
    df = pd.read_csv(filepath)
    dates = pd.to_datetime(df["Date"]).values
    counts = df["Receipt_Count"].values
    return dates, counts

def preprocess_data(dates, counts):
    min_count = np.min(counts)
    max_count = np.max(counts)
    normalized_counts = (counts - min_count) / (max_count - min_count)
    numeric_dates = np.arange(len(dates))
    return numeric_dates, normalized_counts

def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

def split_data(X, y, test_size):
    train_X, train_y = X[:-test_size], y[:-test_size]
    test_X, test_y = X[-test_size:], y[-test_size:]
    return train_X, train_y, test_X, test_y

def create_dataloaders(X, y, batch_size, shuffle=True):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

def month_string_to_index(month_str):
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return months.index(month_str) + 1 if month_str in months else None

def get_prediction_for_month(model, dataloader, counts, month_str):
    month_index = month_string_to_index(month_str)
    if month_index is None:
        print("Invalid month string. Please provide a valid month (Jan, Feb, ..., Dec).")
        return None
    
    model.eval()
    predictions = []
    with torch.no_grad():
        for X, _ in dataloader:
            hidden_state = model.init_hidden(batch_size=X.shape[0])
            output, _ = model(X, hidden_state)
            predictions.append(output.item())
    
    min_count = np.min(counts)
    max_count = np.max(counts)
    predictions = np.array(predictions) * (max_count - min_count) + min_count
    
    return predictions[month_index - 1]

def train_model(model, train_dataloader, test_dataloader, criterion, optimizer, epochs):
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X, y in train_dataloader:
            optimizer.zero_grad()
            hidden_state = model.init_hidden(batch_size=X.shape[0])
            output, _ = model(X, hidden_state)
            y = y.unsqueeze(1)
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_dataloader)
        train_losses.append(train_loss)
        
        model.eval()
        test_loss = 0.0
        with torch.no_grad():
            for X, y in test_dataloader:
                hidden_state = model.init_hidden(batch_size=X.shape[0])
                output, _ = model(X, hidden_state)
                y = y.unsqueeze(1)
                loss = criterion(output, y)
                test_loss += loss.item()
        test_loss /= len(test_dataloader)
        test_losses.append(test_loss)
        
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.3f} - Test Loss: {test_loss:.3f}")
    
    return train_losses, test_losses

def plot_losses(train_losses, test_losses, filename):
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Validation Loss")
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def plot_predictions(predictions, actual, filename):
    plt.plot(predictions, label="Predictions")
    plt.plot(actual, label="Actual")
    plt.legend()
    plt.savefig(filename)
    plt.clf()

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    
    dates, counts = read_data("receipts_data.csv")
    numeric_dates, normalized_counts = preprocess_data(dates, counts)
    seq_length = 1
    X, y = create_sequences(normalized_counts, seq_length)
    train_X, train_y, test_X, test_y = split_data(X, y, test_size=90)
    
    train_dataloader = create_dataloaders(train_X, train_y, batch_size=5)
    test_dataloader = create_dataloaders(test_X, test_y, batch_size=1, shuffle=False)
    
    input_size, hidden_size, output_size = 1, 20, 1
    model = CustomRNN(input_size, hidden_size, output_size)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    epochs = 100
    train_losses, test_losses = train_model(model, train_dataloader, test_dataloader, criterion, optimizer, epochs)
    
    plot_losses(train_losses, test_losses, 'plot1.png')
    
    predictions = []
    model.eval()
    with torch.no_grad():
        for X, _ in test_dataloader:
            hidden_state = model.init_hidden(batch_size=X.shape[0])
            output, _ = model(X, hidden_state)
            predictions.append(output.item())
    
    min_count, max_count = np.min(counts), np.max(counts)
    predictions = np.array(predictions) * (max_count - min_count) + min_count
    
    plot_predictions(predictions, counts[-90:], 'plot2.png')
    
    for i, prediction in enumerate(predictions):
        print(f"Predicted: {prediction:.2f} - Actual: {counts[-90 + i]:.2f}")
    
    month_to_predict = input("Enter the month to predict (e.g., Jan, Feb, ..., Dec): ")
    prediction = get_prediction_for_month(model, test_dataloader, counts, month_to_predict)
    
    if prediction is not None:
        print(f"Predicted count for {month_to_predict}: {prediction:.2f}")
    
    torch.save(model.state_dict(), "trained_model.pth")
    print("Trained model saved at 'trained_model.pth'")

if __name__ == "__main__":
    main()
