import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

from my_rnn import CustomRNN

from torch.utils.data import DataLoader, Dataset, TensorDataset

def read_data(filepath):
    df = pd.read_csv(filepath)
    dates = pd.to_datetime(df["Date"])
    counts = df["Receipt_Count"].values

    # convert the dates and counts to numpy arrays
    dates = np.array(dates)
    counts = np.array(counts)

    return dates, counts

def preprocess_data(dates, counts):
    # normalize the counts to be between 0 and 1
    min_count = np.min(counts)
    max_count = np.max(counts)

    normalized_counts = (counts - min_count) / (max_count - min_count)

    # convert the dates to a numerical format
    numeric_dates = np.arange(len(dates))

    return numeric_dates, normalized_counts

def create_sequences(numeric_dates, normalized_counts, seq_length):
    X, y = [], []

    for i in range(len(normalized_counts) - seq_length):
        seq_in = normalized_counts[i:i + seq_length]
        seq_out = normalized_counts[i + seq_length]
        X.append(seq_in)
        y.append(seq_out)

    X = np.array(X)
    y = np.array(y)

    return X, y

def split_data(X, y, test_size=90):
    # Using 3 months as the test set by default

    train_X = X[:-test_size]
    train_y = y[:-test_size]

    test_X = X[-test_size:]
    test_y = y[-test_size:]

    return train_X, train_y, test_X, test_y



def create_dataloaders(X, y, batch_size, shuffle=True):
    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader

def month_string_to_index(month_str):
    # Convert month string to index (assuming Jan is 1 and Dec is 12)
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    return months.index(month_str) + 1 if month_str in months else None


def get_prediction_for_month(model, test_dataloader, counts, month_str):
    month_index = month_string_to_index(month_str)

    if month_index is not None:
        model.eval()
        predictions = []

        with torch.no_grad():
            for X, y in test_dataloader:
                hidden_state = model.init_hidden(batch_size=X.shape[0])
                output, _ = model(X, hidden_state)
                predictions.append(output.item())

        # Invert the normalization
        min_count = np.min(counts)
        max_count = np.max(counts)

        predictions = np.array(predictions) * (max_count - min_count) + min_count

        # Assuming test set contains the last 3 months
        return predictions[month_index - 1]
    else:
        print("Invalid month string. Please provide a valid month (Jan, Feb, ..., Dec).")
        return None



def main():

    torch.random.manual_seed(42)
    np.random.seed(42)

    # Read data from CSV file
    dates, counts = read_data("receipts_data.csv")

    # Preprocess data
    numeric_dates, normalized_counts = preprocess_data(dates, counts)
     
    # split the data into train and test sets
    numeric_dates_train, normalized_counts_train, numeric_dates_test, normalized_counts_test = split_data(numeric_dates, normalized_counts)

    # Create sequences
    seq_length = 1

    train_X, train_y = create_sequences(numeric_dates_train, normalized_counts_train, seq_length)
    test_X, test_y = create_sequences(numeric_dates_test, normalized_counts_test, seq_length)

    # Create dataloaders
    train_dataloader = create_dataloaders(train_X, train_y,batch_size=5, shuffle=True)
    test_dataloader = create_dataloaders(test_X, test_y, batch_size=1, shuffle=False)


    # Instantiate the Vanilla RNN model
    input_size = 1
    hidden_size = 20
    output_size = 1
    model = CustomRNN(input_size, hidden_size, output_size)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    epochs = 100
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        # Train
        train_loss = 0.0
        model.train()

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

        # Evaluate
        test_loss = 0.0
        model.eval()

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

    # Plot the training and validation losses
    plt.plot(train_losses, label="Training Loss")
    plt.plot(test_losses, label="Validation Loss")
    plt.legend()
    plt.savefig('plot1.png')

    # Make predictions on the test set and denormalize the predictions  
    predictions = []
    model.eval()

    with torch.no_grad():
        for X, y in test_dataloader:
            hidden_state = model.init_hidden(batch_size=X.shape[0])
            output, _ = model(X, hidden_state)
            predictions.append(output.item())
    
    # invert the normalization
    min_count = np.min(counts)
    max_count = np.max(counts)

    predictions = np.array(predictions) * (max_count - min_count) + min_count

    plt.plot(predictions, label="Predictions")
    plt.plot(counts[-90:], label="Actual") # Last 3 months because we used 3 months as the test set
    plt.legend()
    plt.savefig('plot2.png')

    for i in range(len(predictions)):
        print(f"Predicted: {predictions[i]:.2f} - Actual: {counts[-90+i]:.2f}")


    # Get the month to predict from user input
    month_to_predict = input("Enter the month to predict (e.g., Jan, Feb, ..., Dec): ")

    prediction = get_prediction_for_month(model, test_dataloader, counts, month_to_predict)

    if prediction is not None:
        print(f"Predicted count for {month_to_predict}: {prediction:.2f}")
    
    # Save the trained model
    model_save_path = "trained_model.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Trained model saved at {model_save_path}")


if __name__ == "__main__":
    main()

