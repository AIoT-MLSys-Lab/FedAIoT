import torch.nn as nn


class LSTMRegressor(nn.Module):
    def __init__(self, input_size=5, output_size=2, hidden_size=128, num_layers=2):
        super(LSTMRegressor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)
        self.output_activation = nn.Softmax()

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, num_features)
        x, _ = self.lstm(x)  # LSTM output shape: (batch_size, sequence_length, hidden_size)
        x = self.fc(self.dropout(x[:, -1, :]))
        # Use the last LSTM output; shape: (batch_size, output_size)
        return self.output_activation(x)


class CNN_LSTM_Regressor(nn.Module):
    def __init__(self, input_size=5, num_emotions=2):
        super(CNN_LSTM_Regressor, self).__init__()

        self.conv1 = nn.Conv1d(input_size, 64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool1d(kernel_size=2)
        self.lstm = nn.LSTM(64, 128, num_layers=1, batch_first=True)
        self.fc = nn.Linear(128, num_emotions)
        self.output_activation = nn.Softmax()

    def forward(self, x):
        # Input shape: (batch_size, sequence_length, num_features)
        x = x.permute(0, 2, 1)  # Change shape to (batch_size, num_features, sequence_length)

        # 1D Convolution
        x = self.conv1(x)
        x = self.relu(x)
        x = self.max_pool(x)

        x = x.permute(0, 2, 1)  # Change shape to (batch_size, sequence_length, num_channels)

        # LSTM
        x, _ = self.lstm(x)

        # Fully connected layer
        x = self.fc(x[:, -1, :])  # Use the last LSTM output

        return self.output_activation(x)
