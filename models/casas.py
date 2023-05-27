import torch
import loaders.casas
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, no_activities):
        super(LSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, output_dim, padding_idx=0)
        self.lstm = nn.LSTM(output_dim, output_dim, batch_first=True)
        self.fc = nn.Linear(output_dim, no_activities)
        # self.fc = nn.Sequential(nn.Flatten(),
        #                          nn.Dropout(0.2),
        #                          nn.Linear(output_dim, output_dim),
        #                          nn.ReLU(),
        #                          nn.Dropout(0.2),
        #                          nn.Linear(64, no_activities))
    def forward(self, x):
        print(x.shape)
        x = self.embedding(x)
        print(x.shape)
        x, _ = self.lstm(x)
        print(x.shape)
        x = self.fc(x[:, -1, :])
        return x
class BiLSTMModel(nn.Module):
    def __init__(self, input_dim=2000, output_dim=64, max_length=2000, no_activities=12):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, output_dim, padding_idx=0)
        self.lstm = nn.LSTM(output_dim, output_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(output_dim * 2, no_activities)

    def forward(self, x):
        x = self.embedding(x.type(torch.long))
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

class Ensemble2LSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_length, no_activities):
        super(Ensemble2LSTMModel, self).__init__()
        self.model1 = BiLSTMModel(input_dim, output_dim, max_length, no_activities)
        self.model2 = LSTMModel(input_dim, output_dim, max_length, no_activities)
        self.fc = nn.Linear(output_dim * 2, no_activities)

    def forward(self, x):
        x1 = self.model1(x)
        x2 = self.model2(x)
        x = torch.cat((x1, x2), dim=1)
        x = self.fc(x)
        return x

class CascadeEnsembleLSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_length, no_activities):
        super(CascadeEnsembleLSTMModel, self).__init__()
        self.model1 = BiLSTMModel(input_dim, output_dim, max_length, no_activities)
        self.model2 = LSTMModel(input_dim, output_dim, max_length, no_activities)
        self.lstm = nn.LSTM(output_dim * 2, output_dim, batch_first=True)
        self.fc = nn.Linear(output_dim, no_activities)

    def forward(self, x):
        x1 = self.model1.embedding(x)
        x2 = self.model2.embedding(x)
        x1, _ = self.model1.lstm(x1)
        x2, _ = self.model2.lstm(x2)
        x = torch.cat((x1, x2), dim=2)
        x, _ = self.lstm(x)
        x = self.fc(x[:, -1, :])
        return x

class CascadeEnsembleLSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_length, no_activities):
        super(CascadeEnsembleLSTMModel, self).__init__()
        self.embedding1 = nn.Embedding(input_dim, output_dim, padding_idx=0)
        self.embedding2 = nn.Embedding(input_dim, output_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(output_dim, output_dim, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(output_dim, output_dim, batch_first=True)
        self.lstm3 = nn.LSTM(output_dim * 2, output_dim, batch_first=True)
        self.fc = nn.Linear(output_dim, no_activities)

    def forward(self, x):
        x1 = self.embedding1(x)
        x2 = self.embedding2(x)
        x1, _ = self.lstm1(x1)
        x2, _ = self.lstm2(x2)
        x = torch.cat((x1, x2), dim=2)
        x, _ = self.lstm3(x)
        x = self.fc(x[:, -1, :])
        return x

class CascadeLSTMModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_length, no_activities):
        super(CascadeLSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, output_dim, padding_idx=0)
        self.lstm1 = nn.LSTM(output_dim, output_dim, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(output_dim * 2, output_dim, batch_first=True)
        self.fc = nn.Linear(output_dim, no_activities)

    def forward(self, x):
        x = self.embedding(x)
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        x = self.fc(x[:, -1, :])
        return x


