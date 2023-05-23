from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size=18, hidden_size=210, output_size=1):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size * 2)
        self.layer3 = nn.Linear(hidden_size * 2, hidden_size * 4)
        self.layer4 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.layer5 = nn.Linear(hidden_size * 2, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        out = self.relu(self.layer1(x))
        out = self.dropout(out)
        out = self.relu(self.layer2(out))
        out = self.dropout(out)
        out = self.relu(self.layer3(out))
        out = self.dropout(out)
        out = self.relu(self.layer4(out))
        out = self.dropout(out)
        out = self.relu(self.layer5(out))
        out = self.dropout(out)
        out = self.output_layer(out)
        return out
