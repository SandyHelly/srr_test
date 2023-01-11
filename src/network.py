import torch.nn as nn
import torch


class HARNet(nn.Module):
    def __init__(self, window_size: int = 60):
        super(HARNet, self).__init__()
        
        # architecture was stolen from https://github.com/mariusbock/dl-for-har
        # hardcode looks bad, but let it stay like that
        self.nb_filters = 64
        self.filter_width = 11
        self.window_size = window_size
        self.nb_channels = 6
        self.nb_units_lstm = 128
        self.nb_layers_lstm = 1
        self.nb_classes = 1

        self.conv1 = nn.Conv2d(1, self.nb_filters, (self.filter_width, 1))
        self.conv2 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.filter_width, 1))
        self.conv3 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.filter_width, 1))
        self.conv4 = nn.Conv2d(self.nb_filters, self.nb_filters, (self.filter_width, 1))

        self.lstm = nn.LSTM(input_size=self.nb_filters * self.nb_channels, 
                            hidden_size=self.nb_units_lstm, num_layers=self.nb_layers_lstm)

        self.fc = nn.Linear(self.nb_units_lstm, self.nb_classes)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x):

        x = x.view(-1, 1, self.window_size, self.nb_channels)
        
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        
        final_seq_len = x.shape[2]
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(-1, final_seq_len, self.nb_filters * self.nb_channels)

        x, _ = self.lstm(x)
        
        x = x.view(-1, self.nb_units_lstm)
        x = self.fc(x)
        out = x.view(-1, final_seq_len, self.nb_classes)
        return out[:, -1, :]
