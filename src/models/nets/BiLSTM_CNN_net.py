# import torch
# import torch.nn as nn
# import torch.nn.functional as F

# class BiLSTM_CNN_net(nn.Module):
#     def __init__(self, input_size=64, hidden_size=32, num_classes=5):
#         super(BiLSTM_CNN_net, self).__init__()

#         self.hidden_size = hidden_size
#         self.num_classes = num_classes

#         self.conv1d_1 = nn.Conv1d(in_channels=2, out_channels=64, kernel_size=8)
#         self.conv1d_2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=8)
#         self.lstm_1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
#         self.lstm_2 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
#         self.lstm_3 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True)
#         self.dropout = nn.Dropout(p=0.1)
#         self.dense_1 = nn.Linear(192, 40)
#         self.dense_2 = nn.Linear(40, 5)
#         self.loss_fn = nn.CrossEntropyLoss()

#     def forward(self, x):
#         # Apply first convolutional layer
#         conv_output = F.relu(self.conv1d_1(x))

#         # Apply second convolutional layer
#         conv_output = F.relu(self.conv1d_2(conv_output))

#         # Permute dimensions for LSTM input
#         lstm_input = conv_output.permute(0, 2, 1)

#         # Forward pass through LSTM layers
#         lstm_output_1, _ = self.lstm_1(lstm_input)
#         lstm_output_2, _ = self.lstm_2(lstm_input)
#         lstm_output_3, _ = self.lstm_3(lstm_input)

#         # Concatenate LSTM outputs
#         lstm_output = torch.cat([lstm_output_1, lstm_output_2, lstm_output_3], dim=2)

#         # Apply dropout
#         lstm_output = self.dropout(lstm_output)

#         # Global average pooling
#         pooled_output = torch.mean(lstm_output, dim=1)
#         # Apply fully connected layers
#         dense_output = F.relu(self.dense_1(pooled_output))
#         output = self.dense_2(dense_output)
#         return output


import torch
import torch.nn as nn
import torch.nn.functional as F

class BiLSTM_CNN_net(nn.Module):
    def __init__(self, input_size=64, hidden_size=32, num_classes=5):
        super(BiLSTM_CNN_net, self).__init__()

        self.hidden_size = hidden_size
        self.num_classes = num_classes

        self.conv1 = nn.Conv1d(2, 64, kernel_size=8)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=8)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(64, hidden_size, num_layers=3, bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden_size * 2, 40)  # Multiply hidden_size by 2 due to bidirectional LSTM
        self.fc2 = nn.Linear(40, num_classes)  # Multiply hidden_size by 2 due to bidirectional LSTM


    def forward(self, x):
        # Apply the first convolutional layer
        x = self.conv1(x)
        
        # Apply ReLU activation
        x = self.relu(x)
        
        # Apply the second convolutional layer
        x = self.conv2(x)
        
        # Apply ReLU activation
        x = self.relu(x)
        
        # Reshape the tensor for LSTM input
        x = x.permute(0, 2, 1)
        
        # Apply LSTM layer
        outputs, _ = self.lstm(x)

        x = outputs[:, -1, :]
        
        # Apply linear layer
        x = self.fc1(x)

        x = self.fc2(x)
        return x