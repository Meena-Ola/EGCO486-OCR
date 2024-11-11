import torch
import torch.nn as nn
import torch.optim as optim
class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()
        # CNN for feature extraction
        self.cnn = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),  # Input 1 channel, output 64 channels
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.MaxPool2d(kernel_size=2, stride=2),
    
    nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),  # Input 64 channels, output 128 channels
    nn.BatchNorm2d(128),
    nn.ReLU(True),
    nn.MaxPool2d(kernel_size=2, stride=2),

    nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # Final layer: output 64 channels
    nn.BatchNorm2d(64),
    nn.ReLU(True),
    nn.MaxPool2d(kernel_size=2, stride=2),
)


        # RNN for sequence modeling
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, bidirectional=True)
        # Fully connected layer for classification
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Pass input through CNN
        x = self.cnn(x)
        # print("After CNN:", x.shape)
        batch_size,channels,height,width = x.size()  # Reshape to (batch_size, channels, sequence_length) and permute to (sequence_length, batch_size, channels)
        assert channels == 64, f"Expected 64 channels, got {channels}"
        x = x.permute(0, 3, 2, 1).contiguous()  # (batch_size, height, width, channels)
       # print("After permute:", x.shape)
        seq_len = height * width  # Flatten height and width to form sequence length
        x = x.view(batch_size, seq_len, channels)  # (batch_size, seq_len, channels)
       # print("After View:", x.shape)
        x = x.permute(1, 0 ,2)  # Rearrange for RNN input (sequence_length, batch_size, input_size)
      #  print("After premute for long short-term memory:", x.shape)
        x, _ = self.rnn(x)
      #  print("After LSTM:", x.shape)
        x =x[-1] #batch_size, num_classes)
        x = self.fc(x)
        return x
