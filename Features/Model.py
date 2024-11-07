import torch
import torch.nn as nn
import torch.optim as optim

class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()
        # CNN for feature extraction
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Additional CNN layers...
        )
        # RNN for sequence modeling
        self.rnn = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, bidirectional=True)
        # Fully connected layer for classification
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        # Pass input through CNN
        x = self.cnn(x)
        batch_size,channels,height,width = x.size()  # Reshape to (batch_size, channels, sequence_length) and permute to (sequence_length, batch_size, channels)
        x = x.permute(0, 2, 3, 1).contiguous()  # (batch_size, height, width, channels)
        x = x.view(batch_size,height*width,channels)  # Flatten height and width
        x = x.permute(1, 0 ,2)  # Rearrange for RNN input (sequence_length, batch_size, input_size)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x



