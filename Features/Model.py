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
        x = x.permute(1, 2, 0)  # Rearrange for RNN input (sequence_length, batch_size, input_size)
        x, _ = self.rnn(x)
        x = self.fc(x)
        return x

# Instantiate the model
num_classes = 37  # 26 English letters + 10 digits + blank for CTC
model = OCRModel(num_classes)

ctc_loss = nn.CTCLoss()


logits = torch.randn(50, 32, num_classes).log_softmax(2)  # (T, N, C) where T = sequence length, N = batch size, C = num classes
targets = torch.randint(1, num_classes, (32, 20), dtype=torch.long)  # (N, S) where S = target sequence length
input_lengths = torch.full((32,), 50, dtype=torch.long)  # All inputs have max length of 50
target_lengths = torch.randint(10, 20, (32,), dtype=torch.long)  # Target lengths between 10 and 20

loss = ctc_loss(logits, targets, input_lengths, target_lengths)
