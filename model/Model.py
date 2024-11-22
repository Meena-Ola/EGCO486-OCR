import torch
import torch.nn as nn
import torch.optim as optim

class OCRModel(nn.Module):
    def __init__(self, num_classes):
        super(OCRModel, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1) # Input: [N, 1, 64, 256], Output: [N, 32, 64, 256]
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: [N, 32, 32, 128]
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # Output: [N, 64, 32, 128]
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: [N, 64, 16, 64]
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Output: [N, 128, 16, 64]
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Output: [N, 128, 8, 32]
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)# Output: [N, 256, 8, 32]
        self.bn4 = nn.BatchNorm2d(256)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)# Output: [N, 256, 4, 16]
        
        self.dropout = nn.Dropout(p=0.5)
        
        # LSTM layers
        self.lstm = nn.LSTM(256 * 4, 128, bidirectional=True, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        # CNN forward pass
        x = self.pool1(self.bn1(torch.relu(self.conv1(x))))
        x = self.pool2(self.bn2(torch.relu(self.conv2(x))))
        x = self.pool3(self.bn3(torch.relu(self.conv3(x))))
        x = self.pool4(self.bn4(torch.relu(self.conv4(x))))
          # Apply dropout
        x = self.dropout(x)
        
        # Reshape for LSTM
        batch_size, channels, height, width = x.size()
        x = x.permute(0, 3, 1, 2).contiguous().view(batch_size, width, channels * height)
        
        # LSTM forward pass
        x, _ = self.lstm(x)
        
        # Fully connected layer
        x = self.fc(x)
        
        return x

criterion = nn.CTCLoss # Assuming 0 is the blank label


