import torch
import torch.nn as nn

# Instantiate the model
num_classes = 37  # 26 English letters + 10 digits + blank for CTC
batch_size = 32
sequence_length = 50
ctc_loss = nn.CrossEntropyLoss()     # Use CrossEntropy loss instead of CTC loss (assuming this is a classification problem)

