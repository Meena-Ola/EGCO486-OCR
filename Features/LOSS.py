import torch
import torch.nn as nn

# Instantiate the model
num_classes = 37  # 26 English letters + 10 digits + blank for CTC
batch_size = 32
sequence_length = 50
ctc_loss = nn.CTCLoss()


targets = torch.randint(1, num_classes, (batch_size , 20), dtype=torch.long)  # (N, S) where S = target sequence length
input_lengths = torch.full((batch_size ,), 50, dtype=torch.long)  # All inputs have max length of 50
target_lengths = torch.randint(10, 20, (batch_size ,), dtype=torch.long)  # Target lengths between 10 and 20