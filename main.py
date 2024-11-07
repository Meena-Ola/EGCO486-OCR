import torch.optim as optim
import torch
from Features import Model as M
from Features import DataLoader as D
from Features import LOSS as L
num_epochs = 30 
device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Select GPU
model = M.OCRModel(L.num_classes) 
model = model.to(device) # Move model to the GPU (if available)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for images, labels in D.dataloader:
         # Move images and labels to GPU (if available)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = L.ctc_loss(logits, L.targets, L.input_lengths,L.target_lengths)
        loss.backward(retain_graph=True)
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

torch.save(model, 'model_full.pth') 
