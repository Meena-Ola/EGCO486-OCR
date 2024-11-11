import torch.optim as optim
import torch
from Model import Model as M
from Features import DataLoader as D
from Model import LOSS as L
num_epochs = 30
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Select GPU
model = M.OCRModel(L.num_classes)
model = model.to(device)  # Move model to the GPU (if available)
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    model.train()
    Running_loss = 0.0
    
    for images, labels in D.dataloader:
        # Move images and labels to GPU (if available)
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(images)
        
     
        batch_size, num_classes = logits.size()  # Unpacking shape [batch_size, num_classes]
        
    
        loss = L.ctc_loss(logits, labels)  # Modify to use standard loss
        
        # Backward pass and optimization step
        loss.backward()
        optimizer.step()
        
        # Accumulate running loss
        Running_loss += loss.item()

    # Print loss for each epoch
    print(f'Epoch {epoch + 1}, Loss: {Running_loss / len(D.dataloader)}')

# Save the trained model
torch.save(model.state_dict(), 'OCR.pth')
