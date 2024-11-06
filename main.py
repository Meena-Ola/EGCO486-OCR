import torch.optim as optim
from Features import Model as m
from Features import DataLoader as D
num_epochs = 30


optimizer = optim.Adam(m.model.parameters(), lr=0.001)

for epoch in range(num_epochs):
    for images, labels in D.dataloader:
        optimizer.zero_grad()
        logits = m.model(images)
        loss = m.ctc_loss(logits, labels, m.input_lengths, m.target_lengths)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
