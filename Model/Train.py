from model import Validate 

def train(model, train_loader, optimizer, ctc_loss,device, num_epochs=10, print_every=100):
    train_losses= 0.0
    model.train()
    Running_loss = 0.0
    running_count = 0
    for i,(images, labels) in enumerate(train_loader):
            # Move images and labels to GPU (if available)
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(images)
            
        
            loss = ctc_loss(logits, labels)  # Modify to use standard loss
            
            # Backward pass and optimization step
            loss.backward()
            optimizer.step()
            
            # Accumulate running loss
            Running_loss += loss.item()
    return (Running_loss / len(train_loader))



def train_and_validate(model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training phase
        train_loss = train(model, train_loader, optimizer,criterion,device)
        train_losses.append(train_loss)

        # Validation phase
        val_loss = Validate.validate(model,device, val_loader, criterion)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

    return train_losses, val_losses

        
       
