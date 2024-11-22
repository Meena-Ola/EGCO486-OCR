import torch

def train_and_validate(model, criterion, optimizer,scheduler, train_loader, val_loader, num_epochs, device,patience=5):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    best_val_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        correct_train = 0
        total_train = 0
        for images, labels,label_lengths in train_loader:
            images, labels,label_lengths = images.to(device), labels.to(device),label_lengths.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.log_softmax(2).permute(1, 0, 2)
            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to(device)
            loss = criterion(outputs, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        # Calculate accuracy
            _, preds = torch.max(outputs, 2)
            preds = preds.permute(1, 0).contiguous().view(-1)
            labels_flat = labels.view(-1)
            correct_train += (preds == labels_flat).sum().item()
            total_train += labels_flat.size(0)


        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for images, labels,label_lengths in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                outputs = outputs.log_softmax(2).permute(1, 0, 2)
                input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to(device) 
                loss = criterion(outputs, labels, input_lengths, label_lengths)
                val_loss += loss.item()
        # Calculate accuracy
                _, preds = torch.max(outputs, 2)
                preds = preds.permute(1, 0).contiguous().view(-1)
                labels_flat = labels.view(-1)
                correct_val += (preds == labels_flat).sum().item()
                total_val += labels_flat.size(0)
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)
        # Print learning rate
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}, Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.2f}, LR: {current_lr}")
        
        #Learning rate scheduling
        scheduler.step(val_loss)
        
         # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break
    
    return train_losses, val_losses