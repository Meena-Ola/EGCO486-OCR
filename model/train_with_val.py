import torch
from tqdm import tqdm

def train_and_validate(model, criterion, optimizer, scheduler, train_loader, val_loader, num_epochs, device, patience=5):
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
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Training]")
        for images, labels, label_lengths in train_loader_tqdm:
            images, labels, label_lengths = images.to(device), labels.to(device), label_lengths.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            outputs = outputs.permute(1, 0, 2)  # [T, N, C] - Sequence length, batch size, num classes
            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to(device)  # [N] - batch size
            
            # Compute the CTC loss
            loss = criterion(outputs, labels, input_lengths, label_lengths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Decode the outputs to get the predicted labels (argmax)
            _, preds = torch.max(outputs, 2)
            preds = preds.permute(1, 0)  # [N, T] - batch size, sequence length
            
            # Calculate accuracy (per character)
            for pred, label, label_length in zip(preds, labels, label_lengths):
                correct_train += (pred == label).sum().item()
                total_train += label_length.item()
        
        # Average loss and accuracy for the epoch
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        train_accuracy = correct_train / total_train
        train_accuracies.append(train_accuracy)

        # Validation phase
        model.eval()
        val_loss = 0
        correct_val = 0
        total_val = 0
        val_loader_tqdm = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Validation]")
        with torch.no_grad():
            for images, labels, label_lengths in val_loader_tqdm:
                images, labels, label_lengths = images.to(device), labels.to(device), label_lengths.to(device)
                outputs = model(images)
                outputs = outputs.log_softmax(2).permute(1, 0, 2)  # [T, N, C] after log softmax
                input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to(device)  # [N]
                
                # Compute the CTC loss
                loss = criterion(outputs, labels, input_lengths, label_lengths)
                val_loss += loss.item()
                
                # Decode the outputs to get the predicted labels (argmax)
                _, preds = torch.max(outputs, 2)
                preds = preds.permute(1, 0)  # [N, T]
                
                # Calculate accuracy (per character)
                for pred, label, label_length in zip(preds, labels, label_lengths):
                    correct_val += (pred == label).sum().item()
                    total_val += label_length.item()
        
        # Average loss and accuracy for validation
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        val_accuracy = correct_val / total_val
        val_accuracies.append(val_accuracy)

        # Print learning rate
        current_lr = scheduler.optimizer.param_groups[0]['lr']
        print(f'Epoch [{epoch+1}/{num_epochs}], Learning Rate: {current_lr:.6f}')
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}')
        
        # Learning rate scheduling
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
    
    return train_losses, val_losses, train_accuracies, val_accuracies

