import torch
import torch.nn.functional as F

def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct_characters = 0  # For character-level accuracy
    total_characters = 0
    
    with torch.no_grad():
        for images, labels, label_lengths in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)

            # Permute to correct shape: [T, N, C] (time_steps, batch_size, num_classes)
            outputs = outputs.permute(1, 0, 2)  # Change shape from [N, T, C] to [T, N, C]

            # Apply log_softmax across the class dimension (C)
            outputs = F.log_softmax(outputs, dim=2)

            # Compute loss
            input_lengths = torch.full((outputs.size(1),), outputs.size(0), dtype=torch.long).to(device)
            loss = criterion(outputs, labels, input_lengths, label_lengths)
            test_loss += loss.item()

            # Decode predictions using greedy decoding (take argmax along the class dimension)
            _, predicted = outputs.max(2)  # Get the index of max probability at each time step
            predicted = predicted.permute(1, 0)  # Change shape to [N, T] for each sample

            # Loop over each image in the batch
            for i in range(predicted.size(0)):  # Iterate over batch size
                pred_seq = predicted[i]
                true_seq = labels[i]  # True label for the i-th image, just a single character

                # Remove blank token (0) from predictions
                pred_seq = pred_seq[pred_seq != 0]  # Remove blank token

                # For a single character dataset, the sequence is of length 1
                correct_characters += (pred_seq == true_seq).sum().item()  # Character-level accuracy (should match exactly)
                total_characters += 1  # Since it's a single character per image
    
    # Compute average loss
    avg_loss = test_loss / len(test_loader)
    
    # Compute accuracy
    character_accuracy = correct_characters / total_characters * 100
    
    print(f"Test Loss: {avg_loss:.4f}")
    print(f" Accuracy: {character_accuracy:.2f}%")
    
    #return avg_loss, sequence_accuracy, character_accuracy
