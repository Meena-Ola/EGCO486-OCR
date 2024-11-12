def train(model, train_loader, optimizer, ctc_loss,device, num_epochs=10, print_every=100):
    for epoch in range(num_epochs):
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
            running_count += 1

            if (i+1) % print_every == 0:
                    print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Average Loss: {Running_loss/running_count:.4f}")
                    Running_loss = 0.0
                    running_count = 0


        # Print loss for each epoch
        epoch_loss = Running_loss / len(train_loader)
        print(f"\nEnd of Epoch {epoch+1}/{num_epochs}, Average Epoch Loss: {epoch_loss:.4f}")

