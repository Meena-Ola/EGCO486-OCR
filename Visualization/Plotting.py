import matplotlib.pyplot as plt

def T_VLoss(num_epochs, train_losses, val_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
    plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.show()


def T_VAcc(num_epochs,train_acc,val_acc):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), train_acc, label='Train Accuracy')
    plt.plot(range(1, num_epochs + 1), val_acc, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    plt.show()
    
