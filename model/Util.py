import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import csv

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def calculate_accuracy(output, target):
    _, preds = torch.max(output, 1)
    return torch.sum(preds == target).item() / target.size(0)

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
            verbose (bool): If True, prints a message for each validation loss improvement.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.best_model_state = None

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                # Restore the best model
                model.load_state_dict(self.best_model_state)
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_model_state = model.state_dict()
        self.val_loss_min = val_loss
        
def visualize_attention(model, data_loader, device):
    model.eval()
    for inputs, _ in data_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        img = inputs[0].cpu().numpy().transpose(1, 2, 0)
        plt.imshow(img)
        plt.title('Input Image')
        plt.show()
        break
    
def save_metrics(metrics, model_name):
    epochs = range(1, len(metrics['train_losses']) + 1)
    filename = f'{model_name}_metrics.csv'
    with open(filename, 'w', newline='') as csvfile:
        fieldnames = ['Epoch', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for epoch in epochs:
            writer.writerow({
                'Epoch': epoch,
                'Train Loss': metrics['train_losses'][epoch-1],
                'Train Accuracy': metrics['train_accuracies'][epoch-1],
                'Validation Loss': metrics['val_losses'][epoch-1],
                'Validation Accuracy': metrics['val_accuracies'][epoch-1],
            })
            
def plot_metrics(metrics, model_name):
    epochs = range(1, len(metrics['train_losses']) + 1)

    plt.figure(figsize=(14, 5))

    # Plot Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['train_losses'], label='Train Loss')
    plt.plot(epochs, metrics['val_losses'], label='Validation Loss')
    plt.title(f'{model_name} Loss over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics['train_accuracies'], label='Train Accuracy')
    plt.plot(epochs, metrics['val_accuracies'], label='Validation Accuracy')
    plt.title(f'{model_name} Accuracy over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.show()