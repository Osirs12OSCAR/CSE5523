import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import csv
import torchvision
from torch.profiler import profile, ProfilerActivity, record_function

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

    def __call__(self, val_loss, val_acc, model):
        score = -val_loss  # Use loss primarily
        acc_score = val_acc  # Add accuracy for comparison

        if self.best_score is None:
            self.best_score = score
            self.best_acc_score = acc_score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta and acc_score <= self.best_acc_score:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_state_dict(self.best_model_state)
        else:
            self.best_score = score
            self.best_acc_score = acc_score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decreases."""
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        self.best_model_state = model.state_dict()
        self.val_loss_min = val_loss
        
class EarlyStoppingWithDynamicPatience:
    def __init__(self, patience=10, verbose=False, delta=0, max_patience=20):
        self.patience = patience
        self.max_patience = max_patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_val_loss = None
        self.best_val_acc = None
        self.early_stop = False
        self.best_model_state = None

    def __call__(self, val_loss, val_acc, model):
        if self.best_val_loss is None or (val_loss < self.best_val_loss - self.delta):
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter
        elif val_loss >= self.best_val_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
                model.load_state_dict(self.best_model_state)
        # Increase patience dynamically if significant improvement is seen
        if val_loss < self.best_val_loss - self.delta and val_acc > self.best_val_acc:
            self.patience = min(self.patience + 5, self.max_patience)

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased. Saving model...')
        self.best_model_state = model.state_dict()

        
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
    
def visualize_cifar10_classes(data_loader):
    """
    Visualize one image from each class in the CIFAR-10 dataset.

    Args:
        data_loader (DataLoader): DataLoader object for CIFAR-10 dataset.
    """
    class_names = ('airplane', 'automobile', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    class_to_image = {}

    # Loop through the dataset to find one image per class
    for images, labels in data_loader:
        for img, label in zip(images, labels):
            label = label.item()
            if label not in class_to_image:
                class_to_image[label] = img
            if len(class_to_image) == len(class_names):  # Stop if we have all classes
                break
        if len(class_to_image) == len(class_names):
            break

    # Plot images
    plt.figure(figsize=(12, 8))
    for i, class_name in enumerate(class_names):
        plt.subplot(2, 5, i + 1)
        img = class_to_image[i].numpy().transpose(1, 2, 0)
        img = img * np.array((0.2470, 0.2435, 0.2616)) + np.array((0.4914, 0.4822, 0.4465))  # Unnormalize
        img = np.clip(img, 0, 1)
        plt.imshow(img)
        plt.title(class_name)
        plt.axis('off')
    plt.tight_layout()
    plt.show()
    
def compute_topk_accuracy(output, target, topk=(1,)):
    """
    Computes the top-k accuracy for the specified values of k.

    Args:
        output (Tensor): Model predictions (logits).
        target (Tensor): Ground truth labels.
        topk (tuple): Tuple of k values for which to compute top-k accuracy.

    Returns:
        List[float]: Top-k accuracies.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res
    
def measure_model_efficiency(model, input_tensor, device):
    """
    Measure FLOPS, inference time, and memory usage.

    Args:
        model (nn.Module): The model to evaluate.
        input_tensor (Tensor): A single input tensor.
        device (str): Device to perform the evaluation.

    Returns:
        dict: Dictionary with FLOPS, latency, and memory usage.
    """
    model.eval()
    input_tensor = input_tensor.to(device)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True
    ) as prof:
        with record_function("model_inference"):
            model(input_tensor)

    flops_info = prof.key_averages().table(
        sort_by="cuda_time_total" if torch.cuda.is_available() else "cpu_time_total",
        row_limit=10
    )
    memory_used = torch.cuda.memory_allocated(device) / (1024 ** 2) if torch.cuda.is_available() else 0
    return {
        "FLOPS Info": flops_info,
        "Memory (MB)": memory_used
    }

def visualize_attention_maps(model, data_loader, device):
    """
    Visualize attention maps of the model.

    Args:
        model (nn.Module): The model (SE-Net or CBAM).
        data_loader (DataLoader): DataLoader for the dataset.
        device (str): Device to perform the visualization.
    """
    model.eval()
    for inputs, _ in data_loader:
        inputs = inputs.to(device)

        if hasattr(model, "attention_module"):
            attention_map = model.attention_module(inputs)
            attention_map = attention_map[0].detach().cpu().numpy()

            plt.imshow(attention_map, cmap='viridis')
            plt.title("Attention Map")
            plt.colorbar()
            plt.show()
        else:
            print("Model does not have an attention module for visualization.")
        break
