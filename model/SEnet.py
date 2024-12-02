import torch
import torch.nn as nn
import torch.optim as optim
from model.ResNet import BasicBlock, Bottleneck
from model.Hyperparameters import Hyperparameters as hp
from model.Util import set_seed, EarlyStopping, calculate_accuracy,EarlyStoppingWithDynamicPatience

#Class for the Squeeze and Excitation block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.average = nn.AdaptiveAvgPool2d(1)  # Global Average Pooling
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch_size, channels, _, _ = x.size()
        # Squeeze
        y = self.average(x).view(batch_size, channels)
        # Excitation
        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch_size, channels, 1, 1)
        # Scale input
        return x * y
    

class SEBlockWrapper(nn.Module):
    def __init__(self, block, in_planes, planes, stride=1, downsample=None, reduction=16):
        super(SEBlockWrapper, self).__init__()
        self.block = block(in_planes, planes, stride, downsample)
        self.se = SEBlock(planes * block.expansion, reduction=reduction)

    def forward(self, x):
        out = self.block(x)
        out = self.se(out)
        return out
    
    
#Class for the network
class SEResNet(nn.Module):
    def __init__(self, block, layers, num_classes=hp.num_classes, reduction = 16):
        super(SEResNet, self).__init__()
        self.in_planes = 64

        # Initial convolutional layer and max pooling
        self.conv1 = nn.Conv2d(hp.input_channels, 64, kernel_size=hp.initial_kernel_size, stride=hp.initial_stride,
                               padding=hp.initial_padding, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        if hp.dataset == 'CIFAR10':
            self.maxpool = nn.Identity()  # Skip max pooling for CIFAR-10
        else:
            self.maxpool = nn.MaxPool2d(kernel_size=hp.maxpool_kernel_size, stride=hp.maxpool_stride, padding=hp.maxpool_padding)

        # Create the residual layers with CBAM blocks
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1,reduction = reduction)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, reduction=reduction)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,reduction=reduction)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,reduction=reduction)

        # Average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride, reduction):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(SEBlockWrapper(block, self.in_planes, planes, stride, downsample, reduction=reduction))
        self.in_planes = planes * block.expansion
        for _ in range(1, num_blocks):
            layers.append(SEBlockWrapper(block, self.in_planes, planes, reduction=reduction))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def fit(self, train_loader, val_loader, device):
        """Train the model using the provided data loaders."""
        set_seed(hp.seed)
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=hp.learning_rate,
                              momentum=hp.momentum, weight_decay=hp.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        early_stopping = EarlyStoppingWithDynamicPatience(patience=hp.patience,  max_patience=hp.max_patience, verbose=True)

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(hp.num_epochs):
            self.train()
            epoch_loss = 0
            correct = 0
            total = 0

            for inputs, targets in train_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                correct += (outputs.argmax(1) == targets).sum().item()
                total += targets.size(0)

            train_loss = epoch_loss / len(train_loader)
            train_accuracy = correct / total
            train_losses.append(train_loss)
            train_accuracies.append(train_accuracy)

            val_loss, val_accuracy = self.evaluate(val_loader, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f'Epoch {epoch+1}/{hp.num_epochs}, '
                  f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}')

            scheduler.step()

            # Early stopping
            #early_stopping(val_loss, self)
            #if early_stopping.early_stop:
                #print("Early stopping")
                #break
            # Dynamically adjust patience
            if early_stopping.best_val_loss is not None and (early_stopping.best_val_loss - val_loss > early_stopping.delta):
                hp.current_patience = min(hp.current_patience + 5, hp.max_patience)

            early_stopping(val_loss, val_accuracy, self)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies
        }

    def evaluate(self, data_loader, device):
        """Evaluate the model on the provided data loader."""
        self.eval()
        criterion = nn.CrossEntropyLoss()
        loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = self(inputs)
                loss += criterion(outputs, targets).item()
                correct += (outputs.argmax(1) == targets).sum().item()
                total += targets.size(0)

        avg_loss = loss / len(data_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def predict(self, data_loader, device):
        """Make predictions using the trained model."""
        self.eval()
        predictions = []

        with torch.no_grad():
            for inputs in data_loader:
                inputs = inputs.to(device)
                outputs = self(inputs)
                preds = outputs.argmax(1)
                predictions.extend(preds.cpu().numpy())

        return predictions

def senet_model(num_classes=hp.num_classes):
    """Constructs an SE ResNet model based on the specified hyperparameters."""
    if hp.block_type == 'Bottleneck':
        block = Bottleneck
        hp.expansion = 4
    elif hp.block_type == 'BasicBlock':
        block = BasicBlock
        hp.expansion = 1
    else:
        raise ValueError("Invalid block type. Choose 'BasicBlock' or 'Bottleneck'.")

    layers = hp.num_blocks
    return SEResNet(block, layers, num_classes=num_classes)