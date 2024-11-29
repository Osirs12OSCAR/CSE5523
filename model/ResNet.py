import torch
import torch.nn as nn
import torch.optim as optim
from model.Hyperparameters import Hyperparameters as hp
from model.Util import set_seed, EarlyStopping, calculate_accuracy
import sys

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample

    def forward(self, x):
        identity = x  # Save input for shortcut

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Apply the shortcut (identity connection)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Add the shortcut connection
        out = self.relu(out)  # Apply activation

        return out

class Bottleneck(nn.Module):
    expansion = 4  # For Bottleneck block, expansion is 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1 convolution
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        # 3x3 convolution
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 1x1 convolution
        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x  # Save input for shortcut

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # Apply the shortcut (identity connection)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity  # Add the shortcut connection
        out = self.relu(out)  # Apply activation

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=hp.num_classes):
        super(ResNet, self).__init__()
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

        # Create the residual layers
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)   # Conv2_x
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)  # Conv3_x
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)  # Conv4_x
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)  # Conv5_x

        # Average pooling and fully connected layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Global average pooling
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            # Adjust dimensions with a convolutional layer
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        # First block in the layer
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        # Initial convolutional layer and max pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # Classification layer
        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # Flatten the tensor
        x = self.fc(x)

        return x

    def _initialize_weights(self):
        # Initialize weights as per He initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def fit(self, train_loader, val_loader, device):
        """Train the model using the provided data loaders."""
        print("Training the model...")
        set_seed(hp.seed)  # Set random seed for reproducibility
        self.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.parameters(), lr=hp.learning_rate,
                              momentum=hp.momentum, weight_decay=hp.weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        early_stopping = EarlyStopping(patience=hp.patience, verbose=True)

        train_losses = []
        val_losses = []
        train_accuracies = []
        val_accuracies = []

        for epoch in range(hp.num_epochs):
            print(f'Epoch {epoch+1}/{hp.num_epochs}', flush=True)
            # Training phase
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

            # Validation phase
            val_loss, val_accuracy = self.evaluate(val_loader, device)
            val_losses.append(val_loss)
            val_accuracies.append(val_accuracy)

            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}', flush=True)

            scheduler.step()

            # Early stopping
            early_stopping(val_loss, self)
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

def resnet_model(num_classes=hp.num_classes):
    """Constructs a ResNet model based on the specified hyperparameters."""
    if hp.block_type == 'Bottleneck':
        block = Bottleneck
        hp.expansion = 4  # Set expansion for Bottleneck
    elif hp.block_type == 'BasicBlock':
        block = BasicBlock
        hp.expansion = 1  # Set expansion for BasicBlock
    else:
        raise ValueError("Invalid block type. Choose 'BasicBlock' or 'Bottleneck'.")

    layers = hp.num_blocks
    return ResNet(block, layers, num_classes)