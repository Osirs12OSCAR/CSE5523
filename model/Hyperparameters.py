class Hyperparameters:
    # Data parameters
    batch_size = 128  # Adjust based on your hardware capabilities
    num_classes = 10  # For CIFAR-10; set to 1000 for ImageNet
    input_channels = 3  # Number of input channels (e.g., RGB images)

    # Model parameters
    # Block type can be 'BasicBlock' or 'Bottleneck'
    block_type = 'BasicBlock'  # 'Bottleneck' for ResNet-50 or deeper
    expansion = 1  # 1 for BasicBlock, 4 for Bottleneck
    num_blocks = [2, 2, 2, 2]  # For ResNet-18; adjust for other depths
    initial_kernel_size = 3
    initial_stride = 1
    initial_padding = 1
    maxpool_kernel_size = 3
    maxpool_stride = 2
    maxpool_padding = 1
    reduction = 16  # For attention modules
    spatial_kernel_size = 7  # For spatial attention
    pool_types = ['avg', 'max']  # For channel attention pooling

    # Training parameters
    num_epochs = 50
    learning_rate = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    seed = 42

    # Early stopping
    patience = 10

    # Dataset selection
    dataset = 'CIFAR10'  # Options: 'CIFAR10', 'ImageNet'
    use_noise = False
    use_occlusion = False