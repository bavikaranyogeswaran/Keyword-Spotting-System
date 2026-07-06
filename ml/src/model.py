import torch.nn as nn


class KeywordCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # Three convolutional blocks that learn visual patterns in the spectrogram
        # Each block: detect features → normalize → activate → shrink image by half
        # Channels grow from 1 → 16 → 32 → 64 so the network learns richer patterns
        self.conv_layers = nn.Sequential(
            # Block 1: input is a single-channel spectrogram image
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # detect basic patterns
            nn.BatchNorm2d(16),                           # keep values stable
            nn.ReLU(),                                    # allow only positive signals
            nn.MaxPool2d(2),                              # halve the image size

            # Block 2: learn more complex combinations of patterns
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            # Block 3: learn the most abstract high-level features
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        # Flatten the 2D feature map and classify it into one of the command classes
        self.classifier = nn.Sequential(
            nn.Flatten(),           # convert 2D feature map to a 1D vector
            nn.LazyLinear(128),     # fully connected layer (input size inferred automatically)
            nn.ReLU(),
            nn.Dropout(0.3),        # randomly drop 30% of neurons to prevent overfitting
            nn.Linear(128, num_classes),  # output one score per command class
        )

    def forward(self, x):
        # Pass spectrogram through conv blocks to extract features
        x = self.conv_layers(x)

        # Pass features through classifier to get class scores
        x = self.classifier(x)

        return x
