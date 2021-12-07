
import torch.nn as nn
import torch.nn.functional as F
import torch


class Conv3(nn.Module):
    """
     used in QMUL, regression
    """
    def __init__(self, embedding_dim=2916):
        super(Conv3, self).__init__()
        self.layer1 = nn.Conv2d(3, 36, 3, stride=2, dilation=2)
        self.layer2 = nn.Conv2d(36, 36, 3, stride=2, dilation=2)
        self.layer3 = nn.Conv2d(36, 36, 3, stride=2, dilation=2)
        self.out_layer = nn.Linear(embedding_dim, 1, bias=True)

    def forward(self, x):
        raise ValueError("no implement forward, use forward_feature or forward_pred")

    def forward_pred(self, x):
        """
        predict the label with output layer
        :param x:
        :return:
        """
        out = self.forward_feature(x)
        return self.out_layer(out)

    def forward_feature(self, x):
        """
        extract the feature from the last hidden layer
        :param x:
        :return:
        """
        out = F.relu(self.layer1(x))
        out = F.relu(self.layer2(out))
        out = F.relu(self.layer3(out))
        out = out.view(out.shape[0], -1)
        return out


def build_conv_block(in_channels: int, out_channels: int, conv_kernel=3, padding=1, max_pool_kernel_size=2):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=conv_kernel, padding=padding),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=max_pool_kernel_size)
    )


class Conv4(nn.Module):
    """
    used in MAML
    """
    def __init__(self, input_channel=3, embedding_dim=64, out_class_num=5):
        super(Conv4, self).__init__()
        hidden_dim = 64
        self.encoder = nn.Sequential(
            build_conv_block(input_channel, hidden_dim),
            build_conv_block(hidden_dim, hidden_dim),
            build_conv_block(hidden_dim, hidden_dim),
            build_conv_block(hidden_dim, hidden_dim)
        )
        self.output_feature_dim = embedding_dim
        self.out_layer = nn.Linear(embedding_dim, out_class_num, bias=True)

    def forward(self, x):
        raise ValueError("no implement forward, use forward_feature or forward_pred")

    def forward_pred(self, x):
        feature = self.forward_feature(x)
        return self.out_layer(feature)

    def forward_feature(self, x):
        """
        output embedding
        :param x:
        :return:
        """
        x = self.encoder(x)
        return x.view(x.size(0), -1)  # batch_size * feature_d
