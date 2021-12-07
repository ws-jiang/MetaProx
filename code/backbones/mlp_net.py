from torch import nn


class MLP2(nn.Module):
    """
    this net is used for fitting sine function
    """
    def __init__(self, in_dim=1, out_dim=1, embedding_dim=40):
        super(MLP2, self).__init__()
        self.in_dim = in_dim
        hid_dim = embedding_dim

        self.feature_backbone = nn.Sequential(
            nn.Linear(self.in_dim, hid_dim),
            nn.ReLU(),
            nn.Linear(hid_dim, hid_dim),
            nn.ReLU()
        )
        self.output_layer = nn.Linear(embedding_dim, out_dim)

    def forward(self, x):
        raise ValueError("no implement forward, use forward_feature or forward_pred")

    def forward_feature(self, x):
        """
        return the feature before the last relu layer
        :param x:
        :return:
        """
        return self.feature_backbone[0:3](x)  # no relu

    def forward_pred(self, x):
        feature = self.feature_backbone(x)
        return self.output_layer(feature)
