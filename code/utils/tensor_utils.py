import torch

def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]

def split_support_query(x, y, n_support):
    return x[:, : n_support, :], y[:, : n_support].unsqueeze(2), \
           x[:, n_support:, :], y[:, n_support:].unsqueeze(2),


def split_support_query_for_x_in_cls(z, n_support):
    z_support = z[:, :n_support]
    z_query = z[:, n_support:]
    return z_support, z_query

def dump_list_2_json(data_list):
    import json
    return json.dumps(data_list)