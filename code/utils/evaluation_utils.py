import torch
import numpy as np
import torch.nn.functional as F


def count_acc(logits, label):
    """
    :param logits: n * c; n=num of samples, c=number of classes
    :param label: n
    :return: accuracy
    """
    pred = torch.argmax(logits, dim=1)
    return (pred == label).float().mean().item()

def compute_entropy(x):
    b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    b = -1.0 * b.sum()
    return b.item() / x.shape[0]

class Average(object):

    def __init__(self, name=""):
        self.n = 0
        self.name = name
        self.history = []

    def add(self, x):
        self.history.append(x)

    def add_tensor_list(self, x):
        for a in x:
            self.add(a.item())

    def last(self):
        return self.history[-1]

    def item(self):
        return float(np.array(self.history).mean())

    def std(self):
        return np.array(self.history).std()

    def get_history(self):
        return self.history

    def len(self):
        return len(self.history)

    def __repr__(self):
        if len(self.history) == 0:
            return ""

        if self.name in {"acc"} or self.name.find("acc") >= 0:
            return "{}: {:.2f}/{:.2f}/{:.2f}".format(self.name, self.last() * 100, self.item() * 100, np.array(self.history).std())
        else:
            return "{}: {:.4f}".format(self.name,  self.item())

