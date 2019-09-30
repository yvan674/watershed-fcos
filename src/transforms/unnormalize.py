"""Un-Normalize.

Unnormalizes a preds.
"""


class UnNormalize:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, tensor):
        """Un-normalizes a preds."""
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
