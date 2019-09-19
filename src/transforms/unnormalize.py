"""Un-Normalize.

Unnormalizes a tensor.
"""


class UnNormalize:
    def __init__(self, mean, std):
        self.mean, self.std = mean, std

    def __call__(self, tensor):
        """Un-normalizes a tensor."""
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
