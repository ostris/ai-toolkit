import torch


def total_variation(image):
    """
    Compute normalized total variation.
    Inputs:
    - image: PyTorch Variable of shape (N, C, H, W)
    Returns:
    - TV: total variation normalized by the number of elements
    """
    n_elements = image.shape[1] * image.shape[2] * image.shape[3]
    return ((torch.sum(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:])) +
            torch.sum(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]))) / n_elements)


class ComparativeTotalVariation(torch.nn.Module):
    """
    Compute the comparative loss in tv between two images. to match their tv
    """

    def forward(self, pred, target):
        return torch.abs(total_variation(pred) - total_variation(target))
