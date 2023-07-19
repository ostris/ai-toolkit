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


# Gradient penalty
def get_gradient_penalty(critic, real, fake, device):
    with torch.autocast(device_type='cuda'):
        alpha = torch.rand(real.size(0), 1, 1, 1).to(device)
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
        d_interpolates = critic(interpolates)
        fake = torch.ones(real.size(0), 1, device=device)

        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty

