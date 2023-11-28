import torch
from .llvae import LosslessLatentEncoder


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
        real = real.float()
        fake = fake.float()
        alpha = torch.rand(real.size(0), 1, 1, 1).to(device).float()
        interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
        if torch.isnan(interpolates).any():
            print('d_interpolates is nan')
        d_interpolates = critic(interpolates)
        fake = torch.ones(real.size(0), 1, device=device)
            
        if torch.isnan(d_interpolates).any():
            print('fake is nan')
        gradients = torch.autograd.grad(
            outputs=d_interpolates,
            inputs=interpolates,
            grad_outputs=fake,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        # see if any are nan
        if torch.isnan(gradients).any():
            print('gradients is nan')

        gradients = gradients.view(gradients.size(0), -1)
        gradient_norm = gradients.norm(2, dim=1)
        gradient_penalty = ((gradient_norm - 1) ** 2).mean()
        return gradient_penalty.float()


class PatternLoss(torch.nn.Module):
    def __init__(self, pattern_size=4, dtype=torch.float32):
        super().__init__()
        self.pattern_size = pattern_size
        self.llvae_encoder = LosslessLatentEncoder(3, pattern_size, dtype=dtype)

    def forward(self, pred, target):
        pred_latents = self.llvae_encoder(pred)
        target_latents = self.llvae_encoder(target)

        matrix_pixels = self.pattern_size * self.pattern_size

        color_chans = pred_latents.shape[1] // 3
        # pytorch
        r_chans, g_chans, b_chans = torch.split(pred_latents, [color_chans, color_chans, color_chans], 1)
        r_chans_target, g_chans_target, b_chans_target = torch.split(target_latents, [color_chans, color_chans, color_chans], 1)

        def separated_chan_loss(latent_chan):
            nonlocal matrix_pixels
            chan_mean = torch.mean(latent_chan, dim=[1, 2, 3])
            chan_splits = torch.split(latent_chan, [1 for i in range(matrix_pixels)], 1)
            chan_loss = None
            for chan in chan_splits:
                this_mean = torch.mean(chan, dim=[1, 2, 3])
                this_chan_loss = torch.abs(this_mean - chan_mean)
                if chan_loss is None:
                    chan_loss = this_chan_loss
                else:
                    chan_loss = chan_loss + this_chan_loss
            chan_loss = chan_loss * (1 / matrix_pixels)
            return chan_loss

        r_chan_loss = torch.abs(separated_chan_loss(r_chans) - separated_chan_loss(r_chans_target))
        g_chan_loss = torch.abs(separated_chan_loss(g_chans) - separated_chan_loss(g_chans_target))
        b_chan_loss = torch.abs(separated_chan_loss(b_chans) - separated_chan_loss(b_chans_target))
        return (r_chan_loss + g_chan_loss + b_chan_loss) * 0.3333


