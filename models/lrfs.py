import math

import torch
import torch.nn.functional as F


def _fftshift2(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.fftshift(x, dim=(-2, -1))


def _ifftshift2(x: torch.Tensor) -> torch.Tensor:
    return torch.fft.ifftshift(x, dim=(-2, -1))


def radial_frequency_map(height: int, width: int, device: torch.device) -> torch.Tensor:
    fy = torch.fft.fftfreq(height, d=1.0, device=device)
    fx = torch.fft.fftfreq(width, d=1.0, device=device)
    fy = torch.fft.fftshift(fy)
    fx = torch.fft.fftshift(fx)
    grid_y, grid_x = torch.meshgrid(fy, fx, indexing="ij")
    radius = torch.sqrt(grid_y.square() + grid_x.square())
    max_radius = math.sqrt(0.5 ** 2 + 0.5 ** 2)
    return radius / max_radius


def high_frequency_removed(image: torch.Tensor, nu: float) -> torch.Tensor:
    _, _, height, width = image.shape
    radius = radial_frequency_map(height, width, image.device)
    low_pass_mask = (radius < nu).to(dtype=image.dtype).view(1, 1, height, width)
    spectrum = _fftshift2(torch.fft.fft2(image, dim=(-2, -1)))
    filtered = spectrum * low_pass_mask
    restored = torch.fft.ifft2(_ifftshift2(filtered), dim=(-2, -1))
    return restored.abs()


def lipschitz_ratio(
    feature_anchor: torch.Tensor,
    feature_reference: torch.Tensor,
    image_anchor: torch.Tensor,
    image_reference: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    feature_dist = torch.linalg.vector_norm(
        (feature_anchor - feature_reference).flatten(start_dim=1), dim=1
    )
    image_dist = torch.linalg.vector_norm(
        (image_anchor - image_reference).flatten(start_dim=1), dim=1
    )
    return feature_dist / image_dist.clamp_min(eps)


def lrfs_loss(
    feature_anchor: torch.Tensor,
    image_anchor: torch.Tensor,
    encoder_fn,
    nu_mf: float,
    nu_hf: float,
    kappa_mf: float,
    kappa_hf: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    image_hf = high_frequency_removed(image_anchor, nu_hf)
    image_mf = high_frequency_removed(image_anchor, nu_mf)

    feature_hf = encoder_fn(image_hf).detach()
    feature_mf = encoder_fn(image_mf).detach()

    ratio_hf = lipschitz_ratio(feature_anchor, feature_hf, image_anchor, image_hf)
    ratio_mf = lipschitz_ratio(feature_anchor, feature_mf, image_anchor, image_mf)

    loss_hf = F.relu(ratio_hf / kappa_hf - 1.0).mean()
    loss_mf = F.relu(1.0 - ratio_mf / kappa_mf).mean()
    return loss_hf, loss_mf
