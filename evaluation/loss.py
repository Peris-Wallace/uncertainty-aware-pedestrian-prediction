import torch
import torch.nn.functional as F


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device

def relu_evidence(y):
    return F.relu(y)

def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = get_device()
    # ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    ones = torch.ones(1, num_classes, dtype=alpha.dtype, device = alpha.device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + second_term
    return kl

def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    loglikelihood = loglikelihood_loss(y, alpha, device=device)

    # annealing_coef = torch.min(
    #     torch.tensor(1.0, dtype=torch.float32),
    #     torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    # )

    annealing_coef = torch.clamp(
        torch.as_tensor(
            epoch_num / annealing_step,
            dtype=alpha.dtype,
            device=alpha.device,
        ),
        max=1.0,
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)

    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)

    # annealing_coef = torch.min(
    #     torch.tensor(1.0, dtype=torch.float32),
    #     torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    # )
    annealing_coef = torch.clamp(
        torch.as_tensor(
            epoch_num / annealing_step,
            dtype=alpha.dtype,
            device=alpha.device,
        ),
        max=1.0,
    )

    kl_alpha = (alpha - 1) * (1 - y) + 1
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):

    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )

    return loss


def edl_digamma_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    if not device:
        device = get_device()

    evidence = relu_evidence(output)
    alpha = evidence + 1
    loss = torch.mean(
        edl_loss(torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device)
    )

    return loss
