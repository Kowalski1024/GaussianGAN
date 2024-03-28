import torch
import torch.nn.functional as F
import wandb


def discriminator_accuracy(real_logits, fake_logits):
    with torch.no_grad():
        real_accuracy = torch.mean((real_logits > 0.5).float()).item()
        fake_accuracy = torch.mean((fake_logits < 0.5).float()).item()

    return real_accuracy, fake_accuracy


def smooth_label(label, min=0.9, max=1.0):
    return torch.rand_like(label) * (max - min) + min


def noisy_label(label, flip_probability=0.05):
    flip = torch.rand_like(label) < flip_probability
    inverse_label = 1 - label
    label[flip] = inverse_label[flip]

    return label


def calculate_gradient_penalty(discriminator, real_images, fake_images, device):
    alpha = torch.rand(real_images.size(0), 1, 1, 1).to(device)
    alpha = alpha.expand(real_images.size())

    interpolates = alpha * real_images + (1 - alpha) * fake_images
    interpolates = torch.autograd.Variable(interpolates, requires_grad=True).to(device)
    fake = torch.autograd.Variable(
        torch.ones(real_images.size(0), 1).to(device), requires_grad=False
    )

    disc_interpolates = discriminator(interpolates)
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def wgan_loss(real_logit, fake_logit, model_type):
    if model_type == "generator":
        return -torch.mean(fake_logit)

    elif model_type == "discriminator":
        loss_fake = torch.mean(fake_logit)
        loss_real = torch.mean(real_logit)
        loss = loss_fake - loss_real
        return loss


def hinge_loss(real_logit, fake_logit, model_type):
    return NotImplemented


def ls_loss(real_logit, fake_logit, model_type, noise_label=False):
    mse = F.mse_loss

    if model_type == "generator":
        fake_label = torch.ones_like(fake_logit)

        if noise_label:
            fake_label = noisy_label(fake_label)

        device = fake_logit.device
        fake_label = fake_label.to(device)

        return mse(fake_logit, fake_label)

    elif model_type == "discriminator":
        fake_label = torch.zeros_like(fake_logit)

        if noise_label:
            real_label = smooth_label(real_logit)
            real_label = noisy_label(real_label)
        else:
            real_label = torch.ones_like(real_logit)

        device = real_logit.device
        fake_label, real_label = fake_label.to(device), real_label.to(device)

        loss_fake = mse(fake_logit, fake_label)
        loss_real = mse(real_logit, real_label)

        return (loss_fake + loss_real) / 2


def loss(
    real_logit,
    fake_logit,
    model_type,
    type="wgan",
    weight=1.0,
    noise_label=False,
    **kwargs,
):
    match type.lower():
        case "wgan":
            loss = wgan_loss(real_logit, fake_logit, model_type)
        case "wgan-gp":
            loss = wgan_loss(real_logit, fake_logit, model_type)
        case "hinge":
            loss = hinge_loss(real_logit, fake_logit, model_type)
        case "ls":
            loss = ls_loss(real_logit, fake_logit, model_type, noise_label)
        case _:
            raise ValueError(f"Invalid loss type: {type}")

    loss *= weight

    wandb.log({f"{model_type}_loss": loss.item()})

    return loss
