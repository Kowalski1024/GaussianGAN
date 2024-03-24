import numpy as np
from dataset import SyntheticDataset
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
from model import Generator, Discriminator
import PIL.Image


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))

    return pc / m


def load_sparse(path, points=4096):
    ball = np.loadtxt(f"{path}/{points}.xyz")
    return pc_normalize(ball)


def data_loader(path, batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])

    dataset = SyntheticDataset(path, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    return loader


def save_image_grid(img, fname, drange, grid_size):
    lo, hi = drange
    img = np.asarray(img, dtype=np.float32)
    img = (img - lo) * (255 / (hi - lo))
    img = np.rint(img).clip(0, 255).astype(np.uint8)

    gw, gh = grid_size
    _N, C, H, W = img.shape
    img = img.reshape([gh, gw, C, H, W])
    img = img.transpose(0, 3, 1, 4, 2)
    img = img.reshape([gh * H, gw * W, C])

    assert C in [1, 3]
    if C == 1:
        PIL.Image.fromarray(img[:, :, 0], "L").save(fname)
    if C == 3:
        PIL.Image.fromarray(img, "RGB").save(fname)


def pre_train_generator(
    generator, ball, optimizer, loader, epochs=10, device="cuda", batch_size=32
):
    generator.train()
    points = ball.shape[1]
    noise = torch.randn(1, 128).to(device)
    noise = noise.unsqueeze(1).expand(batch_size, points, -1)

    for epoch in range(epochs):
        for i, (real_images, camera) in enumerate(loader):
            real_images, camera = real_images.to(device), camera.to(device)

            optimizer.zero_grad()
            fake_images = generator(ball, noise, camera)
            loss = torch.nn.functional.l1_loss(fake_images, real_images)
            loss.backward()
            optimizer.step()

            if i % 10 == 0:
                print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

        images = torch.cat([fake_images, real_images], dim=0)
        save_image_grid(
            images.detach().cpu().numpy(),
            f"imgs/pretrain_{epoch}.png",
            drange=[-1, 1],
            grid_size=(4, 4),
        )

    return generator


def compute_gradient_penalty(D, real_samples, fake_samples, device="cuda"):
    """Calculates the gradient penalty loss for WGAN GP"""
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(device)
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(
        True
    )
    d_interpolates = D(interpolates)
    fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0).to(device)
    fake = fake.squeeze(1)  # remove the second dimension of 'fake'
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


def train(
    generator,
    discriminator,
    ball,
    optimizer_g,
    optimizer_d,
    loader,
    epochs=10,
    device="cuda",
    batch_size=32,
):
    generator.train()
    discriminator.train()
    points = ball.shape[1]
    noise = torch.normal(0, 0.2, (batch_size, 128)).to(device)
    noise = noise.view(batch_size, 1, 128).expand(-1, points, -1)

    for epoch in range(epochs):
        for i, (real_images, camera) in enumerate(loader):
            real_images, camera = real_images.to(device), camera.to(device)

            # Train the discriminator
            optimizer_d.zero_grad()
            fake_images = generator(ball, noise, camera)
            real_output = discriminator(real_images)
            fake_output = discriminator(fake_images.detach())
            loss_d = -torch.mean(
                torch.log(real_output + 1e-8) + torch.log(1 - fake_output + 1e-8)
            )
            loss_d.backward()
            optimizer_d.step()

            # Train the generator
            optimizer_g.zero_grad()
            fake_output = discriminator(fake_images)
            loss_g = -torch.mean(torch.log(fake_output + 1e-8))
            loss_g.backward()
            optimizer_g.step()

            if i % 10 == 0:
                print(
                    f"Epoch {epoch}, iter {i}, loss_d: {loss_d.item()}, loss_g: {loss_g.item()}"
                )

        # torch.save(generator.state_dict(), f"generator_{epoch}.pt")
        # torch.save(discriminator.state_dict(), f"discriminator_{epoch}.pt")
        images = torch.cat([fake_images[:8], real_images[:8]], dim=0)
        save_image_grid(
            images.detach().cpu().numpy(),
            f"imgs/output_{epoch}.png",
            drange=[-1, 1],
            grid_size=(4, 4),
        )

    return generator, discriminator


def main():
    points = 4096
    batch_size = 8
    ball = load_sparse("balls", points)
    ball = torch.tensor(ball, dtype=torch.float32).cuda()
    ball = ball.unsqueeze(0).expand(batch_size, -1, -1)
    loader = data_loader(
        "/mnt/d/Tomasz/Pulpit/GaussianGAN/datasets/cars_train/car1",
        batch_size=batch_size,
    )

    # Instantiate the generator and the discriminator
    generator = Generator(points, 20, 128).cuda()
    discriminator = (
        Discriminator().cuda()
    )  # Assuming Discriminator is defined elsewhere

    # Create separate optimizers for the generator and the discriminator
    optimizer_g = torch.optim.AdamW(generator.parameters(), lr=1e-3)
    optimizer_d = torch.optim.AdamW(discriminator.parameters(), lr=1e-4)

    # Train the GAN
    # pre_train_generator(generator, ball, optimizer_g, loader, epochs=1, batch_size=batch_size)
    train(
        generator,
        discriminator,
        ball,
        optimizer_g,
        optimizer_d,
        loader,
        epochs=50,
        batch_size=batch_size,
    )


if __name__ == "__main__":
    main()
