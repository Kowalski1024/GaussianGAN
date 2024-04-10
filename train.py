import numpy as np
import torch

from network.gan import ImageGenerator, Discriminator
from network.loss import loss, discriminator_accuracy, calculate_gradient_penalty

from utils import (
    generate_noise,
    save_image_grid,
    load_sphere,
    get_dataloader,
    get_dataset,
    get_optimizers,
)
import wandb
import hydra
from pathlib import Path


def shape_pretrain(generator, sphere, optimizer_g, loader, device, config):
    generator.train()
    B, P, _ = sphere.shape
    mse = torch.nn.L1Loss()

    loader_iter = iter(loader)

    for i in range(config.pretrain.ticks):
        sphere = sphere.to(device)
        noise = generate_noise(B, P).to(device)
        real_images, camera = next(loader_iter)
        real_images, camera = real_images.to(device), camera.to(device)

        fake_images = generator(noise, camera)

        optimizer_g.zero_grad()
        loss_g = mse(real_images, fake_images)
        loss_g.backward()
        optimizer_g.step()

        if i % 5 == 0:
            print(f"#{i}\{config.pretrain.ticks}, Loss G: {loss_g}")

        if i % 10 == 0:
            images = int(B**0.5)
            save_image_grid(
                fake_images[: images**2].detach().cpu().numpy(),
                f"{config.output_dir}/pretrain_{i}.png",
                drange=[-1, 1],
                grid_size=(images, images),
            )


def train_one_tick(
    generator,
    discriminator,
    sphere,
    optimizer_g,
    optimizer_d,
    loader,
    device,
    tick,
    config,
):
    generator.train()
    discriminator.train()
    B, P, _ = sphere.shape

    loader_iter = iter(loader)

    for i in range(config.train.batches_per_tick):
        sphere = sphere.to(device)

        for _ in range(config.train.discriminator_steps):
            noise = generate_noise(B, P).to(device)
            real_images, camera = next(loader_iter)
            real_images, camera = real_images.to(device), camera.to(device)

            with torch.no_grad():
                fake_images = generator(sphere, noise)

            optimizer_d.zero_grad()
            real_logits = discriminator(real_images)
            fake_logits = discriminator(fake_images)

            d_loss = loss(real_logits, fake_logits, "discriminator", **config.train.loss)

            if config.train.loss.type == "wgan-gp":
                gp = calculate_gradient_penalty(discriminator, real_images, fake_images, device)
                d_loss = d_loss + config.train.loss.lambda_gp * gp

            d_loss.backward()
            optimizer_d.step()

            if config.train.loss.type == "wgan":
                for p in discriminator.parameters():
                    p.data.clamp_(-config.train.weight_clip, config.train.weight_clip)

        noise = generate_noise(B, P).to(device)
        fake_images = generator(sphere, noise)

        fake_logits = discriminator(fake_images)
        optimizer_g.zero_grad()
        g_loss = loss(real_logits, fake_logits, "generator", **config.train.loss)
        g_loss.backward()
        optimizer_g.step()

        real_accuracy, fake_accuracy = discriminator_accuracy(real_logits, fake_logits)
        wandb.log(
            {
                "d_loss": d_loss.item(),
                "g_loss": g_loss.item(),
                "real_accuracy": real_accuracy,
                "fake_accuracy": fake_accuracy,
                "tick": tick,
                "iter": i,
            }
        )

        if i % 10 == 0:
            print(
                f"Tick {tick}, Iter {i}, D loss: {d_loss.item()}, G loss: {g_loss.item()}, Real acc: {real_accuracy}, Fake acc: {fake_accuracy}"
            )

    if not tick % config.train.save_images_every:
        images = int(B**0.5)
        save_image_grid(
            fake_images[: images**2].detach().cpu().numpy(),
            f"{config.output_dir}/output_{tick}.png",
            drange=[-1, 1],
            grid_size=(images, images),
        )


def train(
    generator, discriminator, optimizer_g, optimizer_d, sphere, loader, device, config
):
    real_images, _ = next(iter(loader))
    images = int(config.dataloader.batch_size**0.5)
    save_image_grid(
        real_images[: images**2].detach().cpu().numpy(),
        f"{config.output_dir}/real_images.png",
        drange=[-1, 1],
        grid_size=(images, images),
    )

    shape_pretrain(generator, sphere, optimizer_g, loader, device, config)

    # for tick in range(config.train.ticks):
    #     train_one_tick(
    #         generator,
    #         discriminator,
    #         sphere,
    #         optimizer_g,
    #         optimizer_d,
    #         loader,
    #         device,
    #         tick,
    #         config,
    #     )


@hydra.main(config_path="config", config_name="config.yaml", version_base="1.2")
def main(config):
    hydra_output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    )
    config.output_dir = hydra_output_dir
    device = torch.device(config.device)

    # Load the sphere
    sphere = load_sphere(config.sphere.path, config.sphere.points)
    sphere = torch.tensor(sphere, dtype=torch.float32).to(device)
    sphere = sphere.unsqueeze(0).repeat(config.dataloader.batch_size, 1, 1)

    # Create the data loader
    dataset = get_dataset(config.dataset.type, config.dataset.params)
    loader = get_dataloader(
        dataset,
        **config.dataloader,
    )

    # Create the generator and the discriminator
    generator = ImageGenerator().to(device)
    discriminator = Discriminator().to(device)

    # Create the optimizer
    optimizer_g = get_optimizers(generator, config.generator.optimizer)
    optimizer_d = get_optimizers(discriminator, config.discriminator.optimizer)

    wandb.init(project="3d-gan", mode="disabled")

    # Train the model
    train(
        generator,
        discriminator,
        optimizer_g,
        optimizer_d,
        sphere,
        loader,
        device,
        config,
    )


if __name__ == "__main__":
    main()
