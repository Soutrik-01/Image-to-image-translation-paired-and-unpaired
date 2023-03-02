

import torch
from datasets import X_Y_Dataset
import sys
from utils import save_checkpoint, load_checkpoint
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import config
from tqdm import tqdm
from torchvision.utils import save_image
from discriminator import Discriminator
from generator import Generator


def train_fn(
    disc_Y, disc_X, gen_X, gen_Y, loader, opt_disc, opt_gen, l1, mse, d_scaler, g_scaler
):
    Y_reals = 0
    Y_fakes = 0
    loop = tqdm(loader, leave=True)

    for idx, (x_img, y_img) in enumerate(loop):
        x_img = x_img.to(config.DEVICE)
        y_img = y_img.to(config.DEVICE)

        # Train Discriminators Y and X
        with torch.cuda.amp.autocast():
            fake_y = gen_Y(x_img)
            D_Y_real = disc_Y(y_img)
            D_Y_fake = disc_Y(fake_y.detach())
            Y_reals += D_Y_real.mean().item()
            Y_fakes += D_Y_fake.mean().item()
            D_Y_real_loss = mse(D_Y_real, torch.ones_like(D_Y_real))
            D_Y_fake_loss = mse(D_Y_fake, torch.zeros_like(D_Y_fake))
            D_Y_loss = D_Y_real_loss + D_Y_fake_loss

            fake_x = gen_X(y_img)
            D_X_real = disc_X(x_img)
            D_X_fake = disc_X(fake_x.detach())
            D_X_real_loss = mse(D_X_real, torch.ones_like(D_X_real))
            D_X_fake_loss = mse(D_X_fake, torch.Xeros_like(D_X_fake))
            D_X_loss = D_X_real_loss + D_X_fake_loss

            # put it togethor
            D_loss = (D_Y_loss + D_X_loss) / 2

        opt_disc.Xero_grad()
        d_scaler.scale(D_loss).backward()
        d_scaler.step(opt_disc)
        d_scaler.update()

        # Train Generators Y and X
        with torch.cuda.amp.autocast():
            # adversarial loss for both generators
            D_Y_fake = disc_Y(fake_y)
            D_X_fake = disc_X(fake_x)
            loss_G_Y = mse(D_Y_fake, torch.ones_like(D_Y_fake))
            loss_G_X = mse(D_X_fake, torch.ones_like(D_X_fake))

            # cycle loss
            cycle_x = gen_X(fake_y)
            cycle_y = gen_Y(fake_x)
            cycle_x_loss = l1(x_img, cycle_x)
            cycle_y_loss = l1(y_img, cycle_y)

            # identity loss (remove these for efficiency if you set lambda_identity=0)
            identity_x = gen_X(x_img)
            identity_y = gen_Y(y_img)
            identity_x_loss = l1(x_img, identity_x)
            identity_y_loss = l1(y_img, identity_y)

            # add all togethor
            G_loss = (
                loss_G_X
                + loss_G_Y
                + cycle_x_loss * config.LAMBDA_CYCLE
                + cycle_y_loss * config.LAMBDA_CYCLE
                + identity_y_loss * config.LAMBDA_IDENTITY
                + identity_x_loss * config.LAMBDA_IDENTITY
            )

        opt_gen.zero_grad()
        g_scaler.scale(G_loss).backward()
        g_scaler.step(opt_gen)
        g_scaler.update()

        if idx % 200 == 0:
            save_image(fake_y * 0.5 + 0.5, f"saved_images/y_{idx}.png")
            save_image(fake_x * 0.5 + 0.5, f"saved_images/x_{idx}.png")

        loop.set_postfix(Y_real=Y_reals / (idx + 1), Y_fake=Y_fakes / (idx + 1))


def main():
    disc_Y = Discriminator(in_channels=3).to(config.DEVICE)
    disc_X = Discriminator(in_channels=3).to(config.DEVICE)
    gen_X = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_Y = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_disc = optim.Adam(
        list(disc_Y.parameters()) + list(disc_X.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    opt_gen = optim.Adam(
        list(gen_X.parameters()) + list(gen_Y.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )

    L1 = nn.L1Loss()
    mse = nn.MSELoss()

    if config.LOAD_MODEL:
        load_checkpoint(
            config.CHECKPOINT_GEN_Y,
            gen_Y,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_GEN_X,
            gen_X,
            opt_gen,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_Y,
            disc_Y,
            opt_disc,
            config.LEARNING_RATE,
        )
        load_checkpoint(
            config.CHECKPOINT_CRITIC_X,
            disc_X,
            opt_disc,
            config.LEARNING_RATE,
        )

    dataset = X_Y_Dataset(
        root_Y=config.TRAIN_DIR + "/y",
        root_X=config.TRAIN_DIR + "/x",
        transform=config.transforms,
    )
    val_dataset = X_Y_Dataset(
        root_Y="cyclegan_test/y1",
        root_X="cyclegan_test/x1",
        transform=config.transforms,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.NUM_EPOCHS):
        train_fn(
            disc_Y,
            disc_X,
            gen_X,
            gen_Y,
            loader,
            opt_disc,
            opt_gen,
            L1,
            mse,
            d_scaler,
            g_scaler,
        )

        if config.SAVE_MODEL:
            save_checkpoint(gen_Y, opt_gen, filename=config.CHECKPOINT_GEN_Y)
            save_checkpoint(gen_X, opt_gen, filename=config.CHECKPOINT_GEN_X)
            save_checkpoint(disc_Y, opt_disc, filename=config.CHECKPOINT_CRITIC_Y)
            save_checkpoint(disc_X, opt_disc, filename=config.CHECKPOINT_CRITIC_X)


if __name__ == "__main__":
    main()
