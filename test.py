import config
from utils import load_checkpoint
from datasets import X_Y_Dataset
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import tqdm as tqdm
from generator import Generator

def test_fn(loader, gen_X, gen_Y):
    loop = tqdm(loader, leave=True)

    for idx, (x_img, y_img) in enumerate(loop):
        x_img = x_img.to(config.DEVICE)
        y_img = y_img.to(config.DEVICE)
        with torch.cuda.amp.autocast():
            trans_y = gen_Y(x_img)
            trans_x = gen_X(y_img)

            save_image(trans_y * 0.5 + 0.5, f"test_results/y_{idx}.png")
            save_image(trans_x * 0.5 + 0.5, f"test_images/x_{idx}.png")


def main():
    gen_X = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    gen_Y = Generator(img_channels=3, num_residuals=9).to(config.DEVICE)
    opt_gen = optim.Adam(
        list(gen_X.parameters()) + list(gen_Y.parameters()),
        lr=config.LEARNING_RATE,
        betas=(0.5, 0.999),
    )
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

    test_fn(val_loader, gen_X, gen_Y)






    