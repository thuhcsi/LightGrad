# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from LightGrad import LightGrad
from utils import plot_tensor, save_plot
import yaml
import argparse
import random
import pathlib

from LightGrad.dataset import Dataset, collateFn


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, help="path to config file")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    print("Initializing data loaders...")
    with open(args.config) as f:
        config = yaml.load(f, yaml.SafeLoader)
    log_dir = pathlib.Path(config["log_dir"])
    torch.manual_seed(config["random_seed"])
    np.random.seed(config["random_seed"])
    train_dataset = Dataset(
        config["train_datalist_path"],
        config["phn2id_path"],
        config["sample_rate"],
        config["n_fft"],
        config["n_mels"],
        config["f_min"],
        config["f_max"],
        config["hop_size"],
        config["win_size"],
    )
    val_dataset = Dataset(
        config["valid_datalist_path"],
        config["phn2id_path"],
        config["sample_rate"],
        config["n_fft"],
        config["n_mels"],
        config["f_min"],
        config["f_max"],
        config["hop_size"],
        config["win_size"],
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        collate_fn=collateFn,
        num_workers=16,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config["batch_size"], shuffle=True, collate_fn=collateFn
    )

    print("Initializing model...")
    model = LightGrad.build_model(config, train_dataset.get_vocab_size())
    print(f"Total parameters: {model.nparams}")
    start_epoch = 1
    start_steps = 1
    if config["ckpt"]:
        print("loading ", config["ckpt"])
        epoch, steps, state_dict = torch.load(config["ckpt"], map_location="cpu")
        start_epoch = epoch + 1
        start_steps = steps + 1
        model.load_state_dict(state_dict)

    model = model.cuda()

    print("Initializing optimizer...")
    optimizer = torch.optim.Adam(params=model.parameters(), lr=config["learning_rate"])

    print("Initializing logger...")
    logger = SummaryWriter(log_dir=log_dir)

    ckpt_dir = log_dir / "ckpt"
    pic_dir = log_dir / "pic"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pic_dir.mkdir(parents=True, exist_ok=True)
    print("Start training...")
    iteration = start_steps
    out_size = config["out_size"] * config["sample_rate"] // config["hop_size"]

    for epoch in range(start_epoch, start_epoch + 10000):
        model.train()
        dur_losses = []
        prior_losses = []
        diff_losses = []
        with tqdm(
            train_loader, total=len(train_dataset) // config["batch_size"]
        ) as progress_bar:
            for batch_idx, batch in enumerate(progress_bar):
                model.zero_grad()
                x, x_lengths = batch["x"].cuda(), batch["x_lengths"].cuda()
                y, y_lengths = batch["y"].cuda(), batch["y_lengths"].cuda()
                dur_loss, prior_loss, diff_loss = model.compute_loss(
                    x, x_lengths, y, y_lengths, out_size=out_size
                )
                loss = sum([dur_loss, prior_loss, diff_loss])
                loss.backward()

                enc_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.encoder.parameters(), max_norm=1
                )
                dec_grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.decoder.parameters(), max_norm=1
                )
                optimizer.step()

                logger.add_scalar(
                    "training/duration_loss", dur_loss.item(), global_step=iteration
                )
                logger.add_scalar(
                    "training/prior_loss", prior_loss.item(), global_step=iteration
                )
                logger.add_scalar(
                    "training/diffusion_loss", diff_loss.item(), global_step=iteration
                )
                logger.add_scalar(
                    "training/encoder_grad_norm", enc_grad_norm, global_step=iteration
                )
                logger.add_scalar(
                    "training/decoder_grad_norm", dec_grad_norm, global_step=iteration
                )

                dur_losses.append(dur_loss.item())
                prior_losses.append(prior_loss.item())
                diff_losses.append(diff_loss.item())

                if batch_idx % 5 == 0:
                    msg = (
                        f"LightGrad Epoch: {epoch}, iteration: {iteration} | "
                        f" dur_loss: {dur_loss.item()}, "
                        f"prior_loss: {prior_loss.item()}, "
                        f"diff_loss: {diff_loss.item()}"
                    )
                    progress_bar.set_description(msg)

                iteration += 1
                if iteration >= config["max_step"]:
                    torch.save(
                        [epoch, iteration, model.state_dict()],
                        f=ckpt_dir / f"LightGrad_{epoch}_{iteration}.pt",
                    )
        model.eval()
        with torch.no_grad():
            all_dur_loss = []
            all_prior_loss = []
            all_diffusion_loss = []
            for _, item in enumerate(val_loader):
                x, x_lengths = batch["x"].cuda(), batch["x_lengths"].cuda()
                y, y_lengths = batch["y"].cuda(), batch["y_lengths"].cuda()

                dur_loss, prior_loss, diff_loss = model.compute_loss(
                    x, x_lengths, y, y_lengths, out_size=out_size
                )
                loss = sum([dur_loss, prior_loss, diff_loss])
                all_dur_loss.append(dur_loss)
                all_prior_loss.append(prior_loss)
                all_diffusion_loss.append(diff_loss)
            average_dur_loss = sum(all_dur_loss) / len(all_dur_loss)
            average_prior_loss = sum(all_prior_loss) / len(all_prior_loss)
            average_diffusion_loss = sum(all_diffusion_loss) / len(all_diffusion_loss)
            logger.add_scalar("val/duration_loss", average_dur_loss, global_step=epoch)
            logger.add_scalar("val/prior_loss", average_prior_loss, global_step=epoch)
            logger.add_scalar(
                "val/diffusion_loss", average_diffusion_loss, global_step=epoch
            )
            print(
                f"val duration_loss: {average_dur_loss}, "
                f"prior_loss: {average_prior_loss}, "
                f"diffusion_loss: {average_diffusion_loss}"
            )
            y_enc, y_dec, attn = model(x, x_lengths, n_timesteps=10)
            idx = random.randrange(0, y_enc.shape[0])
            y_enc = y_enc[idx].cpu()
            y_dec = y_dec[idx].cpu()
            y = y[idx].cpu()
            attn = attn[idx][0].cpu()
            logger.add_image(
                "image/generated_enc",
                plot_tensor(y_enc),
                global_step=epoch,
                dataformats="HWC",
            )
            logger.add_image(
                "image/generated_dec",
                plot_tensor(y_dec),
                global_step=epoch,
                dataformats="HWC",
            )
            logger.add_image(
                "image/alignment",
                plot_tensor(attn),
                global_step=epoch,
                dataformats="HWC",
            )
            logger.add_image(
                "image/ground_truth",
                plot_tensor(y),
                global_step=epoch,
                dataformats="HWC",
            )
            save_plot(y_enc, pic_dir / f"generated_enc_{epoch}.png")
            save_plot(y_dec, pic_dir / f"generated_dec_{epoch}.png")
            save_plot(attn, pic_dir / f"alignment_{epoch}.png")
            save_plot(y, pic_dir / f"ground_truth_{epoch}.png")
        torch.save(
            [epoch, iteration, model.state_dict()],
            f=ckpt_dir / f"LightGrad_{epoch}_{iteration}.pt",
        )
