import argparse
import pickle as pkl
import time
import os
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from model import Generator
from Util.network_util import Build_Generator_From_Dict
from torchvision import utils

def generated_samples(g, ckpt_name, batch_size, n_sample, device, args):
    if args.truncation < 0.9999:
        mean_noise = torch.randn(10000, 512).to(device)
        g_mean = g.style(mean_noise).mean(dim=0)

    base_path = "generated_samples"
    sample_path = os.path.basename(os.path.splitext(ckpt_name)[0])
    sample_path = os.path.join(base_path, sample_path)
    os.makedirs(sample_path, exist_ok=True)
    with torch.no_grad():
        n_batch = n_sample // batch_size
        resid = n_sample - (n_batch - 1) * batch_size
        batch_sizes = [batch_size] * (n_batch - 1) + [resid]
        logits = []

        for idx, batch in enumerate(tqdm(batch_sizes)):
            latent = torch.randn(batch, 512, device=device)
            if args.truncation > 0.9999:
                img = g([latent])
            else:
                img = g([latent], truncation=args.truncation, truncation_latent=g_mean)
            # save_image
            for i in range(batch):
                utils.save_image(
                    img[i],
                    os.path.join(sample_path, f'{str(idx * batch_size + i).zfill(5)}.png'),
                    normalize=True,
                    range=(-1, 1),
                )

if __name__ == '__main__':
    device = 'cuda'

    parser = argparse.ArgumentParser()
    parser.add_argument('--batch', type=int, default=1)
    parser.add_argument('--n_sample', type=int, default=100)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--ckpt', type=str, default="./")
    parser.add_argument('--truncation', type=float, default=1.)

    args = parser.parse_args()
    ckpt = torch.load(args.ckpt, map_location=device)
    g = Build_Generator_From_Dict(ckpt['g_ema'], size=args.size).to(device)

    generated_samples(
        g, args.ckpt, args.batch, args.n_sample, device, args
    )