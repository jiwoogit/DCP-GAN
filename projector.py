import argparse
import math
import os

import torch
import lpips
from torch import optim
from torch.nn import functional as F
from PIL import Image
from tqdm import tqdm

from torchvision import transforms, utils
from Util.network_util import Build_Generator_From_Dict, Convert_Tensor_To_Image


def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss

def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()
        noise.data.add_(-mean).div_(std)

def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


if __name__ == "__main__":
    device = "cuda"

    parser = argparse.ArgumentParser(
        description="Image projector to the generator latent spaces"
    )
    parser.add_argument(
        "--ckpt", type=str, required=True, help="path to the model checkpoint"
    )
    parser.add_argument(
        "--size", type=int, default=256, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--batch", type=int, default=4, help="output image sizes of the generator"
    )
    parser.add_argument(
        "--lr_rampup",
        type=float,
        default=0.05,
        help="duration of the learning rate warmup",
    )
    parser.add_argument(
        "--lr_rampdown",
        type=float,
        default=0.25,
        help="duration of the learning rate decay",
    )
    parser.add_argument("--lr", type=float, default=0.1, help="learning rate")
    parser.add_argument(
        "--noise", type=float, default=0.05, help="strength of the noise level"
    )
    parser.add_argument(
        "--noise_ramp",
        type=float,
        default=0.75,
        help="duration of the noise level decay",
    )
    parser.add_argument("--step", type=int, default=1000, help="optimize iterations")
    parser.add_argument(
        "--noise_regularize",
        type=float,
        default=1e5,
        help="weight of the noise regularization",
    )
    parser.add_argument("--mse", type=float, default=1.0, help="weight of the mse loss")
    parser.add_argument(
        "--w_plus",
        action="store_true",
        help="allow to use distinct latent codes to each layers",
    )
    parser.add_argument(
        "--dir", type=str, help="path to image files to be projected"
    )

    args = parser.parse_args()

    n_mean_latent = 10000
    resize = min(args.size, 256)
    base_path = "projected_samples"
    sample_path = os.path.basename(os.path.splitext(args.ckpt)[0])
    sample_path = os.path.join(base_path, sample_path)
    os.makedirs(sample_path, exist_ok=True)

    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    model_dict = torch.load(args.ckpt, map_location='cpu')
    g_ema = Build_Generator_From_Dict(model_dict['g_ema'], size=args.size).to(device)
    g_ema.eval()


    img_list = os.listdir(args.dir)
    img_list.sort()
    batch_size = args.batch
    n_batch = int(len(img_list) / batch_size )
    resid = len(img_list) - batch_size * n_batch
    batch_size_list = [batch_size] * (n_batch)
    if resid > 0:
        batch_size_list += [resid]

    percept = lpips.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=device.startswith("cuda")
    )

    mse_loss_list = []
    p_loss_list = []
    for idx, batch_size in enumerate(batch_size_list):
        print(f'{idx}/{len(batch_size_list)}')
        imgs = []
        for imgfile in img_list[idx * batch_size: idx*batch_size + batch_size]:
            img = transform(Image.open(os.path.join(args.dir, imgfile)).convert("RGB"))
            imgs.append(img)

        imgs = torch.stack(imgs, 0).to(device)
        with torch.no_grad():
            noise_sample = torch.randn(n_mean_latent, 512, device=device)
            latent_out = g_ema.style(noise_sample)
            latent_mean = latent_out.mean(0)
            latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

        noises_single = g_ema.make_noise()
        noises = []
        for noise in noises_single:
            noises.append(noise.repeat(imgs.shape[0], 1, 1, 1).normal_())

        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(imgs.shape[0], 1)
        if args.w_plus:
            latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)

        latent_in.requires_grad = True

        for noise in noises:
            noise.requires_grad = True

        optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

        pbar = tqdm(range(args.step))
        latent_path = []

        for i in pbar:
            t = i / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]["lr"] = lr
            noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
            latent_n = latent_noise(latent_in, noise_strength.item())
            img_gen = g_ema(None, input_is_latent=True, latent_styles=[latent_n], noise=noises)

            batch, channel, height, width = img_gen.shape

            if height > 256:
                factor = height // 256

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])

            p_loss = percept(img_gen, imgs).sum()
            n_loss = noise_regularize(noises)
            mse_loss = F.mse_loss(img_gen, imgs)

            loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            noise_normalize_(noises)

            if (i + 1) % 100 == 0:
                latent_path.append(latent_in.detach().clone())

            pbar.set_description(
                (
                    f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                    f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
                )
            )

        with torch.no_grad():
            img_gen = g_ema(None, input_is_latent=True, latent_styles=[latent_path[-1]], noise=noises)
            if height > 256:
                factor = height // 256

                img_gen = img_gen.reshape(
                    batch, channel, height // factor, factor, width // factor, factor
                )
                img_gen = img_gen.mean([3, 5])
            mse_loss = F.mse_loss(img_gen, imgs)
            p_loss = percept(img_gen, imgs).mean()
            mse_loss_list.append(mse_loss)
            p_loss_list.append(p_loss)

            for g_idx in range(batch):
                utils.save_image(
                    img_gen[g_idx],
                    os.path.join(sample_path, f'{str(idx * args.batch + g_idx).zfill(5)}.png'),
                    normalize=True,
                    nrow=1,
                    range=(-1, 1),
                )

print(sum(mse_loss_list) / len(mse_loss_list))
print(sum(p_loss_list) / len(p_loss_list))
