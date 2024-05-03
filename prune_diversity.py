# Content-aware pruning on a single GPU
# Author: Yuchen Liu

import os

import click
import dnnlib
import numpy as np
import torch

import legacy

from pruning_util.diversity_aware_pruning import get_diversity_pruning_score
from pruning_util.network_util import Build_Generator_From_Dict, Get_Network_Shape
from pruning_util.pruning_util import Get_Uniform_RmveList, Generate_Prune_Mask_List
from pruning_util.mask_util import Mask_the_Generator

device = 'cuda:0'

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR')
@click.option('--n_sample', help='Where to save the output images', type=click.IntRange(min=1), default=5000)
@click.option('--batch_size', help='Where to save the output images', type=click.IntRange(min=1), default=5)
@click.option('--pruning_ratio', help='Where to save the output images', type=click.FloatRange(min=0.1, max=0.9), default=0.7)
@click.option('--edit_strength', help='Where to save the output images', type=click.FloatRange(min=0.1, max=20.0), default=10.0)
@click.option('--n_direction', help='Where to save the output images', type=click.IntRange(min=1), default=10)
def pruning_diversity(
    network_pkl: str,
    outdir: str,
    n_sample: int,
    batch_size: int,
    pruning_ratio: float,
    edit_strength: float,
    n_direction: int
):

    print('Loading networks from "%s"...' % network_pkl)
    with dnnlib.util.open_url(network_pkl) as f:
        teacher_pkl = legacy.load_network_pkl(f) # type: ignore
    G = teacher_pkl['G_ema'].to(device)

    os.makedirs(outdir, exist_ok=True)

    # Labels.
    label = torch.zeros([1, G.c_dim], device=device)
    if G.c_dim != 0:
        raise click.ClickException('pruning not check in cGAn')
    grad_score_list = get_diversity_pruning_score(
        g=G,
        n_sample=n_sample,
        batch_size=batch_size,
        device=device,
        noise_path=None,
        edit_strength=edit_strength,
        n_direction=n_direction,
    )
    grad_score_array = np.array([np.array(grad_score) for grad_score in grad_score_list])
    content_aware_pruning_score = np.sum(grad_score_array, axis=0)

    # Generator Pruning
    net_shape = Get_Network_Shape(G)
    rmve_list = Get_Uniform_RmveList(net_shape, pruning_ratio)
    prune_net_mask = Generate_Prune_Mask_List(content_aware_pruning_score, net_shape, rmve_list)

    pruned_generator_dict = Mask_the_Generator(G.state_dict(), prune_net_mask)

    ckpt_file = os.path.join(outdir, f'div_pruned_{"{:.2f}".format(pruning_ratio)}.pth')
    torch.save(pruned_generator_dict, ckpt_file)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    pruning_diversity() # pylint: disable=no-value-for-parameter
#----------------------------------------------------------------------------