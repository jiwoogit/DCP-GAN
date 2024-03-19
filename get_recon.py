from tqdm import tqdm
import torch
import torch.nn as nn
from PIL import Image
import numpy as np

import time
import argparse
from matplotlib import pyplot as plt

from Util.network_util import Build_Generator_From_Dict, Convert_Tensor_To_Image
import lpips
import os

device = 'cuda:0'
gpu_device_ids = [0]

# Arg Parsing

parser = argparse.ArgumentParser()

parser.add_argument('--generated_img_size', type=int, default=256)
parser.add_argument('--t', type=str, default='''./Model/teacher_model/256px_full_size.pt''')
parser.add_argument('--s', type=str, default='''./Model/student_model/dcp_ffhq256.pt''')
parser.add_argument('--batch', type=int, default=4)
parser.add_argument('--n_sample', type=int, default=10000)
parser.add_argument('--info_print', action='store_true', default=False)

args = parser.parse_args()

# Model Preparation
model_dict = torch.load(args.t, map_location=device)
teacher_g = Build_Generator_From_Dict(model_dict['g_ema'], size=args.generated_img_size).to(device)
teacher_g = nn.DataParallel(teacher_g, device_ids=gpu_device_ids)
teacher_g.eval();

model_dict = torch.load(args.s, map_location=device)
student_g = Build_Generator_From_Dict(model_dict['g_ema'], size=args.generated_img_size).to(device)
student_g = nn.DataParallel(student_g, device_ids=gpu_device_ids)
student_g.eval();

total_iter = args.n_sample // args.batch

mean = []
for i in tqdm(range(total_iter)):
    z = torch.randn(args.batch, 512).to(device)
    t_img = teacher_g([z])
    s_img = student_g([z])
    mean.append(torch.abs(t_img - s_img).mean().detach().cpu())

print(sum(mean) / total_iter)
    


