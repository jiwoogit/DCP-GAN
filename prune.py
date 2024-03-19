# Diversity-aware pruning on a single GPU

import torch
import numpy as np
import time
import datetime
import argparse
import os

from Util.mask_util import Mask_the_Generator
from Util.diversity_aware_pruning import Get_Diversity_Pruning_Score
from Util.network_util import Build_Generator_From_Dict, Get_Network_Shape
from Util.pruning_util import Get_Uniform_RmveList, Generate_Prune_Mask_List

device = 'cuda:0'

# Parameter Parsing
parser = argparse.ArgumentParser()

parser.add_argument('--generated_img_size', type=int, default=256)
parser.add_argument('--ckpt', type=str, default='''./Model/teacher_model/256px_full_size.pt''')
parser.add_argument('--n_sample', type=int, default=5000)
parser.add_argument('--batch_size', type=int, default=5)
parser.add_argument('--remove_ratio', type=float, default=0.7)
parser.add_argument('--edit_strength', type=float, default=5.0)
parser.add_argument('--n_direction', type=int, default=10)
parser.add_argument('--truncation_trick', type=float, default=1.0)
parser.add_argument('--noise_path', type=str)
args = parser.parse_args()


# Generator Loading
model_dict = torch.load(args.ckpt, map_location=device)
g_ema = Build_Generator_From_Dict(model_dict['g_ema'], size=args.generated_img_size).to(device)

# Generator Scoring
start_time = time.time()
grad_score_list = Get_Diversity_Pruning_Score(generator=g_ema, 
                                                  n_sample=args.n_sample, 
                                                  batch_size=args.batch_size, 
                                                  edit_strength=args.edit_strength,
                                                  n_direction=args.n_direction,
                                                  truncation_trick=args.truncation_trick,
                                                  noise_path=args.noise_path,
                                                  device=device)

grad_score_array = np.array([np.array(grad_score) for grad_score in grad_score_list])
diversity_aware_pruning_score = np.sum(grad_score_array, axis=0)

end_time = time.time()

print('The diverstiy-aware metric scoring takes: ' + str(round(end_time - start_time, 4)) + ' seconds')

# Generator Pruning
net_shape = Get_Network_Shape(g_ema.state_dict())
rmve_list = Get_Uniform_RmveList(net_shape, args.remove_ratio)
prune_net_mask = Generate_Prune_Mask_List(diversity_aware_pruning_score, net_shape, rmve_list, False)

pruned_generator_dict = Mask_the_Generator(g_ema.state_dict(), prune_net_mask)

pruned_ckpt = {'g': pruned_generator_dict, 'g_ema': pruned_generator_dict}
m_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

ckpt_dir = './Model/pruned_model'
os.makedirs(ckpt_dir, exist_ok=True)
ckpt_file = f'dcp_{str(args.remove_ratio)}_{str(args.generated_img_size)}px_a{str(args.edit_strength)}_n{str(int(args.n_direction))}_t{"{:.2f}".format(args.truncation_trick)}_model.pt'
ckpt_file = os.path.join(ckpt_dir, ckpt_file)
torch.save(pruned_ckpt, ckpt_file)
