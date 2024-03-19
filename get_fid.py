import torch
import torch.nn as nn
import time
import argparse

from Util.network_util import Build_Generator_From_Dict
from Evaluation.fid import Get_Model_FID_Score

device = 'cuda'
gpu_device_ids = [0]

# Arg Parsing

parser = argparse.ArgumentParser()

parser.add_argument('--generated_img_size', type=int, default=256)
parser.add_argument('--ckpt', type=str, default='''./Model/teacher_model/256px_full_size.pt''')
parser.add_argument('--n_sample', type=int, default=50000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--info_print', action='store_true', default=False)
parser.add_argument('--data_type', type=str, choices=['ffhq','church','car','horse'], default='ffhq')

args = parser.parse_args()


model_dict = torch.load(args.ckpt, map_location=device)
g_ema = Build_Generator_From_Dict(model_dict['g_ema'], size=args.generated_img_size).to(device)
g_ema = nn.DataParallel(g_ema, device_ids=gpu_device_ids)
g_ema.eval();


print('Number of samples: ' + str(args.n_sample))
fid = Get_Model_FID_Score(generator=g_ema, batch_size=args.batch_size, num_sample=args.n_sample, device=device, 
                              gpu_device_ids=gpu_device_ids, info_print=args.info_print, train_mode=args.data_type)
print('FID Scores: ' + str(fid) + '\n')
