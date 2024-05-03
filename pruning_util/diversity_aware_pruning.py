import torch
from torch import nn
from torchvision import utils, transforms
import torch.nn.functional as F

import numpy as np
from PIL import Image

from pruning_util.face_parsing.BiSeNet import BiSeNet

from .network_util import Convert_Tensor_To_Image
from .estimator import get_estimator
from pathlib import Path
from tqdm import tqdm
file_path = Path(__file__).parent

def Get_Parsing_Net(device):
    '''
    Usage:
        Obtain the network for parsing and its preprocess method
    '''
    
    PRETRAINED_FILE = (file_path / '''./face_parsing/pretrained_model/79999_iter.pth''').resolve()
    
    n_classes = 19
    parsing_net = BiSeNet(n_classes=n_classes).to(device)
    pretrained_weight = torch.load(PRETRAINED_FILE, map_location=device)
    parsing_net.load_state_dict(pretrained_weight)
    parsing_net.eval();

    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    return parsing_net, to_tensor


def Extract_Face_Mask(pil_image, parsing_net, to_tensor, device):
    '''
    Usage:
        Extract the face foreground from an pil image
        
    Args:
        pil_image:   (PIL.Image) a single image
        parsing_net: (nn.Module) the network to parse the face images
        to_tensor:   (torchvision.transforms) the image transformation function
        device:      (str) device to place the networks
    '''
    
    with torch.no_grad():
        image = pil_image.resize((512, 512), Image.BILINEAR)
        img = to_tensor(image)
        img = torch.unsqueeze(img, 0)
        img = img.to(device)
        out = parsing_net(img)[0]
        parsing = out.squeeze(0).cpu().numpy().argmax(0)
    
    return parsing


def Batch_Img_Parsing(img_tensor, parsing_net, device):
    '''
    Usage:
        Parse the image tensor in a batch format
    
    Args:
        img_tensor:  (torch.Tensor) of the image tensor generated from generator in format of [N, C, H, W]
        parsing_net: (nn.Module) of the deep network for parsing
    '''
    CHANNEL_MEAN = [0.485, 0.456, 0.406]
    CHANNEL_STD = [0.229, 0.224, 0.225]
    PARSING_SIZE = 512
    
    scale_factor = PARSING_SIZE / img_tensor.shape[-1]
    transformed_tensor = ((img_tensor + 1 ) / 2).clamp(0,1) # Rescale tensor to [0,1]
    transformed_tensor = F.interpolate(transformed_tensor, 
                                       scale_factor=scale_factor, 
                                       mode='bilinear', 
                                       align_corners=False) # Scale to 512
    for i in range(transformed_tensor.shape[1]):
        transformed_tensor[:,i,...] = (transformed_tensor[:,i,...] - CHANNEL_MEAN[i]) / CHANNEL_STD[i]
        
    transformed_tensor = transformed_tensor.to(device)
    with torch.no_grad():
        img_parsing = parsing_net(transformed_tensor)[0]
    
    parsing = img_parsing.argmax(1)
    return parsing

def Get_Masked_Tensor(img_tensor, batch_parsing, device, mask_grad=False):
    '''
    Usage:
        To produce the masked img_tensor in a differentiable way
    
    Args:
        img_tensor:    (torch.Tensor) generated 4D tensor of shape [N,C,H,W]
        batch_parsing: (torch.Tensor) the parsing result from SeNet of shape [N,512,512] (the net fixed the parsing to be 512)
        device:        (str) the device to place the tensor
        mask_grad:     (bool) whether requires gradient to flow to the masked tensor or not
    '''
    PARSING_SIZE = 512
    
    mask = (batch_parsing > 0) * (batch_parsing != 16) 
    mask_float = mask.unsqueeze(0).type(torch.FloatTensor) # Make it to a 4D tensor with float for interpolation
    scale_factor = img_tensor.shape[-1] / PARSING_SIZE
    
    resized_mask = F.interpolate(mask_float, scale_factor=scale_factor, mode='bilinear', align_corners=False)
    resized_mask = (resized_mask.squeeze() > 0.5).type(torch.FloatTensor).to(device)
    
    if mask_grad:
        resized_mask.requires_grad = True
    
    masked_img_tensor = torch.zeros_like(img_tensor).to(device)
    for i in range(img_tensor.shape[0]):
        masked_img_tensor[i] = img_tensor[i] * resized_mask[i]
    
    return masked_img_tensor



def vis_parsing_maps(im, parsing_anno, stride):
    import cv2
    # Colors for all 20 parts
    part_colors = [[255, 0, 0], [255, 85, 0], [255, 170, 0],
                   [255, 0, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0],
                   [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255],
                   [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170],
                   [255, 0, 255], [255, 85, 255], [255, 170, 255],
                   [0, 255, 255], [85, 255, 255], [170, 255, 255]]

    im = np.array(im)
    vis_im = im.copy().astype(np.uint8)
    vis_parsing_anno = parsing_anno.copy().astype(np.uint8)
    vis_parsing_anno = cv2.resize(vis_parsing_anno, None, fx=stride, fy=stride, interpolation=cv2.INTER_NEAREST)
    vis_parsing_anno_color = np.zeros((vis_parsing_anno.shape[0], vis_parsing_anno.shape[1], 3)) + 255

    num_of_class = np.max(vis_parsing_anno)

    for pi in range(1, num_of_class + 1):
        index = np.where(vis_parsing_anno == pi)
        vis_parsing_anno_color[index[0], index[1], :] = part_colors[pi]

    vis_parsing_anno_color = vis_parsing_anno_color.astype(np.uint8)
    # print(vis_parsing_anno_color.shape, vis_im.shape)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, vis_parsing_anno_color, 0.6, 0)
    return vis_im


def Get_Salt_Pepper_Noisy_Image(img_tensor, mask, prob):
    '''
    Usage:
        Obtain the salt & pepper noisy image 
    
    Args:
        img_tensor: (torch.Tensor) a single generated image from the model
        mask:       (np.array) of type (bool) indicating the fore-/back-ground of the image
        prob:       (float) the probability of salt and pepper noise to appear in foreground
    '''
    img_size = img_tensor.shape[-1]
    salt_pepper_noise = np.random.randint(low=0, high=2,size=(img_size,img_size)) * 2 - 1 # s/p noise will be -1 and 1
        
    noisy_img = img_tensor.clone()
    for h in range(img_size):
        for w in range(img_size):
            if mask[h,w] == True and (np.random.random() < prob):
                noisy_img[:,:,h,w] = salt_pepper_noise[h,w]
    
    return noisy_img


def Get_Weight_Gradient(noisy_img, img_tensor, generator):
    '''
    Usage:
        Obtain the gradients of all filters' weights in the feed-forward path
    
    Args:
        noisy_img:  (torch.Tensor) of the noisy image
        img_tensor: (torch.Tensor) of the original generated image
        generator:  (nn.Module) of the generator
    '''
    loss = torch.sum(torch.abs(noisy_img - img_tensor))
    loss.backward()
    
    layer_names = generator.synthesis.layer_names
    
    # module_list = [generator.synthesis.input] + \
    module_list = [getattr(generator.synthesis, layer_name) for layer_name in layer_names]
    # for i, m in enumerate(module_list):
    #     print(layer_names[i])
    #     print(m.weight.shape)
    grad_list = [module.weight.grad for module in module_list]
    # for i in range(len(layer_names)):
    #     print(layer_names[i], grad_list[i].shape)
    
    grad_score_list = [(torch.mean(torch.abs(grad), axis=[0,2,3])).cpu().numpy() for grad in grad_list]
    # grad_score_list = [(torch.mean(grad, axis=[0,2,3])).cpu().numpy() for grad in grad_list]
    return grad_score_list

def get_diversity_pruning_score(g, n_sample, batch_size, device, \
    edit_strength, n_direction, noise_path=None, info_print=False):
    '''
    Usage:
        Obtain the network score
    
    Args:
        g:             (Module) of a generator
        n_sample:      (int) of the number of samples for estimation
        batch_size:    (int) of the size of the batch
        device:        (str) the device to place for the operations
        edit_strength: (float) of the strength of the perturbations
        n_direction:   (int) of the number of perturbation latent vectors.
        noise_path:    (str) the path of the z (reproduce result)
    '''
    # noise and batch setup
    LATENT_DIM = 512 
    n_components = n_direction
    alpha = edit_strength
    n_batch = n_sample // batch_size
    batch_size_list = [batch_size] * (n_batch - 1) + [batch_size + n_sample % batch_size]
    grad_score_list = []
    transformer = get_estimator('pca', LATENT_DIM, None)


    noise_z = torch.randn(10000, LATENT_DIM).to(device)
    latents = g.mapping(noise_z, None)[:, 0, :]
    transformer.fit(latents.detach().cpu().numpy())
    comp, stddev, var_ratio = transformer.get_components()
    comp /= np.linalg.norm(comp, axis=1, keepdims=True)
    comp = torch.from_numpy(comp).to(device)
    num_ws = g.synthesis.num_ws
    print('total var sum: ',  sum(var_ratio))
    if noise_path is not None:
        noise_z_load = torch.load(noise_path).to(device)
    
    for (idx,batch) in enumerate(tqdm(batch_size_list)):
        if info_print:
            print('Processing Batch: ' + str(idx))
        noise_z = torch.randn(batch, LATENT_DIM).to(device)
        if noise_path is not None:
            noise_z = noise_z_load[idx*batch_size:idx*batch_size+batch]

        grad_score = []
        comp_list = np.random.choice(512, n_components, p=var_ratio)
        comp_list = comp[comp_list]

        for i in range(n_components):
            latents = g.mapping(noise_z, None)[:, 0, :]
            direction = comp_list[i].unsqueeze(0).repeat(batch, 1)
            latents_pca = latents.clone().detach() + (direction * alpha)
            input_ws_pca = latents_pca.unsqueeze(1).repeat(1, num_ws, 1)
            input_ws = latents.unsqueeze(1).repeat(1, num_ws, 1)

            # stop grad
            g.requires_grad_(False)
            img_pca = g.synthesis(input_ws_pca)
            g.requires_grad_(True)
            img = g.synthesis(input_ws)

            grad_score_1 = Get_Weight_Gradient(img_pca, img, g)
            g.zero_grad()

            grad_score.append(grad_score_1)

        n_layer = len(grad_score[0])
        mean_grad = []
        for n in range(n_layer):
            all_other_grad = []
            for i in range(len(grad_score)):
                all_other_grad.append(grad_score[i][n])
            all_other_grad = np.stack(all_other_grad, axis=0).mean(axis=0)
            mean_grad.append(all_other_grad)

        for i in range(len(grad_score)):
            tmp = []
            for n in range(len(grad_score[0])):
                tmp.append((grad_score[i][n] - mean_grad[n])**2)
            grad_score_list.append(tmp)
        
    return grad_score_list