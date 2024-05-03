import random
from copy import deepcopy
from torch_utils import misc


def get_random_mask(size, mask_ratio):
    return [random.random() > mask_ratio for i in range(size)]


def Mask_the_Generator(model_dict, net_mask_list):
    '''
    Usage:
        Produce a pruned generator dictionary based on a mask list
    
    Args:
        model_dict:    (dict) of the state of the generator
        net_mask_list: (list) of layer_mask which is a (list) of (bool) indicating each channel's KEPT/PRUNED 
    '''
    
    # Getting the styled_conv key to be masked
    styled_affine_key = []
    for key in model_dict.keys():
        if ('affine' in key) and not ('input' in key):
            styled_affine_key.append(key)
    styled_conv_key = []
    for key in model_dict.keys():
        if ('weight' in key or 'bias' in key) and not ('affine' in key or 'mapping' in key or 'input' in key):
            styled_conv_key.append(key)
    print(styled_conv_key)
    
    # The dictionary of the final pruned model
    pruned_dict = deepcopy(model_dict)
    
    # Masking operation
    pruned_dict['synthesis.input.weight'] = model_dict['synthesis.input.weight'][net_mask_list[0], ...]
    Mask_Styled_Conv_Key(model_dict, pruned_dict, net_mask_list, styled_conv_key)
    Mask_Styled_Aff_Key(model_dict, pruned_dict, net_mask_list, styled_affine_key)
    
    return pruned_dict


def Mask_Styled_Conv_Key(model_dict, pruned_dict, net_mask_list, styled_conv_key):
    '''
    Usage:
        Update the conv weights of styled convolution
    
    Args:
        model_dict:      (dict) of the original model state
        pruned_dict:     (dict) of the pruned model state
        net_mask_list:   (list) of layer_mask which is a (list) of (bool) indicating each channel's KEPT/PRUNED 
        styled_conv_key: (list) of key for the styled conv kernel
    '''
    NUM_KEY_EACH_LAYER = 2
    max_mask = len(styled_conv_key) // NUM_KEY_EACH_LAYER
    for idx in range(len(styled_conv_key) // NUM_KEY_EACH_LAYER):
        input_layer_mask = net_mask_list[idx]
        if idx + 1 == max_mask:
            weight_key = styled_conv_key[idx * 2]
            pruned_dict[weight_key] = model_dict[weight_key].cpu()[:, input_layer_mask, ...]
        else:
            output_layer_mask = net_mask_list[idx+1]
            weight_key, bias_key = styled_conv_key[idx * 2], styled_conv_key[idx * 2 + 1]
            pruned_dict[weight_key] = model_dict[weight_key].cpu()[:, input_layer_mask, ...][output_layer_mask, ...]
            pruned_dict[bias_key]   = model_dict[bias_key].cpu()[output_layer_mask]
    # print(pruned_dict['synthesis.L14_512_3.weight'].shape)
    # print(pruned_dict['synthesis.L14_512_3.bias'].shape)


def Mask_Styled_Aff_Key(model_dict, pruned_dict, net_mask_list, styled_affine_key):
    '''
    Usage:
        Update the weights of the affine transformation in the styled convolution
    
    Args:
        model_dict:     (dict) of the original model state
        pruned_dict:    (dict) of the pruned model state
        net_mask_list:  (list) of layer_mask which is a (list) of (bool) indicating each channel's KEPT/PRUNED 
        styled_affine_key: (list) of key for the affine transformation to get styles 
    '''
    NUM_KEY_EACH_LAYER = 2
    for idx in range(len(styled_affine_key) // NUM_KEY_EACH_LAYER):
        layer_mask = net_mask_list[idx]
        weight_key, bias_key = styled_affine_key[idx * 2], styled_affine_key[idx * 2 + 1]
        pruned_dict[weight_key] = model_dict[weight_key].cpu()[layer_mask, ...]
        pruned_dict[bias_key]   = model_dict[bias_key].cpu()[layer_mask]

