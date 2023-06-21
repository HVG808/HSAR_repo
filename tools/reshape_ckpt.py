import torch
import ipdb

def replace_keys(self, old_dict):
    list_oldkey = old_dict.keys()
    new_dict = {}
    for key in list_oldkey:
        newkey = key.replace('visual.', '', 1)
        new_dict[newkey] = old_dict[key]

    return new_dict

Act_file = 'Act_sthv2_v7.ckpt'
x_clip_file = 'k400_32_16.pth'

Act_ckpt = torch.load(Act_file)
x_ckpt = torch.load(x_clip_file)

x_model_weight = x_ckpt['model']
Act_model_weight = Act_ckpt['state_dic']

ipdb.set_Trace()



