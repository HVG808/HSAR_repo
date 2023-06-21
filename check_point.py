import torch
import ipdb


def replace_keys(old_dict):
    list_oldkey = old_dict.keys()
    new_dict = {}
    for key in list_oldkey:
        newkey = key.replace('encoder.', '', 1)
        new_dict[newkey] = old_dict[key]

    return new_dict

checkpoint1 = 'saves/Vit_sthv2_v7.ckpt'
checkpoint2 = 'saves/checkpoint_teacher.pth'
checkpoint3 = 'saves/timesformer_divST_8x32x1_15e_kinetics400_rgb-3f8e5d03.pth'

Vit_checkpoint = torch.load(checkpoint1)
iBot_checkpoint = torch.load(checkpoint2)
times_checkpoint = torch.load(checkpoint3)
Vit_checkpoint['state_dict'] = replace_keys(Vit_checkpoint['state_dict'])

ipdb.set_trace()



