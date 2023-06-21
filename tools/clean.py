import torch
import ipdb


def replace_keys(old_dict):
    list_oldkey = old_dict.keys()
    new_dict = {}
    for key in list_oldkey:
        newkey = key.replace('encoder.', '', 1)
        new_dict[newkey] = old_dict[key]

    return new_dict


def main():
    old_file = 'OUTPUT/HSAR_linear_tiny_finetune_ssv2_atten/epoch_200.pth'
    old_checkpoint = torch.load(old_file)
    ipdb.set_trace()

    Act_checkpoint['state_dict'] = replace_keys(Act_checkpoint['state_dict'])
    self.temporal_enc.load_state_dict(Act_checkpoint['state_dict'], strict=False)
    self.spactial_enc.load_state_dict(Vit_checkpoint['state_dict'], strict=False)
    print("AcT Loaded: ", self.temporal_enc.to_patch_embedding[1].weight[12][20])
    print("ViT Loaded: ", self.spactial_enc.to_patch_embedding[1].weight[12][20])


if __name__ == '__main__':
    main()
