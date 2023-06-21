# HSAR_repo

Implementation for Less is More: Decoupled High-Semantic Encoding for Action Recognition (ACM ICMR 2023) 

[[paper]](https://doi.org/10.1145/3591106.3592233)

## Abstract
This paper focuses on how to improve the efficiency of the action recognition framework by optimizing its complicated feature extraction pipelines and enhancing explainability, benefiting future adaptation to more complex visual understanding tasks (e.g. video captioning). To achieve this task, we propose a novel decoupled two-stream framework for action recognition - HSAR, which utilizes high-semantic features for increased efficiency and provides well-founded explanations in terms of spatial-temporal perceptions that will benefit further expansions on visual understanding tasks. The inputs are decoupled into spatial and temporal streams with designated encoders aiming to extract only the pinnacle of representations, gaining high-semantic features while reducing computation costs greatly. A lightweight Temporal Motion Transformer (TMT) module is proposed for globally modeling temporal features through self-attention, omitting redundant spatial features. Decoupled spatial-temporal embeddings are further merged dynamically by an attention fusion model to form a joint high-semantic representation. The visualization of the attention in each module offers intuitive interpretations of HSAR’s explainability. Extensive experiments on three widely-used benchmarks (Kinetics400, 600, and Sthv2) show that our framework achieves high prediction accuracy with significantly reduced computation (only 64.07 GFLOPs per clip), offering a great trade-off between accuracy and computational costs.

![framework](https://github.com/HVG808/HSAR/blob/main/HSAR_whole.png)

**This repo supports:**
* Action Recognition on videos datasets including kinetics 400, kinetics 600, and somethingsomethingV2.

## Preparation
Environment: Linux, cudatoolkit >= 11.3.1, Python>=3.8, PyTorch>=1.10.2

1. Clone the repo

2. Create vitual environment by conda
```bash
conda create -n HSAR python=3.8
codna activate HSAR
pip install -r requirement.txt
```

3. Setup mmaction2 toolbox following their [instruction](https://mmaction2.readthedocs.io/en/latest/get_started/installation.html)

## Training and Validation

### Download Video

Please download Raw videos from cooresponding datasets. You can also follow instructions [here](https://github.com/open-mmlab/mmaction2/blob/main/tools/data/kinetics/README.md)

### Download Pre-trained weights from iBot and XCLIP
To initialize weights in HSAR, please download the pre-trained model from [iBot](https://github.com/bytedance/ibot). We chose the ViT-B/16 option.

Also remember to download weights pre-trained by [XCLIP](https://github.com/microsoft/VideoX/tree/master/X-CLIP). We chose the X-CLIP-B/16 option.

### Load pre-trained HSAR model
HSAR-Base [K400](https://drive.google.com/file/d/19fHYKMGaaSDPfwQ5qVU49MRprYMDn7ZA/view?usp=sharing) 
          [K600](https://drive.google.com/file/d/11soN45asc0hp70avBBdBWODBh3rH2b7c/view?usp=sharing) 
          [Sthv2](https://drive.google.com/file/d/1T8ct-uJ0SrP4z63HfFmoisHWy9YtixFC/view?usp=sharing)

HSAR-Large [K400](https://drive.google.com/file/d/1zMIr90WYnER8DtExOiUWzzVLsBI59u4K/view?usp=sharing)


### Action Recognition
```
# Training
python tools/train.py configs/recognition/HS/k400/HSAR_Base_K400.py --gpus 1

# Evaluation
python tools/test.py configs/recognition/HS/k400/HSAR_Base_K400_test.py saves/HSAR_Base_kinetics400.pth --eval 'top-k accuracy'
```

### Visualizting Attention Map
Please refer to [code](https://github.com/bytedance/ibot/blob/main/analysis/visualize_attn_map.sh) published by iBot.

## Citation
If you find our repo useful, please cite
```
@inproceedings{10.1145/3591106.3592233,
author = {Zhang, Chun and Ren, Keyan and Bian, Qingyun and Shi, Yu},
title = {Less is More: Decoupled High-Semantic Encoding for Action Recognition},
year = {2023},
isbn = {9798400701788},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3591106.3592233},
doi = {10.1145/3591106.3592233},
booktitle = {Proceedings of the 2023 ACM International Conference on Multimedia Retrieval},
pages = {262–271},
numpages = {10},
keywords = {Computer Vision, High-Semantics, Decoupled Feature Extraction, Machine Learning, Action Recognition, Explainable AI},
location = {Thessaloniki, Greece},
series = {ICMR '23}
}
```

## Acknowledgement

HSAR is implemented with [MMaction2](https://github.com/open-mmlab/mmaction2) toolbox. 
We thank the authors for their efforts.
