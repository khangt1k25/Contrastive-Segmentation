

## Requirements
requirements.txt
## Setup
```
git clone -b akien https://github.com/khangt1k25/Contrastive-Segmentation.git

cd Contrastive-Segmentation/
```
## Training

1. Change data path at data/util/mypath.py [automatically download]

2. Change result path at configs/env.yml [save result]

3. Change config at configs/VOCSegmentation_unsupervised_saliency_model or VOCSegmentation_supervised_saliency_model

4. Run this script to train with unsupervised saliency model

```shell
cd pretrain

python main.py --config_env configs/env.yml --config_exp configs/VOCSegmentation_unsupervised_saliency_model.yml 
```
## Evaluate
Change segmentation result path at configs/env.yml [The same path as step 3]

### Linear Finetune
  
1. Change pretraining path at configs/linear_finetune/linear_finetune_VOCSegmentation_unsupervised_saliency to point our pretrained model path

2. Run this script

```shell
cd segmentation

python linear_finetune.py --config_env configs/env.yml --config_exp configs/linear_finetune/linear_finetune_VOCSegmentation_unsupervised_saliency.yml
```

### 2. Kmeans

1. Change pretraining path at configs/kmeans/kmeans_VOCSegmentation_unsupervised_saliency to point our pretrained model path 

2. Run this script
```shell
cd segmentation

python kmeans.py --config_env configs/env.yml --config_exp configs/kmeans/kmeans_VOCSegmentation_unsupervised_saliency.yml
```

### 3. Retrieval


1. Change pretraining path at configs/retrieval/retrieval_VOCSegmentation_unsupervised_saliency to point our pretrained model path 

2. Run this script
```shell
cd segmentation

python retrieval.py --config_env configs/env.yml --config_exp configs/retrieval/retrieval_VOCSegmentation_unsupervised_saliency.yml
```

### 4. Visualize [Updating ...]

```shell
cd segmentation

python eval.py --config_env configs/env.yml --config_exp configs/VOCSegmentation_supervised_saliency_model.yml --state-dict $PATH_TO_MODEL
```


## Citation
This code is based on the [SCAN](https://github.com/wvangansbeke/Unsupervised-Classification) and [MoCo](https://github.com/facebookresearch/moco) repositories.
If you find this repository useful for your research, please consider citing the following paper(s):

```bibtex
@article{vangansbeke2020unsupervised,
  title={Unsupervised Semantic Segmentation by Contrasting Object Mask Proposals},
  author={Van Gansbeke, Wouter and Vandenhende, Simon and Georgoulis, Stamatios and Van Gool, Luc},
  journal={arxiv preprint arxiv:2102.06191},
  year={2021}
}
@inproceedings{vangansbeke2020scan,
  title={Scan: Learning to classify images without labels},
  author={Van Gansbeke, Wouter and Vandenhende, Simon and Georgoulis, Stamatios and Proesmans, Marc and Van Gool, Luc},
  booktitle={Proceedings of the European Conference on Computer Vision},
  year={2020}
}
@inproceedings{he2019moco,
  title={Momentum Contrast for Unsupervised Visual Representation Learning},
  author={Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
  booktitle = {Conference on Computer Vision and Pattern Recognition},
  year={2019}
}
```
For any enquiries, please contact the main authors.

For an overview on self-supervised learning, have a look at the [overview repository](https://github.com/wvangansbeke/Self-Supervised-Learning-Overview).

## License

This software is released under a creative commons license which allows for personal and research use only. For a commercial license please contact the authors. You can view a license summary [here](http://creativecommons.org/licenses/by-nc/4.0/).

## Acknoledgements
This work was supported by Toyota, and was carried out at the TRACE Lab at KU Leuven (Toyota Research on Automated Cars in Europe - Leuven).
