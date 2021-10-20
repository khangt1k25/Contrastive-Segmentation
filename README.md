

## Requirements
kornia 

## Setup 
1. Create folder named USC in drive
2. Upload notebook 
3. Change path in pretrain/data/util/mypath.py. The PASCAL VOC dataset will be saved to this path[automatically download]
4. Specify output dir in configs/env.yml
## Training


Read the config in VOCSegmentation_unsupervised_saliency_model.yml 

```shell
cd pretrain
python main.py --config_env configs/env.yml --config_exp configs/VOCSegmentation_unsupervised_saliency_model.yml
```

## Evaluation
### Linear Classifier (LC)
We freeze the weights of the pre-trained model and train a 1 x 1 convolutional layer to predict the class assignments from the generated feature representations. Since the discriminative power of a linear classifier is low, the pixel embeddings need to be informative of the semantic class to solve the task in this way. To train the classifier run the following command:
```shell
cd segmentation
python linear_finetune.py --config_env configs/env.yml --config_exp configs/linear_finetune/linear_finetune_VOCSegmentation_supervised_saliency.yml
```
Note, make sure that the `pretraining` variable in `linear_finetune_VOCSegmentation_supervised_saliency.yml` points to the location of your pre-trained model.
You should get the following results:
```
mIoU is 63.95
IoU class background is 90.95
IoU class aeroplane is 83.78
IoU class bicycle is 30.66
IoU class bird is 78.79
IoU class boat is 64.57
IoU class bottle is 67.31
IoU class bus is 84.24
IoU class car is 76.77
IoU class cat is 79.10
IoU class chair is 21.24
IoU class cow is 66.45
IoU class diningtable is 46.63
IoU class dog is 73.25
IoU class horse is 62.61
IoU class motorbike is 69.66
IoU class person is 72.30
IoU class pottedplant is 40.15
IoU class sheep is 74.70
IoU class sofa is 30.43
IoU class train is 74.67
IoU class tvmonitor is 54.66
```
Unsurprisingly, the model has not learned a good representation for every class since some classes are hard to distinguish, e.g. `chair` or `sofa`.

We visualize a few examples after CRF post-processing below.
<p align="left">
    <img src="images/examples.jpg" width="950"/>

### Clustering (K-means)
The feature representations are clustered with K-means. If the pixel embeddings are disentangled according to the defined class labels, we can match the predicted clusters with the ground-truth classes using the Hungarian matching algorithm. 


```shell
cd segmentation
python kmeans.py --config_env configs/env.yml --config_exp configs/kmeans/kmeans_VOCSegmentation_supervised_saliency.yml
```
Remarks: Note that we perform the complete K-means fitting on the validation set to save memory and that the reported results were averaged over 5 different runs. 
You should get the following results (21 clusters):
```
IoU class background is 88.17
IoU class aeroplane is 77.41
IoU class bicycle is 26.18
IoU class bird is 68.27
IoU class boat is 47.89
IoU class bottle is 56.99
IoU class bus is 80.63
IoU class car is 66.80
IoU class cat is 46.13
IoU class chair is 0.73
IoU class cow is 0.10
IoU class diningtable is 0.57
IoU class dog is 35.93
IoU class horse is 48.68
IoU class motorbike is 60.60
IoU class person is 32.24
IoU class pottedplant is 23.88
IoU class sheep is 36.76
IoU class sofa is 26.85
IoU class train is 69.90
IoU class tvmonitor is 27.56
```

### Semantic Segment Retrieval
We examine our representations on PASCAL through segment retrieval. First, we compute a feature vector for every object mask in the `val` set by averaging the pixel embeddings within the predicted mask. Next, we retrieve the nearest neighbors on the `train_aug` set for each object.

```shell
cd segmentation
python retrieval.py --config_env configs/env.yml --config_exp configs/retrieval/retrieval_VOCSegmentation_unsupervised_saliency.yml
```

| Method                    | MIoU (7 classes) | MIoU (21 classes)|
| ------------------------- | ---------------- | ---------------- |
| MoCo v2                   | 48.0             | 39.0             |
| MaskContrast* (unsup sal.)| 53.4             | 43.3             |
| MaskContrast* (sup sal.)  | 62.3             | 49.6             |

_\* Denotes MoCo init._


## Model Zoo
Download the pretrained and linear finetuned models here.

| Dataset            | Pixel Grouping Prior    | mIoU (LC)     | mIoU (K-means)   |Download link |
|------------------  | ----------------------  |---------------|---------  |--------------|
| PASCAL VOC         |  Supervised Saliency    |   -           |   44.2    |[Pretrained Model 🔗](https://drive.google.com/file/d/1UkzAZMBG1U8kTqO3yhO2nTtoRNtEvyRq/view?usp=sharing) | 
| PASCAL VOC         |  Supervised Saliency    |   63.9 (65.5*)  |   44.2    |[Linear Finetuned 🔗](https://drive.google.com/file/d/1C2iv8wFV8MNLYLKw2E0Do2aeO-eaWNw3/view?usp=sharing)  |
| PASCAL VOC         |  Unsupervised Saliency   |   -           |  35.0     |[Pretrained Model 🔗](https://drive.google.com/file/d/1efL1vWVcrGAqeC6OLalX8pwec41c6NZj/view?usp=sharing) |
| PASCAL VOC         |  Unsupervised Saliency   |   58.4 (59.5*) |  35.0     |[Linear Finetuned 🔗](https://drive.google.com/file/d/1y-HZTHHTyAceiFDLAraLXooGOdyQqY2Z/view?usp=sharing)  |

_\* Denotes CRF post-processing._

To evaluate and visualize the predictions of the finetuned model, run the following command:
```shell
cd segmentation
python eval.py --config_env configs/env.yml --config_exp configs/VOCSegmentation_supervised_saliency_model.yml --state-dict $PATH_TO_MODEL
```
You can optionally append the `--crf-postprocess` flag. 


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
