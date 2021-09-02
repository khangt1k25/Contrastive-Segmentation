from copy import deepcopy
from kornia.geometry import transform
from numpy.core.fromnumeric import size
from torchvision import transforms
from data.dataloaders.transforms_v2 import MyAugmentation
from data.dataloaders.dataset import TwoTransformDataset
from utils.common_config import get_train_dataset,get_base_transformations, get_next_transformations
from PIL import Image
import torch
from utils.collate import collate_custom

toPIL = transforms.ToPILImage()
# img = Image.open('../examples/pic.jpg')
# sal = torch.rand(size=(224, 224))

# sample = {'image': img, 'sal': sal}

# aug = MyAugmentation()

# aug(sample)

base_transform = get_base_transformations()
next_transform = get_next_transformations()   
print(base_transform)
print(next_transform)

p = {'train_db_name':'VOCSegmentation', 'train_db_kwargs': {'saliency':'unsupervised_model'}}
train_dataset = TwoTransformDataset(get_train_dataset(p, transform = None), base_transform, next_transform, type_kornia=1)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, collate_fn=collate_custom)

for i, batch in enumerate(train_dataloader):
    # Forward pass
    im_q = batch['query']['image']
    im_k = batch['key']['image']
    sal_q = batch['query']['sal']
    sal_k = batch['key']['sal']

    state_dict = batch['T']
    transform = batch['transform']

    print(len(transform))
    print(len(state_dict))
    break
# sample = train_dataset[1234]

# key = sample['key']
# query = sample['query']


# print(torch.unique(key['sal']))

# toPIL(key['image']).show(title='key')
# # toPIL(key['sal'].float()).show()
# toPIL(query['image']).show(title='query')
# # toPIL(query['sal'].float()).show()


# next = next_transform.forward_with_params(deepcopy(query), state_dict=sample['T'])

# inv = next_transform.inverse(deepcopy(key), transform=sample['transform'])

# toPIL(inv['image']).show(title='inv')
# toPIL(next['image']).show()

# toPIL(inv['sal'].float()).show()

# toPIL(next['sal'].float()).show()