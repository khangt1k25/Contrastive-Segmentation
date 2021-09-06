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
train_dataset = TwoTransformDataset(get_train_dataset(p, transform = None), base_transform, next_transform, type_kornia=2, min_area=0.01, max_area=0.99)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32, collate_fn=collate_custom)

for i, batch in enumerate(train_dataloader):
    # Forward pass
    im_q = batch['query']['image']
    im_k = batch['key']['image']
    sal_q = batch['query']['sal']
    sal_k = batch['key']['sal']

    state_dict = batch['T']
    transform = batch['transform']
    augmented_k = []
    augmented_sal = []
    i = 1
    # toPIL(im_q[i]).show()
    # toPIL(im_k[i]).show()

    toPIL(sal_q[i].float()).show()
    
    
    # toPIL(sal_k[i].float()).show()
    
    print(torch.unique(sal_q[i]))
    print(torch.unique(sal_k[i]))
    print(sal_k.shape)
    print(sal_q.shape)
    # for i in range(len(state_dict)):

    #     sample = {"image": deepcopy(im_k[i]), 'sal': deepcopy(sal_k[i])}
    #     new_sample = next_transform.forward_with_params(sample, state_dict[i])
    #     augmented_sal.append(new_sample['sal'].squeeze(0))
    #     augmented_k.append(new_sample['image'].squeeze(0))

    # augmented_k = torch.stack(augmented_k, dim=0).squeeze(0)
    # # augmented_k = augmented_k.permute((0, 2, 3, 1))
    # augmented_sal = torch.stack(augmented_sal, dim=0)
   

    # print(augmented_sal.shape)
    # print(torch.unique(augmented_sal))
    # toPIL(augmented_sal[i]).show()
    # toPIL(augmented_k[i]).show()

           
    # toPIL(im_q[0]).show()
    # toPIL(im_k[0]).show()
    # inverse_k = []
    # inverse_sal = []
    # for i in range(im_q.shape[0]):

    #     sample = {"image": deepcopy(im_k[i]), 'sal': deepcopy(sal_k[i])}
    #     new_sample = next_transform.inverse(sample, transform[i])
    #     inverse_k.append(new_sample['image'].squeeze(0))
    #     inverse_sal.append(new_sample['sal'].squeeze(0))

    # inverse_k = torch.stack(inverse_k, dim=0).squeeze(0)
    # # inverse_k = inverse_k.permute((0, 2, 3, 1))                  
    # inverse_sal = torch.stack(inverse_sal, dim=0)
    # # print(inverse_k.shape)
    # # print(inverse_sal.shape)
    # # print(im_k.shape)
    
    

    

    # print(len(transform))
    # print(len(state_dict))
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