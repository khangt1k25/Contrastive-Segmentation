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
from modules.losses import ConsistencyLoss

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
train_dataset = TwoTransformDataset(get_train_dataset(p, transform = None), base_transform, next_transform, type_kornia=1, min_area=0.01, max_area=0.99)
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
    
    
    for i in range(len(state_dict)):

        sample = {"image": deepcopy(im_k[i]), 'sal': deepcopy(sal_k[i])}
        new_sample = next_transform.forward_with_params(sample, state_dict[i])
        augmented_sal.append(new_sample['sal'].squeeze(0))
        augmented_k.append(new_sample['image'].squeeze(0))

    augmented_k = torch.stack(augmented_k, dim=0).squeeze(0)
    
    augmented_k = augmented_k.permute((0, 2, 3, 1))
    augmented_sal = torch.stack(augmented_sal, dim=0)
   
    i = 21
    

    
    
    
    q_selected = im_q.permute((0, 2, 3, 1))

    

    # mask = augmented_sal.unsqueeze(-1)
    
    # x = augmented_k * mask
    # y = q_selected * mask

    # z = (augmented_k-q_selected) * mask
    # # print(z.shape)
    # loss = (z**2).sum(dim=-1)
    # print(loss.shape)
    # print(loss.mean())

    # x =  x.reshape((-1, x.shape[-1]))
    # y =  y.reshape((-1, y.shape[-1]))
    # loss2 = ((x-y)**2).sum(dim=-1).mean()

    # print(loss2)
    # print(x.shape)
    
    
    # print(x.shape)
    # print(y.shape)

    # augmented_k = augmented_k.permute((0, 3, 1, 2))
    # # toPIL(augmented_sal[i].float()).show()
    # # toPIL(augmented_k[i]).show()
    # x = x.permute((0, 3, 1, 2))
    # y = y.permute((0, 3, 1, 2))

    # toPIL(x[i]).show()
    # toPIL(y[i]).show()
   


    

    # return (((x-y) ** 2).sum(dim=-1)).mean()
    # consistency = ConsistencyLoss(type='l2')

    # consistency_loss = consistency(augmented_k, q_selected, mask=augmented_sal)
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