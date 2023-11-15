import os
import shutil
import torch
import torchvision
import random
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

torch.manual_seed(0)

# To check Pytorch version

# print(f"Using torch version : {torch.__version__}")

class_name = ['Normal', 'Viral', 'CovidPositive']
root_dir = 'COVID-19_Radiography_Dataset'
source_dirs = ['Normal', 'Viral Pneumonia', 'COVID']


# To create sample datasets

# if os.path.isdir(os.path.join(root_dir, source_dirs[1])):
#     os.mkdir(os.path.join(root_dir, 'test'))

#     for i, d in enumerate(source_dirs):
#         os.rename(os.path.join(root_dir, d), os.path.join(root_dir, classes_name[i]))

#     for c in classes_name:
#         os.mkdir(os.path.join(root_dir, 'test', c))
        
#     for c in classes_name:
#         images = [x for x in os.listdir(os.path.join(root_dir, c)) if x.lower().endswith('png')]
#         selected_images = random.sample(images, 30)
#         for image in selected_images:
#             source_path = os.path.join(root_dir, c, image)
#             target_path = os.path.join(root_dir, 'test', c, image)
#             shutil.move(source_path, target_path)


# To work with images

class ChestXRayCustomDataset(torch.utils.data.Dataset):
    def __init__(self, image_dirs, transform):
        def get_images(class_name):
            images = [x for x in os.listdir(image_dirs[class_name]) if x.lower().endswith('.png')]
            print(f"Found {len(images)} {class_name} examples")
            return images
        
        self.images = {}
        self.class_names = ['Normal', 'Viral', 'CovidPositive']

        for c in self.class_names:
            self.images[c] = get_images(c)

        self.image_dirs = image_dirs
        self.transform = transform

    def __len__(self):
        return sum([len(self.images[c]) for c in self.class_names])
    
    def __getItem__(self, index):
        class_name = random.choice(self.class_names)
        index = index % len(self.images[class_name])
        image_name = self.images[class_name][index]
        image_path = os.path.join(self.image_dirs[class_name], image_name)
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.class_names.index(class_name)

# To transform images

train_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])

test_transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(size=(224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
])


train_dirs = {
    'Normal': 'COVID-19_Radiography_Dataset/Normal',
    'Viral': 'COVID-19_Radiography_Dataset/Viral',
    'CovidPositive': 'COVID-19_Radiography_Dataset/CovidPositive'
}

train_Dataset = ChestXRayCustomDataset(train_dirs, train_transform)


test_dirs = {
    'Normal': 'COVID-19_Radiography_Dataset/test/Normal',
    'Viral': 'COVID-19_Radiography_Dataset/test/Viral',
    'CovidPositive': 'COVID-19_Radiography_Dataset/test/CovidPositive'
}

test_Dataset = ChestXRayCustomDataset(test_dirs, test_transform)

batch_size = 6

dl_train = torch.utils.data.DataLoader(train_Dataset, batch_size=1, shuffle=False)

dl_test = torch.utils.data.DataLoader(test_Dataset, batch_size=1, shuffle=False)

print(f"The length of trainDataset is {len(dl_train)}")
print(f"The length of testDataset is {len(dl_test)}")

