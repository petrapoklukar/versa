import numpy as np
import os
import re
from PIL import Image
import pandas as pd
pd.options.mode.chained_assignment = None
import torchvision.transforms as transforms
import pickle


def transform(image):
    """
    Equivalent to transform = transforms.Compose([
        transforms.CenterCrop(168),
        transforms.Resize(self.img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        lambda t: t.unsqueeze(0)])
    :param image: numpy array of shape [height, width, 3]
    :return: image: numpy array of shape [1, 84, 84, 3]
    """
    trans = transforms.Compose([transforms.CenterCrop(168), transforms.Resize(84)])
    image = trans(image)
    image = np.float32(np.transpose(image, [2, 0, 1])) / 255.
    for channel in range(image.shape[0]):
        image[channel] = (image[channel] - 0.5) / 0.5
    image = image.transpose(1, 2, 0) # 84, 84, 3
    image = image[np.newaxis]
    return image


def get_transform_images(df):
    imgs = []
    for img_name in list(df.index):
        img = Image.open(os.path.join(path, base_folder, img_name))
        img = transform(img)
        imgs.append(img)
    return np.concatenate(imgs)
    
path = '../datasets/celeba/celeba/'
attribute_filename = 'list_attr_celeba.txt'
base_folder = 'img_align_celeba/'
attr = pd.read_csv(os.path.join(path, attribute_filename), delim_whitespace=True, header=1)
attr = (attr + 1) // 2

path_to_train_imgs = os.path.join(path, 'train_imgs')
path_to_validation_imgs = os.path.join(path, 'validation_imgs')
path_to_test_imgs = os.path.join(path, 'test_imgs')
if not os.path.exists(path_to_train_imgs):
    os.makedirs(path_to_train_imgs)
if not os.path.exists(path_to_validation_imgs):
    os.makedirs(path_to_validation_imgs)
if not os.path.exists(path_to_test_imgs):
    os.makedirs(path_to_test_imgs)

n_train_imgs = 5
n_validation_imgs = 3
n_test_imgs = 9

train_attr_names = ['Oval_Face', 'Attractive', 'Mustache', 'Male', 'Pointy_Nose', 'Bushy_Eyebrows', 
                    'Blond_Hair', 'Rosy_Cheeks', 'Receding_Hairline', 'Eyeglasses', 'Goatee', 'Brown_Hair', 
                    'Narrow_Eyes', 'Chubby', 'Big_Lips', 'Wavy_Hair', 'Bags_Under_Eyes', 'Arched_Eyebrows',
                    'Wearing_Earrings', 'High_Cheekbones', 'Black_Hair', 'Bangs', 'Wearing_Lipstick', 'Sideburns', 'Bald']
validation_attr_names = ['Wearing_Necklace', 'Smiling', 'Pale_Skin', 'Wearing_Necktie', 'Big_Nose']
test_attr_names = ['Straight_Hair', '5_o_Clock_Shadow', 'Wearing_Hat', 'Gray_Hair', 'Heavy_Makeup', 
                   'Young', 'Blurry', 'Double_Chin', 'Mouth_Slightly_Open', 'No_Beard']

train_attr_idx = [list(attr).index(a) for a in train_attr_names]
validation_attr_idx = [list(attr).index(a) for a in validation_attr_names]
test_attr_idx = [list(attr).index(a) for a in test_attr_names]

train_attr_pd = attr[:n_train_imgs]
validation_attr_pd = attr.iloc[n_train_imgs:n_train_imgs + n_validation_imgs]
test_attr_pd = attr[n_train_imgs + n_validation_imgs:n_train_imgs + n_validation_imgs + n_test_imgs]

a = get_transform_images(train_attr_pd)
train_dict = {
    'img_array': get_transform_images(train_attr_pd),
    'attr_pd': train_attr_pd,
    'attr_list': train_attr_idx}
with open(os.path.join(path_to_train_imgs, 'train.pkl'), 'wb') as f:
    pickle.dump(train_dict, f)
    

validation_dict = {
    'img_array': get_transform_images(validation_attr_pd),
    'attr_pd': validation_attr_pd,
    'attr_list': validation_attr_idx}
with open(os.path.join(path_to_validation_imgs, 'validation.pkl'), 'wb') as f:
    pickle.dump(validation_dict, f)
    

test_dict = {
    'img_array': get_transform_images(test_attr_pd),
    'attr_pd': test_attr_pd,
    'attr_list': test_attr_idx}
with open(os.path.join(path_to_test_imgs, 'test.pkl'), 'wb') as f:
    pickle.dump(test_dict, f)
    
print('done')