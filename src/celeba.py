import numpy as np
import os
import re
from PIL import Image
import pandas as pd
pd.options.mode.chained_assignment = None
import torchvision.transforms as transforms

def onehottify_2d_array(a):
    """
    https://stackoverflow.com/questions/36960320/convert-a-2d-matrix-to-a-3d-one-hot-matrix-numpy
    :param a: 2-dimensional array.
    :return: 3-dim array where last dim corresponds to one-hot encoded vectors.
    """

    # https://stackoverflow.com/a/46103129/ @Divakar
    def all_idx(idx, axis):
        grid = np.ogrid[tuple(map(slice, idx.shape))]
        grid.insert(axis, idx)
        return tuple(grid)

    num_columns = a.max() + 1
    out = np.zeros(a.shape + (num_columns,), dtype=int)
    out[all_idx(a, axis=2)] = 1
    return out


class CelebAData(object):
    """
    Data Generator capable of generating batches of sinusoid or Omniglot data.
    A "class" is considered a class of omniglot digits or a particular sinusoid function.
    """
    def __init__(self, path, seed):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.image_height = 84
        self.image_width = 84
        self.image_channels = 3

        self.path = '/local_storage/datasets/celeba/celeba'# '/home/petra/Documents/PhD/Repos/datasets/celeba/celeba'
        self.base_folder = "img_align_celeba/"
        self.attribute_filename = 'list_attr_celeba.txt'
        self.attr = pd.read_csv(os.path.join(self.path, self.attribute_filename), delim_whitespace=True, header=1)
        self.attr = (self.attr + 1) // 2  # map from {-1, 1} to {0, 1}
        self.attr_names = list(self.attr.columns)
        self.num_total_attr = len(self.attr_names)
        self.test_attr_idx = np.array([32, 0, 35, 17, 18, 39, 10, 14, 21, 24])
        self.train_attr_idx = np.setdiff1d(np.arange(self.num_total_attr), self.test_attr_idx)
        self.id_to_name_fn = lambda x: self.attr_names[x]


    def get_image_height(self):
        return self.image_height

    def get_image_width(self):
        return self.image_width

    def get_image_channels(self):
        return self.image_channels
    
    def get_batch(self, source, tasks_per_batch, shot, way, eval_samples):
        """
        Returns a batch of tasks from miniImageNet. Values are np.float32 and scaled to [0,1]
        :param source: one of `train`, `test`, `validation` (i.e. from which classes to pick)
        :param tasks_per_batch: number of tasks to include in batch.
        :param shot: number of training examples per class.
        :param way: number of classes per task.
        :param eval_samples: number of evaluation samples to use.
        :return: [train_images, test_images, train_labels, test_labels]

        shapes:
            * Images: [tasks_per_batch, way * shot, height, width, channels]
            * Labels: [tasks_per_batch, way * shot, way]
                      (one-hot encoded in last dim)
        """

        train_images, test_images, train_labels, test_labels = self._sample_batch(
            source, tasks_per_batch, shot, way, eval_samples)

        train_images, train_labels = self._shuffle_batch(train_images, train_labels)

        return [train_images, test_images, train_labels, test_labels]

    def _sample_batch(self, source, tasks_per_batch, shot, way, eval_samples):
        """
        Sample a k-shot batch from images.
        :param images: Data to sample from [way, samples, h, w, c] (either of train, val, test)
        :param tasks_per_batch: number of tasks to include in batch.
        :param shot: number of training examples per class.
        :param way: number of classes per task.
        :param eval_samples: number of evaluation samples to use.
        :return: A list [train_images, test_images, train_labels, test_labels]

        shapes:
            * Images: [tasks_per_batch, way * (shot or eval_samples), h, w, c]
            * Labels: [tasks_per_batch, way * (shot or eval_samples), way]
                      (one-hot encoded in last dim)
        """
        xs = np.zeros((tasks_per_batch, way * shot, self.image_height, self.image_width, self.image_channels),
                      dtype=np.float32)
        ys = np.zeros((tasks_per_batch, way * shot), dtype=np.int32)
        xq = np.zeros((tasks_per_batch, way * eval_samples, self.image_height, self.image_width, self.image_channels),
                      dtype=np.float32)
        yq = np.zeros((tasks_per_batch, way * eval_samples), dtype=np.int32)
        # labels = np.zeros((self.batch_size, num_attr), dtype=np.int64)
        for i in range(tasks_per_batch):
            positive_samples, negative_samples, positive_attr = self.filter_attr(source, 2,
                                                                                 shot + eval_samples, 
                                                                                 np.array([-1, -1]))
            xs[i], ys[i] = self.get_imgs(positive_samples, negative_samples, shot)
            xq[i], yq[i] = self.get_imgs(positive_samples, negative_samples, eval_samples)
            # labels[i] = positive_attr

        # labels to one-hot encoding
        ys = onehottify_2d_array(ys)
        yq = onehottify_2d_array(yq)
        return [xs, xq, ys, yq]
    
    def _shuffle_batch(self, train_images, train_labels):
        """
        Randomly permute the order of the second column
        :param train_images: [tasks_per_batch, way * shot, height, width, channels]
        :param train_labels: [tasks_per_batch, way * shot, way]
        :return: permuted images and labels.
        """
        for i in range(train_images.shape[0]):
            permutation = np.random.permutation(train_images.shape[1])
            train_images[i, ...] = train_images[i, permutation, ...]
            train_labels[i, ...] = train_labels[i, permutation, ...]
        return train_images, train_labels
    
    @staticmethod
    def sorted_nicely(l):
        """ Sorts numbers in strings numerically as opposed to alphabetically
        """
        convert = lambda text: int(text) if text.isdigit() else text
        alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
        return sorted(l, key=alphanum_key)

    def transform(self, image):
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


    def get_imgs(self, positive_samples, negative_samples, num_imgs):
        df = pd.concat([positive_samples.sample(n=num_imgs),
                        negative_samples.sample(n=num_imgs)])
        df_reshuffled = df.reindex(np.random.permutation(df.index))

        imgs = []
        for img_name in list(df_reshuffled.index):
            img = Image.open(os.path.join(self.path, self.base_folder, img_name))
            img = self.transform(img)
            imgs.append(img)
        #reshaped_imgs = np.concatenate(imgs)[np.newaxis]
        return np.concatenate(imgs), df_reshuffled['nway_label'].values

    def filter_attr(self, source, num_attr, min_num, positive_attr=np.array([-1, -1])):
        if positive_attr[0] < 0 and source == 'train':
            positive_attr = np.random.choice(self.train_attr_idx, size=num_attr)
        elif positive_attr[0] < 0 and source != 'train':
            positive_attr = np.random.choice(self.test_attr_idx, size=num_attr)
        positive_condition = (self.attr.iloc[:, positive_attr] == 1).sum(axis=1) == num_attr
        negative_condition = (self.attr.iloc[:, positive_attr] != 1).sum(axis=1) == num_attr
        if positive_condition.sum() < min_num or negative_condition.sum() < min_num:
            return self.filter_attr(source, num_attr, min_num)
        else:
            positive_samples = self.attr[positive_condition]
            positive_samples['nway_label'] = np.repeat(1, len(positive_samples))
            negative_samples = self.attr[negative_condition]
            negative_samples['nway_label'] = np.repeat(0, len(negative_samples))
            return positive_samples, negative_samples, positive_attr

    def ids_to_names(self, attr_ids):
        attr_names = []
        for _tuple in attr_ids:
            name = [self.id_to_name_fn(_tuple[0]), self.id_to_name_fn(_tuple[1])]
            if len(_tuple) == 3:
                name.append('(xor)')
            attr_names.append(name)
        joined = []
        for elem in attr_names:
            joined.append('+'.join(elem))
        return '/'.join(joined)
