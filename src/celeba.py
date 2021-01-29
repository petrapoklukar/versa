import numpy as np
import os
import re
# from PIL import Image
import pandas as pd
pd.options.mode.chained_assignment = None
# import torchvision.transforms as transforms
import pickle
from collections import OrderedDict, defaultdict
from itertools import combinations

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
    def __init__(self, seed=1201):
        """
        Args:
            num_samples_per_class: num samples to generate per class in one batch
            batch_size: size of meta batch size (e.g. number of functions)
        """
        self.image_height = 84
        self.image_width = 84
        self.image_channels = 3

        self.path = 'datasets/celeba'
        # self.path = '/local_storage/datasets/celeba/celeba'# 
        # self.path = '/home/petra/Documents/PhD/Repos/datasets/celeba/celeba'
        # self.base_folder = "img_align_celeba/"
        # self.attribute_filename = 'list_attr_celeba.txt'
        
        # Load all pkl
        with open(os.path.join(self.path, 'train.pkl'), 'rb') as f:
            train_dict = pickle.load(f)
            self.train_attr_df = train_dict['attr_pd'] # Already scaled to 0, 1
            self.train_attr_idx = train_dict['attr_list'] 
            self.train_imgs = train_dict['img_array']
        
        with open(os.path.join(self.path, 'validation.pkl'), 'rb') as f:
            validation_dict = pickle.load(f)
            self.validation_attr_df = validation_dict['attr_pd'] # Already scaled to 0, 1
            self.validation_attr_idx = validation_dict['attr_list'] 
            self.validation_imgs = validation_dict['img_array']
        
        with open(os.path.join(self.path, 'test.pkl'), 'rb') as f:
            test_dict = pickle.load(f)
            self.test_attr_df = test_dict['attr_pd'] # Already scaled to 0, 1
            self.test_attr_idx = test_dict['attr_list']
            self.test_imgs = test_dict['img_array']
        


        self.attr_names = list(self.train_attr_df.columns)
        self.num_total_attr = len(self.attr_names)
        self.id_to_name_fn = lambda x: self.attr_names[x]
        np.random.seed(seed)

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
        
        for i in range(tasks_per_batch):
            # Support set gets 3 attributes - amiguous setting 
            positive_images, negative_images, manual_attr = self.filter_attr(source, 3, shot)
            assert(positive_images.shape == negative_images.shape)
            
            support_imgs = np.concatenate([positive_images, negative_images])
            support_labels = np.concatenate([np.repeat(1, shot), np.repeat(0, shot)])
            support_permutation = np.arange(shot * way) # Permute positive and negative samples
            np.random.shuffle(support_permutation)
            xs[i], ys[i] = support_imgs[support_permutation], support_labels[support_permutation]

            # Query set gets 2 attributes - non-amiguous setting 
            positive_images, negative_images, _ = self.filter_attr(source, 2, eval_samples, manual_attr=manual_attr)
            assert(positive_images.shape == negative_images.shape)
            
            query_imgs = np.concatenate([positive_images, negative_images])
            query_labels = np.concatenate([np.repeat(1, eval_samples), np.repeat(0, eval_samples)])
            query_permutation = np.arange(eval_samples * way) # Permute positive and negative samples
            np.random.shuffle(query_permutation)
            xq[i], yq[i] = query_imgs[query_permutation], query_labels[query_permutation]

        # labels to one-hot encoding
        ys = onehottify_2d_array(ys)
        yq = onehottify_2d_array(yq)
        return [xs, xq, ys, yq]
    
    def _sample_batch_with_2_attr(self, source, tasks_per_batch, shot, way, eval_samples):
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
        
        for i in range(tasks_per_batch):
            positive_images, negative_images = self.filter_attr(source, 2, shot + eval_samples)
            assert(positive_images.shape == negative_images.shape)
            
            support_idx = np.random.choice(np.arange(len(positive_images)), size=shot, replace=False)
            query_idx = np.setdiff1d(np.arange(len(positive_images)), support_idx)
            assert(len(query_idx) == eval_samples)
            
            support_imgs = np.concatenate([positive_images[support_idx], negative_images[support_idx]])
            support_labels = np.concatenate([np.repeat(1, shot), np.repeat(0, shot)])
            support_permutation = np.arange(shot * way)
            np.random.shuffle(support_permutation)
            xs[i], ys[i] = support_imgs[support_permutation], support_labels[support_permutation]


            query_imgs = np.concatenate([positive_images[query_idx], negative_images[query_idx]])
            query_labels = np.concatenate([np.repeat(1, eval_samples), np.repeat(0, eval_samples)])
            query_permutation = np.arange(eval_samples * way)
            np.random.shuffle(query_permutation)
            
            xq[i], yq[i] = query_imgs[query_permutation], query_labels[query_permutation]

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


    def filter_attr(self, source, num_attr, min_num, manual_attr=np.array([-1, -1])):
        if source == 'train':
            attr_df = self.train_attr_df
            img_array = self.train_imgs
        elif source == 'validation':
            attr_df = self.validation_attr_df
            img_array = self.validation_imgs
        elif source == 'test':
            attr_df = self.test_attr_df
            img_array = self.test_imgs
            
        # Randomly sample the attributes
        if manual_attr[0] < 0 and source == 'train':
            positive_attr_idx = np.random.choice(self.train_attr_idx, size=num_attr)
        elif manual_attr[0] < 0 and source == 'validation':
            positive_attr_idx = np.random.choice(self.validation_attr_idx, size=num_attr)
        elif manual_attr[0] < 0 and source == 'test':
            positive_attr_idx = np.random.choice(self.test_attr_idx, size=num_attr)
        # Ensure that query gets the same attributes
        elif len(manual_attr) > num_attr:
            positive_attr_idx = np.random.choice(manual_attr, num_attr, replace=False)
                
        
        positive_condition = (attr_df.iloc[:, positive_attr_idx] == 1).sum(axis=1) == num_attr
        negative_condition = (attr_df.iloc[:, positive_attr_idx] != 1).sum(axis=1) == num_attr
        
        # For some combinations of attributes it might happen that we don't get enough images
        if positive_condition.sum() < min_num or negative_condition.sum() < min_num:
            return self.filter_attr(source, num_attr, min_num)
        else:
            positive_imgs_idx = np.random.choice(attr_df[positive_condition]['id'], 
                                                 size=min_num, replace=False)
            negative_imgs_idx = np.random.choice(attr_df[negative_condition]['id'], 
                                                 size=min_num, replace=False)
            
            positive_imgs = np.float32(img_array[positive_imgs_idx] / 255.)
            negative_imgs = np.float32(img_array[negative_imgs_idx] / 255.)
            return positive_imgs, negative_imgs, positive_attr_idx

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
    
    def _not_idx(self, attr_id, labels):
        """
        This takes in a list of attributes, and returns a list of attributes that has neither of the them.
        """
        attr_boolean = np.logical_or.reduce([labels[:, attr] == 1 for attr in attr_id], axis=0)
        attr_boolean = np.logical_not(attr_boolean)
        attr_idx = np.where(attr_boolean)[0]
        return attr_idx

    def _has_idx(self, attr_id, labels):
        has_all_attr = np.all([labels[:, attr] == 1 for attr in attr_id], axis=0)
        attr_idx = np.where(has_all_attr)[0]
        return attr_idx

    def _xor_attribute(self, attr_id, labels):
        assert len(attr_id) == 2
        has_xor_attr = []
        has_xor_attr.append(labels[:, attr_id[0]] == 1)
        has_xor_attr.append(labels[:, attr_id[1]] == 0)
        has_xor_attr = np.all(has_xor_attr, axis=0)
        attr_idx = np.where(has_xor_attr)[0]

        has_xor_attr = []
        has_xor_attr.append(labels[:, attr_id[1]] == 1)
        has_xor_attr.append(labels[:, attr_id[0]] == 0)
        has_xor_attr = np.all(has_xor_attr, axis=0)
        neg_idx = np.where(has_xor_attr)[0]
        return attr_idx, neg_idx


    def get_attribute_combinations(self, attributes, labels, way, orders=[2]):
        """
        attributes: list
        order : the k in the n choose k sense
        """
        attr_combinations = OrderedDict()
        ambig_attr_combinations = OrderedDict()
        for order in orders:
            for attr_id in combinations(attributes, order):
                # sample classes with both attributes and none
                attr_idx = self._has_idx(attr_id, labels)
                # sample the not class
                ambig_attr_idx = self._not_idx(attr_id, labels)
                # check there are are enough examples
                if order == 3:
                    # one set for adaptation and three for evaluation
                    min_examples = 3 * way
                else:
                    min_examples = 2 * way

                if len(attr_idx) >= min_examples and len(ambig_attr_idx) >= min_examples:
                    attr_combinations[attr_id] = attr_idx
                    ambig_attr_combinations[attr_id] = ambig_attr_idx
                
                # higher order xor is also possible not but not the goal of the evaluation              
                if order == 2:
                    # sample attributes that have only one of the attribute
                    attr_idx, neg_idx = self._xor_attribute(attr_id, labels)
                    if len(attr_idx) >= 2 * way and len(neg_idx) >= 2 * way:
                        attr_combinations[attr_id + ('xor',)] = attr_idx
                        ambig_attr_combinations[attr_id + ('xor',)] = neg_idx

        return attr_combinations, ambig_attr_combinations

    def init_testing_params(self, way):
        # Used for testing
        self.metatest_attr_doubles, self.metatest_attr_ambig = self.get_attribute_combinations(
            self.test_attr_idx, np.array(self.test_attr_df), way)
        
        self.metatest_attr_triplet, self.meta_attr_triplet_neg = self.get_attribute_combinations(
            self.test_attr_idx, np.array(self.test_attr_df), way, orders=[3])
        self.n_test_triplets = len(self.metatest_attr_triplet)
        

    def get_test_triplet_batch_new(self, shot, way, eval_samples, input_idx):
        """
        Generate a tasks which contains dataset containing three test attributes
        """
        tasks_per_batch = 3
        # attr_names = []
        xs = np.zeros((tasks_per_batch, way * shot, self.image_height, self.image_width, self.image_channels),
                      dtype=np.float32)
        ys = np.zeros((tasks_per_batch, way * shot), dtype=np.int32)
        xq = np.zeros((tasks_per_batch, way * eval_samples, self.image_height, self.image_width, self.image_channels),
                      dtype=np.float32)
        yq = np.zeros((tasks_per_batch, way * eval_samples), dtype=np.int32)
        
        triplet_key = list(self.metatest_attr_triplet.keys())[input_idx]
        # attr_names.append(" ".join(map(self.id_to_name_fn, triplet_key)))
        attr_idx_3 = self.metatest_attr_triplet[triplet_key] # idx having these attributes
        negative_attr_idx_3 = self.meta_attr_triplet_neg[triplet_key] # idx not having these attributes 
        
        amb_idx = []
        # Support set gets 3 attributes - amiguous setting 
        positive_idx = np.random.choice(attr_idx_3, size=shot, replace=False)
        positive_images = self.test_imgs[positive_idx].astype('float32') / 255. # shot, H, W, C 
        amb_idx.append(positive_idx)
        
        negative_idx = np.random.choice(negative_attr_idx_3, size=shot, replace=False)
        negative_images = self.test_imgs[negative_idx].astype('float32') / 255. # shot, H, W, C 
        amb_idx.append(negative_idx)
        assert(positive_images.shape == negative_images.shape)
        
        support_imgs = np.concatenate([positive_images, negative_images])
        support_labels = np.concatenate([np.repeat(1, shot), np.repeat(0, shot)])
        
        for i in range(tasks_per_batch):
            support_permutation = np.arange(shot * way) # Permute positive and negative samples
            np.random.shuffle(support_permutation)
            xs[i], ys[i] = support_imgs[support_permutation], support_labels[support_permutation]
            
        # Get all possible combinations of 2 attributes out of 3
        for t, sub_combs in enumerate(combinations(triplet_key, 2)):
            pos_attr_idx = self.metatest_attr_doubles[sub_combs]
            negative_attr_idx = self.metatest_attr_ambig[sub_combs]
            # remove any examples that were sampled above
            pos_attr_idx = np.setdiff1d(pos_attr_idx, amb_idx[0]) 
            negative_attr_idx = np.setdiff1d(negative_attr_idx, amb_idx[1]) 
            
            positive_idx_query = np.random.choice(pos_attr_idx, size=eval_samples, replace=False)
            positive_images = self.test_imgs[positive_idx_query].astype('float32') / 255.
            
            negative_idx_query = np.random.choice(negative_attr_idx, size=eval_samples, replace=False)
            negative_images = self.test_imgs[negative_idx_query].astype('float32') / 255.
            assert(positive_images.shape == negative_images.shape)
            
            query_imgs = np.concatenate([positive_images, negative_images])
            query_labels = np.concatenate([np.repeat(1, eval_samples), np.repeat(0, eval_samples)])
            query_permutation = np.arange(eval_samples * way) # Permute positive and negative samples
            np.random.shuffle(query_permutation)
        
            xq[t], yq[t] = query_imgs[query_permutation], query_labels[query_permutation]
        
        
        # labels to one-hot encoding
        ys = onehottify_2d_array(ys)
        yq = onehottify_2d_array(yq)
        return [xs, xq, ys, yq]
    
    def get_test_triplet_batch(self, shot, way, eval_samples, input_idx):
        """
        Generate a tasks which contains dataset containing three test attributes
        """
        tasks_per_batch = 3
        # attr_names = []
        inputa = np.zeros((tasks_per_batch, way * shot, self.image_height, self.image_width, self.image_channels),
                    dtype=np.float32)
        labela = np.zeros((tasks_per_batch, way * shot, way), dtype=np.int32)
        
        triplet_key = list(self.metatest_attr_triplet.keys())[input_idx]
        # attr_names.append(" ".join(map(self.id_to_name_fn, triplet_key)))
        attr_idx_3 = self.metatest_attr_triplet[triplet_key] # idx having these attributes
        negative_attr_idx_3 = self.meta_attr_triplet_neg[triplet_key] # idx not having these attributes 
        amb_idx = []
        for i, idx in enumerate([attr_idx_3, negative_attr_idx_3]):
            sampled_idx = np.random.choice(idx, size=shot, replace=False)
            sampled_data = self.test_imgs[sampled_idx][np.newaxis] # 1, shot, H, W, C
            amb_idx.append(sampled_idx)
            inputa[:, i::way] = sampled_data.astype('float32') / 255.
            labela[:, i::way, i] = 1.
        # support_permutation = np.arange(shot * way) # Permute positive and negative samples
        # np.random.shuffle(support_permutation)
        # xs[i], ys[i] = support_imgs[support_permutation], support_labels[support_permutation]
            
        # d val
        inputb = np.zeros((tasks_per_batch, way * eval_samples, self.image_height, self.image_width, self.image_channels),
                    dtype=np.float32)
        labelb = np.zeros((tasks_per_batch, way * eval_samples, way), dtype=np.int32)
        # Get all possible combinations of 2 attributes out of 3
        for t, sub_combs in enumerate(combinations(triplet_key, 2)):
            # attr_names.append(" ".join(map(self.id_to_name_fn, sub_combs)))
            pos_attr_idx = self.metatest_attr_doubles[sub_combs]
            negative_attr_idx = self.metatest_attr_ambig[sub_combs]
            # remove any examples that were sampled above
            pos_attr_idx = np.setdiff1d(pos_attr_idx, amb_idx[0]) 
            negative_attr_idx = np.setdiff1d(negative_attr_idx, amb_idx[1]) 
            for i, idx in enumerate([pos_attr_idx, negative_attr_idx]):
                sampled_idx_query = np.random.choice(idx, size=eval_samples, replace=False)
                sampled_data = self.test_imgs[sampled_idx_query][np.newaxis] # 1, shot, H, W, C   
                inputb[t, i::way] = sampled_data.astype('float32') / 255.
                labelb[t, i::way, i] = 1.
        
        # labels to one-hot encoding
        # labela = onehottify_2d_array(labela)
        # labelb = onehottify_2d_array(labelb)
        # return [xs, xq, ys, yq]
        return [inputa, inputb, labela, labelb]


dataset = CelebAData(123)
dataset.init_testing_params(2)
dataset.get_test_triplet_batch_new(5, 2, 7, 0)
r = dataset._sample_batch('test', 3, 5, 2, 7)

def test():
    print('done')
    dataset = CelebAData(123)
    dataset.get_batch('train', 10, 4, 2, 7)