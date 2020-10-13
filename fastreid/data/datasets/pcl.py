# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import os, glob
import os.path as osp
import re
import warnings

from .bases import ImageDataset
from ..datasets import DATASET_REGISTRY


@DATASET_REGISTRY.register()
class PCL(ImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    dataset_dir = 'PCL'
    dataset_name = 'pcl'
    def __init__(self, root='datasets', **kwargs):
        # self.root = osp.abspath(osp.expanduser(root))
        self.root = root
        self.dataset_dir = osp.join(self.root, self.dataset_dir)

        self.train_dir = osp.join(self.dataset_dir, 'train')
        self.query_dir = osp.join(self.dataset_dir, 'image_A', 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'image_A', 'gallery')

        required_files = [
            self.dataset_dir,
            self.train_dir,
            self.query_dir,
            self.gallery_dir,
        ]
        self.check_before_run(required_files)

        train = self.process_train(self.train_dir)
        query = self.process_test(self.query_dir)
        gallery = self.process_test(self.gallery_dir)

        super(PCL, self).__init__(train, query, gallery, **kwargs)


    def process_train(self, dir_path, filter_thres=3):
        label_file = os.path.join(dir_path, 'label.txt')
        with open(label_file, 'r') as f:
            lines = f.readlines()
        im_dir = os.path.join(dir_path, 'images')
        pid_counts = {}
        data = []
        for line in lines:
            name, pid = line.strip().split(':')
            path = os.path.join(im_dir, name)
            pid = self.dataset_name + '_' + pid
            data.append([path, pid, 1])
            if not pid in pid_counts:
                pid_counts[pid] = 1
            else:
                pid_counts[pid] += 1
        data1 = []
        for d in data:
            if pid_counts[d[1]] > filter_thres:
                data1.append(d)
        data = sorted(data1, key=lambda x: int(x[1].split('_')[-1]))
        return data

    def process_test(self, dir_path):
        ims = sorted(os.listdir(dir_path))
        data = []
        for im in ims:
            data.append([osp.join(dir_path, im), '1', '1'])
        return data

