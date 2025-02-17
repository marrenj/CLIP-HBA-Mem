# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp

from .base_sr_dataset import BaseSRDataset
from .registry import DATASETS

import pandas as pd
import scipy.io
import numpy as np
import os


@DATASETS.register_module()
class HebartDataset(BaseSRDataset):

    def __init__(self,
                 img_folder,
                 pipeline,
                 ann_file,
                 scale=1,
                 test_mode=False):
        super().__init__(pipeline, scale, test_mode)
        self.img_folder = str(img_folder)
        # self.ann_file = pd.read_csv(ann_file, error_bad_lines=True)
        self.ann_file = pd.read_csv(ann_file, on_bad_lines='error')
        if test_mode:
            self.data_infos = self.ann_file[self.ann_file.set=='test'].reset_index()
            # Extracting all dimension scores for the test set
            self.dimension_scores = self.ann_file[self.ann_file.set=='test'].drop(columns=['image_name', 'set'])
        else:
            self.data_infos = self.ann_file[self.ann_file.set=='training'].reset_index()
            # Extracting all dimension scores for the training set
            self.dimension_scores = self.ann_file[self.ann_file.set=='training'].drop(columns=['image_name', 'set'])

    def load_annotations(self):
        return 0

    def __getitem__(self, idx):
        """Get item at each call.
        Args:
            idx (int): Index for getting each item.
        """
  
        results = dict(
            lq_path=osp.join(self.img_folder, self.data_infos['image_name'][idx]),
            gt=self.dimension_scores[idx]/100)
        results['scale'] = self.scale
        return self.pipeline(results)