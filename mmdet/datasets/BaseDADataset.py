import os.path as osp

import mmcv
import numpy as np
from torch.utils.data import Dataset

from .pipelines import Compose
from .registry import DATASETS


@DATASETS.register_module
class BaseDADataset(Dataset):
    """Domain Adaption dataset for detection.

    Annotation format:
    [
        {
            'filename': 'a.jpg',
            'width': 1280,
            'height': 720,
            'ann': {
                'bboxes': <np.ndarray> (n, 4),
                'labels': <np.ndarray> (n, ),
                'bboxes_ignore': <np.ndarray> (k, 4), (optional field)
                'labels_ignore': <np.ndarray> (k, 4) (optional field)
            }
        },
        ...
    ]

    The `ann` field is optional for testing.
    """

    CLASSES = None

    def __init__(self,
                 ann_file,
                 t_ann_file,
                 s_pipeline,
                 t_pipeline,
                 data_root=None,
                 img_prefix='',
                 seg_prefix=None,
                 t_data_root=None,
                 t_img_prefix='',
                 t_seg_prefix=None,
                 test_mode=False):
        self.ann_file = {'source': ann_file, 'target': t_ann_file}
        self.data_root = {'source': data_root, 'target': t_data_root}
        self.img_prefix = {'source': img_prefix, 'target': t_img_prefix}
        self.seg_prefix = {'source': seg_prefix, 'target': t_seg_prefix}
        self.test_mode = test_mode

        # join paths if data_root is specified
        self.join_path('source')
        self.join_path('target')
        # load annotations (and proposals)
        self.img_infos = dict(source=self.load_annotations(self.ann_file['source'], 'source'),
                        target=self.load_annotations(self.ann_file['target'], 'target'))
        # filter images with no annotation during training
        if not test_mode:
            valid_inds_s = self._filter_imgs(key='source')
            valid_inds_t = self._filter_imgs(key='target')
            self.img_infos['source'] = [self.img_infos['source'][i] for i in valid_inds_s]
            self.img_infos['target'] = [self.img_infos['target'][i] for i in valid_inds_t]
        # set group flag for the sampler
        if not self.test_mode:
            self._set_group_flag()
        # processing pipeline
        self.s_pipeline = Compose(s_pipeline)
        self.t_pipeline = Compose(t_pipeline)

    def join_path(self, key):
        if self.data_root[key] is not None:
            if not osp.isabs(self.ann_file[key]):
                self.ann_file[key] = osp.join(self.data_root[key], self.ann_file[key])
            if not (self.img_prefix[key] is None or osp.isabs(self.img_prefix[key])):
                self.img_prefix[key] = osp.join(self.data_root[key], self.img_prefix[key])
            if not (self.seg_prefix[key] is None or osp.isabs(self.seg_prefix[key])):
                self.seg_prefix[key] = osp.join(data_root, self.seg_prefix[key])

    def __len__(self):
        return len(self.img_infos['source'])

    def load_annotations(self, ann_file, key):
        return mmcv.load(ann_file)

    def load_proposals(self, proposal_file):
        return mmcv.load(proposal_file)

    def get_ann_info(self, idx, key):
        return self.img_infos[key][idx]['ann']

    def pre_pipeline(self, results, key):
        results['img_prefix'] = self.img_prefix[key]
        results['seg_prefix'] = self.seg_prefix[key]
        results['bbox_fields'] = []
        results['mask_fields'] = []

    def _filter_imgs(self, key, min_size=32):
        """Filter images too small."""
        valid_inds = []
        for i, img_info in enumerate(self.img_infos[key]):
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
        for i in range(len(self)):
            img_info = self.img_infos['source'][i]
            if img_info['width'] / img_info['height'] > 1:
                self.flag[i] = 1

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_img(idx)
        while True:
            data = self.prepare_train_img(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def rand_sample_target(self):
        idx = np.random.choice(range(len(self.img_infos['target'])), 1)[0]
        return idx

    def prepare_train_img(self, idx):
        s_img_info = self.img_infos['source'][idx]
        ann_info = self.get_ann_info(idx, key='source')
        s_results = dict(img_info=s_img_info, ann_info=ann_info)
        target_idx = self.rand_sample_target()
        t_img_info = self.img_infos['target'][target_idx]
        t_results = dict(img_info=t_img_info)
        self.pre_pipeline(s_results, key='source')
        self.pre_pipeline(t_results, key='target')
        return self.s_pipeline(s_results), self.t_pipeline(t_results)
