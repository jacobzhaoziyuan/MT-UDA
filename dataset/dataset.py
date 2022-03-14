import glob
import torch.utils.data as data
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import re


def sort_vol_slice(path):
    vol = re.findall('[a-z]+_([0-9]+)_.+?\.npy', path.split('/')[-1])[0]
    slice_ = re.findall('[a-z]+_[0-9]+_([0-9]+).+', path.split('/')[-1])[0]
    return int(vol)*1000+int(slice_)


class WHS_dataset(data.Dataset):
    def __init__(self, data_dir, supervised=False, transforms=None):
        print(data_dir)
        if '.txt' in data_dir:
            with open(data_dir, 'r') as fp:
                self.raw_data = [f.strip() for f in fp.readlines()]
        else:
            self.raw_data = [f for f in glob.glob(data_dir+'/*') if '_label.npy' not in f]
        self.raw_data = sorted(self.raw_data, key = lambda x: sort_vol_slice(x))
        self.transform = transforms
        self.supervised = supervised
        self.idx_mapping = {i: i for i in range(len(self.raw_data))}
        
    def update_idx_mapping(self, idx_mapping=None):
        """
        For targetlike_dataloaders, first half of the data and the second half is of different modality but have same origin, (e.g. `fake_mr` and `ct`)
        and in order to train structure loss, we need to align slices from both modalities in pairs.
        So we disable the default shuffle in outside dataloader, and instead use `idx_mapping` to shuffle slices. 
        Here only the first half of data is shuffled, then copy and shift the assignment to the second half. 
        """
        if idx_mapping is not None:
            self.idx_mapping = idx_mapping
        else:
            num_label_slices = len(self.raw_data)
            self.idx_mapping = {i: i_ for i, i_ in zip(range(num_label_slices), np.random.permutation(range(num_label_slices)))}

    def __getitem__(self, idx):
        idx =  self.idx_mapping[idx]
        img_path = self.raw_data[idx]
        img = np.load(img_path)
        img_modal = re.findall('(.+?)_[0-9]+.+', img_path.split('/')[-1])[0]
        img_vol = re.findall('.+_([0-9]+)_[0-9]+.+', img_path.split('/')[-1])[0]

        if self.supervised:
            if 'cyc' in img_path:
                gt_path = re.sub('\.npy', '_label.npy', img_path)
                if 'cyc_mr' in gt_path:
                    gt_path = re.sub('cyc_mr/cyc_mr', 'org_mr/mr', gt_path)
                elif 'cyc_ct' in gt_path:
                    gt_path = re.sub('cyc_ct/cyc_ct', 'org_ct/ct', gt_path)
            else:
                gt_path = re.sub('\.npy','_label.npy',img_path)
            gt = np.load(gt_path)


        if self.transform and self.supervised:
            seq = iaa.Affine(scale=(0.9, 1.1), rotate=(-10, 10))
            seq_det = seq.to_deterministic()
            gt = gt.astype(np.uint8)
            segmap = SegmentationMapsOnImage(gt, shape=gt.shape)
            img, segmaps_aug = seq_det(image=img, segmentation_maps=segmap)
            gt = segmaps_aug.arr.squeeze().astype(np.uint8)
        return {"img": img[np.newaxis,...], "gt": gt.astype(np.uint8), "path": img_path} if self.supervised else {'img': img[np.newaxis,...], 'gt': 0, "path": img_path}

    def __len__(self):
        return len(self.raw_data)
