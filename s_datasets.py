import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from pycocotools.coco import COCO

class CocoSegmentationDataset(Dataset):
    def __init__(self, root, annFile, transforms=None, image_size=(128, 128), n_classes=81):
        self.root = root
        self.coco = COCO(annFile)
        self.ids = list(self.coco.imgs.keys())
        self.transforms = transforms
        self.image_size = image_size
        self.n_classes = n_classes

        # Build a mapping from COCO category_id to contiguous index (0=background, 1...N=classes)
        cats = self.coco.loadCats(self.coco.getCatIds())
        self.catid2trainid = {cat['id']: idx+1 for idx, cat in enumerate(cats)}  # 0 for background

    def __getitem__(self, index):
        img_id = self.ids[index]
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        mask = np.zeros(self.image_size, dtype=np.uint8)  # 0 is background

        for ann in anns:
            cat_id = ann['category_id']
            train_id = self.catid2trainid.get(cat_id, 0)  # 0 for background/unlabeled
            m = self.coco.annToMask(ann)
            m = Image.fromarray(m).resize(self.image_size, resample=Image.NEAREST)
            mask = np.where(np.array(m), train_id, mask)

        img = img.resize(self.image_size)
        if self.transforms:
            img = self.transforms(img)
        else:
            from torchvision import transforms as T
            img = T.ToTensor()(img)
        mask = torch.from_numpy(mask).long()
        return img, mask

    def __len__(self):
        return len(self.ids)
