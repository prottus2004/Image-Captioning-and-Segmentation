import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import nltk

class CocoCaptionDataset(Dataset):
    def __init__(self, image_folder, annotations, word2idx, transform=None):
        self.image_folder = image_folder
        self.annotations = annotations
        self.word2idx = word2idx
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_id = ann['image_id']
        caption = ann['caption']
        image_filename = f"{image_id:012d}.jpg"
        image_path = os.path.join(self.image_folder, image_filename)
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        tokens = ['<start>'] + nltk.tokenize.word_tokenize(caption.lower()) + ['<end>']
        caption_indices = [self.word2idx.get(token, self.word2idx['<unk>']) for token in tokens]
        return image, torch.tensor(caption_indices)
