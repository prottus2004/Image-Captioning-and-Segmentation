import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2 as transforms_v2
from PIL import Image
import pickle
import random
import json
import nltk
from nltk.corpus import wordnet
from torch.nn.utils.rnn import pad_sequence
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ---- 1. Load Vocabulary ----
with open('word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)
with open('idx2word.pkl', 'rb') as f:
    idx2word = pickle.load(f)

# ---- 2. Define Augmentation Transforms ----
train_transform = transforms_v2.Compose([
    transforms_v2.ToImage(),
    transforms_v2.RandomResizedCrop((224, 224)),
    transforms_v2.RandomHorizontalFlip(p=0.5),
    transforms_v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms_v2.Compose([
    transforms_v2.ToImage(),
    transforms_v2.Resize((224, 224)),
    transforms_v2.ToDtype(torch.float32, scale=True),
    transforms_v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ---- 3. Text Augmentation ----
def synonym_replacement(tokens, n=1):
    new_tokens = tokens.copy()
    for _ in range(n):
        idxs = [i for i, word in enumerate(new_tokens) if word not in ['<start>', '<end>', '<pad>', '<unk>']]
        if not idxs:
            break
        idx = random.choice(idxs)
        synonyms = set()
        for syn in wordnet.synsets(new_tokens[idx]):
            for lemma in syn.lemmas():
                synonym = lemma.name().replace('_', ' ')
                if synonym != new_tokens[idx]:
                    synonyms.add(synonym)
        if synonyms:
            new_tokens[idx] = random.choice(list(synonyms))
    return new_tokens

# ---- 4. Custom Dataset with Text Augmentation ----
class AugmentedCocoCaptionDataset(Dataset):
    def __init__(self, image_folder, annotations, word2idx, transform, augment_text=False):
        self.image_folder = image_folder
        self.annotations = annotations
        self.word2idx = word2idx
        self.transform = transform
        self.augment_text = augment_text
        self.captions = [
            ['<start>'] + nltk.word_tokenize(annot['caption'].lower()) + ['<end>']
            for annot in annotations
        ]
        self.image_ids = [annot['image_id'] for annot in annotations]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.image_folder, f'{image_id:012d}.jpg')
        image = Image.open(image_path).convert('RGB')
        image = self.transform(image)
        tokens = self.captions[idx]
        if self.augment_text and random.random() < 0.5:
            tokens = synonym_replacement(tokens)
        caption_indices = [self.word2idx.get(word, self.word2idx['<unk>']) for word in tokens]
        return image, torch.tensor(caption_indices), len(caption_indices)

# ---- 5. Load Annotations and Split ----
with open(r'C:\Users\Pratyush\image_captioning_segmentation\data\coco2017\annotations_trainval2017\annotations\captions_train2017.json', 'r') as f:
    captions_data = json.load(f)
annotations = captions_data['annotations']
train_ann, val_ann = train_test_split(annotations, test_size=0.2, random_state=42)

# ---- 6. Collate Function ----
def collate_fn(batch):
    images, captions, lengths = zip(*batch)
    images = torch.stack(images)
    lengths = torch.tensor(lengths)
    captions = pad_sequence(captions, batch_first=True, padding_value=word2idx['<pad>'])
    return images, captions, lengths

# ---- 7. Model Setup ----
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN

def save_checkpoint(state, filename='finetune_checkpoint5.pth'):
    torch.save(state, filename)

def load_checkpoint(filename='finetune_checkpoint5.pth', device='cuda'):
    if os.path.exists(filename):
        print(f"Loading checkpoint from {filename}")
        return torch.load(filename, map_location=device)
    return None

def evaluate_on_validation(encoder, decoder, val_loader, criterion, device):
    encoder.eval()
    decoder.eval()
    val_loss = 0
    with torch.no_grad():
        for images, captions, lengths in val_loader:
            images = images.to(device)
            captions = captions.to(device)
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                features = encoder(images)
                outputs = decoder(features, captions[:, :-1])  # Only features and captions
                loss = criterion(outputs.reshape(-1, outputs.size(2)), captions[:, 1:].reshape(-1))
            val_loss += loss.item()
    return val_loss / len(val_loader)

if __name__ == "__main__":

    # ---- 8. DataLoader (must be inside main on Windows) ----
    train_dataset = AugmentedCocoCaptionDataset(
        image_folder='data/coco2017/train2017',
        annotations=train_ann,
        word2idx=word2idx,
        transform=train_transform,
        augment_text=True
    )
    val_dataset = AugmentedCocoCaptionDataset(
        image_folder='data/coco2017/train2017',
        annotations=val_ann,
        word2idx=word2idx,
        transform=val_transform,
        augment_text=False
    )

    train_loader = DataLoader(
        train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn,
        num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn,
        num_workers=2, pin_memory=True
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    if device.type == 'cuda':
        print("CUDA device name:", torch.cuda.get_device_name(0))

    encoder = EncoderCNN(embed_size=256).to(device)
    decoder = DecoderRNN(embed_size=256, hidden_size=512, vocab_size=len(word2idx)).to(device)

    if os.path.exists('encoder_best_finetuned5.pth'):
        encoder.load_state_dict(torch.load('encoder_best_finetuned5.pth', map_location=device))
    if os.path.exists('decoder_best_finetuned5.pth'):
        decoder.load_state_dict(torch.load('decoder_best_finetuned5.pth', map_location=device))

    criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
    params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
    optimizer = optim.Adam(params, lr= 1e-3)
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)

    scaler = torch.amp.GradScaler("cuda")

    num_epochs = 30
    patience = 5
    start_epoch = 0
    best_val_loss = float('inf')
    epochs_no_improve = 0

    checkpoint = load_checkpoint(device=device)
    if checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        epochs_no_improve = checkpoint['epochs_no_improve']
        print(f"Resuming from epoch {start_epoch}")

    print("Starting fine-tuning with augmented data...")

    for epoch in range(start_epoch, num_epochs):
        encoder.train()
        decoder.train()
        total_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (images, captions, lengths) in progress_bar:
            images = images.to(device, non_blocking=True)
            captions = captions.to(device, non_blocking=True)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                features = encoder(images)
                outputs = decoder(features, captions[:, :-1])  # Only features and captions
                loss = criterion(outputs.reshape(-1, outputs.size(2)), captions[:, 1:].reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), max_norm=2.0)
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            progress_bar.set_description(f"Epoch [{epoch+1}/{num_epochs}] Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        avg_val_loss = evaluate_on_validation(encoder, decoder, val_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] finished. Avg Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        scheduler.step(avg_val_loss)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(encoder.state_dict(), 'encoder_best_finetuned5.pth')
            torch.save(decoder.state_dict(), 'decoder_best_finetuned5.pth')
            print("Validation loss improved, model saved!")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
            if epochs_no_improve >= patience:
                print("Early stopping triggered.")
                break

        save_checkpoint({
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'epochs_no_improve': epochs_no_improve
        })

    print("Fine-tuning complete!")
