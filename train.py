# train.py
import os
import torch
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Name:", torch.cuda.get_device_name(0))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)
    captions = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions

import json
from scripts.dataset import CocoCaptionDataset

# Load annotations
with open(r'C:\Users\Pratyush\image_captioning_segmentation\data\coco2017\annotations_trainval2017\annotations\captions_train2017.json', 'r') as f:
    captions_data = json.load(f)
annotations = captions_data['annotations']

# Load word2idx and idx2word
import pickle
with open('word2idx.pkl', 'rb') as f:
    word2idx = pickle.load(f)
with open('idx2word.pkl', 'rb') as f:
    idx2word = pickle.load(f)

# --- NEW: Split into training and validation sets ---
from sklearn.model_selection import train_test_split
train_ann, val_ann = train_test_split(annotations, test_size=0.2, random_state=42)

# --- NEW: Create train and validation datasets/loaders ---
train_dataset = CocoCaptionDataset(
    image_folder='data/coco2017/train2017',
    annotations=train_ann,
    word2idx=word2idx,
    transform=transform
)
val_dataset = CocoCaptionDataset(
    image_folder='data/coco2017/train2017',
    annotations=val_ann,
    word2idx=word2idx,
    transform=transform
)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

# Test: Iterate over one batch
for images, captions in train_loader:
    print("Images batch shape:", images.shape)
    print("Captions batch shape:", captions.shape)
    print("First caption indices:", captions[0])
    break  # Only check the first batch

# Import your models
from models.encoder import EncoderCNN
from models.decoder import DecoderRNN

embed_size = 256
hidden_size = 512
vocab_size = len(word2idx)

encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

import torch.nn as nn
import torch.optim as optim

criterion = nn.CrossEntropyLoss(ignore_index=word2idx['<pad>'])
params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
optimizer = optim.Adam(params, lr=1e-3)

num_epochs = 58
patience = 8  # Early stopping patience

# --- RESUME FROM CHECKPOINT IF AVAILABLE ---
start_epoch = 0
best_val_loss = float('inf')
epochs_no_improve = 0

#Checkpoint integration

if os.path.exists('checkpoint.pth'):
    print("Loading checkpoint...")
    checkpoint = torch.load('checkpoint.pth', map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_val_loss = checkpoint['best_val_loss']
    epochs_no_improve = checkpoint['epochs_no_improve']
    print(f"Resuming training from epoch {start_epoch}")
else:
    print("No checkpoint found, starting from scratch.")

# --- NEW: Validation evaluation function ---
def evaluate_on_validation(encoder, decoder, val_loader, criterion, device):
    encoder.eval()
    decoder.eval()
    val_loss = 0
    with torch.no_grad():
        for images, captions in val_loader:
            images = images.to(device)
            captions = captions.to(device)
            features = encoder(images)
            outputs = decoder(features, captions[:, :-1])
            loss = criterion(outputs.reshape(-1, outputs.size(2)), captions[:, 1:].reshape(-1))
            val_loss += loss.item()
    return val_loss / len(val_loader)

print("Starting training...")
for epoch in range(start_epoch, num_epochs):  # Critical fix - starts from saved epoch
    encoder.train()
    decoder.train()
    total_loss = 0
    for i, (images, captions) in enumerate(train_loader):
        images = images.to(device)
        captions = captions.to(device)
        features = encoder(images)
        outputs = decoder(features, captions[:, :-1])
        loss = criterion(outputs.reshape(-1, outputs.size(2)), captions[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if (i+1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
    avg_loss = total_loss / len(train_loader)

    # --- VALIDATION ---
    avg_val_loss = evaluate_on_validation(encoder, decoder, val_loader, criterion, device)
    print(f"Epoch [{epoch+1}/{num_epochs}] finished. Avg Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # --- EARLY STOPPING ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(encoder.state_dict(), 'encoder_best.pth')
        torch.save(decoder.state_dict(), 'decoder_best.pth')
        print("Validation loss improved, model saved!")
    else:
        epochs_no_improve += 1
        print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= patience:
            print("Early stopping triggered.")
            break

# --- SAVE CHECKPOINT AFTER EACH EPOCH ---
    torch.save({
        'epoch': epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_val_loss': best_val_loss,
        'epochs_no_improve': epochs_no_improve
    }, 'checkpoint.pth')

print("Training complete!")