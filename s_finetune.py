import os
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from s_datasets import CocoSegmentationDataset
from Unet import UNetResNet
from utils import mean_iou
import numpy as np
from tqdm import tqdm

def main():
    # Settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_classes = 81
    num_epochs = 100
    batch_size = 8
    checkpoint_path = "unet_coco_checkpoint1.pth"
    best_model_path = "unet_coco_finetuned.pth"
    early_stopping_patience = 10
    image_size = (128, 128)

    print("="*60)
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
    print("="*60)

    # Data
    train_dataset = CocoSegmentationDataset(
        root='data/coco2017/train2017',
        annFile='data/coco2017/annotations_trainval2017/annotations/instances_train2017.json',
        transforms=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]),
        image_size=image_size
    )
    val_dataset = CocoSegmentationDataset(
        root='data/coco2017/val2017/val2017',
        annFile='data/coco2017/annotations_trainval2017/annotations/instances_val2017.json',
        transforms=transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]),
        image_size=image_size
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model, optimizer, loss
    model = UNetResNet(n_classes=n_classes, out_size=image_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    start_epoch = 0
    best_iou = 0.0
    epochs_no_improve = 0
    early_stop = False

    # Resume from checkpoint if exists
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_iou = checkpoint['best_iou']
        epochs_no_improve = checkpoint.get('epochs_no_improve', 0)
        print(f"Resumed from epoch {start_epoch}, best IoU so far: {best_iou:.4f}")

    for epoch in range(start_epoch, num_epochs):
        if early_stop:
            print("Early stopping triggered. Training halted.")
            break

        model.train()
        train_loss = 0.0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False)
        for images, masks in train_bar:
            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            train_bar.set_postfix(loss=loss.item())
        train_loss /= len(train_loader.dataset)

        # Validation
        model.eval()
        iou_scores = []
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False)
        with torch.no_grad():
            for images, masks in val_bar:
                images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                outputs = model(images)
                iou = mean_iou(outputs, masks, n_classes)
                iou_scores.append(iou)
                val_bar.set_postfix(iou=iou)
        mean_val_iou = np.nanmean(iou_scores)
        print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Mean IoU: {mean_val_iou:.4f}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_iou': best_iou,
            'epochs_no_improve': epochs_no_improve,
        }
        torch.save(checkpoint, checkpoint_path)

        # Save best model and update early stopping
        if mean_val_iou > best_iou:
            best_iou = mean_val_iou
            torch.save(model.state_dict(), best_model_path)
            print(f"Best model saved at epoch {epoch+1} with IoU: {best_iou:.4f}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            print(f"No improvement for {epochs_no_improve} epoch(s).")

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping: No improvement in IoU for {early_stopping_patience} consecutive epochs.")
            early_stop = True

    print("Training complete. Best IoU achieved:", best_iou)

if __name__ == "__main__":
    main()
