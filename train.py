import torch
from torch.utils.data import DataLoader  
from configs.defaults import Config      
from data.dataloader import SegmentationDataset
from data.transforms import get_train_transforms, get_val_transforms
from models.dinknet import DinkNet34
from models.losses import DiceBCELoss
from utils.trainer import Trainer



config = Config()
train_tfs = get_train_transforms(config.img_size)
val_tfs = get_val_transforms(config.img_size)

# Datasets
train_ds = SegmentationDataset(config.train_image_dir, config.train_mask_dir, train_tfs)
val_ds = SegmentationDataset(config.val_image_dir, config.val_mask_dir, val_tfs)

# Dataloaders
train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=config.batch_size)

# Model & Training
model = DinkNet34()
criterion = DiceBCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    criterion=criterion,
    device="cuda",
    config=config
)

for epoch in range(config.epochs):
    train_metrics = trainer.train_epoch()
    val_metrics = trainer.validate()
    
    print(f"Epoch {epoch+1}/{config.epochs}")
    print(f"Train Loss: {train_metrics['train_loss']:.4f} | Dice: {train_metrics['train_dice']:.4f}")
    print(f"Val Loss: {val_metrics['val_loss']:.4f} | Dice: {val_metrics['val_dice']:.4f}\n")