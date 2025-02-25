import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from models.metrics import dice_score

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, device, config):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.best_score = 0

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        total_dice = 0
        
        for images, masks in tqdm(self.train_loader, desc="Training"):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, masks)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            total_dice += dice_score(outputs.detach(), masks)
            
        return {
            'train_loss': total_loss / len(self.train_loader),
            'train_dice': total_dice / len(self.train_loader)
        }

    def validate(self):
        self.model.eval()
        total_loss = 0
        total_dice = 0
        
        with torch.no_grad():
            for images, masks in tqdm(self.val_loader, desc="Validation"):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                total_loss += loss.item()
                total_dice += dice_score(outputs, masks)
                
        metrics = {
            'val_loss': total_loss / len(self.val_loader),
            'val_dice': total_dice / len(self.val_loader)
        }
        
        # Save best model
        if metrics['val_dice'] > self.best_score:
            self.save_model()
            self.best_score = metrics['val_dice']
            
        return metrics

    def save_model(self, path="best_model.pth"):
        torch.save(self.model.state_dict(), path)