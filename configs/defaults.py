from dataclasses import dataclass

@dataclass
class Config:
    # Data
    img_size: tuple = (512, 512)
    batch_size: int = 8
    num_workers: int = 4
    
    # Training
    epochs: int = 50
    lr: float = 1e-4
    weight_decay: float = 1e-5
    
    # Model
    model_name: str = "DinkNet34"
    num_classes: int = 1
    
    # Paths
    train_image_dir: str = "data/train/images"
    train_mask_dir: str = "data/train/masks"
    val_image_dir: str = "data/val/images"
    val_mask_dir: str = "data/val/masks"