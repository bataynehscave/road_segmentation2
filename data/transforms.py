import albumentations as A

def get_train_transforms(img_size=(512, 512)):
    return A.Compose([
        A.Resize(*img_size),
        A.HueSaturationValue(20, 30, 30, p=0.5),
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=45,
            p=0.5
        )
    ])


def get_val_transforms(img_size=(512, 512)):
    return A.Compose([
        A.Resize(*img_size),
        # Typically just resizing for validation
    ])