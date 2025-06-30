from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from PIL import Image
import os

class CTDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir  = root_dir          # dataset_split/train/COVID
        self.transform = transform
        self.paths     = [os.path.join(root_dir, f) for f in os.listdir(root_dir)
                          if f.lower().endswith(('.png','.jpg','.jpeg'))]

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img_path = self.paths[idx]
        label    = 0 if 'non-COVID' in img_path else 1
        img      = Image.open(img_path).convert('L')   # 1-channel
        if self.transform:
            img  = self.transform(img)
        return img, label

def build_dataloaders(split_dir, batch_size=32, num_workers=4):
    from transforms import train_tf, val_tf

    train_dataset = datasets.ImageFolder(
        root=os.path.join(split_dir, 'train'),
        transform=train_tf
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(split_dir, 'val'),
        transform=val_tf
    )
    test_dataset = datasets.ImageFolder(
        root=os.path.join(split_dir, 'test'),
        transform=val_tf
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader