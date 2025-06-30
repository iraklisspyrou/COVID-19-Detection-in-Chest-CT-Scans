from torchvision import transforms

train_tf = transforms.Compose([
    # 1) Φόρτωση σε γκρι (single channel)
    transforms.Grayscale(num_output_channels=1),

    # 2) Άλλα standard transforms
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),

    # 3) Normalize με mean/std για 1 κανάλι
    transforms.Normalize(mean=[0.5], std=[0.5])
])

val_tf = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
