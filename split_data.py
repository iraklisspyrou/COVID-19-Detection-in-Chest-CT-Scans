import os
import shutil
import random

# Paths
base_dir = r"C:\Users\INFOLYSiS\Desktop\MSc AI\Multimodal_ML"
covid_src = os.path.join(base_dir, "COVID")
non_covid_src = os.path.join(base_dir, "non-COVID")

# Output structure
output_dir = os.path.join(base_dir, "dataset_split")
splits = ['train', 'val', 'test']
categories = ['COVID', 'non-COVID']

# Δημιουργία φακέλων εξόδου
for split in splits:
    for category in categories:
        path = os.path.join(output_dir, split, category)
        os.makedirs(path, exist_ok=True)

def split_and_copy_images(src_folder, category_name):
    images = [f for f in os.listdir(src_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(images)
    total = len(images)
    
    train_split = int(0.7 * total)
    val_split = int(0.15 * total)
    
    train_imgs = images[:train_split]
    val_imgs = images[train_split:train_split + val_split]
    test_imgs = images[train_split + val_split:]

    print(f"\n[{category_name}]")
    print(f"Total: {total} | Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

    for file in train_imgs:
        src_path = os.path.join(src_folder, file)
        dest_path = os.path.join(output_dir, 'train', category_name, file)
        shutil.copy2(src_path, dest_path)

    for file in val_imgs:
        src_path = os.path.join(src_folder, file)
        dest_path = os.path.join(output_dir, 'val', category_name, file)
        shutil.copy2(src_path, dest_path)

    for file in test_imgs:
        src_path = os.path.join(src_folder, file)
        dest_path = os.path.join(output_dir, 'test', category_name, file)
        shutil.copy2(src_path, dest_path)

# Τρέχουμε για κάθε κατηγορία
split_and_copy_images(covid_src, "COVID")
split_and_copy_images(non_covid_src, "non-COVID")
