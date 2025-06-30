import os, random, torch, numpy as np

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dirs(*dirs):
    for d in dirs:
        os.makedirs(d, exist_ok=True)
