# gradcam_vit_morf_localCAM_dual.py
# ------------------------------------------------
# Grad‑CAM + MoRF για ViT (COVID  vs  non‑COVID)
# Παράγει:
#   • morf_curve_both.png   (δύο καμπύλες p̂ vs fraction removed)
#   • aopc_covid_curve.png  (AOPC COVID)
#   • aopc_noncovid_curve.png (AOPC non‑COVID)

import os, torch, numpy as np, matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from torchcam.methods import SmoothGradCAMpp

from vit_model import ViTClassifier
from utils import set_seed
from data_loading import build_dataloaders

# ------------------------------------------------------------------
# 1) CONFIG
# ------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(42)

split_dir  = "/content/dataset_split"
vit_ckpt   = "/content/best_vit.pth"
output_dir = "outputs/morf_gradcam_vit"
os.makedirs(output_dir, exist_ok=True)

BATCH   = 1
WORKERS = 2
STEPS   = 20

# ------------------------------------------------------------------
# 2) DATA & MODEL
# ------------------------------------------------------------------
_, _, test_loader = build_dataloaders(split_dir, BATCH, WORKERS)
test_ds = test_loader.dataset
print("Detected classes:", test_ds.classes)
print("Class→Index mapping:", test_ds.class_to_idx)

model = ViTClassifier(pretrained=False).to(device)
model.load_state_dict(torch.load(vit_ckpt, map_location=device))
model.eval()

try:
    TARGET_LAYER = model.vit.conv_proj
except AttributeError:
    TARGET_LAYER = model.vit.encoder.layers[-1].ln_1

# ------------------------------------------------------------------
# 3) PREPROCESS
# ------------------------------------------------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

# ------------------------------------------------------------------
# 4) HELPERS
# ------------------------------------------------------------------

def _unnorm(img_t):
    x = img_t.clone().squeeze(0).cpu().numpy()
    for i in range(3):
        x[i] = x[i] * IMAGENET_STD[i] + IMAGENET_MEAN[i]
    return np.clip(x.transpose(1, 2, 0), 0, 1)


def _get_cam_heatmap(img_t, cls):
    cam = SmoothGradCAMpp(model, TARGET_LAYER)
    logits = model(img_t.to(device))
    h = cam(cls, logits)[0].detach().cpu().numpy()
    cam.remove_hooks()
    h = np.squeeze(h)
    if h.ndim == 1:            # vector → square
        side = int(np.sqrt(h.shape[0]))
        h = h.reshape(side, side)
    h = np.maximum(h, 0)
    h /= h.max() if h.max() > 0 else 1
    h = np.asarray(Image.fromarray((h*255).astype(np.uint8))
                   .resize((224, 224), Image.BICUBIC), np.float32) / 255.0
    return h


def save_overlay(img_t, cls, path, alpha=0.6):
    heat = _get_cam_heatmap(img_t, cls)
    bg   = _unnorm(img_t)
    cmap = plt.get_cmap('viridis')(heat)[..., :3]
    out  = bg*(1-alpha*heat[..., None]) + cmap*(alpha*heat[..., None])
    Image.fromarray((out*255).astype(np.uint8)).save(path)


def morf_confidences(img_t, cls, steps=20):
    with torch.no_grad():
        p0 = torch.softmax(model(img_t.to(device)), 1)[0, cls].item()
    heat = _get_cam_heatmap(img_t, cls)
    flat = heat.flatten()
    confs = [p0]
    img_np = img_t.cpu().numpy().astype(np.float32)
    for k in range(1, steps+1):
        thr = np.percentile(flat, 100*(1 - k/steps))
        mask = heat >= thr
        pert = img_np.copy()
        pert[0, :, mask] = 0
        with torch.no_grad():
            p = torch.softmax(model(torch.tensor(pert, device=device)), 1)[0, cls].item()
        confs.append(p)
    return confs

# ------------------------------------------------------------------
# 5) MAIN LOOP
# ------------------------------------------------------------------
covid_idx     = test_ds.class_to_idx['COVID']
noncovid_idx  = test_ds.class_to_idx['non-COVID']

covid_curves    = []
noncovid_curves = []

for pth, gt in test_ds.samples:
    img = Image.open(pth).convert('L').convert('RGB')
    x   = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        pred = model(x.to(device)).argmax(1).item()
    if pred != gt:
        continue  # skip misclassified

    save_overlay(x, pred, os.path.join(output_dir, f"gradcam_{os.path.basename(pth)}"))
    conf = morf_confidences(x, pred, STEPS)

    if gt == covid_idx:
        covid_curves.append(conf)
    else:  # non‑COVID
        noncovid_curves.append(conf)

# ------------------------------------------------------------------
# 6) PLOTS
# ------------------------------------------------------------------
frac = np.linspace(0, 1, STEPS+1)

if covid_curves and noncovid_curves:
    arr_covid = np.array(covid_curves, np.float32)
    arr_non   = np.array(noncovid_curves, np.float32)

    # MoRF curves (mean probability)
    plt.figure(figsize=(6,4))
    plt.plot(frac, arr_covid.mean(0), marker='o', label='COVID')
    plt.plot(frac, arr_non.mean(0),  marker='s', label='non‑COVID')
    plt.xlabel('Fraction of top pixels removed')
    plt.ylabel('Mean predicted probability')
    plt.title('MoRF Curves (ViT)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'morf_curve_both.png'), dpi=120)
    plt.show()

    # AOPC COVID
    diffs_covid = arr_covid[:,0,None] - arr_covid[:,1:]
    aopc_covid  = diffs_covid.mean(0)
    plt.figure(figsize=(6,4))
    plt.plot(range(1,STEPS+1), aopc_covid, marker='o')
    plt.fill_between(range(1,STEPS+1), aopc_covid, alpha=0.3)
    plt.xlabel('Masking step (k)')
    plt.ylabel('Mean AOPC')
    plt.title('AOPC Curve – COVID')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aopc_covid_curve.png'), dpi=120)
    plt.show()

    # AOPC non‑COVID
    diffs_non  = arr_non[:,0,None] - arr_non[:,1:]
    aopc_non   = diffs_non.mean(0)
    plt.figure(figsize=(6,4))
    plt.plot(range(1,STEPS+1), aopc_non, marker='s', color='tab:orange')
    plt.fill_between(range(1,STEPS+1), aopc_non, alpha=0.3, color='tab:orange')
    plt.xlabel('Masking step (k)')
    plt.ylabel('Mean AOPC')
    plt.title('AOPC Curve – non‑COVID')
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'aopc_noncovid_curve.png'), dpi=120)
    plt.show()

else:
    print("Δεν βρέθηκαν αρκετές σωστά ταξινομημένες εικόνες και για τις δύο κλάσεις.")


