import os
import random
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms
from torchcam.methods import SmoothGradCAMpp

from ResNet import ResNet18               # Assumes ResNet18 class is defined in ResNet.py
from utils import set_seed                # Assumes set_seed in utils.py
from data_loading import build_dataloaders  # Assumes build_dataloaders in data_loading.py
from transforms import val_tf             # Assumes val_tf in transforms.py

# ----------------------------
# 1) BASIC CONFIGURATION
# ----------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
set_seed(42)

split_dir  = "/content/dataset_split"   # folder containing train/val/test subfolders
ckpt_path  = "/content/best_resnet.pth" # path to the trained ResNet18 weights
output_dir = "outputs/gradcam_full"     # where heatmaps and plots will be saved
os.makedirs(output_dir, exist_ok=True)

batch_size  = 1
num_workers = 2
num_steps   = 20  # L = 20 MoRF steps

# ----------------------------
# 2) LOAD DATALOADER & MODEL
# ----------------------------
train_loader, val_loader, test_loader = build_dataloaders(
    split_dir=split_dir,
    batch_size=batch_size,
    num_workers=num_workers
)

test_dataset = test_loader.dataset
test_samples = test_dataset.samples   # list of (img_path, label)

# Inspect how ImageFolder assigned class indices:
print("Detected classes:", test_dataset.classes)
print("Class to index mapping:", test_dataset.class_to_idx)
# For example: ['COVID', 'NonCOVID'] and {'COVID': 0, 'NonCOVID': 1}

model = ResNet18(pretrained=False).to(device)
checkpoint = torch.load(ckpt_path, map_location=device)
model.load_state_dict(checkpoint)
model.eval()

target_layer = model.net.layer4[-1]  # layer for Grad-CAM

# ----------------------------
# 3) HELPER FUNCTION: GENERATE GRAD-CAM OVERLAY
# ----------------------------
def generate_gradcam_viridis_from_tensor(img_tensor: torch.Tensor,
                                         model: torch.nn.Module,
                                         target_class: int,
                                         alpha: float = 0.6):
    """
    Produce a Viridis Grad-CAM overlay for the specified target_class
    from a normalized tensor (1,1,224,224). Returns a PIL Image.
    """
    model.eval()
    cam_extractor = SmoothGradCAMpp(model, target_layer)

    logits = model(img_tensor.to(device))
    # No need to recompute predicted class; we pass target_class explicitly
    acts = cam_extractor(target_class, logits)
    amap = acts[0].squeeze().cpu().detach().numpy()

    # ReLU + normalize to [0,1]
    amap = np.maximum(amap, 0)
    if amap.max() > 0:
        amap = amap / amap.max()
    else:
        amap = np.zeros_like(amap)

    # Resize to 224×224
    heatmap_resized = np.array(
        Image.fromarray((amap * 255).astype(np.uint8))
             .resize((224, 224), resample=Image.BICUBIC),
        dtype=np.float32
    ) / 255.0

    cam_extractor.remove_hooks()

    # Create grayscale background from img_tensor
    gray_norm = img_tensor[0, 0].cpu().numpy()    # normalized [-1,1]
    gray_orig = (gray_norm * 0.5) + 0.5           # in [0,1]
    bg_rgb = np.stack([gray_orig]*3, axis=-1)     # shape (224,224,3)

    # Viridis colormap
    cmap = plt.get_cmap('viridis')
    heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # (224,224,3)

    # Blend: background * (1 - α*heatmap) + heatmap_colored * (α*heatmap)
    alpha_map = (heatmap_resized[..., None] * alpha)   # (224,224,1)
    blended = bg_rgb * (1 - alpha_map) + heatmap_colored * alpha_map
    blended = np.clip(blended, 0, 1)

    blended_pil = Image.fromarray((blended * 255).astype(np.uint8))
    return blended_pil

# ----------------------------
# 4) HELPER FUNCTION: COMPUTE MoRF FOR A SPECIFIED CLASS PROBABILITY
# ----------------------------
def compute_morf_for_tensor_confidence(img_tensor: torch.Tensor,
                                       model: torch.nn.Module,
                                       target_class: int,
                                       num_steps: int = 20):
    """
    Returns a list of target_class probabilities:
    [p(x^(0)), p(x^(1)), ..., p(x^(L))]
    where x^(k) is the image after masking the top k/L most relevant pixels
    for the target_class.
    """
    model.eval()
    cam_extractor = SmoothGradCAMpp(model, target_layer)

    img_tensor = img_tensor.to(device=device, dtype=torch.float32)
    logits     = model(img_tensor)
    probs      = torch.nn.functional.softmax(logits, dim=1)

    orig_confidence = probs[0, target_class].item()

    acts = cam_extractor(target_class, logits)
    amap = acts[0].squeeze().cpu().detach().numpy()

    amap = np.maximum(amap, 0)
    if amap.max() > 0:
        amap = amap / amap.max()
    else:
        amap = np.zeros_like(amap)

    heatmap_resized = np.array(
        Image.fromarray((amap * 255).astype(np.uint8))
             .resize((224, 224), resample=Image.BICUBIC),
        dtype=np.float32
    ) / 255.0

    cam_extractor.remove_hooks()
    h_flat = heatmap_resized.flatten()

    confidences = [orig_confidence]
    inp_norm_np = img_tensor.cpu().numpy().astype(np.float32)

    for k in range(1, num_steps + 1):
        perc = 100 * (1 - k / num_steps)
        thr  = np.percentile(h_flat, perc)
        mask_map = (heatmap_resized >= thr)

        perturbed = inp_norm_np.copy()
        perturbed[0, 0, mask_map] = 0.0

        perturbed_tensor = torch.tensor(
            perturbed, dtype=torch.float32, device=device
        ).contiguous()

        with torch.no_grad():
            new_logits = model(perturbed_tensor)
            new_probs  = torch.nn.functional.softmax(new_logits, dim=1)
            new_conf   = new_probs[0, target_class].item()

        confidences.append(new_conf)

        if k % 50 == 0:
            torch.cuda.empty_cache()

    return confidences

# ----------------------------
# 5) PROCESS ALL TEST IMAGES (ONLY CORRECTLY PREDICTED)
# ----------------------------
covid_curves    = []
noncovid_curves = []
skipped = 0

for idx, (img_path, gt_label) in enumerate(test_samples):
    print(f"\nProcessing {idx+1}/{len(test_samples)}: {os.path.basename(img_path)} (gt_label={gt_label})")

    # Load and preprocess image
    try:
        pil_img = Image.open(img_path).convert('L')
    except Exception as e:
        print(f"  [Error] Cannot open image: {e}. Skipping.")
        skipped += 1
        continue

    img_tensor = val_tf(pil_img).unsqueeze(0)  # shape (1,1,224,224)

    # 5.1) Check model prediction
    with torch.no_grad():
        logits = model(img_tensor.to(device))
        probs  = torch.nn.functional.softmax(logits, dim=1)
        pred   = probs.argmax(dim=1).item()

    if pred != gt_label:
        print("  Misclassified by the model; skipping MoRF & Grad-CAM.")
        skipped += 1
        continue

    # 5.2) Generate Grad-CAM overlay for the predicted (correct) class
    overlay = generate_gradcam_viridis_from_tensor(img_tensor, model, target_class=pred, alpha=0.6)
    out_name = f"gradcam_{os.path.basename(img_path)}"
    overlay.save(os.path.join(output_dir, out_name))

    # 5.3) Compute MoRF on the correctly predicted class probability
    confidences = compute_morf_for_tensor_confidence(
        img_tensor=img_tensor,
        model=model,
        target_class=pred,
        num_steps=num_steps
    )

    confidences_np = np.array(confidences, dtype=np.float32)
    if np.isnan(confidences_np).any():
        print("  Warning: NaN values in MoRF. Skipping.")
        skipped += 1
        continue

    # 5.4) Append to respective list
    if gt_label == test_dataset.class_to_idx['COVID']:
        covid_curves.append(confidences)
    else:
        noncovid_curves.append(confidences)

    torch.cuda.empty_cache()
    print("  Finished MoRF for this image.")

print(f"\nDone. Skipped {skipped} images due to misclassification or errors.")

# ----------------------------
# 6) CONVERT TO NUMPY ARRAYS & COMPUTE AOPC
# ----------------------------
covid_array = np.array(covid_curves)     # shape = (n_covid_correct, L+1)
non_array   = np.array(noncovid_curves)  # shape = (n_noncovid_correct, L+1)

n_covid, total_cols_covid  = covid_array.shape if covid_array.size else (0, 0)
n_noncovid, total_cols_non = non_array.shape if non_array.size else (0, 0)
L = num_steps

# Compute AOPC for COVID class (only correctly predicted)
if n_covid > 0:
    base_conf_covid = covid_array[:, 0]      # shape (n_covid,)
    all_conf_covid  = covid_array[:, 1:]     # shape (n_covid, L)
    diff_covid      = base_conf_covid[:, None] - all_conf_covid   # (n_covid, L)
    AOPC_per_covid_image = diff_covid.mean(axis=1)                 # (n_covid,)
    AOPC_COVID = AOPC_per_covid_image.mean()
else:
    AOPC_COVID = float('nan')

# Compute AOPC for non-COVID class (only correctly predicted)
if n_noncovid > 0:
    base_conf_non = non_array[:, 0]        # shape (n_noncovid,)
    all_conf_non  = non_array[:, 1:]       # shape (n_noncovid, L)
    diff_non      = base_conf_non[:, None] - all_conf_non          # (n_noncovid, L)
    AOPC_per_non_image = diff_non.mean(axis=1)                     # (n_noncovid,)
    AOPC_NonCovid = AOPC_per_non_image.mean()
else:
    AOPC_NonCovid = float('nan')

# Overall AOPC across all correctly predicted images
all_AOPC_values = []
if n_covid > 0:
    all_AOPC_values.extend(AOPC_per_covid_image.tolist())
if n_noncovid > 0:
    all_AOPC_values.extend(AOPC_per_non_image.tolist())

AOPC_overall = np.mean(all_AOPC_values) if all_AOPC_values else float('nan')

print("\n--- AOPC RESULTS (only correctly predicted) ---")
print(f"AOPC (COVID, correct preds)       = {AOPC_COVID:.4f}")
print(f"AOPC (Non-COVID, correct preds)   = {AOPC_NonCovid:.4f}")
print(f"AOPC (all correctly predicted)    = {AOPC_overall:.4f}\n")

# ----------------------------
# 7) PLOT MoRF CURVES (English labels, all correctly predicted)
# ----------------------------
plt.figure(figsize=(6, 5))

covid_array_for_plot = covid_array if n_covid > 0 else np.zeros((1, L+1))
non_array_for_plot   = non_array if n_noncovid > 0 else np.zeros((1, L+1))

mean_conf_covid = covid_array_for_plot.mean(axis=0)   # shape (L+1,)
mean_conf_non   = non_array_for_plot.mean(axis=0)     # shape (L+1,)

# x-axis: fraction from 0.0 to 1.0 in L+1 steps
fractions_plot = np.linspace(0, 1, L+1)  # [0.0, 0.05, 0.10, …, 1.0]

plt.plot(fractions_plot, mean_conf_covid, marker='o', label='COVID (mean)')
plt.plot(fractions_plot, mean_conf_non,   marker='s', label='Non-COVID (mean)')

plt.xlabel('Fraction of top pixels removed')
plt.ylabel('Mean class probability')
plt.title('MoRF: COVID vs Non-COVID (correctly predicted images)')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(os.path.join(output_dir, 'morf_full_covid_vs_non_correct.png'))
plt.show()

# ----------------------------
# 8) PLOT AOPC CURVES WITH X-AXIS = STEP NUMBER
# ----------------------------
# Compute AOPC curves (for k = 1..L):
AOPC_curve_COVID   = diff_covid.mean(axis=0)     if n_covid > 0 else np.zeros(L)
AOPC_curve_NonCovid = diff_non.mean(axis=0)      if n_noncovid > 0 else np.zeros(L)

steps = np.arange(1, L+1)  # [1, 2, 3, …, L]

# ---- Plot for COVID ----
plt.figure(figsize=(6, 5))
plt.plot(steps, AOPC_curve_COVID, marker='o', color='tab:blue', label='AOPC COVID')
plt.fill_between(steps,
                 AOPC_curve_COVID,
                 color='tab:blue',
                 alpha=0.3)

plt.xlabel('Masking step (k)')
plt.ylabel('Mean AOPC')
plt.title('AOPC Curve for COVID (correctly predicted)')
plt.xticks(steps)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'aopc_curve_full_covid_correct.png'))
plt.show()

# ---- Plot for non-COVID ----
plt.figure(figsize=(6, 5))
plt.plot(steps, AOPC_curve_NonCovid, marker='s', color='tab:orange', label='AOPC Non-COVID')
plt.fill_between(steps,
                 AOPC_curve_NonCovid,
                 color='tab:orange',
                 alpha=0.3)

plt.xlabel('Masking step (k)')
plt.ylabel('Mean AOPC')
plt.title('AOPC Curve for Non-COVID')
plt.xticks(steps)
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'aopc_curve_full_non_covid_correct.png'))
plt.show()

