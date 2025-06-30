import yaml
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

import pandas as pd
from io import StringIO
from sklearn.metrics import (
    confusion_matrix,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    matthews_corrcoef
)
import seaborn as sns
import matplotlib.pyplot as plt

from data_loading import build_dataloaders
from ResNet import ResNet18
# from models.vit import ViTBase  # ξεCommentάρετε αν χρησιμοποιείτε ViT
from engine import train_one_epoch, validate
from utils import set_seed, ensure_dirs


def get_all_preds_and_targets(model, loader, device):
    """
    Κάνει inference σε όλο το loader (π.χ. test_loader)
    Επιστρέφει δύο λίστες: all_preds, all_targets
    """
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().tolist())
            all_targets.extend(y.cpu().tolist())
    return all_preds, all_targets


def main():
    # ---------------------------------------------------
    # 1. Φόρτωση ρυθμίσεων από config.yaml
    # ---------------------------------------------------
    cfg = yaml.safe_load(open('config.yaml', 'r'))
    cfg['lr'] = float(cfg['lr'])

    # ---------------------------------------------------
    # 2. Set seed & Δημιουργία φακέλων outputs
    # ---------------------------------------------------
    set_seed()
    base_output = "outputs"
    ckpt_dir    = os.path.join(base_output, "checkpoints")
    fig_dir     = os.path.join(base_output, "figures")
    log_dir     = os.path.join(base_output, "logs")
    ensure_dirs(ckpt_dir, fig_dir, log_dir)

    # ---------------------------------------------------
    # 3. TensorBoard Writer
    # ---------------------------------------------------
    tb_log_dir = os.path.join(log_dir, "runs")
    writer = SummaryWriter(log_dir=tb_log_dir)

    # ---------------------------------------------------
    # 4. DataLoaders (παράδειγμα: σε Colab /content/dataset_split)
    # ---------------------------------------------------
    split_path = "/content/dataset_split"
    train_loader, val_loader, test_loader = build_dataloaders(
        split_path, batch_size=cfg['batch'], num_workers=4
    )

    # ---------------------------------------------------
    # 5. Επιλογή μοντέλου & μεταφορά σε συσκευή
    # ---------------------------------------------------
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if cfg['model'] == 'resnet':
        model = ResNet18(pretrained=True).to(device)
    else:
        model = ViTBase(pretrained=True).to(device)

    # ---------------------------------------------------
    # 6. Ορισμός optimizer & loss function
    # ---------------------------------------------------
    optimiser = optim.Adam(model.parameters(), lr=cfg['lr'])
    criterion = nn.CrossEntropyLoss()

    # ---------------------------------------------------
    # 7. Early Stopping setup (βασισμένο στο val loss)
    # ---------------------------------------------------
    best_val_loss = float('inf')
    patience = cfg.get('patience', 5)
    epochs_no_improve = 0

    # ---------------------------------------------------
    # 8. Training / Validation Loop
    # ---------------------------------------------------
    for epoch in range(cfg['epochs']):
        # --- Train step (επιστρέφει μέσο train_loss) ---
        train_loss = train_one_epoch(model, train_loader, optimiser, criterion, device)

        # --- Validate step (επιστρέφει μέσο val_loss + metrics) ---
        val_loss, val_metrics = validate(model, val_loader, criterion, device)
        val_acc = val_metrics['acc']
        val_f1  = val_metrics['f1']

        # --- TensorBoard logging των losses (Train vs Val στο ίδιο plot) ---
        writer.add_scalars('Loss', {
            'Train': train_loss,
            'Val':   val_loss
        }, epoch)

        # --- Προαιρετικά: ξεχωριστά plots για Train/Loss και Val/Loss ---
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Val/Loss',   val_loss,   epoch)

        print(f"[Epoch {epoch:02d}] "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

        # --- Early Stopping based on Val Loss ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            # Αποθήκευση βέλτιστου βάρους
            torch.save(model.state_dict(), os.path.join(ckpt_dir, f"best_{cfg['model']}.pth"))
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping: δεν μειώθηκε το Val Loss για {patience} διαδοχικές εποχές.")
            break

    # ---------------------------------------------------
    # 9. Τελική αξιολόγηση στο Test Set
    # ---------------------------------------------------
    best_ckpt = os.path.join(ckpt_dir, f"best_{cfg['model']}.pth")
    model.load_state_dict(torch.load(best_ckpt))
    model.to(device)

    test_loss, test_metrics = validate(model, test_loader, criterion, device)
    test_acc, test_f1 = test_metrics['acc'], test_metrics['f1']

    print("\n--- Τελικά Metrics στο Test Set ---")
    print(f"Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

    # --- Log Test Loss & Test F1 στο TensorBoard ---
    writer.add_scalar('Test/Loss', test_loss)
    writer.add_scalar('Test/F1',   test_f1)

    # ---------------------------------------------------
    # 10. Υπολογισμός και εμφάνιση Metrics Table (Test Set)
    # ---------------------------------------------------
    all_preds, all_targets = get_all_preds_and_targets(model, test_loader, device)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(all_targets, all_preds).ravel()

    # Βασικές μετρικές
    acc         = accuracy_score(all_targets, all_preds)
    prec        = precision_score(all_targets, all_preds)
    rec         = recall_score(all_targets, all_preds)
    f1_score_val = f1_score(all_targets, all_preds)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    fpr         = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    mcc         = matthews_corrcoef(all_targets, all_preds)
    dice        = 2 * tp / float(2*tp + fp + fn) if (2*tp + fp + fn) > 0 else 0.0

    # Δημιουργία DataFrame
    metrics_dict = {
        'Accuracy':             acc,
        'Precision':            prec,
        'Recall':               rec,
        'F1-Score':             f1_score_val,
        'Specificity':          specificity,
        'False_Positive_Rate':  fpr,
        'MCC':                  mcc,
        'Dice':                 dice
    }
    metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=['Value'])

    # Εκτύπωση του πίνακα στον κονσόλα
    print("\n=== Metrics Table (Test Set) ===")
    print(metrics_df)

    # Προαιρετικά: Log του πίνακα σαν text στο TensorBoard
    buffer = StringIO()
    metrics_df.to_csv(buffer)
    metrics_table_str = buffer.getvalue()
    writer.add_text('Metrics/Test', metrics_table_str)

    # ---------------------------------------------------
    # 11. Confusion Matrix Heatmap (Test Set)
    # ---------------------------------------------------
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(5,4))
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=['non-COVID', 'COVID'],
        yticklabels=['non-COVID', 'COVID']
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix (Test)")
    cm_path = os.path.join(fig_dir, "confusion_matrix_test.png")
    plt.savefig(cm_path)
    writer.add_figure('Confusion Matrix/Test', plt.gcf())
    plt.close()

    # ---------------------------------------------------
    # 12. Κλείσιμο TensorBoard writer
    # ---------------------------------------------------
    writer.close()


if __name__ == "__main__":
    main()

