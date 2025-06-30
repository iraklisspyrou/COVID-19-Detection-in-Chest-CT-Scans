import torch, torch.nn as nn
from evaluate import calc_metrics

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train(); total_loss=0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        y_hat = model(x)
        loss = criterion(y_hat, y)
        loss.backward(); optimizer.step()
        total_loss += loss.item()*x.size(0)
    return total_loss/len(loader.dataset)

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval(); total_loss=0; preds, targets = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        total_loss += loss.item()*x.size(0)
        preds.extend(y_hat.argmax(1).cpu()); targets.extend(y.cpu())
    metrics = calc_metrics(preds, targets)
    return total_loss/len(loader.dataset), metrics
