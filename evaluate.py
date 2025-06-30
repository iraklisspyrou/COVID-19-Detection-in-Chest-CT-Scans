from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def calc_metrics(preds, targets):
    return {
        'acc': accuracy_score(targets, preds),
        'f1' : f1_score(targets, preds),
        'prec': precision_score(targets, preds),
        'rec': recall_score(targets, preds),
    }
