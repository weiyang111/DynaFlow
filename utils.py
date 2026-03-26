import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_score, f1_score

class Evaluator:
    @staticmethod
    def get_metrics(y_true, y_scores, threshold=0.5):
        y_true = np.array(y_true)
        y_scores = np.array(y_scores)
        y_pred = (y_scores >= threshold).astype(int)
        
        auc = roc_auc_score(y_true, y_scores)
        ap = average_precision_score(y_true, y_scores)
        precision = precision_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        return {
            'AUC': auc,
            'AP': ap,
            'Precision': precision,
            'F1': f1,
            'TPR': tpr,
            'TNR': tnr
        }

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False