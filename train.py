import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset import DynaFlowDataset, dynaflow_collate_fn
from model import DynaFlow
from utils import Evaluator, EarlyStopping, set_seed

def train():
    set_seed(42)
    BATCH_SIZE = 32
    EPOCHS = 50 
    LR = 1e-4
    PATIENCE = 10
    CLIP_VAL = 5.0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Experimental Device: {device}")
    
    dataset = DynaFlowDataset(data_path="processed_uci.pt")
    
    train_size = int(0.5 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=dynaflow_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=dynaflow_collate_fn)
    
    model = DynaFlow(in_channels=dataset.feature_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCELoss() 
    
    early_stopping = EarlyStopping(patience=PATIENCE, verbose=True)
    evaluator = Evaluator()
    
    print("\nDynaFlow Training Started")
    
    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        for batch_window, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
            batch_window = [g.to(device) for g in batch_window]
            labels = labels.to(device)
            
            optimizer.zero_grad()
            probs = model(batch_window).squeeze()
            loss = criterion(probs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP_VAL)
            optimizer.step()
            epoch_loss += loss.item()
            
        model.eval()
        all_true, all_probs = [], []
        
        with torch.no_grad():
            for batch_window, labels in tqdm(test_loader, desc="Evaluating"):
                batch_window = [g.to(device) for g in batch_window]
                probs = model(batch_window).squeeze()
                all_true.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        metrics = evaluator.get_metrics(all_true, all_probs)
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | AUC: {metrics['AUC']:.4f} | "
              f"AP: {metrics['AP']:.4f}")
        
        early_stopping(metrics['AUC'])
            
        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break

    print("\nExperiment Finished")
    print(f"Best Test AUC reached: {early_stopping.best_score:.4f}")

if __name__ == "__main__":
    train()