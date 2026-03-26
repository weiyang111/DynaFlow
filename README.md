# DynaFlow
Official implementation of "Frequency-Temporal-Enhanced Graph Neural Architecture for Dynamic Anomaly Detection"

**Key Features:**
- **Frequency-Domain Enhancement (GFT Layer)**: Feature decomposition via graph Fourier transform
- **Spatial Aggregation (GNN-GRU Layers)**: Multi-layer message passing for neighborhood information
- **Temporal Modeling (GARU Cell)**: Graph-aware GRU for capturing temporal dependencies
- **End-to-End Classifier**: Binary classification for anomaly detection

## Project Structure

```
├── model.py                 # DynaFlow model definition
├── layers.py               # GFT, GNN-GRU, GARU layer implementations
├── train.py                # Training script
├── dataset.py              # Dataset loading and runtime feature encoding
├── data_preprocess.py      # Offline data preprocessing
├── utils.py                # Evaluation, early stopping, utilities
├── best_dynaflow_model.pth # Pre-trained model weights
├── processed_*.pt          # Preprocessed datasets (PyTorch format)
└── data/                   # Raw dataset directory
```

## Quick Start

### Installation

```bash
pip install torch torch-geometric pandas numpy scipy scikit-learn tqdm
```

### Training

#### Step 1: Data Preprocessing (One-time)

**Input**: Raw edge lists in `data/` directory (format: `[source, target, timestamp]`)

**Output**: files containing temporally windowed graph subsequences

Supported datasets: bitcoinalpha, bitcoinotc, email-dnc, topology, uci, digg

#### Step 2: Train Model

```bash
python train.py
```

**Key Hyperparameters** (in `train.py`):
- `BATCH_SIZE`: 32
- `EPOCHS`: 50
- `LR`: 1e-4
- `PATIENCE`: 10

**Output**:
- Training logs with AUC, F1 metrics
- Best model saved as `best_dynaflow_model.pth`

## Model Architecture

### DynaFlow (model.py)

```python
model = DynaFlow(
    in_channels=22,      # Node feature dimension (2*(max_dist+1))
    gru_hidden=32,       # GNN-GRU hidden dim
    garu_hidden=256,     # GARU hidden dim
    k_nodes=10,          # SortPooling top-k nodes
    num_layers=4         # Number of GNN-GRU layers
)
```

**Processing Pipeline**:
1. **GFT Layer**: Graph Fourier transform + SortPooling for top-k nodes
2. **Spatial Layers**: 4-layer GNN-GRU for neighborhood aggregation
3. **Temporal Layer**: GARU cell for sequence modeling
4. **Classifier**: MLP for anomaly probability output

### Node Features

Node features encode distance information:
$$R_t(v) = \text{OneHot}(\eta_v) \oplus \text{OneHot}(\eta_{ab})$$

Where:
- $\eta_v$: shortest distance from node $v$ to target edge endpoints
- $\eta_{ab}$: shortest distance between nodes in the target edge

### Layers (layers.py)

- `GFTLayer`: Graph Fourier Transform + SortPooling
- `GNN_GRU_Layer`: Graph convolution + GRU hybrid layer
- `GARUCell`: Graph-attention enhanced GRU

## Dataset

### Supported Datasets

| Dataset | Nodes | Snapshots | Description |
|---------|-------|-----------|-------------|
| BitcoinAlpha | ~3.8K | 21 | Bitcoin trust network |
| BitcoinOTC | ~5.8K | 63 | Bitcoin OTC trading network |
| Email-DNC | ~1.1K | 20 | Email communication network |
| Topology | ~7.5K | 63 | Internet AS topology |
| UCI | ~1.9K | 190 | UCI social network |
| Digg | ~2.7K | 16 | Social news network |

### Data Format

Preprocessed data (`.pt` files) contains:
```python
[
    {
        'window': [Data(...), Data(...), ...],  # Sequence of k graphs
        'label': 0 or 1                         # 0: normal, 1: anomaly
    },
    ...
]
```

## Usage Example

### Inference with Pre-trained Model

```python
import torch
from model import DynaFlow
from dataset import DynaFlowDataset

# Load dataset
dataset = DynaFlowDataset(data_path="processed_bitcoinalpha.pt")

# Load model
model = DynaFlow(in_channels=dataset.feature_dim)
model.load_state_dict(torch.load("best_dynaflow_model.pth"))
model.eval()

# Predict
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

with torch.no_grad():
    sample = dataset[0]
    anomaly_prob = model(sample)  # Output: probability in [0, 1]
```

## Training Configuration
- **Batch Size**: 32
- **Optimizer**: Adam (LR=1e-4)
- **Loss Function**: Binary Cross Entropy
- **Gradient Clipping**: 5.0
- **Early Stopping**: Patience=10

## Evaluation Metrics

- **AUC**: Area Under Curve
- **F1-Score**: Harmonic mean of precision and recall
- **Precision**: True positive rate among predicted positives
- **Recall**: True positive rate among actual positives

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `FileNotFoundError` for `.pt` files | Run data preprocessing first |
| CUDA out of memory | Reduce `BATCH_SIZE` to 16 or 8 |
| Slow training | Verify CUDA availability; use GPU |

## License

[Add license information as needed]
