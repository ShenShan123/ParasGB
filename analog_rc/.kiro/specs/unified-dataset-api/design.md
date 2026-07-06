# Design Document - Unified Dataset API

## 1. Overview

This design document describes the architecture for a unified dataset API that provides consistent interfaces for RC circuit graph data loading, splitting, evaluation, and testing. The design follows PyTorch Geometric (PyG) API style for simplicity and familiarity.

### Design Goals

1. **Unified Interface**: Provide PyG-style unified API to simplify dataset operations
2. **Modularity**: Independent components for easy maintenance and extension
3. **Backward Compatibility**: Create new module without modifying existing code
4. **High Performance**: Support caching, multi-process loading, and other optimizations
5. **Flexible Configuration**: Support multiple task types and configuration options

### Core Design Principles

- **Single Responsibility**: Each class has one clear responsibility
- **Open-Closed**: Open for extension, closed for modification
- **Dependency Inversion**: Depend on abstractions, not concrete implementations
- **Principle of Least Surprise**: API design follows user intuition and PyG conventions

## 2. Architecture

### Module Structure

```
rcg_unified/
├── __init__.py              # Module entry point, exports main classes
├── dataset.py               # RCDataset core class
├── dataloader.py            # DataLoader factory class
├── evaluator.py             # Evaluator for model evaluation
├── config.py                # Configuration management
├── normalizer.py            # Label normalization utilities
└── utils.py                 # Utility functions
```

### High-Level Data Flow

```
Raw Data (.pt files)
    ↓
RCDataset.load_and_process()
    ↓
Cached Processed Data
    ↓
RCDataset.get_idx_split() → train/val/test indices
    ↓
RCDataset.get_dataloader() → NeighborLoader / LinkNeighborLoader
    ↓
Model Training
    ↓
Evaluator.evaluate() → Metrics
```

## 3. Component Design

### 3.1 RCDataset (dataset.py)

**Responsibility**: Dataset loading, processing, and splitting

**PyG-Style API**:
```python
from rcg_unified import RCDataset

# Create dataset
dataset = RCDataset(
    root='data/',
    train_cases=['1', '5', '7'],
    test_cases=['2', '6'],
    task_level='node',  # 'node' or 'edge'
    task_type='regression',  # 'regression' or 'classification'
    use_cache=True
)

# Get split indices (PyG style)
split_idx = dataset.get_idx_split()
# Returns: {'train': tensor([...]), 'val': tensor([...]), 'test': {'case2': tensor([...]), 'case6': tensor([...])}}

# Get DataLoader (PyG style)
train_loader = dataset.get_dataloader(
    split='train',
    batch_size=32,
    num_neighbors=[10, 10],
    num_hops=2,
    shuffle=True
)

# Access data
print(f"Train size: {len(split_idx['train'])}")
print(f"Test cases: {list(split_idx['test'].keys())}")  # ['case2', 'case6']
```

**Core Attributes**:
- `root`: Data directory path
- `train_cases`: List of training case IDs
- `test_cases`: List of test case IDs  
- `task_level`: 'node' (predict net node capacitance) or 'edge' (predict pin-to-pin resistance)
- `task_type`: 'regression' or 'classification'
- `train_graph`: Merged training graph
- `test_graphs`: Dict of test graphs {case_id: graph}
- `normalizer`: LabelNormalizer instance

**Core Methods**:

```python
def __init__(self, root, train_cases, test_cases, task_level, task_type, 
             val_ratio=0.2, use_cache=True, class_boundaries=None):
    """Initialize dataset"""
    
def get_idx_split(self) -> dict:
    """Get train/val/test split indices
    
    Returns:
        {
            'train': tensor([...]),  # Training indices
            'val': tensor([...]),    # Validation indices  
            'test': {                # Test indices per case
                'case2': tensor([...]),
                'case6': tensor([...])
            }
        }
    """
    
def get_dataloader(self, split, batch_size, num_neighbors, num_hops, 
                   shuffle=False, num_workers=0):
    """Get DataLoader for specified split
    
    Args:
        split: 'train', 'val', or test case ID (e.g., 'case2')
        batch_size: Batch size
        num_neighbors: List of neighbor sampling sizes per hop
        num_hops: Number of hops
        shuffle: Whether to shuffle
        num_workers: Number of worker processes
        
    Returns:
        NeighborLoader (for node tasks) or LinkNeighborLoader (for edge tasks)
    """
```

**Internal Methods**:
- `_load_and_process_graph(case_id)`: Load and process single graph
- `_merge_train_graphs()`: Merge training graphs into one
- `_create_train_val_split()`: Split training data into train/val
- `_filter_power_nets(hg)`: Remove power supply nets (VDD/VSS/GND)
- `_process_node_labels(g, hg)`: Process node-level labels
- `_process_edge_labels(g, hg)`: Process edge-level labels

### 3.2 DataLoaderFactory (dataloader.py)

**Responsibility**: Create appropriate DataLoaders based on task level

**API Design**:
```python
from rcg_unified import DataLoaderFactory

factory = DataLoaderFactory(task_level='node')

# Create node-level DataLoader
loader = factory.create_loader(
    graph=train_graph,
    indices=train_idx,
    batch_size=32,
    num_neighbors=[10, 10],
    num_hops=2,
    shuffle=True
)
```

**Core Methods**:
```python
def create_node_loader(self, graph, indices, batch_size, num_neighbors, 
                       num_hops, shuffle, num_workers):
    """Create NeighborLoader for node-level tasks"""
    
def create_edge_loader(self, graph, edge_label_index, edge_label, 
                       batch_size, num_neighbors, num_hops, shuffle, num_workers):
    """Create LinkNeighborLoader for edge-level tasks"""
```

### 3.3 Evaluator (evaluator.py)

**Responsibility**: Compute evaluation metrics for model predictions

**API Design**:
```python
from rcg_unified import Evaluator

evaluator = Evaluator(task_type='regression')

# Evaluate predictions
metrics = evaluator.evaluate(y_pred, y_true)
# Returns: {'mae': 0.123, 'mse': 0.045, 'rmse': 0.212, 'r2': 0.876}

# Batch evaluation on multiple test sets
test_results = evaluator.evaluate_multiple(
    model=model,
    test_loaders={'case2': loader2, 'case6': loader6},
    device='cuda'
)
# Returns: {'case2': {...}, 'case6': {...}}
```

**Core Methods**:
```python
def evaluate(self, y_pred, y_true) -> dict:
    """Compute metrics for single prediction
    
    For regression: MAE, MSE, RMSE, R2
    For classification: Accuracy, F1, Precision, Recall
    """
    
def evaluate_multiple(self, model, test_loaders, device) -> dict:
    """Evaluate model on multiple test sets"""
    
def _compute_regression_metrics(self, y_pred, y_true) -> dict:
    """Compute regression metrics"""
    
def _compute_classification_metrics(self, y_pred, y_true) -> dict:
    """Compute classification metrics"""
```

### 3.4 LabelNormalizer (normalizer.py)

**Responsibility**: Normalize and denormalize labels

**API Design**:
```python
from rcg_unified import LabelNormalizer

# Node-level normalizer (capacitance ~1e-13 F)
node_normalizer = LabelNormalizer(
    task_level='node',
    method='log',
    max_value=8e-13
)

# Normalize
normalized = node_normalizer.normalize(raw_labels)  # [0, 1]

# Denormalize
original = node_normalizer.denormalize(normalized)  # Original scale

# Convert to classes
classes = node_normalizer.to_classes(normalized, boundaries=[0.2, 0.4, 0.6, 0.8])
```

**Core Methods**:
```python
def normalize(self, labels) -> torch.Tensor:
    """Normalize labels to [0, 1] range"""
    
def denormalize(self, normalized_labels) -> torch.Tensor:
    """Denormalize labels to original scale"""
    
def to_classes(self, normalized_labels, boundaries) -> torch.Tensor:
    """Convert normalized labels to discrete classes"""
```

**Normalization Methods**:
- **Node labels (capacitance)**: `log1p(y * 1e15) / log1p(MAX * 1e15)`
- **Edge labels (resistance)**: `log1p(y) / log1p(700)`

### 3.5 Config (config.py)

**Responsibility**: Configuration management and validation

**API Design**:
```python
from rcg_unified import Config

# Load from file
config = Config.from_file('config.yaml')

# Load from dict
config = Config.from_dict({
    'root': 'data/',
    'train_cases': ['1', '5', '7'],
    'test_cases': ['2', '6'],
    'task_level': 'node',
    'task_type': 'regression',
    'batch_size': 32,
    'num_neighbors': [10, 10],
    'num_hops': 2
})

# Access config
print(config.train_cases)
print(config.batch_size)

# Validate config
config.validate()  # Raises ValueError if invalid
```

**Core Methods**:
```python
@classmethod
def from_file(cls, filepath) -> 'Config':
    """Load config from YAML/JSON file"""
    
@classmethod
def from_dict(cls, config_dict) -> 'Config':
    """Create config from dictionary"""
    
def validate(self):
    """Validate configuration parameters"""
    
def to_dict(self) -> dict:
    """Convert config to dictionary"""
```

## 4. Data Models

### 4.1 Graph Data Structure

After processing, each graph has the following attributes:

**Node-level task**:
```python
Data(
    x=tensor([...]),              # Node type IDs [N, 1]
    node_attr=tensor([...]),      # Node features [N, 16]
    edge_index=tensor([...]),     # Edge connectivity [2, E]
    edge_type=tensor([...]),      # Edge type IDs [E]
    y=tensor([...]),              # Node labels [N, 2]
                                  #   [:, 0]: regression label (normalized)
                                  #   [:, 1]: classification label (class ID)
    train_node_mask=tensor([...]) # Valid training nodes [N]
)
```

**Edge-level task**:
```python
Data(
    x=tensor([...]),                  # Node type IDs [N, 1]
    node_attr=tensor([...]),          # Node features [N, 16]
    edge_index=tensor([...]),         # Structural edges [2, E]
    edge_type=tensor([...]),          # Edge type IDs [E]
    edge_label_index=tensor([...]),   # Target edges [2, E_target]
    edge_label_y=tensor([...])        # Edge labels [E_target, 2]
                                      #   [:, 0]: regression label (normalized)
                                      #   [:, 1]: classification label (class ID)
)
```

### 4.2 Label Format

All labels follow a unified format `[N, 2]`:
- **Column 0**: Normalized regression label (0-1 range)
- **Column 1**: Discrete classification label (class ID)

This allows the same dataset to support both regression and classification tasks.

## 5. Correctness Properties

Based on prework analysis, the following correctness properties must be maintained:

### 5.1 Data Integrity

**Property 1.1**: Power net filtering
- GIVEN a graph with net nodes
- WHEN filtering power nets (VDD/VSS/GND)
- THEN net.x[:, 0]==1 OR net.x[:, 1]==1 nodes are removed
- AND remaining nodes maintain correct indices

**Property 1.2**: Label validity
- Node labels: 0 < capacitance <= 8e-13 F
- Edge labels: 0 < resistance <= 700 Ω
- Invalid labels are filtered out

**Property 1.3**: Feature padding
- All node features padded to 16 dimensions
- Padding uses zeros for missing dimensions

### 5.2 Data Splitting

**Property 2.1**: No data leakage
- Train, val, test sets are disjoint
- No overlap between case IDs in train and test

**Property 2.2**: Validation split
- Val set is 20% of train set by default
- Split is deterministic (random_state=42)

### 5.3 Label Normalization

**Property 3.1**: Normalization range
- All normalized labels in [0, 1] range
- Denormalization recovers original scale

**Property 3.2**: Classification boundaries
- Default boundaries: [0.2, 0.4, 0.6, 0.8] → 5 classes
- Classes are 0-indexed: {0, 1, 2, 3, 4}

### 5.4 DataLoader Behavior

**Property 4.1**: Shuffle behavior
- Train loader: shuffle=True
- Val/test loaders: shuffle=False

**Property 4.2**: Batch consistency
- Node task: batches contain sampled subgraphs
- Edge task: batches contain edge pairs with labels

## 6. Error Handling Strategy

### 6.1 File Not Found
```python
if not os.path.exists(filepath):
    raise FileNotFoundError(f"Data file not found: {filepath}")
```

### 6.2 Invalid Labels
```python
# Filter invalid labels and log warning
valid_mask = (labels > 0) & (labels <= MAX_VALUE)
if valid_mask.sum() < len(labels):
    logger.warning(f"Filtered {len(labels) - valid_mask.sum()} invalid labels")
labels = labels[valid_mask]
```

### 6.3 Configuration Validation
```python
def validate(self):
    if self.task_level not in ['node', 'edge']:
        raise ValueError(f"Invalid task_level: {self.task_level}")
    if self.task_type not in ['regression', 'classification']:
        raise ValueError(f"Invalid task_type: {self.task_type}")
    if not self.train_cases:
        raise ValueError("train_cases cannot be empty")
```

### 6.4 Empty Batch Handling
```python
# Skip empty batches during training
if pred.numel() == 0:
    logger.warning("Empty batch encountered, skipping")
    continue
```

## 7. Performance Optimization

### 7.1 Caching Strategy

**Processed data caching**:
```
data/
├── case1_RC.pt                    # Raw data
├── processed_for_node/            # Node task cache
│   └── case1_processed.pt
└── processed_for_edge/            # Edge task cache
    └── case1_processed.pt
```

**Cache invalidation**: Delete processed files when raw data changes

### 7.2 Multi-Process Loading

```python
train_loader = NeighborLoader(
    ...,
    num_workers=4,  # Use 4 worker processes
    persistent_workers=True  # Keep workers alive
)
```

### 7.3 Memory Optimization

- Use `torch.float32` instead of `float64` where possible
- Delete temporary attributes after processing
- Use in-place operations when safe

## 8. Testing Strategy

### 8.1 Unit Tests

Test each component independently:
- `test_dataset.py`: Test RCDataset loading and splitting
- `test_dataloader.py`: Test DataLoader creation
- `test_evaluator.py`: Test metric computation
- `test_normalizer.py`: Test normalization/denormalization
- `test_config.py`: Test configuration validation

### 8.2 Integration Tests

Test component interactions:
- `test_end_to_end.py`: Full pipeline from loading to evaluation
- `test_multiple_cases.py`: Multiple case handling
- `test_cache.py`: Cache behavior

### 8.3 Property-Based Tests

Use hypothesis for property testing:
- Normalization is reversible
- Split indices are disjoint
- Label ranges are valid

## 9. Usage Examples

### Example 1: Node Regression Task

```python
from rcg_unified import RCDataset, Evaluator

# Create dataset
dataset = RCDataset(
    root='data/',
    train_cases=['1', '5', '7'],
    test_cases=['2', '6'],
    task_level='node',
    task_type='regression'
)

# Get split
split_idx = dataset.get_idx_split()

# Get loaders
train_loader = dataset.get_dataloader('train', batch_size=32, 
                                      num_neighbors=[10, 10], num_hops=2, shuffle=True)
val_loader = dataset.get_dataloader('val', batch_size=32,
                                    num_neighbors=[10, 10], num_hops=2)

# Train model
model = YourModel()
for epoch in range(100):
    for batch in train_loader:
        # Training code
        pass

# Evaluate
evaluator = Evaluator(task_type='regression')
test_results = evaluator.evaluate_multiple(model, 
    {case_id: dataset.get_dataloader(case_id, batch_size=32, 
                                     num_neighbors=[10, 10], num_hops=2)
     for case_id in dataset.test_cases}
)

print(test_results)
# {'case2': {'mae': 0.123, 'r2': 0.876}, 'case6': {'mae': 0.145, 'r2': 0.854}}
```

### Example 2: Edge Classification Task

```python
from rcg_unified import RCDataset, Evaluator

# Create dataset
dataset = RCDataset(
    root='data/',
    train_cases=['1', '5', '7'],
    test_cases=['2', '6'],
    task_level='edge',
    task_type='classification',
    class_boundaries=[0.2, 0.4, 0.6, 0.8]  # 5 classes
)

# Get loaders
train_loader = dataset.get_dataloader('train', batch_size=64,
                                      num_neighbors=[15, 15], num_hops=2, shuffle=True)

# Train and evaluate
evaluator = Evaluator(task_type='classification')
# ... training code ...
metrics = evaluator.evaluate(y_pred, y_true)
print(metrics)
# {'accuracy': 0.892, 'f1': 0.875, 'precision': 0.881, 'recall': 0.869}
```

### Example 3: Using Configuration File

```yaml
# config.yaml
root: data/
train_cases: ['1', '5', '7']
test_cases: ['2', '6']
task_level: node
task_type: regression
batch_size: 32
num_neighbors: [10, 10]
num_hops: 2
val_ratio: 0.2
use_cache: true
```

```python
from rcg_unified import Config, RCDataset

# Load config
config = Config.from_file('config.yaml')

# Create dataset from config
dataset = RCDataset(
    root=config.root,
    train_cases=config.train_cases,
    test_cases=config.test_cases,
    task_level=config.task_level,
    task_type=config.task_type,
    val_ratio=config.val_ratio,
    use_cache=config.use_cache
)
```

## 10. Migration from Existing Code

### From change/dataset.py

```python
# Old way
from change.dataset import RCDataset as OldDataset
g = OldDataset.load_and_process('data/case1_RC.pt', task_level='node')

# New way
from rcg_unified import RCDataset
dataset = RCDataset(root='data/', train_cases=['1'], test_cases=[], 
                    task_level='node', task_type='regression')
split_idx = dataset.get_idx_split()
```

### From CircuitGCL/sram_dataset.py

```python
# Old way
from CircuitGCL.sram_dataset import SealSramDataset
dataset = SealSramDataset(name='1+5+7', root='data/', task_level='edge')

# New way
from rcg_unified import RCDataset
dataset = RCDataset(root='data/', train_cases=['1', '5', '7'], 
                    test_cases=[], task_level='edge', task_type='regression')
```

## 11. Future Extensions

### 11.1 Additional Task Types

- Multi-task learning (regression + classification simultaneously)
- Graph-level tasks (predict circuit properties)

### 11.2 Advanced Sampling

- Importance sampling based on label distribution
- Hard negative mining for edge tasks

### 11.3 Data Augmentation

- Random edge dropout
- Node feature perturbation
- Subgraph sampling

### 11.4 Distributed Training

- Support for DistributedDataParallel
- Sharded dataset loading

## 12. Summary

This design provides a unified, PyG-style API for RC circuit graph datasets that:

1. **Simplifies usage** with intuitive `get_idx_split()` and `get_dataloader()` methods
2. **Maintains compatibility** by creating a new module without modifying existing code
3. **Ensures correctness** through well-defined properties and validation
4. **Optimizes performance** with caching and multi-process loading
5. **Supports flexibility** through configuration management

The modular design allows easy extension and maintenance while providing a consistent interface for all RC circuit graph tasks.
