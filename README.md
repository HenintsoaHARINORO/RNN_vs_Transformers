## Experimental Setup

### Datasets
- **WikiText-103**: Partial sample used for language modeling
- **MNIST**: Full dataset used for image classification

### Models
- **GPT-nano** 
- **GRU** 

### Perturbation Types
- **Remove Random**: Randomly removing elements from input with 50% probability
- **Replace Random**: Randomly replacing elements in input with 50% probability

## Results

### WikiText-103 Language Modeling Results

#### GRU Performance
| Perturbation Type | Train Loss | Train PPL | Val Loss | Val PPL |
|-------------------|------------|-----------|----------|---------|
| Replace Random    | 2.305      | 10.03     | 2.515    | 12.37   |
| Remove Random     | 1.602      | 4.96      | 2.077    | 7.98    |

**Model Configuration:**
- Vocabulary Size: 50,257
- Embedding Dimension: 256
- Hidden Dimension: 512
- Number of Layers: 2
- Dropout: 0.1

#### GPT-nano Performance
| Perturbation Type | Train Loss | Train PPL | Val Loss | Val PPL |
|-------------------|------------|-----------|----------|---------|
| Replace Random    | 2.621      | 13.75     | 2.580    | 13.20   |
| Remove Random     | 1.925      | 6.86      | 1.837    | 6.28    |

**Model Configuration:**
- Vocabulary Size: 50,257
- Number of Layers: 6
- Number of Heads: 6
- Embedding Dimension: 384
- Block Size: 256
- Dropout: 0.1

### MNIST Classification Results

#### GPT Performance
| Perturbation Type | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Best Val Acc (%) |
|-------------------|------------|---------------|----------|-------------|------------------|
| Remove Random     | 0.019      | 99.38         | 0.040    | 98.89       | 98.98           |
| Replace Random    | 0.018      | 99.38         | 0.038    | 98.86       | 99.03           |

**Model Configuration:**
- Number of Classes: 10
- Dropout: 0.1
- Total Parameters: 1,199,882

#### GRU Performance
| Perturbation Type | Train Loss | Train Acc (%) | Val Loss | Val Acc (%) | Best Val Acc (%) |
|-------------------|------------|---------------|----------|-------------|------------------|
| Replace Random    | 0.016      | 99.52         | 0.024    | 99.24       | 99.27           |
| Remove Random     | 0.017      | 99.48         | 0.037    | 98.97       | 99.26           |

**Model Configuration:**
- Number of Classes: 10
- Dropout: 0.1

 