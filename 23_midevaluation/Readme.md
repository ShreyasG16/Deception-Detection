# Deception Detection- Baseline

## Project Overview
This Baseline implements Bidirectional LSTM with attention mechanism and additional contextual features.
## Key Features
- Bidirectional LSTM  with attention
- Contextual feature integration.
- Focal Loss for handling class imbalance
- Weighted sampling to address dataset skew
- Comprehensive evaluation metrics

## Libraries
- Python 3.8+
- PyTorch
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- tqdm

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/diplomacy-deception-detection.git
cd diplomacy-deception-detection
```



3. Install required packages:
```bash
pip install -r requirements.txt
```

## Dataset Preparation
- Ensure your dataset is in JSONL format
- Place train, validation, and test files in the specified paths:
  - `/kaggle/input/deception/train (1).jsonl`
  - `/kaggle/input/deception/validation.jsonl`
  - `/kaggle/input/deception/test.jsonl`

## Hyperparameters
- `MAX_VOCAB_SIZE`: Maximum vocabulary size
- `MAX_SEQ_LENGTH`: Maximum sequence length
- `EMBEDDING_DIM`: Word embedding dimension
- `HIDDEN_DIM`: LSTM hidden layer dimension
- `BATCH_SIZE`: Training batch size
- `LEARNING_RATE`: Optimizer learning rate
- `NUM_EPOCHS`: Training epochs
- `DROPOUT_RATE`: Regularization dropout rate
- `FOCAL_LOSS_GAMMA`: Focal loss gamma parameter

## Running the Model
```bash
Run Jupyter Notebook ( We used Kaggle Notebook )
```

## Output Files
- `best_model.pt`: Saved best-performing model
- `confusion_matrix.png`: Model performance visualization
- `test_predictions.csv`: Detailed test set predictions
- `train_processed.csv`: Preprocessed Training Dataset
- `val_processed`: Preprocessed Validation Dataset
- `training history`: Validation-Training loss and F1-Accuracy score

## Model Architecture
The model combines:
- Embedding Layer
- Bidirectional LSTM
- Attention Mechanism
- Additional Feature Processing


## Performance Metrics
- Accuracy
- F1 Score
- Confusion Matrix



