# Toxic Comment Classification with RNN and LSTM

This project uses PyTorch to classify Wikipedia comments into six possible labels: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. A comment can have more than one label.

## Dataset

The data comes from the [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/datasets/julian3833/jigsaw-toxic-comment-classification-challenge). Download `train.csv`, `test.csv`, and `test_labels.csv` and place them in the repository's main folder. The scripts match test comments to labels by `id` and exclude test rows whose labels are `-1`.

## Approach

Both models lowercase and tokenize comments with NLTK, convert tokens to IDs from a training vocabulary, and use learned word embeddings. Sequences are limited to 100 tokens and padded within each batch. The models combine the final recurrent hidden state with max pooling to predict the six labels. Training uses `BCEWithLogitsLoss`; evaluation reports micro F1, accuracy, and a per-label classification report.

## Run

From the repository's main folder, install the dependencies and NLTK tokenizer data:

```bash
pip install -r requirements.txt
python -m nltk.downloader punkt_tab
```

Then run either model:

```bash
python RNN/Toxic_RNN.py
python LSTM/Toxic_LSTM.py
```

If your script filenames or folder names differ, adjust these commands. PyTorch with CUDA may need an installation command from the [PyTorch installation guide](https://docs.pytorch.org/get-started/locally/).

## Results

My model achieved 0.6595 micro F1 (65.95%) across all six toxicity labels on the 63,978 valid Jigsaw test examples. Rows marked -1 in test_labels.csv were excluded from evaluation.

For context, a published study evaluating on the same valid test set reported 0.65 micro F1 for an LSTM, 0.66 for a GRU, and 0.67 for a BiLSTM + CNN. This places my result alongside published recurrent-model results for this dataset.
