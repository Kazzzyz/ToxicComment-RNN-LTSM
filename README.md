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

Earlier runs achieved approximately **0.63 test micro F1**. The exact score depends on the model and run; each script prints its own final score and results for all six labels. The valid labeled test set contains **63,978 comments** after excluding `-1` rows. Kaggle's leaderboard uses ROC AUC, while this project reports F1.
