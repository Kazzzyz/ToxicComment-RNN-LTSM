
import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import nltk 
from collections import Counter
import pandas as pd 
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence ,pad_packed_sequence
from sklearn.metrics import f1_score ,classification_report


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 


#hyperparameters
embed_dim = 100
input_size=embed_dim
hidden_size = 250
num_layers=1
learning_rate = 0.001
batch_size = 100
num_epochs = 6
output_size = 6 

label_columns = [
    "toxic",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate"
]


df=pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")
test_labels_df = pd.read_csv("test_labels.csv")


df['Tokenized']=df['comment_text'].apply(lambda x: nltk.word_tokenize(x.lower()))
test_df['Tokenized']=test_df['comment_text'].apply(lambda x: nltk.word_tokenize(x.lower()))

test_df = test_df.merge(test_labels_df,on="id",validate="one_to_one")


invalid_rows = (test_df[label_columns] == -1).any(axis=1)
test_df = test_df[~invalid_rows].reset_index(drop=True)


vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}


token_counts = Counter()

for tokens in df["Tokenized"]:
    token_counts.update(tokens)

max_vocab_size = 50000

for token, count in token_counts.most_common(max_vocab_size - 2):
    vocab[token] = len(vocab)


df["Token_IDs"] = df["Tokenized"].apply(lambda tokens: [vocab.get(token, vocab["<UNK>"]) for token in tokens]) 
test_df["Token_IDs"] = test_df["Tokenized"].apply(lambda tokens: [vocab.get(token, vocab["<UNK>"]) for token in tokens])


class TextDataset(Dataset):
    def __init__(self, dataframe, vocab):
        self.x = dataframe["Token_IDs"].tolist() 
        self.y = torch.tensor(dataframe[["toxic","severe_toxic","obscene","threat","insult","identity_hate"]].to_numpy(),dtype=torch.float32) 
        self.vocab = vocab
        self.n_samples = len(dataframe)

    def __getitem__(self, index):
        token_ids = self.x[index]

        if len(token_ids) == 0:
            token_ids = [self.vocab["<UNK>"]] 

        token_ids = torch.tensor(token_ids, dtype=torch.long) 
        labels = self.y[index]  

        return token_ids, labels 

    def __len__(self):
        return self.n_samples


def collate_fn(batch): 
    token_ids, labels = zip(*batch)
    token_ids = [ids[:100] for ids in token_ids]
    padded_ids = pad_sequence(token_ids, batch_first=True, padding_value=vocab["<PAD>"])
    labels = torch.stack(labels)
    return padded_ids, labels


train_dataset=TextDataset(df,vocab)
train_loader=DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=0)

test_dataset=TextDataset(test_df,vocab)
test_loader=DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=0)



padded_token_ids, labels = next(iter(train_loader)) 

print("Padded comments shape:", padded_token_ids.shape)
print("Labels shape:", labels.shape)




class LSTM(nn.Module):
    def __init__(self,vocab_size,embed_dim,hidden_size,num_layers,output_size,padding_id):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embed_dim,padding_idx=padding_id) 
        self.lstm = nn.LSTM(input_size=embed_dim,hidden_size=hidden_size,num_layers=num_layers,batch_first=True)
        self.fc = nn.Linear(hidden_size * 2, output_size) 
        self.padding_id = padding_id

    def forward(self, token_ids):
        lengths = (token_ids != self.padding_id).sum(dim=1)
        embedded = self.embedding(token_ids)

        packed = pack_padded_sequence(embedded,lengths.cpu(),batch_first=True,enforce_sorted=False)

        packed_output, (final_hidden, final_cell) = self.lstm(packed)

        lstm_output, _ = pad_packed_sequence(packed_output,batch_first=True)

        real_word_mask = (token_ids[:, :lstm_output.size(1)] != self.padding_id)
        lstm_output = lstm_output.masked_fill(~real_word_mask.unsqueeze(-1),float("-inf"))
        max_pooled = lstm_output.max(dim=1).values
        last_hidden = final_hidden[-1]

        combined = torch.cat((last_hidden, max_pooled),dim=1)

        logits = self.fc(combined)

        return logits

model = LSTM(vocab_size=len(vocab),embed_dim=embed_dim,hidden_size=hidden_size,num_layers=num_layers,output_size=output_size,padding_id=vocab["<PAD>"]).to(device)



#Loss and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


#Training loop
n_total_steps=len(train_loader)
for epoch in range(num_epochs):
    for i, (padded_ids, labels) in enumerate(train_loader):
        token_ids = padded_ids.to(device)
        labels = labels.to(device)  
        
        #forward pass
        outputs = model(token_ids)
        loss = criterion(outputs, labels)
        
        #backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1) % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{n_total_steps}], Loss: {loss.item():.4f}')



# Evaluation
model.eval()

correct = 0
total = 0

all_predictions = []
all_labels = []

with torch.no_grad():
    for padded_ids, labels in test_loader:
        token_ids = padded_ids.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(token_ids)

        # Logits → probabilities → predictions
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities >= 0.45).int()

        # Accuracy over all six label decisions
        correct += (predictions == labels).sum().item()
        total += labels.numel()

        # Save predictions and labels for F1
        all_predictions.append(predictions.cpu())
        all_labels.append(labels.cpu())

# Combine all batches
all_predictions = torch.cat(all_predictions).numpy()
all_labels = torch.cat(all_labels).numpy().astype(int)

# Calculate accuracy
accuracy = 100 * correct / total

# Calculate F1
f1 = f1_score(
    all_labels,
    all_predictions,
    average="micro",
    zero_division=0
)

print(f"Test accuracy: {accuracy:.2f}%")
print(f"Micro F1 score: {f1:.4f}")

print(f"Actual positive percentage: {all_labels.mean() * 100:.2f}%")
print(f"Predicted positive percentage: {all_predictions.mean() * 100:.2f}%")

all_zero_accuracy = (all_labels == 0).mean() * 100
print(f"Accuracy from predicting only zeros: {all_zero_accuracy:.2f}%")

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=label_columns,
        zero_division=0
    )
)


torch.save(model.state_dict(), "lstm_Final.pth")
