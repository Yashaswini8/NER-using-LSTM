# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
The objective of this experiment is to build a Named Entity Recognition (NER) system using an LSTM-based model to identify and classify entities such as person names, locations, organizations, and other important terms from text. The model should learn the contextual relationships between words in a sentence and assign the correct entity tag to each word.

The dataset used for this task is the CoNLL-2003 Dataset, which contains annotated sentences with entity labels such as PER (Person), LOC (Location), ORG (Organization), and MISC (Miscellaneous). Each word in the dataset is tagged with its corresponding entity label, which is used to train the model to recognize named entities in new text.

## DESIGN STEPS

### STEP 1:
Load the dataset and preprocess the text by tokenizing sentences and converting words and labels into numerical representations using vocabularies or indexing.
### STEP 2:
Build the LSTM-based neural network model using an embedding layer, LSTM layers, and a fully connected layer to predict the entity tag for each word.
### STEP 3:
Train the model using the training dataset, evaluate its performance on the validation or test dataset, and use the trained model to identify named entities in unseen text.
Write your own steps

## PROGRAM
### Name:Yashaswini S
### Register Number:212224220123
```python
class BiLSTMTagger(nn.Module):

    def __init__(self, vocab_size=len(word2idx)+1, tagset_size=len(tag2idx), embedding_dim=128, hidden_dim=128):
        super(BiLSTMTagger, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(hidden_dim * 2, tagset_size)

    def forward(self, input_ids):

        x = self.embedding(input_ids)

        lstm_out, _ = self.lstm(x)

        logits = self.fc(lstm_out)

        return logits


model = BiLSTMTagger().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):

    train_losses = []
    val_losses = []

    for epoch in range(epochs):

        model.train()
        total_train_loss = 0

        for batch in train_loader:

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)

            outputs = outputs.view(-1, outputs.shape[-1])
            labels = labels.view(-1)

            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        total_val_loss = 0

        with torch.no_grad():

            for batch in test_loader:

                input_ids = batch["input_ids"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(input_ids)

                outputs = outputs.view(-1, outputs.shape[-1])
                labels = labels.view(-1)

                loss = loss_fn(outputs, labels)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(test_loader)
        val_losses.append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    return train_losses, val_losses

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot
![NER Output](https://raw.githubusercontent.com/Yashaswini8/NER-using-LSTM/9cd690d7889cb6068cce1e85df3a6097863418be/Screenshot%202026-03-15%20221607.png)


### Sample Text Prediction

![NER Output](https://raw.githubusercontent.com/Yashaswini8/NER-using-LSTM/b7b4f385b63d62f66f88a22cdc7cb5f30e20ae79/Screenshot%202026-03-15%20221819.png)
## RESULT
The Named Entity Recognition (NER) model based on an LSTM network was successfully implemented using PyTorch.
