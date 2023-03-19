import os 
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from torchtext.data.functional import to_map_style_dataset
from tqdm import tqdm
from torch.utils.data.dataset import random_split
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sentiment_dict = {1: "Negative",
                2: "Neutral",
                3: "Positive"}
def rating_to_sentiment(rating):
    if rating>3 and rating<=5:
        return 3
    elif rating == 3:
        return 2
    else:
        return 1
def yield_tokens(data_iter,tokenizer):
    for _, text in data_iter:
        yield tokenizer(text)

def collate_batch(batch,tokenizer,vocab):
    # Pipelines for processing text and labels
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x) - 1
    label_list, text_list, offsets = [], [], [0]
    # Iterate through batch, processing text and adding text, labels and offsets to lists
    for (label, text) in batch:
        label_list.append(label_pipeline(label))
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        text_list.append(processed_text)
        offsets.append(processed_text.size(0))
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    return label_list.to(device), text_list.to(device), offsets.to(device) 


def file_processing():
    batch_size = 32
    file = pd.read_csv(r'./tripadvisor_hotel_reviews.csv')
    file['Sentiment'] = file['Rating'].apply(rating_to_sentiment)
    
    data = [(label,text)for label,text in zip(file['Sentiment'].to_list(),file['Review'].to_list())]
    train_iter,test_iter = train_test_split(data,test_size=0.1)

    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)

    num_train = int(len(train_dataset) * 0.95)
    split_train_dataset, split_val_dataset = random_split(train_dataset, [num_train, len(train_dataset) - num_train])

    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(data,tokenizer), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])
    # Create training, validation and test set DataLoaders using custom collate_batch function
    train_dataloader = DataLoader(split_train_dataset, batch_size=batch_size,
                                shuffle=True, collate_fn = lambda x: collate_batch(x,tokenizer,vocab))
    val_dataloader = DataLoader(split_val_dataset, batch_size=batch_size,
                                shuffle=True, collate_fn = lambda x: collate_batch(x,tokenizer,vocab))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=True, collate_fn = lambda x: collate_batch(x,tokenizer,vocab))
    print(len(split_train_dataset),len(split_val_dataset))
    return train_dataloader,val_dataloader,test_dataloader,vocab

# Define the LSTM model
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(LSTMModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim,padding_idx=0, sparse=True)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        embedded = self.embedding(text, offsets).unsqueeze(1)
        lstm_output, (ht, ct) = self.lstm(embedded)
        return self.fc(ht[-1])
    
def train_model():
    # Model parameters
    train_dataloader,val_dataloader,test_dataloader,vocab= file_processing()
    vocab_size = len(vocab)
    embedding_dim = 128
    hidden_dim = 64
    num_classes = 3

    # Create the LSTM model, loss function, and optimizer
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    num_epochs = 100

    for epoch in range(num_epochs):
        # Training loop
        train_loss = 0.0
        train_acc = 0.0
        model.train()

        for (labels, text, offsets) in tqdm(train_dataloader):
            text = text.to(device)
            labels = labels.to(device)
            offsets = offsets.to(device)
            optimizer.zero_grad()
            outputs = model(text, offsets)
            
            loss = criterion(outputs, labels)
    
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()
            

        train_loss /= len(train_dataloader)
        train_acc /= len(train_dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}')

        # Validation loop
        val_loss = 0.0
        val_acc = 0.0
        model.eval()
        
        with torch.no_grad():
            for (labels, text, offsets) in val_dataloader:
                text = text.to(device)
                labels = labels.to(device)
                offsets = offsets.to(device)
                outputs = model(text, offsets)
                
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == labels).sum().item()
                
        val_loss /= len(val_dataloader)
        val_acc /= len(val_dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}')

    return model

def save_model(model):
    save_model_path = os.path.join('../models','model_LSTM.pth')
    torch.save(model.state_dict(), save_model_path)
    
def evaluate(dataloader, model):
    model.eval()
    all_labels, all_predictions = [], []

    with torch.no_grad():
        for idx, (label, text, offset) in enumerate(dataloader):
            predict = model(text, offset)
            predicted_labels = predict.argmax(1)
            all_labels.extend(label.tolist())
            all_predictions.extend(predicted_labels.tolist())

    report = classification_report(all_labels, all_predictions)
    return report

def load_model():

    save_model_path = os.path.join('../models','model_LSTM.pth')
    loaded_model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_classes)
    loaded_model.load_state_dict(torch.load(save_model_path))
    return loaded_model
if __name__ == "__main__":

    train_dataloader,test_dataloader,vocab = file_processing()
    trainmodel=train_model()
    save_model=save_model(trainmodel)
    loaamodel = load_model()
    print(evaluate(train_dataloader,loaamodel))