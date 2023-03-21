'''
Reference: https://pytorch.org/text/stable/
https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM
'''
## Library Imports
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
from sklearn.metrics import classification_report

from config import sentiment_dict
from config import model_path
from config import embedding_dim, hidden_dim, num_classes, batch_size, num_epochs
from config import data_path
from architecture import LSTMModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def rating_to_sentiment(rating):
    """
    Convert the rating to sentiment (0: negative, 1: positive)

    Args:
        rating (int): rating of the review
    
    Returns:
        sentiment (int): sentiment of the review
    """
    if rating>4:
        ## Positive Sentiment
        return 1
    elif rating < 3:
        ## Negative Sentiment
        return 0

def yield_tokens(data_iter,tokenizer):
    """
    Yield tokens from data iterator

    Args:
        data_iter (iter): data iterator
        tokenizer (function): tokenizer function
    
    Returns:
        tokens (iter): iterator of tokens
    """
    for _, text in data_iter:
        yield tokenizer(text)

def collate_batch(batch,tokenizer,vocab):
    """
    Custom function to Collate batch of data

    Args:
        batch (iter): batch of data
        tokenizer (function): tokenizer function
        vocab (Vocab): vocab object

    Returns:
        label_list (Tensor): tensor of labels
        text_list (Tensor): tensor of text
        offsets (Tensor): tensor of offsets
    """

    # Pipelines for processing text and labels
    text_pipeline = lambda x: vocab(tokenizer(x))
    label_pipeline = lambda x: int(x)

    label_list, text_list, offsets = [], [], [0]
    # Iterate through batch, processing text and adding text, labels and offsets to lists
    for (label, text) in batch:
        # Save labels
        label_list.append(label_pipeline(label))
        # Process text
        processed_text = torch.tensor(text_pipeline(text), dtype=torch.int64)
        # Save text
        text_list.append(processed_text)
        # Save offsets
        offsets.append(processed_text.size(0))
    
    # Convert lists to tensors
    label_list = torch.tensor(label_list, dtype=torch.int64)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)
    text_list = torch.cat(text_list)
    
    return label_list.to(device), text_list.to(device), offsets.to(device) 


def train_data_processing(data_path, batch_size=32):
    """
    Process the data and create train, validation and test dataloaders

    Args:
        data_path (str): path to the data
        batch_size (int): batch size for the dataloader
    
    Returns:
        train_dataloader (DataLoader): train dataloader
        val_dataloader (DataLoader): validation dataloader
        test_dataloader (DataLoader): test dataloader
        vocab (Vocab): vocab object
    """
    
    # Read the data
    file = pd.read_csv(data_path)
    # Convert the rating to sentiment
    file['Sentiment'] = file['Rating'].apply(rating_to_sentiment)
    # Drop the rows with missing values
    d_file = file.dropna()

    # Split the data into train and test
    data = [(label,text)for label,text in zip(d_file['Sentiment'].to_list(),d_file['Review'].to_list())]
    train_iter,test_iter = train_test_split(data, test_size=0.1)

    # Create train, validation and test datasets
    train_dataset = to_map_style_dataset(train_iter)
    test_dataset = to_map_style_dataset(test_iter)

    # Split the train dataset into train and validation
    num_train = int(len(train_dataset) * 0.95)
    split_train_dataset, split_val_dataset = random_split( train_dataset, [num_train, len(train_dataset) - num_train])

    # Create vocab using training data
    tokenizer = get_tokenizer('basic_english')
    vocab = build_vocab_from_iterator(yield_tokens(train_iter,tokenizer), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    # Create training, validation and test set DataLoaders using custom collate_batch function
    train_dataloader = DataLoader(split_train_dataset, batch_size=batch_size,
                                shuffle=True, collate_fn = lambda x: collate_batch(x,tokenizer,vocab))
    val_dataloader = DataLoader(split_val_dataset, batch_size=batch_size,
                                shuffle=True, collate_fn = lambda x: collate_batch(x,tokenizer,vocab))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                shuffle=True, collate_fn = lambda x: collate_batch(x,tokenizer,vocab))
    
    # Print the length of train, validation and test datasets
    print(len(split_train_dataset),len(split_val_dataset))

    return train_dataloader, val_dataloader, test_dataloader, vocab
    
def train_model(train_dataloader, val_dataloader, vocab, embedding_dim, hidden_dim, num_classes, num_epochs=10):
    """
    Train the LSTM model on the training set and evaluate on the validation set

    Args:
        train_dataloader (DataLoader): train dataloader
        val_dataloader (DataLoader): validation dataloader
        vocab (Vocab): vocab object
        embedding_dim (int): embedding dimension
        hidden_dim (int): hidden dimension
        num_classes (int): number of classes
        num_epochs (int): number of epochs to train the model
    
    Returns:
        model (LSTMModel): trained LSTM model
    """

    # Calculate the vocab size
    vocab_size = len(vocab)

    # Create the LSTM model, loss function, and optimizer
    model = LSTMModel(vocab_size, embedding_dim, hidden_dim, num_classes)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    
    #  Train the model
    for epoch in range(num_epochs):
        train_loss = 0.0
        train_acc = 0.0

        # Set the model to training mode
        model.train()

        # Training loop
        for (labels, text, offsets) in tqdm(train_dataloader):
            # Move data to GPU
            text = text.to(device)
            labels = labels.to(device)
            offsets = offsets.to(device)

            # Clear the gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(text, offsets)
            
            # Calculate the loss
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Calculate the accuracy
            train_loss += loss.item()
            train_acc += (outputs.argmax(1) == labels).sum().item()
            
        # Calculate the average loss and accuracy
        train_loss /= len(train_dataloader.dataset)
        train_acc /= len(train_dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, train_loss: {train_loss:.4f}, train_acc: {train_acc:.4f}')

        # Validation loop
        val_loss = 0.0
        val_acc = 0.0
        model.eval()
        
        # Disable gradient calculation
        with torch.no_grad():
            for (labels, text, offsets) in val_dataloader:
                # Move data to GPU
                text = text.to(device)
                labels = labels.to(device)
                offsets = offsets.to(device)

                # Forward pass
                outputs = model(text, offsets)
                
                # Calculate the loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_acc += (outputs.argmax(1) == labels).sum().item()
        
        # Calculate the average loss and accuracy
        val_loss /= len(val_dataloader.dataset)
        val_acc /= len(val_dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, val_loss: {val_loss:.4f}, val_acc: {val_acc:.4f}')

    return model

def save_model(model, model_path):
    """
    Saves the model to disk

    Args:
        model: Trained model
    
    Returns:
        None
    """

    # Export the model to TorchScript
    model_scripted = torch.jit.script(model)

    # Save the model to disk
    model_scripted.save(model_path)
    
    
def evaluate(dataloader, model):
    """
    Evaluates the model on the test set

    Args:
        dataloader: Test set dataloader
        model: Trained model
    
    Returns:
        report: Classification report for the test set from Sklearn
    """

    # Evaluate the model on the test set
    model.eval()
    all_labels, all_predictions = [], []

    # Disable gradient calculation
    with torch.no_grad():
        # Iterate over the test set
        for idx, (label, text, offset) in enumerate(dataloader):
            # Predict the labels
            predict = model(text, offset)
            predicted_labels = predict.argmax(1)
            all_labels.extend(label.tolist())
            all_predictions.extend(predicted_labels.tolist())

    # Classification report
    report = classification_report(all_labels, all_predictions)

    return report

def load_model():
    """
    Loads the trained model

    Args:
        None

    Returns:
        model: Trained model
    """
    model = torch.jit.load('./models/model_LSTM.pt')

    return model


if __name__ == "__main__":

    # Process the data for training
    train_dataloader, val_dataloader, test_dataloader, vocab = train_data_processing(data_path, batch_size)
    
    # Train the model
    model = train_model(train_dataloader, val_dataloader, vocab, embedding_dim, hidden_dim, num_classes, num_epochs)
    
    # Save the trained model
    save_model(model, model_path)
    
    # Load the trained model
    lod_model = load_model()

    # Evaluate the model
    print(evaluate(test_dataloader,lod_model))
