import torch.nn as nn

class LSTMModel(nn.Module):
    """"
    LSTM Model for Sentiment Analysis
    """
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes):
        super(LSTMModel, self).__init__()
        # Embedding layer
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim,mode = 'mean', sparse=True)
        # LSTM layer
        self.lstm = nn.LSTM(embedding_dim, hidden_dim,num_layers=1, batch_first=True)
        # Fully connected layer
        self.fc = nn.Linear(hidden_dim, num_classes)
        # Initialize weights
        self.init_weights()
    
    def init_weights(self):
        initrange = 0.5
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc.weight.data.uniform_(-initrange, initrange)
        self.fc.bias.data.zero_()

    def forward(self, text, offsets):
        # Forward pass
        embedded = self.embedding(text, offsets).unsqueeze(1)
        lstm_output, (ht, ct) = self.lstm(embedded)
        
        return self.fc(ht[-1])