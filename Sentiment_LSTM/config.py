sentiment_dict = {0: "Negative",1: "Positive"}

model_path = "./models/model_LSTM.pt"
data_path = './data/tripadvisor_hotel_reviews.csv'

embedding_dim = 128
hidden_dim = 64
num_classes = 2
batch_size = 32
num_epochs = 10