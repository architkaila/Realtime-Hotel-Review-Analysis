device_type = "cuda"
device_type_inference = "cpu"
batch_size = 32
epochs = 10
raw_data_path = "./data/tripadvisor_hotel_reviews.csv"
trained_weights_path='./models/fine_tuned_weights_bert.pt'
inference_data="./data/X_test.pkl"
infer_sample = 0