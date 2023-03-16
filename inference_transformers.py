import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import warnings
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
from transformers import logging
logging.set_verbosity_error()
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from arch import BERT_Arch
device = torch.device("cpu")

def load_data():
	df=pd.read_csv("../data/tripadvisor_hotel_reviews.csv")
	text=df['Review'].loc[0:10].to_numpy()
	return (text)
def encode(text):
	# Load the BERT tokenizer
	tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
	tokens = tokenizer.batch_encode_plus(
    text.tolist(),
    max_length = None,
    padding=True,
    truncation=True)
	return (tokens)
def load_model(wt_file_path):
	# import BERT-base pretrained model
	bert = AutoModel.from_pretrained('bert-base-uncased')
	model = BERT_Arch(bert)
	model=model.to(device)
	model.load_state_dict(torch.load(wt_file_path))
	return (model)

def predict():
	test_text=load_data()
	tokens_test=encode(test_text)
	test_seq = torch.tensor(tokens_test['input_ids'])
	test_mask = torch.tensor(tokens_test['attention_mask'])
	path='fine_tuned_weights.pt'
	sentiment=load_model(path)
	with torch.no_grad():
		preds = sentiment(test_seq.to(device), test_mask.to(device))
		preds = preds.detach().cpu().numpy()
	preds = np.argmax(preds, axis = 1)
	counts = np.bincount(preds)
	majority_vote = np.argmax(counts)
	print ("Positive" if majority_vote==1 else "Negative")

if __name__ == "__main__":
	predict()