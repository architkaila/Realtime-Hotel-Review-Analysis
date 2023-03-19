# Library imports
import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoModel, BertTokenizerFast
import os
import urllib.request
from transformers import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore")

# Local imports
from architecture import BERT_Arch
from config import device_type_inference
from config import inference_data
from config import trained_weights_path
from config import infer_sample

# Set the device type
device = torch.device(device_type_inference)

def load_data(instance):
	"""
	This function loads the test data

	Args:
		instance (int): The index of the sample to be inferred
	
	Returns:
		text (list): The list of text to be encoded
	"""

	# Load the test data
	X_test_df = pd.read_pickle(inference_data)
	text = []

	# Append the text to the list
	text.append(X_test_df['Review'].to_numpy()[instance])
	
	return (text)

def encode(text):
	"""
	This function encodes the text into tokens, masks, and segment flags.

	Args:
		text (list): The list of text to be encoded
	
	Returns:
		tokens (dict): The dictionary of tokens, masks, and segment flags
	"""
	# Load the BERT tokenizer
	tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')

	# Encode text
	tokens = tokenizer.batch_encode_plus(
    									text,
										max_length = None,
										padding=True,
										truncation=True)
	return (tokens)

def load_model(wt_file_path):
	"""
	This function loads the trained model

	Args:
		wt_file_path (str): The path to the trained weights
	
	Returns:
		model (torch.nn.Module): The trained model
	"""
	
	# import BERT-base pretrained model
	bert = AutoModel.from_pretrained('bert-base-uncased')
	# pass the pre-trained BERT to our define architecture
	model = BERT_Arch(bert)
	# push the model to GPU
	model = model.to(device)
	# load weights of best model
	model.load_state_dict(torch.load(wt_file_path, map_location=device))
	
	return (model)

def run_inference(instance):
	"""
	This function runs the inference on the trained model

	Args:
		instance (int): The index of the sample to be inferred
	
	Returns:
		None
	"""
	# Load the test data
	test_text=load_data(instance)
	# Encode the test data
	tokens_test=encode(test_text)
	
	test_seq = torch.tensor(tokens_test['input_ids'])
	test_mask = torch.tensor(tokens_test['attention_mask'])
	
	sentiment=load_model(trained_weights_path)
	
	with torch.no_grad():
		preds = sentiment(test_seq.to(device), test_mask.to(device))
		preds = preds.detach().cpu().numpy()
	
	preds = np.argmax(preds, axis = 1)
	for i, review in enumerate(test_text):
		print(f"[INFO] RAW Hotel Review: {infer_sample}\n")
		print(review, "\n")
		print("Sentiment: ", preds[i], "\n\n")

if __name__ == "__main__":

	# Download the fine-tuned weights if not present
	if not os.path.isfile('./models/fine_tuned_weights_bert.pt'):
		print("[INFO] Downloading the fine-tuned weights...")
		urllib.request.urlretrieve("https://duke.box.com/shared/static/ra736456g3p0369hssm0uoeriimko91v.pt", "./models/fine_tuned_weights_bert.pt")
		print("[INFO] Download complete, running inference...")
	else:
		print("[INFO] Fine-tuned weights already present, running inference...")

	run_inference(infer_sample)