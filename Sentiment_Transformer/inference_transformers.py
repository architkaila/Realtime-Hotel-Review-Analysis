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
		urllib.request.urlretrieve("https://dl2.boxcloud.com/d/1/b1!sUY1ikLMHLPx1fbNKZAQ6uCbWnWR_V5khjs097g6dVTwtE6E5vXidZRfoh33VOg2bJlrAiFzntH8Rg2FPQtwR5hrJxsA2teTlGpE7DXObmvnv-xwovcb_04VvAqLe98vZcmgrZNnOs8lS-D1WQH4ZJpExzFt2DxyAj44TKrYib819Q8JmDHgbQM9t8gSD3vzMiQWn7vnBd3jnLLHa94bThkzb89p4XQnOtwNpKTlSwxSJ0_28bvTtkUCTDgaQjwfDWvB32_sDE2Zo5mKO3b3lYUfh_rZe3MOPv2vpbsSFJwq91GCbIi78vwKJGiwG6dsDA4HLUD9tc6W3LVrIagLnkIo7CkPnNPztZtklJ1E8vIBGrPqLQ9uhmzwH2v60cqHd_EMEq916x1H6ra6R6SzaG7OkQKMnB96s2iBtxcQVr6t_mC3AgFTTVYqleJADOSjF8DgTRuVJ1jnpKZtnJc2WZtth8VasLKjB6-Y_ntuzYmBuxZBi9Z6j4KgVWYuCPgzxp9haFaKFp3caGz117uWludFQeEygxqcIwkE5Ab3qoK2r3a8aoMV5TeOVphxkIfb9PRuHWJU5Lpo1sg21v1Ds8H71Q-QMPWNrVQt_FgPBqcT7iadDAijbcFvOdnMgs4bAYAbqL1K7MAeUewXPVgoyNs-Dmah9ce2b8d10JBNjjHum-WrRO1zzsoLPDEVDYIbGbKds2_P-EoC_pqiQ2PMKVl1cnzrAm5INRRUWlykpnkE_F_oNiMVLylk79KsbfKGqqEWJ9c_Q6DNcMlaosU4womWunoP_ug-T6vW1Mv10j-oi-ksQxHQn10NlQEreDD4UeCTIlJCnd26779_geZUykR9kJe-JGYU3vDLoqR_QVesIilAD-OSBjnhRQ_ByCMhzl_800XVSZgAKHbzSTueN06X6G06QPf2qTqXKC5vgeqwy3bYiigleNqaNQvVz3kLPui6rDjzt3SUj0FL4iHZB7Yop3Mq6ruOfJs7MESVz0YdN9dWbRgDuRtIB7-yUtee1ehYKsNX6YGqNc77TCw2NPU5_fe8-9seRtudGr3lD6r0aGdotg8NdWE3Oai1sdJSVTre3hkJomfHqV5C2OAJVHJvPfYMc8ihgGRtNvY0hb_9JdHg9MQV5mJ_ercELSEs_rmfhboeNVzvL8ChmbgKV-C1XA5RZtGxcpXld1Mb1Q5nKEYW1F6eGKIy_qlD9QXnqvL6hyv4kKO2UIi9qSJ7ssYFtyuwzGFmjqV6SDixEJlbuwDE8Vanance74vPVAe0XoTsJ5TjW1DPywEq_6vtiMQC2HFoMWjVALErG0ayzGZq2YMFrczRUhoT2HQLfYr-rXx8V0dFKtEIoGwazANs5gZBm_leSAZGksqOQPPd4MtLuAKO8xr8VAzKFXh60S8DGJB2eVaMD3hecgiaTuzZZ2pwtzPNjtqNNN64RTJH29IcLV8./download", "./models/fine_tuned_weights_bert.pt")
		print("[INFO] Download complete, running inference...")
	else:
		print("[INFO] Fine-tuned weights already present, running inference...")

	run_inference(infer_sample)