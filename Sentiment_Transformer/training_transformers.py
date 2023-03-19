import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import transformers
from transformers import AutoModel, BertTokenizerFast
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils.class_weight import compute_class_weight
from architecture import BERT_Arch
from transformers import AdamW

from transformers import logging
logging.set_verbosity_error()

import warnings
warnings.filterwarnings("ignore")

# specify GPU
from config import device_type
device = torch.device(device_type)

from config import batch_size
from config import epochs
from config import best_valid_loss
from config import raw_data_path

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

def load_data():
    
	df=pd.read_csv(raw_data_path)
    
	df['Sentiment'] = df['Rating'].apply(rating_to_sentiment)
	df.dropna(inplace=True)
    
	df['Sentiment']=df['Sentiment'].astype(int)
	
	df.reset_index(drop=True, inplace=True)
	
	return (df)

def encode(df):
        
	train_text, temp_text, train_labels, temp_labels = train_test_split(
              														df['Review'], df['Sentiment'], 
                                                                    random_state=2018, 
                                                                    test_size=0.3, 
                                                                    stratify=df['Sentiment']
                                                                    )


	val_text, test_text, val_labels, test_labels = train_test_split(temp_text, temp_labels, 
                                                                random_state=2018, 
                                                                test_size=0.5, 
                                                                stratify=temp_labels)
	# Load the BERT tokenizer
	tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        
	# tokenize and encode sequences in the training set
	tokens_train = tokenizer.batch_encode_plus(
		train_text.tolist(),
		max_length = None,
		padding=True,
		truncation=True
	)

	# tokenize and encode sequences in the validation set
	tokens_val = tokenizer.batch_encode_plus(
		val_text.tolist(),
		max_length = None,
		padding=True,
		truncation=True
	)

	# tokenize and encode sequences in the test set
	tokens_test = tokenizer.batch_encode_plus(
		test_text.tolist(),
		max_length = None,
		padding=True,
		truncation=True
	)
	return (tokens_train, tokens_val, tokens_test, train_labels, val_labels, test_labels)

def to_tensor(token,labels):
        
	seq=torch.tensor(token['input_ids'])
        
	mask=torch.tensor(token['attention_mask'])
        
	label_tensor=torch.tensor(labels.tolist())
        
	return (seq, mask,label_tensor)

def initialize_model():
                
	# import BERT-base pretrained model
	bert = AutoModel.from_pretrained('bert-base-uncased')
	# pass the pre-trained BERT to our define architecture
	model = BERT_Arch(bert)
    
	# freeze all the parameters
	for param in bert.parameters():
		param.requires_grad = False

	# push the model to GPU
	model = model.to(device)
	return (model)

# function to train the model
def train(model, train_dataloader, cross_entropy, optimizer):
  
    model.train()

    total_loss, total_accuracy = 0, 0
  
	# empty list to save model predictions
    total_preds=[]
  
	# iterate over batches
    for step, batch in enumerate(train_dataloader):
    
    # progress update after every 50 batches.
        if step % 50 == 0 and not step == 0:
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))

        # push the batch to gpu
        batch = [r.to(device) for r in batch]
        
        sent_id, mask, labels = batch
        
        # clear previously calculated gradients 
        model.zero_grad()        

        # get model predictions for the current batch
        preds = model(sent_id,mask)

        # compute the loss between actual and predicted values
        loss = cross_entropy(preds, labels)

        # add on to the total loss
        total_loss = total_loss + loss.item()

        # backward pass to calculate the gradients
        loss.backward()

        # clip the the gradients to 1.0. It helps in preventing the exploding gradient problem
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # update parameters
        optimizer.step()

        # model predictions are stored on GPU. So, push it to CPU
        preds=preds.detach().cpu().numpy()

        # append the model predictions
        total_preds.append(preds)

	# compute the training loss of the epoch
    avg_loss = total_loss / len(train_dataloader)
  
  	# predictions are in the form of (no. of batches, size of batch, no. of classes).
  	# reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

  	# returns the loss and predictions
    return avg_loss, total_preds

# function for evaluating the model
def evaluate(model, val_dataloader, cross_entropy):
  
    print("\nEvaluating...")
  
	# deactivate dropout layers
    model.eval()

    total_loss, total_accuracy = 0, 0
  
	# empty list to save the model predictions
    total_preds = []

	# iterate over batches
    for step,batch in enumerate(val_dataloader):
    
		# Progress update every 50 batches.
        if step % 50 == 0 and not step == 0:
      
      		# Calculate elapsed time in minutes.
            # elapsed = format_time(time.time() - t0)
            
      		# Report progress.
            print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(val_dataloader)))

    	# push the batch to gpu
        batch = [t.to(device) for t in batch]

        sent_id, mask, labels = batch

    	# deactivate autograd
        with torch.no_grad():
      
     		# model predictions
            preds = model(sent_id, mask)

      		# compute the validation loss between actual and predicted values
            loss = cross_entropy(preds,labels)

            total_loss = total_loss + loss.item()

            preds = preds.detach().cpu().numpy()

            total_preds.append(preds)

 	# compute the validation loss of the epoch
    avg_loss = total_loss / len(val_dataloader) 

  	# reshape the predictions in form of (number of samples, no. of classes)
    total_preds  = np.concatenate(total_preds, axis=0)

    return avg_loss, total_preds

def run_pipeline(df):
    
	tokens_train, tokens_val, tokens_test, train_labels, val_labels, test_labels = encode(df)
	train_seq,train_mask,train_y = to_tensor(tokens_train,train_labels)
	val_seq,val_mask,val_y = to_tensor(tokens_val,val_labels)
	test_seq,test_mask,test_y = to_tensor(tokens_test,test_labels)

	# wrap tensors
	train_data = TensorDataset(train_seq, train_mask, train_y)

	# sampler for sampling the data during training
	train_sampler = RandomSampler(train_data)

	# dataLoader for train set
	train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

	# wrap tensors
	val_data = TensorDataset(val_seq, val_mask, val_y)

	# sampler for sampling the data during training
	val_sampler = SequentialSampler(val_data)

	# dataLoader for validation set
	val_dataloader = DataLoader(val_data, sampler = val_sampler, batch_size=batch_size)

	model = initialize_model()

	# define the optimizer
	optimizer = AdamW(model.parameters(),
						lr = 1e-5)          # learning rate

	#compute the class weights
	class_wts = compute_class_weight(class_weight='balanced', classes= np.unique(train_labels), y=train_labels)
	# convert class weights to tensor
	weights= torch.tensor(class_wts,dtype=torch.float)
	weights = weights.to(device)

	# loss function
	cross_entropy  = nn.NLLLoss(weight=weights) 

	# empty lists to store training and validation loss of each epoch
	train_losses=[]
	valid_losses=[]

	#for each epoch
	for epoch in range(epochs):
			
		print('\n Epoch {:} / {:}'.format(epoch + 1, epochs))
		
		#train model
		train_loss, _ = train(model, train_dataloader, cross_entropy, optimizer)
		
		#evaluate model
		valid_loss, _ = evaluate(model, val_dataloader, cross_entropy)
		
		#save the best model
		if valid_loss < best_valid_loss:
			best_valid_loss = valid_loss
			torch.save(model.state_dict(), 'fine_tuned_weights.pt')
		
		# append training and validation loss
		train_losses.append(train_loss)
		valid_losses.append(valid_loss)
		
		print(f'\nTraining Loss: {train_loss:.3f}')
		print(f'Validation Loss: {valid_loss:.3f}')

if __name__ == "__main__":
    df=load_data()
    run_pipeline(df)