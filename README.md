# # Realtime-Hotel-Review-Analysis
> #### _Archit, Neha, Zenan | Spring '23 | Duke AIPI 540 NLP Project_
&nbsp;

## Project Description
When travelling to a new place one of the most hectic tasks is to search for a good place to stay. On various travel websites, one finds multiple listings of hotels which can generally be filtered by the ratings. But even so, one would have to read through many reviews to make sure that the amenities that they are looking for are up to the mark for a given hotel. This is where our project comes in...

We train a model on a dataset that has user reviews for a hotel and the corresponding ratings. What is expected of the model is to understand what a positive review looks like and then predict the nature of the reviews that it has not seen before. Once we have found out the sentiment of the review, we employ techniques like dependency parsing and SHAP analysis to find the relevant attributes that make these sentiments positive. So basically we predict if the stay at the hotel will be great and if so, what is it that actually makes it great.

&nbsp;
## Data Processing:
For this project we use the Trip Advisor Hotel Reviews dataset available on Kaggle-https://www.kaggle.com/datasets/andrewmvd/trip-advisor-hotel-reviews 
The dataset is highly skewed in the favour of positive reviews which is why for training we have chosen to use rating 5 as positve and ratings 1 and 2 as negetive reviews. We have also ignored the neutral category of reviews. 

For the deep learning models that we have fine tuned there is little to no pre-processing of data done.
For the non-deep learning model, we have created vectors of the words using TF-IDF technique. 
Dependency parsing is more accurate when the provided content is grammatically correct. To ensure this we have used the gpt-3 api and prompted it to correct the grammar for the given review.


&nbsp;
## Model Training and Evaluation
We trained three models for comparison - BERT uncased transformer, LSTM and Random Forest and below were the results:

| Model          |  Accuracy (Test) |
| -------------- | :--------------: |
| BERT           |        87%       |
| Random Forest  |        87%       |
| LSTM           |        85%       |


&nbsp;
### Model 1: BERT uncased Transformer
We performed transfer learning using a pre-trained BERT model and fine tuned it using our data. With a basic GPU, the training took approximately 2 hours for 10 epochs. Again with a basic GPU where the allocated is small, it may run out of memory for inference. So inference on CPU takes around 15-30 seconds where we also consider the time to load the pre-trained model weights. The accuracy of the BERT model was nothing special with 87%. The precision for the negative class, which has low volume of data, is not great.
![image](https://user-images.githubusercontent.com/110474064/226507974-2d8f3b7f-80e1-423a-bf74-c6e510565c71.png)

### Model 2: LSTM
Here the torch nn module is used to create embeddings and train the model on the given data. This model doesnt perform all that impressively coming in at 85% accuracy but a slightly better precision for the negative classes. 
![image](https://user-images.githubusercontent.com/110474064/226509098-a8183774-3224-4fab-8b05-4455bb7cbe6d.png)

### Model 3: Random Forest
We used TF-IDF technique and the built in function to vectorize the given reviews. These were then fed to a random forest algorithm for training. This was by far the quickest model to train and for inference. It matched BERT's accuracy at 87% and showed good metrics for both precision for both classes. So instead of the fine tuning the transformer model further we decided to go forward with the Random Forest model. Another advantage of the Random Forest model is that we can use SHAP analysis on it and explain the predicted outcome of the model. So we effectively traded off an even higher accuracy for explainability.
![image](https://user-images.githubusercontent.com/110474064/226509834-f8da27fb-8362-4902-90d8-52b975818bbc.png)


&nbsp;
## Running the demo (StreamLit)

**1. Clone this repository and switch to the streamlit-demo branch**
```
git clone https://github.com/architkaila/Realtime-Hotel-Review-Analysis.git
git checkout streamlit-demo
```
**2. Create a conda environment:** 
```
conda create --name environ python=3.7.16
conda activate environ
```
**3. Install requirements:** 
```
pip install -r requirements.txt
```
**4. Run the application**
```
streamlit run streamlit_app.py
```
**5. StreamLit Appication:**
* You can find the code for streamlit application on the [`streamlit-demo`](https://github.com/architkaila/Realtime-Hotel-Review-Analysis/tree/streamlit_demo) branch
* [Click Here](https://github.com/architkaila/Realtime-Hotel-Review-Analysis/blob/streamlit_demo/README.md) for the streamlit documentation 
* Here you can play around with the streamlit demo 

&nbsp;
# Future Enhancements:

1. To make results more robust, we would have to collect more data for the negative class.
2. The inference scripts( including the SHAP analysis and dependency parsing functionality) and the saved model can be pushed to a serverless cloud component like Azure functions or AWS lambda and an api can be called for faster inference time. With serverless deployment, we would make sure that the resources are being used efficiently.
3. We plan to integrate the model with the google review api, so that we can have user feed the hotel name in real time. Then we can use a weighted metric (like number of likes or upvotes on a review) to extract the top "n" review and run the above model on those "n" reviews to find the general sentiment and highest rated attributes for the given hotel.

&nbsp;
# References

1. "BERT, RoBERTa, DistilBERT, XLNet — which one to use?" by Suleiman Khan (2019) : compares BERT, RoBERTa, and XLNet to LSTM and GRU on sequence modeling tasks. Used to select the models we would want to fine tune and compare for this project.

2. "How to Fine-Tune BERT for Text Classification?" by Chi Sun et al. (2019) : used as a guideline for fine-tuning a pre-trained BERT model. Also used this paper to determine the value in trying to fine-tune ROBERTA and XLNET as compared to BERT.

3. "A Comparative Study on Transformer vs RNN in Speech Applications" by Shigeki Karita et al. (2019)


