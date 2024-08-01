
# Twitter Sentiment Analysis

![App Screenshot](https://upload.wikimedia.org/wikipedia/commons/0/04/Twitter_and_X_logo_side_by_side.jpg)

## Objective
The objective of this project is to determine which method of the sentiment analysis provides better results when analyzing X data. We compared the performance of traditional Machine learning models, Deep neural networks and Transformers by leveraging Natural Lnaguage Processing(NLP) techniques. We evualte these models using Accuracy, Confusion Matrix, Precision, Loss.

## Dataset
The dataset used is Kaggle's **Sentiment140**. Consists of 1.6 million tweets labelled as positive and negative.You can access the dataset on Kaggle at the following link:
https://www.kaggle.com/datasets/kazanova/sentiment140


## Exploratory Data Analysis
- **Distribution of Positive and Negative Tweets**
![download](https://github.com/user-attachments/assets/d7f7de11-6c31-4ba6-b10a-bf755d7ecccd)

- **Distribution of @UserMentions , Links & #Hashtags**
![download](https://github.com/user-attachments/assets/a784cdd8-9bc7-4f45-bf55-566a028ad925)

## Word Clouds
- **Frequency of Positive words**
![download](https://github.com/user-attachments/assets/69f3a701-ed76-44e1-a566-9bae140c92da)

- **Frequency of Negative words** 
![download](https://github.com/user-attachments/assets/4da5419a-0860-48d7-adf9-568c91c35c9f)

## Natural Language Processing (NLP)
NLP plays a pivotal role in extracting, processing, and understanding textual data to determine the sentiment expressed within it.NLP techniques are used to clean and prepare textual data for analysis. This includes removing noise (e.g., stopwords, punctuation), normalizing text (e.g., lowercasing, stemming, and lemmatization), and handling slang and abbreviations.

## Sentiment Analysis models
- **ML Models:** [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html) , [Naive Bayes](https://scikit-learn.org/stable/modules/naive_bayes.html)
- **Neural Network:** [LSTM](https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM)
- **Transformer:** [BERT](https://huggingface.co/google-bert/bert-base-uncased)

## Machine Learning Models
- **Logistic Regression**
Widely used statistical model for binary classification. It predicts the probability that a given input belongs to a particular class by applying a logistic function (also known as the sigmoid function) to a linear combination of input features.

**Tokenizer:** [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

![download](https://github.com/user-attachments/assets/689000b5-f74e-4bfd-8d48-3d015779ffbc)

- **Naive Bayes**
Probablistic classifier based on Bayes theorem.

**Tokenizer:** [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)

![download](https://github.com/user-attachments/assets/44529d60-25ef-4b22-b933-6f5ff1fc6014)

- **Evaluation Table**

     | Model | Accuracy     | Dataset size                      |
     | :-------- | :------- | :-------------------------------- |
     | `Logistic Regression`      | `78.1` | 1,600,000 |
     | `Naive Bayes`      | `76.7` | 1,600,000 |


## Deep Neural Network
- **LSTM (Long Short-Term Memory)**
A type of Recurrent Neural Network, an LSTM recurrent unit tries to “remember” all the past knowledge that the network is seen so far and to “forget” irrelevant data.

**Embeddings:** [Word2Vec](https://www.tensorflow.org/text/tutorials/word2vec)

- **Evaluation Table**
| Model | Accuracy     | Dataset size                      | Number of Epochs | 
| :-------- | :------- | :-------------------------------- |:------------  |                   
| `LSTM`      | `79` | 300,000 | 8            

## Transformer
- **BERT(Bidirectional Encoder Representations from Transformer)**
BERT is a deep bidirectional, unsupervised language representation, pre-trained using a plain text corpus.BERT converts words into numbers. This process is important because machine learning models use numbers, not words, as inputs. This allows you to train machine learning models on your textual data.

**BERT model used:** [bert-base-multilingual-uncased-sentiment ](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)

I was able to use only limited number of epochs due to its long training time and the size of the dataset is greatly reduced for this purpose.

- **Evaluation Table**
| Model | No of Epochs     | Train Loss | Precision| Recall     | F1 score | Dataset size |
| :-------- | :------- | :-------------------------------- |:-------- | :------- | :-------------------------------- |:-------------------------------- |
| `BERT`      | `3` | 38.2 |`80`      | `77.6` | 75.4 |20,000 |

## Sentiment Analyzer

A simple sentiment analyzer is built to check the sentiment of the tweets/Reviews pasted in the text box. Sentiment score will be show according to the sentiment along with probability of the sentiment being positive , negative or neutral.

**Model used:** [bert-base-multilingual-uncased-sentiment ](https://huggingface.co/nlptown/bert-base-multilingual-uncased-sentiment)

- Demo video

https://github.com/user-attachments/assets/a5310205-fb54-42c0-a92b-4eeaf5d92fa1

In order to use this analyzer run  ***App.ipynb*** on your local python  IDE

## References

- [Training Models](https://www.kaggle.com/datasets/jp797498e/twitter-entity-sentiment-analysis)
- [Research Blog posts](https://www.analytixlabs.co.in/blog/twitter-sentiment-analysis/#:~:text=Twitter%20sentiment%20analysis%20is%20a,positive%2C%20negative%2C%20or%20neutral.)








