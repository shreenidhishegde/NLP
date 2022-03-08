#!/usr/bin/env python
# coding: utf-8

# In[212]:
#Python3 version is used


import pandas as pd
import numpy as np
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
import re
from bs4 import BeautifulSoup
import contractions
import warnings
warnings.filterwarnings("ignore")
 


# In[61]:


#! pip install bs4 # in case you don't have it installed

# Dataset: https://s3.amazonaws.com/amazon-reviews-pds/tsv/amazon_reviews_us_Kitchen_v1_00.tsv.gz


# ## Read Data

# In[213]:


"""
Here we are reading the data from tsv file as pandas dataframe.
We use '\t' to seperate columns by a tab and "error_bad_lines = False" to drop bad lines from the DataFrame

"""  
text_data = pd.read_csv("data.tsv",error_bad_lines = False, sep = '\t', warn_bad_lines = False)


# ## Keep Reviews and Ratings

# In[215]:


#We are using dropna to look for missing values in the rows which has no review body and drop the corresponding rows.

text_data.dropna(subset = ['review_body'], inplace= True)
text_data.dropna(subset = ['star_rating'], inplace= True)


# We Keep only the reviews and ratings of the initial data
text_data = text_data[["star_rating","review_body"]]

# We are including 3 sample reviews with the corresponding rating
# print ("\n -x--x- Three sample reviews with corresponding ratings -x--x- \n")
# print (text_data.sample(n=3, random_state=100))

# We are reporting statistics of the ratings
star_counts = text_data["star_rating"].value_counts()



# print("\n -x--x- Statistics of the ratings -x--x- \n")
# print(star_counts)



# # Labelling Reviews:
# ## The reviews with rating 4,5 are labelled to be 1 and 1,2 are labelled as 0. Discard the reviews with rating 3'

# In[216]:


# Using np.where we are putting conditions to label the ratings >=4 as 1 and ratins <=2 as 0. 
#The remainign ratings which is labelled as 3 will be labelled as -1.

text_data['label'] = np.where(text_data["star_rating"] >=4,1, np.where(text_data["star_rating"] <= 2,0,-1))
rating_count = text_data['label'].value_counts()

result_list = []
for data in rating_count.iteritems():
    result_list.append(str(data[1]))
print(','.join(result_list))

#We are getting the counts of all review labels before removing the reviews with rating 3
# print("\n -x--x- Counts of all review labels before removing the reviews with rating 3 -x--x-\n")
# print(rating_count)

#Discarding the reviews with rating 3
text_data = text_data[text_data["label"]!= -1]

#We are getting the counts of all review labels after removing the reviews with rating 3
# print("\n -x--x- Counts of all review labels after removing the reviews with rating 3 -x--x- \n")
# print(text_data['label'].value_counts())


#  ## We select 200000 reviews randomly with 100,000 positive and 100,000 negative reviews.
# 
# 

# In[220]:


#We select sample of 100,000 positive and 100,000 negative reviews
negative = text_data.label[text_data.label.eq(0)].sample(100000,random_state =100).index
positive = text_data.label[text_data.label.eq(1)].sample(100000, random_state =100).index

#We combine both the selected samples
text_data = text_data.loc[negative.union(positive)]

# We are getting mean of each row based on characters in each row and then finding the mean of all the rows
review_mean = text_data.review_body.apply(lambda x : len(str(x))).mean()
# print 3 sample reviews

# print(f'\n -x--x- The average length of the reviews in terms of character length in the dataset before cleaning is -x--x- \n{review_mean:.2f}')
# print("\n -x--x- The sample reviews before cleaning -x--x- \n")
# print(text_data["review_body"].sample(n=3, random_state=100))


                                          
                                                                                                                 


# # Data Cleaning
# 
# ## Convert the all reviews into the lower case.

# In[221]:


#We use str.lower() to convert all the characters into lower characters
text_data["review_body"] = text_data["review_body"].str.lower()


#  ## Remove the HTML and URLs from the reviews

# In[222]:


#We use Beatiful soup to remove all the HTML Tags from the dataframe
text_data["review_body"] = text_data["review_body"].apply(lambda x: BeautifulSoup(str(x),"html.parser").get_text())

# We are here removing the URLs from the reviews
Url_pattern = r'\s*(https?://|www\.)+\S+(\s+|$)'
text_data["review_body"] = text_data["review_body"].apply(lambda x: re.sub(Url_pattern, " ", str(x), flags=re.UNICODE))


# ## remove non-alphabetical characters

# In[223]:


#First we remove all the words starts with digits
text_data["review_body"] = text_data["review_body"].apply(lambda x: re.sub(r"[^\D']+", " ", str(x), flags=re.UNICODE))

#Next we remove all the words which starts with non alphabetic characters
text_data["review_body"] = text_data["review_body"].apply(lambda x: re.sub(r"[^\w']+", " ", str(x), flags=re.UNICODE))


# ## Remove the extra spaces between the words

# In[224]:


#We remove the extra spaces from the dataset
# text_data["review_body"] = text_data["review_body"].replace('\s+', ' ', regex=True)
text_data["review_body"] = text_data["review_body"].apply(lambda x: re.sub(r"\s+", " ", str(x), flags=re.UNICODE))


# ## perform contractions on the reviews.

# In[225]:


#We perform contractions using contractions.fix

def contractionfunction(i):
    i = i.apply(lambda x: contractions.fix(x))
    return i

text_data["review_body"] = contractionfunction(text_data["review_body"])
#We convert the characters into lowercase again as after contractions some words will get capitalized Ex: "i'm will" become "I am" after contraction.
text_data["review_body"] = text_data["review_body"].str.lower()


# In[227]:


review_mean_new = text_data.review_body.apply(lambda x :len(str(x))).mean()
# print("\n -x--x- The average length of the reviews in terms of character length in the dataset after cleaning is -x--x-\n", review_mean_new)
# print("\n -x--x- The sample reviews after cleaning -x--x-\n")
# print(text_data["review_body"].sample(n=3, random_state=100))

print(str(review_mean) + "," + str(review_mean_new))


# # Pre-processing

# ## remove the stop words 

# In[228]:


# We are removing stop words 

from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

text_data["review_body"] = text_data["review_body"].apply(lambda x: " ".join([i for i in x.split() if i not in stop_words]))


# ## perform lemmatization  

# In[229]:


#We perform lemmatization on the data

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
text_data["review_body"] = text_data["review_body"].apply(lambda x: " ".join(lemmatizer.lemmatize(i) for i in x.split()))


# In[230]:


review_mean_3 = text_data.review_body.apply(lambda x : np.mean(len(str(x)))).mean()
# print("\n -x--x- The average length of the reviews in terms of character length in the dataset after pre-processing is -x--x- \n", review_mean_3)
# print("\n -x--x- The sample reviews after pre-processing -x--x-\n")
# print(text_data["review_body"].sample(n=3, random_state=100))

print(str(review_mean_new) + "," + str(review_mean_3))


# # TF-IDF Feature Extraction

# In[231]:


# We use train_test_split from sklearn to split the data into 80% training and 20% testing sets
from sklearn.model_selection import train_test_split 
from sklearn.feature_extraction.text import TfidfVectorizer

X_train, X_test, y_train, y_test = train_test_split(text_data["review_body"], text_data["label"], test_size=0.2, random_state=100)

#We ignore terms that have a document frequency strictly higher 0.7 and document frequency strictly lower than 1.
vectorizer = TfidfVectorizer(min_df=1, max_df=0.7)
Xtrain = vectorizer.fit_transform(X_train)
Xtest = vectorizer.transform(X_test)
    


# In[232]:


# we standardize the data using StandardScaler from sklearn
from sklearn.preprocessing import StandardScaler

#Create the instance
sc = StandardScaler(with_mean=False)

#We fit the scaler to the training feauture set only
sc.fit(Xtrain)

#Scale or Transform the training and the testing tests using the scaler that was fitted to training data
Xtrain_std = sc.transform(Xtrain)
Xtest_std = sc.transform(Xtest)


# # Perceptron

# In[234]:


#implementation of perceptron
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Create a perceptron object with the parameters: 40 iterations (epochs) over the data, and a learning rate of 0.1
ppn = Perceptron(max_iter=40, eta0=0.1, random_state=100)

# Fit the model to the standardized data
ppn.fit(Xtrain_std, y_train)


# Apply the trained perceptron on the X data to make predicts for the Y test data
y_pred_test = ppn.predict(Xtest_std)

#We measure the performance using the "accuracy_score,f1_score,precision_score and recall_score"

# print("\n -x--x- Accuracy, Precision, Recall, and f1-score on test data -x--x-\n")

# print(f'Testing Accuracy:{accuracy_score(y_test, y_pred_test):.4f}')
# print(f'Testing f1_Score:{f1_score(y_test, y_pred_test):.4f}')
# print(f'Testing Precision:{precision_score(y_test, y_pred_test):.4f}')
# print(f'Testing recall_score:{recall_score(y_test, y_pred_test):.4f}')

#Apply the trained perceptron on the data to make predicts for the trained data
y_pred_tarin = ppn.predict(Xtrain_std)


# print("\n -x--x- Accuracy, Precision, Recall, and f1-score on train data -x--x-\n")

# print(f'Training Accuracy:{accuracy_score(y_train, y_pred_tarin):.4f}')
# print(f'Training f1_Score:{f1_score(y_train, y_pred_tarin):.4f}')
# print(f'Training Precision:{precision_score(y_train, y_pred_tarin):.4f}')
# print(f'Training recall_score:{recall_score(y_train, y_pred_tarin):.4f}')

print(str(accuracy_score(y_train, y_pred_tarin)) + "," + str(precision_score(y_train, y_pred_tarin)) + "," + str(recall_score(y_train, y_pred_tarin)) + "," + str(f1_score(y_train, y_pred_tarin)) + "," + str(accuracy_score(y_test, y_pred_test)) + "," + str(precision_score(y_test, y_pred_test)) + "," + str(recall_score(y_test, y_pred_test)) + "," + str(f1_score(y_test, y_pred_test)))


# # SVM

# In[235]:


#implementation of SVM
from sklearn import svm

#Create a Classifier for svm
clf = svm.LinearSVC() # We are using Linear Kernel

#Train the model using the training sets
clf.fit(Xtrain, y_train)

#Apply the trained svm on Xtrain_std data to make predictions for the test data
y_pred_test = clf.predict(Xtest)


# print("\n -x--x- Accuracy, Precision, Recall, and f1-score on test data -x--x- \n")

# print(f'Testing Accuracy:{accuracy_score(y_test, y_pred_test):.4f}')
# print(f'Testing f1_Score:{f1_score(y_test, y_pred_test):.4f}')
# print(f'Testing Precision:{precision_score(y_test, y_pred_test):.4f}')
# print(f'Testing recall_score:{recall_score(y_test, y_pred_test):.4f}')

#Apply the trained perceptron on the data to make predicts for the trained data
y_pred_tarin = clf.predict(Xtrain)

#We measure the performance using the "accuracy_score,f1_Score,Precision_score and recall_score"

# print("\n -x--x- Accuracy, Precision, Recall, and f1-score on train data -x--x- \n")

# print(f'Training Accuracy:{accuracy_score(y_train, y_pred_tarin):.4f}')
# print(f'Training f1_Score:{f1_score(y_train, y_pred_tarin):.4f}')
# print(f'Training Precision:{precision_score(y_train, y_pred_tarin):.4f}')
# print(f'Training recall_score:{recall_score(y_train, y_pred_tarin):.4f}')

print(str(accuracy_score(y_train, y_pred_tarin)) + "," + str(precision_score(y_train, y_pred_tarin)) + "," + str(recall_score(y_train, y_pred_tarin)) + "," + str(f1_score(y_train, y_pred_tarin)) + "," + str(accuracy_score(y_test, y_pred_test)) + "," + str(precision_score(y_test, y_pred_test)) + "," + str(recall_score(y_test, y_pred_test)) + "," + str(f1_score(y_test, y_pred_test)))


# # Logistic Regression

# In[236]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# instantiate the model 
logreg = LogisticRegression()

# fit the model with data
logreg.fit(Xtrain,y_train)

#Apply the trained Logistic Regression on Xtrain_std data to make predictions for the test data
y_pred_test=logreg.predict(Xtest)

#We measure the performance using the "accuracy_score,f1_Score,Precision_score and recall_score"
# print("\n-x--x- Accuracy, Precision, Recall, and f1-score on test data --x--x-\n")

# print(f'Testing Accuracy:{accuracy_score(y_test, y_pred_test):.4f}')
# print(f'Testing f1_Score:{f1_score(y_test, y_pred_test):.4f}')
# print(f'Testing Precision:{precision_score(y_test, y_pred_test):.4f}')
# print(f'Testing recall_score:{recall_score(y_test, y_pred_test):.4f}')

#Apply the trained perceptron on the data to make predicts for the trained data
y_pred_tarin = logreg.predict(Xtrain)

#We measure the performance using the "accuracy_score"

# print("\n-x--x- Accuracy, Precision, Recall, and f1-score on train data -x--x-\n")

# print(f'Training Accuracy:{accuracy_score(y_train, y_pred_tarin):.4f}')
# print(f'Training f1_Score:{f1_score(y_train, y_pred_tarin):.4f}')
# print(f'Training Precision:{precision_score(y_train, y_pred_tarin):.4f}')
# print(f'Training recall_score:{recall_score(y_train, y_pred_tarin):.4f}')

print(str(accuracy_score(y_train, y_pred_tarin)) + "," + str(precision_score(y_train, y_pred_tarin)) + "," + str(recall_score(y_train, y_pred_tarin)) + "," + str(f1_score(y_train, y_pred_tarin)) + "," + str(accuracy_score(y_test, y_pred_test)) + "," + str(precision_score(y_test, y_pred_test)) + "," + str(recall_score(y_test, y_pred_test)) + "," + str(f1_score(y_test, y_pred_test)))



# # Naive Bayes

# In[237]:


#Import Gaussian Naive Bayes model
from sklearn.naive_bayes import MultinomialNB

#Create a Gaussian Classifier
model = MultinomialNB()

# Train the model using the training sets
model.fit(Xtrain, y_train)

#Predict Output
y_pred_test=model.predict(Xtest)

#We measure the performance using the "accuracy_score,f1_Score,Precision_score and recall_score"
# print("\n -x--x- Accuracy, Precision, Recall, and f1-score on test data -x--x- \n")

# print(f'Testing Accuracy:{accuracy_score(y_test, y_pred_test):.4f}')
# print(f'Testing f1_Score:{f1_score(y_test, y_pred_test):.4f}')
# print(f'Testing Precision:{precision_score(y_test, y_pred_test):.4f}')
# print(f'Testing recall_score:{recall_score(y_test, y_pred_test):.4f}')

#Apply the trained perceptron on the data to make predicts for the trained data
y_pred_tarin = model.predict(Xtrain)

#We measure the performance using the "accuracy_score"

# print("\n -x--x- Accuracy, Precision, Recall, and f1-score on train data -x--x- \n")

# print(f'Training Accuracy:{accuracy_score(y_train, y_pred_tarin):.4f}')
# print(f'Training f1_Score:{f1_score(y_train, y_pred_tarin):.4f}')
# print(f'Training Precision:{precision_score(y_train, y_pred_tarin):.4f}')
# print(f'Training recall_score:{recall_score(y_train, y_pred_tarin):.4f}')

print(str(accuracy_score(y_train, y_pred_tarin)) + "," + str(precision_score(y_train, y_pred_tarin)) + "," + str(recall_score(y_train, y_pred_tarin)) + "," + str(f1_score(y_train, y_pred_tarin)) + "," + str(accuracy_score(y_test, y_pred_test)) + "," + str(precision_score(y_test, y_pred_test)) + "," + str(recall_score(y_test, y_pred_test)) + "," + str(f1_score(y_test, y_pred_test)))


# In[ ]:




