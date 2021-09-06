# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import pandas as pd
import numpy as np
import os
import xlrd
print(os.getcwd())
import nltk
nltk.download('punkt')
from PIL import Image
from wordcloud import ImageColorGenerator
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from collections import Counter
from nltk.corpus import stopwords
from string import punctuation
import re,string,unicodedata
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score


# %%
True_News = pd.read_csv('TrueNews.csv')
True_News.head()
True_News.shape


# %%
Fake_News = pd.read_csv('FakeNews.csv')
Fake_News.head()
Fake_News.shape


# %%
True_News['target'] = 1
Fake_News['target'] = 0 


# %%
True_News.head()
# True_News.count()
True_News.info()


# %%
Fake_News.head()
Fake_News.count()


# %%
News_data = pd.concat([True_News, Fake_News], ignore_index=True, sort=False)
News_data.tail()
News_data.count()
News_data.info()


# %%
import string
# def punctuation_removal(text):
#    all_list = [char for char in text if char not in string.punctuation]
#    clean_str = ''.join(all_list)
#    return clean_str
News_data['text'] = News_data['text'].apply(lambda y: ''.join([char for char in y if char not in string.punctuation]))
Fake_News['text'] = Fake_News['text'].apply(lambda y: ''.join([char for char in y if char not in string.punctuation]))
True_News['text'] = True_News['text'].apply(lambda y: ''.join([char for char in y if char not in string.punctuation]))


# %%
News_data.head()


# %%
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk import tokenize
useless = stopwords.words('english')
News_data['text'] = News_data['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (useless) ]))
News_data['text'][0]
Fake_News['text'] = Fake_News['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (useless) ]))
True_News['text'] = True_News['text'].apply(lambda x: ' '.join([word for word in x.split() if word not in (useless) ]))
Count_vect_test = CountVectorizer()
# token_space_test = tokenize.WhitespaceTokenizer()
# word_token = tokenize(News_data['text'])

#print(News_data.target[0])

#  for i, row in News_data.iterrows():
#    if News_data.target[i] ==1:
#         print("ok")
#        # all_words_test_fake = ' '.join([text for text in News_data['text']])    
#    else:
#        print("11111111")
    
# all_words_test_fake = ' '.join([text for text in News_data['text']])

# token_phrase_test = word_tokenize(all_words_test_fake)


# %%
# for i, row in News_data.iterrows():
#    if News_data.target[i] ==1:
#        all_words_test_fake = ' '.join(News_data['text'])
#    else:
#        all_words_test_real = ' '.join([News_data['text']])    

# all_words_test_fake = ' '.join([text for text in News_data['text'] if News_data.target.value==1])

# all_words_test_fake=  ' '.join([text for text in News_data['text']])


# token_phrase_fake = word_tokenize(all_words_test_fake)
# print(token_phrase_test[:100])
# token_phrase_real = word_tokenize(all_words_test_real)

all_words_test_fake= ' '.join([x for x in Fake_News['text'] ])
all_words_test_true = ' '.join([x for x in True_News['text'] ])
token_phrase = tokenize.WhitespaceTokenizer()
token_phrase_fake = token_phrase.tokenize(all_words_test_fake)
token_phrase_true = token_phrase.tokenize(all_words_test_true)

print(token_phrase_fake[:100])
print(token_phrase_true[:100])
# token_phrase_real = word_tokenize(all_words_test_real)


# %%
frequency_test_fake = nltk.FreqDist(token_phrase_fake)
# print(type(frequency_test))
df_frequency_test_fake = pd.DataFrame({"Word": list(frequency_test_fake.keys()),
                                   "Frequency": list(frequency_test_fake.values())})
df_frequency_test_fake = df_frequency_test_fake.nlargest(columns = "Frequency",n= 50)
print(df_frequency_test_fake)
frequency_test_true = nltk.FreqDist(token_phrase_true)
# print(type(frequency_test))
df_frequency_test_true = pd.DataFrame({"Word": list(frequency_test_true.keys()),
                                   "Frequency": list(frequency_test_true.values())})
df_frequency_test_true = df_frequency_test_true.nlargest(columns = "Frequency",n= 50)
print(df_frequency_test_true)
# NEED TO DIVIDE BET


# %%
plt.figure(figsize=(12,8))
ax = sns.barplot(data = df_frequency_test_fake, x = "Word", y = "Frequency", color="grey")
ax.set(ylabel = "Count")
plt.xticks(rotation='vertical')
plt.show()


# %%
plt.figure(figsize=(12,8))
ax = sns.barplot(data = df_frequency_test_true, x = "Word", y = "Frequency", color="grey")
ax.set(ylabel = "Count")
plt.xticks(rotation='vertical')
plt.show()

print(Fake_News.groupby(['subject'])['text'].count())
Fake_News.groupby(['subject'])['text'].count().plot(kind="bar")
plt.show()
print(True_News.groupby(['subject'])['text'].count())
True_News.groupby(['subject'])['text'].count().plot(kind="bar")
plt.show()


# %%
# token_space_test2 = tokenize.WhitespaceTokenizer()

# all_words_test2 = ' '.join([text for text in News_data['text']])
# token_phrase_test2 = token_space_test2.tokenize(all_words_test2)
# print(len(token_phrase_test2))
# print(type(token_phrase_test2))
# frequency2 = nltk.FreqDist(token_phrase_test2)
# df_frequency_test2 = pd.DataFrame({"Word": list(frequency2.keys()),
#                                    "Frequency": list(frequency2.values())})
# df_frequency_test2 = df_frequency_test2.nlargest(columns = "Frequency",n= 50)
# print(df_frequency_test2)


# %%



# %%
print(News_data.groupby(['subject'])['text'].count())
News_data.groupby(['subject'])['text'].count().plot(kind="bar")
plt.show()


# %%
print(News_data.groupby(['target'])['text'].count())
News_data.groupby(['target'])['text'].count().plot(kind='bar')
plt.show()


# %%
from wordcloud import WordCloud
fake_data = News_data[News_data['target'] == 0]
all_words = ' '.join([text for text in fake_data.text])
wordcloud = WordCloud(width= 800, height= 500,
                          max_font_size = 110,
                          collocations = False).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# %%
from wordcloud import WordCloud
real_data = News_data[News_data['target'] == 1]
all_words = ' '.join([text for text in real_data.text])
wordcloud = WordCloud(width= 800, height= 500, max_font_size = 110,
 collocations = False).generate(all_words)
plt.figure(figsize=(10,7))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.show()


# %%
# Most frequent words counter (Code adapted from https://www.kaggle.com/rodolfoluna/fake-news-detector)   
from nltk import tokenize
token_space = tokenize.WhitespaceTokenizer()
def counter(text, column_text, quantity):
    all_words = ' '.join([text for text in text[column_text]])
    token_phrase = token_space.tokenize(all_words)
    frequency = nltk.FreqDist(token_phrase)
    df_frequency = pd.DataFrame({"Word": list(frequency.keys()),
                                   "Frequency": list(frequency.values())})
    df_frequency = df_frequency.nlargest(columns = "Frequency", n = quantity)
    print(df_frequency)
    plt.figure(figsize=(12,8))
    ax = sns.barplot(data = df_frequency, x = "Word", y = "Frequency", color = 'blue')
    ax.set(ylabel = "Count")
    plt.xticks(rotation='vertical')
    plt.show()


# %%
counter(News_data[News_data['target'] == 0], 'text', 50)


# %%
counter(News_data[News_data['target'] == 1], 'text', 50)


# %%
# Function to plot the confusion matrix (code from https://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html)
from sklearn import metrics
import itertools
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


# %%

X_train,X_test,y_train,y_test = train_test_split(News_data['text'], News_data.target, test_size=0.2, random_state=42)


# %%
# Vectorizing and applying TF-IDF
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', LogisticRegression())])
# Fitting the model
model = pipe.fit(X_train, y_train)
# Accuracy
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# %%
cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=[0, 1])


# %%
# Vectorizing and applying TF-IDF
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score

pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', MultinomialNB())])
# Fitting the model
model = pipe.fit(X_train, y_train)
# Accuracy
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# %%
cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=[0, 1])


# %%
from sklearn.tree import DecisionTreeClassifier
# Vectorizing and applying TF-IDF
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', DecisionTreeClassifier(criterion= 'entropy',
                                           max_depth = 20, 
                                           splitter='best', 
                                           random_state=42))])
# Fitting the model
model = pipe.fit(X_train, y_train)
# Accuracy
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# %%
cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=[0,1])


# %%
from sklearn.ensemble import RandomForestClassifier
pipe = Pipeline([('vect', CountVectorizer()),
                 ('tfidf', TfidfTransformer()),
                 ('model', RandomForestClassifier(n_estimators=50, criterion="entropy"))])
model = pipe.fit(X_train, y_train)
prediction = model.predict(X_test)
print("accuracy: {}%".format(round(accuracy_score(y_test, prediction)*100,2)))


# %%
cm = metrics.confusion_matrix(y_test, prediction)
plot_confusion_matrix(cm, classes=[0,1])


# %%



# %%



