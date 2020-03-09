
# coding: utf-8

# In[91]:


import pandas as pd
import random


# In[92]:


ls


# In[93]:


train_data = pd.read_csv("train.csv")
train_data


# In[94]:


test_data = pd.read_csv("test.csv")
test = test_data[['target','title','content']]


# In[95]:


test


# In[96]:


import matplotlib.pyplot as plt
fig = plt.figure(figsize=(8,6))
train_data.groupby('target').title.count().plot.bar(ylim=0)
train_data.shape
plt.show()


# <h4>Data is not skewed towards one class.Its BALANCED.
# Because it is not biased we can use conventional algorithms.When biased those algorithms tend to go towards majority classes.

# In[97]:


# taking a percentage of data
#p = 0.005
#train_data = pd.read_csv("train.csv",header=0,skiprows=lambda i: i>0 and random.random() > p)
#train_data
# should add column
#t_data["act_class"]
#with open("classes.txt") as classes:
#    act_class = classes.readlines()
#act_class
train_data


# <h4>Data Exploration : The given problem is an Multi-Class Text Classification.Total there are 14 unique classes in the given data.
#                    They are considered in the dataset in order to sort them.
#                    

# <h5>Intially the data is sampled because the computation is taking more time(in terms of CPU),Later full_data is considered

# In[98]:


train_data.loc[train_data['target'] == 1,'target_id'] = 'Company'
train_data.loc[train_data['target'] == 2,'target_id'] = 'EducationalInstitution'
train_data.loc[train_data['target'] == 3,'target_id'] = 'Artist'
train_data.loc[train_data['target'] == 4,'target_id'] = 'Athlete'
train_data.loc[train_data['target'] == 5,'target_id'] = 'OfficeHolder'
train_data.loc[train_data['target'] == 6,'target_id'] = 'MeanOfTransportation'
train_data.loc[train_data['target'] == 7,'target_id'] = 'Building'
train_data.loc[train_data['target'] == 8,'target_id'] = 'NaturalPlace'
train_data.loc[train_data['target'] == 9,'target_id'] = 'Village'
train_data.loc[train_data['target'] == 10,'target_id'] = 'Animal'
train_data.loc[train_data['target'] == 11,'target_id'] = 'Plant'
train_data.loc[train_data['target'] == 12,'target_id'] = 'Album'
train_data.loc[train_data['target'] == 13,'target_id'] = 'Film'
train_data.loc[train_data['target'] == 14,'target_id'] = 'WrittenWork'


# In[99]:


train_data


# 
# <h2> Pre-Processing
# 
# <h4>Removing the duplicates and Sorting them acc to Target Variables

# In[100]:


from io import StringIO
col = ['target_id', 'content']
train_data = train_data[col]
train_data = train_data[pd.notnull(train_data['content'])]
train_data.columns = ['target_id', 'content']
train_data['target'] = train_data['target_id'].factorize()[0]
category_id_df = train_data[['target_id', 'target']].drop_duplicates().sort_values('target')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['target', 'target_id']].values)
train_data


# ROC-AUC curve

# In[204]:


def roc_auc(predicted_data):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes= 14
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(np.array(pd.get_dummies(y_test))[:, i], np.array(pd.get_dummies(predicted_data))[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])


    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

    lw=2
    plt.figure(figsize=(8,5))
    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='green', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                 ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--',color='red', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.annotate('Random Guess',(.5,.48),color='red')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()


# <h3> Feature Extraction and Selection

# <h5>Count Vectorizer

# In[168]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
X_train = train_data['content']
y_train = train_data['target_id']


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)


# In[104]:


print(clf.predict(count_vect.transform([" The Vaidnei River is a tributary of the Scroafa River in Romania."])))


# In[105]:


test.loc[test['target'] == 1,'target_id'] = 'Company'
test.loc[test['target'] == 2,'target_id'] = 'EducationalInstitution'
test.loc[test['target'] == 3,'target_id'] = 'Artist'
test.loc[test['target'] == 4,'target_id'] = 'Athlete'
test.loc[test['target'] == 5,'target_id'] = 'OfficeHolder'
test.loc[test['target'] == 6,'target_id'] = 'MeanOfTransportation'
test.loc[test['target'] == 7,'target_id'] = 'Building'
test.loc[test['target'] == 8,'target_id'] = 'NaturalPlace'
test.loc[test['target'] == 9,'target_id'] = 'Village'
test.loc[test['target'] == 10,'target_id'] = 'Animal'
test.loc[test['target'] == 11,'target_id'] = 'Plant'
test.loc[test['target'] == 12,'target_id'] = 'Album'
test.loc[test['target'] == 13,'target_id'] = 'Film'
test.loc[test['target'] == 14,'target_id'] = 'WrittenWork'


# In[106]:


import numpy as np


# In[169]:


X_test = test['content']
y_test = test['target_id']

predicted_data = clf.predict(count_vect.transform(X_test))
accuracy = np.mean(predicted_data == y_test)
accuracy


# In[170]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


Train_accuracy = accuracy_score(y_train, clf.predict(X_train_tfidf))
Train_accuracy


# In[171]:


Test_accuracy = accuracy_score(y_test, predicted_data)
Test_accuracy


# In[172]:


conf = confusion_matrix(y_test, predicted_data)
conf


# In[173]:


roc_auc(predicted_data)


# In[174]:


from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

classifier = LinearSVC()

text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', classifier),])
text_clf.fit(X_train, y_train)

predicted_data = text_clf.predict(X_test)
accuracy= np.mean(predicted_data == y_test)
accuracy


# In[175]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


Train_accuracy = accuracy_score(y_train, clf.predict(X_train_tfidf))
Train_accuracy


# In[176]:


Test_accuracy = accuracy_score(y_test, predicted_data)
Test_accuracy


# In[177]:


conf = confusion_matrix(y_test, predicted_data)
conf


# In[178]:


roc_auc(predicted_data)


# <h4> Base line models considered are NaiveBayesian(Multinomial) , SVM

# <h5>TF-IDF VECTORIZER

# In[179]:


from sklearn.feature_extraction.text import TfidfVectorizer


Tfidf_vect = TfidfVectorizer(max_features=50000)
Tfidf_vect.fit(X_train)
Train_X_Tfidf = Tfidf_vect.transform(X_train)
Test_X_Tfidf = Tfidf_vect.transform(X_test)


# In[180]:


print(Tfidf_vect.vocabulary_)


# In[181]:


from sklearn import model_selection, naive_bayes, svm

Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf,y_train)
# predict the labels on validation dataset
predictions_NB = Naive.predict(Test_X_Tfidf)


# In[182]:


predictions_NB


# In[183]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


Train_accuracy = accuracy_score(y_train, clf.predict(X_train_tfidf))
Train_accuracy


# In[184]:


Accuracy = accuracy_score(predictions_NB, y_test)*100
Accuracy


# In[185]:


roc_auc(predicted_data)


# In[186]:


SVM = LinearSVC()
SVM.fit(Train_X_Tfidf,y_train)
predictions_SVM = SVM.predict(Test_X_Tfidf)


# In[187]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


Train_accuracy = accuracy_score(y_train, clf.predict(X_train_tfidf))
Train_accuracy


# In[188]:


Accuracy = accuracy_score(predictions_SVM, y_test)*100
Accuracy


# In[189]:


roc_auc(predicted_data)


# <h5> Removing stop_words and other pre-processing works

# In[190]:


import re
from bs4 import BeautifulSoup
replace_symbols= re.compile('[/(){}\[\]\|@,;]')
other_symbols = re.compile('[^0-9a-z #+_]')
def clean_text(text):
    text = replace_symbols.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = other_symbols.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    return text
train_data['content'] = train_data['content'].apply(clean_text)


# In[191]:


train_data['content'].dropna(inplace=True)
train_data['content'] = [entry.lower() for entry in train_data['content']]


# In[192]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
X_train = train_data['content']
y_train = train_data['target_id']


count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)


# In[151]:


X_test = test['content']
y_test = test['target_id']

predicted_data = clf.predict(count_vect.transform(X_test))
accuracy = np.mean(predicted_data == y_test)
accuracy


# In[193]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


Train_accuracy = accuracy_score(y_train, clf.predict(X_train_tfidf))
Train_accuracy


# In[194]:


Test_accuracy = accuracy_score(y_test, predicted_data)
Test_accuracy


# In[195]:


conf = confusion_matrix(y_test, predicted_data)
conf


# In[196]:


roc_auc(predicted_data)


# In[197]:


from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

classifier = LinearSVC()

clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', classifier),])
clf.fit(X_train, y_train)

predicted_data = clf.predict(X_test)
accuracy= np.mean(predicted_data == y_test)
accuracy


# In[163]:


from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

classifier = LinearSVC()

text_clf = Pipeline([('vect', CountVectorizer()),('tfidf', TfidfTransformer()),('clf', classifier),])
text_clf.fit(X_train, y_train)

predicted_data = text_clf.predict(X_test)
accuracy= np.mean(predicted_data == y_test)
accuracy


# In[199]:


Test_accuracy = accuracy_score(y_test, predicted_data)
Test_accuracy


# In[200]:


conf = confusion_matrix(y_test, predicted_data)
conf


# In[201]:


roc_auc(predicted_data)


# <h3>Using both types of Vectorizers we are getting the maximum accuracy for the SVM classifier

# <h2>Final Model : SVM Classifier

# <h5> Inputs from Medium Towards_Data_Science Analytics-Vidhya GitHub
