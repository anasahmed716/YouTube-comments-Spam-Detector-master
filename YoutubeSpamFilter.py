# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 11:13:42 2019

@author:

"""

import numpy as np
import pandas as pd
import seaborn as sns  # For visualization
import matplotlib.pyplot as plt   # For visualization

sns.set(style='ticks', context='talk')
na_values = ['NO CLUE', 'N/A', '0']

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', -1)

ytdspam_psy = pd.read_csv(r'C:\Users\anasa\Desktop\YouTube-comments-Spam-Detector-master\Youtube01-Psy.csv')
ytdspam_psy.head()
ytdspam_psy.shape

ytdspam_kattyparry = pd.read_csv(r'C:\Users\anasa\Desktop\YouTube-comments-Spam-Detector-master\Youtube02-KatyPerry.csv')
ytdspam_kattyparry.head()
ytdspam_kattyparry.shape

ytdspam_LMFAO = pd.read_csv(r'C:\Users\anasa\Desktop\YouTube-comments-Spam-Detector-master\Youtube03-LMFAO.csv')
ytdspam_LMFAO.head()
ytdspam_LMFAO.shape

ytdspam_Emnm = pd.read_csv(r'C:\Users\anasa\Desktop\YouTube-comments-Spam-Detector-master\Youtube04-Eminem.csv')
ytdspam_Emnm.head()
ytdspam_Emnm.shape

ytd_data = pd.concat([ytdspam_psy,ytdspam_kattyparry,ytdspam_LMFAO,ytdspam_Emnm])
ytd_data.head()
ytd_data.shape
ytd_data.loc[:,'CONTENT':'CLASS'].head()

#Checking spam comments
spam = ytd_data[ytd_data['CLASS']==1]
spam.isnull().sum()
spam.shape
spam.head()

SpmMatch = ["check my video", "Follow me", "watch my videos","subscribe","Please share",
            "Check out","my channel","my page", "giftcard","promos","sex","channel",
            "new track","ATTENTION","HTTP","subs","check","like them","new album","Hack",
            "VOTE","please listen","join me","help me","help","youtube","gay","share",
            "fuck","make money","visit","Donate", "trailer","free","channel","instagram",
            "facebook","soundcloud","support","website"]

# Count the number of spam words in the comment using the spam dictionary
ytd_data['SPM_CNT'] = ytd_data['CONTENT'].str.upper().str.count(str.upper("|".join(SpmMatch)))
ytd_data.shape
ytd_data['SPM_CNT']

ytd_data['IS_URL'] = ytd_data['CONTENT'].str.upper().str.contains(str.upper('http|https|www|.com'))
ytd_data.shape
ytd_data['IS_URL']

sns.barplot(x='CLASS', y='IS_URL', data=ytd_data) #checking the spam length against the feature is_url to find the spam words probability.

stop_words = [ "a","i","me","my","we","our","for", "ours","ourselves", "you", "your",
              "yourself","yourselves", "he","him", "his", "himself","her","hers",
              "herself","it", "its", "itself","them","their", "theirs","themselves",
              "what", "which","whom","this", "that","these", "those", "am","are","was", 
              "were","be", "been", "being","has","had", "having","do", "does", "did",
              "would","should", "could","ought", "i'm", "you're","she's","it's", "we're",
              "they're", "i've", "you've","they've","i'd", "you'd","he'd", "she'd",
              "we'd","i'll","you'll", "he'll","she'll", "we'll", "they'll","aren't",
              "wasn't", "weren't","hasn't", "haven't", "hadn't","don't","didn't",
              "won't","wouldn't", "shan't", "shouldn't","cannot","couldn't", "mustn't",
              "let's", "that's", "who's","here's","there's", "when's","where's", "why's",
              "how's","an","the", "and","but", "if", "or","as","until", "while","of", 
              "at", "by","with","about", "against","between", "into", "through","before",
              "after", "above","below", "to", "from","down","in", "out","on", "off", 
              "over","again","further", "then","once", "here", "there","where","why",
              "how","all", "any", "both","few","more", "most","other", "some", "such",
              "nor","not", "only","own", "same", "so","too","very"]
len(stop_words) 
#%%
from nltk.corpus import stopwords
sr = stopwords.words('english')
print(sr) 
len(sr)

#%%
ytd_data['CONTENT_FLTR'] = ytd_data['CONTENT'].str.lower().apply(lambda x: [item for item in str.split(x) if item not in stop_words])
ytd_data['CONTENT_FLTR']

ytd_data['COMMENT_LEN'] =  ytd_data['CONTENT_FLTR'].str.len()
ytd_data['COMMENT_LEN']

ytd_data.loc[:,['CONTENT','CONTENT_FLTR']].head()

ytd_data['SPM_to_COMNT'] = (ytd_data['SPM_CNT']/ytd_data['COMMENT_LEN'])
ytd_data['SPM_to_COMNT']

sns.barplot(orient='v',data=ytd_data,y='SPM_to_COMNT', x='CLASS')

ytd_data['spm_len'] = np.where(ytd_data['COMMENT_LEN'] > 50, 1, 0)
ytd_data['spm_len']
ytd_data['spm_len'].unique()

ytd_data.columns
sns.factorplot(y="spm_len",x="CLASS",data=ytd_data,kind="bar")
sns.factorplot(y="COMMENT_LEN",x="CLASS",data=ytd_data,kind="bar")

list(ytd_data.columns.values)

#%%Import Library of Gaussian Naive Bayes model
from sklearn.naive_bayes import GaussianNB
import numpy as np

NB = GaussianNB()

# Train the model using the training sets 
NB.fit(ytd_data.values[:,(5,6,8,9,10)],ytd_data.CLASS)

ytdspam_shakira = pd.read_csv(r'C:\Users\anasa\Desktop\YouTube-comments-Spam-Detector-master\Youtube05-Shakira.csv')

ytdspam_shakira.shape

# Count the number of spam words in the comment using the spam dictionary
ytdspam_shakira['SPM_CNT'] = ytdspam_shakira['CONTENT'].str.upper().str.count(str.upper("|".join(SpmMatch)))

ytdspam_shakira['IS_URL'] = ytdspam_shakira['CONTENT'].str.upper().str.contains(str.upper('http|https|www|.com'))

ytdspam_shakira['CONTENT_FLTR'] = ytdspam_shakira['CONTENT'].str.lower().apply(lambda x: [item for item in str.split(x) if item not in stop_words])

ytdspam_shakira['COMMENT_LEN'] =  ytdspam_shakira['CONTENT_FLTR'].str.len()

ytdspam_shakira['SPM_to_COMNT'] = (ytdspam_shakira['SPM_CNT']/ytdspam_shakira['COMMENT_LEN'])

ytdspam_shakira['spm_len'] = np.where(ytdspam_shakira['COMMENT_LEN'] > 50, 1, 0)

ytdspam_shakira.head()

#ytdspam_shakira.values[:,(5,6,8,9,10)]

# Predict the test data with NB Model
predicted = NB.predict(ytdspam_shakira.values[:,(5,6,8,9,10)])
predicted

print(list(zip(ytdspam_shakira.CLASS,predicted)))

from sklearn.metrics import accuracy_score

ytdspam_shakira.values[:,4]

from sklearn.metrics import confusion_matrix
confusion_matrix(ytdspam_shakira.CLASS, predicted)

accuracy_score(ytdspam_shakira.CLASS, predicted, normalize = True)

from sklearn import metrics

fpr, tpr, _ = metrics.roc_curve(ytdspam_shakira.CLASS,  predicted)
auc = metrics.roc_auc_score(ytdspam_shakira.CLASS,  predicted)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

y_pred_prob=NB.predict_proba(ytdspam_shakira.values[:,(5,6,8,9,10)])
print(y_pred_prob)

for a in np.arange(0,1,0.05):                         #(0,1,0.01)-----for incrementing in steps of 0.01
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(ytdspam_shakira.CLASS, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", 
        cfm[1,0]," , type 1 error:", cfm[0,1])
    
# Now changing the threshold value to 0.45
 
y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value>0.45:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)

confusion_matrix(ytdspam_shakira.CLASS, y_pred_class)

accuracy_score(ytdspam_shakira.CLASS, y_pred_class, normalize = True)
#%%
from sklearn import svm

C = 1  # SVM regularization parameter

svc = svm.SVC(kernel='linear', C=C).fit(ytd_data.values[:,(5,6,8,9,10)], ytd_data.CLASS)

predicted_svm = svc.predict(ytdspam_shakira.values[:,(5,6,8,9,10)])
predicted_svm

from sklearn.metrics import confusion_matrix
confusion_matrix(ytdspam_shakira.CLASS, predicted_svm)

accuracy_score(ytdspam_shakira.CLASS, predicted_svm, normalize = True)

fpr, tpr, _ = metrics.roc_curve(ytdspam_shakira.CLASS,  predicted_svm)
auc = metrics.roc_auc_score(ytdspam_shakira.CLASS,  predicted_svm)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

#%%

from sklearn.tree import DecisionTreeClassifier

model_DecisionTree = DecisionTreeClassifier(random_state=10)

model_DecisionTree = model_DecisionTree.fit(ytd_data.values[:,(5,6,8,9,10)], ytd_data.CLASS)

predicted_DT = model_DecisionTree.predict(ytdspam_shakira.values[:,(5,6,8,9,10)])
predicted_DT

print(list(zip(ytdspam_shakira.CLASS,predicted_DT)))

from sklearn.metrics import confusion_matrix
confusion_matrix(ytdspam_shakira.CLASS, predicted_DT)

accuracy_score(ytdspam_shakira.CLASS, predicted_DT, normalize = True)

fpr, tpr, _ = metrics.roc_curve(ytdspam_shakira.CLASS,  predicted_DT)
auc = metrics.roc_auc_score(ytdspam_shakira.CLASS,  predicted_DT)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

y_pred_prob=model_DecisionTree.predict_proba(ytdspam_shakira.values[:,(5,6,8,9,10)])
print(y_pred_prob)

for a in np.arange(0,1,0.05):                         #(0,1,0.01)-----for incrementing in steps of 0.01
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(ytdspam_shakira.CLASS, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", 
        cfm[1,0]," , type 1 error:", cfm[0,1])
    
# Now changing the threshold value to 0.45
 
y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value>0.45:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)

confusion_matrix(ytdspam_shakira.CLASS, y_pred_class)

accuracy_score(ytdspam_shakira.CLASS, y_pred_class, normalize = True)
#%%

from sklearn.linear_model import LogisticRegression

classifier = LogisticRegression()

classifier = classifier.fit(ytd_data.values[:,(5,6,8,9,10)], ytd_data.CLASS)

predicted_LR = classifier.predict(ytdspam_shakira.values[:,(5,6,8,9,10)])
predicted_LR

from sklearn.metrics import confusion_matrix
confusion_matrix(ytdspam_shakira.CLASS, predicted_LR)

accuracy_score(ytdspam_shakira.CLASS, predicted_LR, normalize = True)

print(list(zip(ytdspam_shakira.CLASS,predicted_LR)))

fpr, tpr, _ = metrics.roc_curve(ytdspam_shakira.CLASS,  predicted_LR)
auc = metrics.roc_auc_score(ytdspam_shakira.CLASS,  predicted_LR)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()

y_pred_prob=classifier.predict_proba(ytdspam_shakira.values[:,(5,6,8,9,10)])
print(y_pred_prob)

for a in np.arange(0,1,0.05):                         #(0,1,0.01)-----for incrementing in steps of 0.01
    predict_mine = np.where(y_pred_prob[:,1] > a, 1, 0)
    cfm=confusion_matrix(ytdspam_shakira.CLASS, predict_mine)
    total_err=cfm[0,1]+cfm[1,0]
    print("Errors at threshold ", a, ":",total_err, " , type 2 error :", 
        cfm[1,0]," , type 1 error:", cfm[0,1])
    
# Now changing the threshold value to 0.1
 
y_pred_class=[]
for value in y_pred_prob[:,1]:
    if value>0.1:
        y_pred_class.append(1)
    else:
        y_pred_class.append(0)
print(y_pred_class)

confusion_matrix(ytdspam_shakira.CLASS, y_pred_class)

accuracy_score(ytdspam_shakira.CLASS, y_pred_class, normalize = True) 
#%%Proper Ensemble Modelling

#Ensemble Modelling

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier

# create the sub models
estimators = []

model1 = LogisticRegression()
estimators.append(('log', model1))

model2 = DecisionTreeClassifier(criterion='gini',random_state=10)
estimators.append(('cart', model2))

model3 = SVC(kernel="rbf", C=100,gamma=0.1)
estimators.append(('svm', model3))

model4 = GaussianNB()
estimators.append(('nb', model4))
print(estimators) 

# create the ensemble model
ensemble = VotingClassifier(estimators)
ensemble.fit(ytd_data.values[:,(5,6,8,9,10)], ytd_data.CLASS)
Y_pred=ensemble.predict(ytdspam_shakira.values[:,(5,6,8,9,10)])
print(Y_pred) 

from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
#confusion matrix
print(confusion_matrix(ytdspam_shakira.CLASS, Y_pred))
print(classification_report(ytdspam_shakira.CLASS, Y_pred))
print(accuracy_score(ytdspam_shakira.CLASS, Y_pred))
