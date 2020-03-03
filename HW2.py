"""
This homework is an assignment for machine learning II course @SMU data science master program. 
It is a data scientist interview problem on medical insurance cliams. The dataset has around half
a million claims and each has about 30 features. A medical claim is denoted by a claim number 
('Claim.Number'). Each claim consists of one or more medical lines denoted by a claim line number
('Claim.Line.Number').

Note: The assignment requires the use of structured array instead of Pandas dataframes.
"""

#import libraries

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from itertools import product
import matplotlib.pyplot as plt
from adjustText import adjust_text

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
Loading the data as structured array and extracting the J-code claims 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#read in the claims data into a structured numpy array
CLAIMS = np.genfromtxt("claim.sample.csv", delimiter=",", dtype=None, names=True, usecols=range(29))
#print dtypes and field names
print('Datatypes for each features: ', CLAIMS.dtype)
#number of records
print('\nThe number of total claims: ', CLAIMS.shape)
#Using startswith() from numpy to find all claims that their procedure code starts with J (encoded)
indice = np.char.startswith(CLAIMS["ProcedureCode"], 'J'.encode(), start=0, end=None)
J_CLAIMS = CLAIMS[indice]
#number of claims with J procedure code
print('\nThe number of total J-code claims: ', J_CLAIMS.shape)

##Anpother way to subset the matrix using np.char.find()
#indice = np.char.find(CLAIMS['ProcedureCode'], 'J'.encode(), start = 0, end = 1)>=0

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
QUESTION 1: Find the number of claim lines that have J procedure code
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

print("\nQ1. The number of J-code claims: ", J_CLAIMS.shape[0])

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
QUESTION 2: How much was paid for to providers for 'in network' claims with J-code?
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#extracting the in network J-code claims, I needs to be encoded
J_inNetwork = J_CLAIMS[J_CLAIMS["InOutOfNetwork"] == 'I'.encode()]

#number of in network claims
print("\nThe number of in network J-code claims: ", J_inNetwork.shape[0])

#total amount paid to providers by in network claims with J-procedure code
print("\nQ2. Total amount paid to providers by in network J-code claims: ", 
     round(sum(J_inNetwork["ProviderPaymentAmount"]),2))

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""" 
QUESTION 3: What are the top five J-codes based on the payment to providers?
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#extract the procedure code and provider payment columns
J_payment = zip(J_CLAIMS["ProcedureCode"], J_CLAIMS["ProviderPaymentAmount"])

#generate a dictionary with jcodes as key and payment to provider as values
dic = {}
for J_code, payment in J_payment:
    if J_code in dic:
        dic[J_code] += payment
    else:
        dic[J_code] = payment

#sort the payment (from large to small) and print out the top 5 J codes
sorted_J_payment = sorted(dic.items(), key=lambda x: x[1], reverse=True)
print("\nQ3. Top five paid J-codes and their amount: \n")
for i in range(5):
    print(sorted_J_payment[i][0], ": $", round(sorted_J_payment[i][1], 2))

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
QUESTION 4: Create a scatter plot that displays the number of unpaid J-code claims 
(lines where the ‘Provider.Payment.Amount’ field is equal to zero) versus the number of
paid J-code claims for each provider.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#extracting the two relevant columns
payment_by_provider = zip(J_CLAIMS["ProviderID"], J_CLAIMS["ProviderPaymentAmount"])

#Use a nested dictionary that records the counts of paid and unpaid claims for each provider
provider_payment_dic = {}

for provider, payment in payment_by_provider:
    if provider in provider_payment_dic:
        if payment > 0:
            provider_payment_dic[provider]["paid"] += 1
        else:
            provider_payment_dic[provider]["unpaid"] += 1
    else:
        provider_payment_dic[provider] = {}
        if payment > 0:
            provider_payment_dic[provider]["paid"] = 1
            provider_payment_dic[provider]["unpaid"] = 0
        else:
            provider_payment_dic[provider]["paid"] = 0
            provider_payment_dic[provider]["unpaid"] = 1

#filter out providers with no paid claims (paid claims counts = 0)
providers = provider_payment_dic.keys()
claims_count = np.array([(p, provider_payment_dic[p]["paid"], provider_payment_dic[p]["unpaid"])
        for p in providers if provider_payment_dic[p]["paid"] > 0])

#print the providers with at least one paid J-code claims
print("\nThese providers have at least one paid J-code claims:\n")
for p in claims_count:
    print(p[0])

#scatterplot of the number of 'paid' vs 'unpaid' claims for each provider
plt.plot([0, 10], [0, 10], "red")#draw a diagonal red line to easily identify providers with more unpaid claims
plt.scatter(np.log(claims_count[:,1].astype(int)), np.log(claims_count[:,2].astype(int)), marker="o", s=7)
plt.xlabel("Paid claim counts (log)")
plt.ylabel("Unpaid claim counts (log)")
lab = [plt.text(np.log(p[1].astype(int)), np.log(p[2].astype(int)), str(p[0][5:])) for p in claims_count]
adjust_text(lab )#adjust the label position for providers
plt.savefig("paid_vs_unpaid.png")
plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
QUESTION 5: What insights can you suggest from the graph?
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

print('\nQ5. Insight from the graph: all except one provider (FA1000455101) are above the red line, indicating that almost all providers have more unpaid J-code claims.')

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
QUESTION 6: Based on the graph, is the behavior of any of the providers concerning? Explain.
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

print('\nQ6. Yes, the main concern is that the paid rate of J-code claims are too low because all but one providers have more unpaid claims. Next, I identified the top underpaid J-codes')

#The J-code claims for these providers
J_claims_paid = J_CLAIMS[np.isin(J_CLAIMS["ProviderID"], claims_count[:,0])]

#paid vs unpaid J-code claims 
payment_by_jcode = zip(J_claims_paid["ProcedureCode"], J_claims_paid["ProviderPaymentAmount"])

jcode_dic = {}
for code, payment in payment_by_jcode:
    if code in jcode_dic:
        if payment > 0:
            jcode_dic[code]["paid"] += 1
        else:
            jcode_dic[code]["unpaid"] += 1
    else:
        jcode_dic[code] = {}
        if payment > 0:
            jcode_dic[code]["paid"] = 1
            jcode_dic[code]["unpaid"] = 0
        else:
            jcode_dic[code]["unpaid"] = 1
            jcode_dic[code]["paid"] = 0

#j-code with more unpaid claims
codes = jcode_dic.keys()
jcodes = np.array([c for c in codes if jcode_dic[c]["paid"] > 0])
print("\nThe following J-codes are underpaid: \n\n", jcodes)

#top five unpaid j-codes (according to counts)
jcode_unpaid = sorted(jcode_dic.items(), key=lambda x: x[1]["unpaid"], reverse=True)
print("\nTop five unpaid J-code claims: \n\n", jcode_unpaid[:5])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
QUESTION 7: What percentage of J-code claim lines were unpaid (consider all claim lines with a 
J-code for Q7-10)?
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

print("\nQ7. The percentage of unpaid J-code claims: ", sum(J_CLAIMS["ProviderPaymentAmount"] == 0) / J_CLAIMS.shape[0])

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
QUESTION 8: Create a model to predict when a J-code is unpaid. 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#extracting the target labels (ProviderPaymentAmount) 
labels = (J_CLAIMS["ProviderPaymentAmount"] > 0).astype(int)

#EDA: Print out unique values for each columns, several features have high cardinality, definitely a concern
columns = J_CLAIMS.dtype.names
print("\nExplorative Data Analysis: number of unique values for each feature. A lot more data exploration is done in a Jupyter Notebook.")
for c in columns:
    print(c, len(np.unique(J_CLAIMS[c])))

#Detailed EDA is performed in a Jupyter notebook and cleaned data saved as J_ claims.csv
raw_data = np.genfromtxt('J_claims.csv', names = True, delimiter = ',', usecols = range(521))
colnames = raw_data.dtype.names #for feature importance analysis
raw_data = np.vstack(raw_data.tolist())#1d array to 2d array

#cross-validation folds
n_folds = 5
#a tuple is used as data container
data = (raw_data, labels, n_folds)

#define run() to train and cross validate different models and return average accuracy, AUC and recall scores 
def run(a_clf, data, clf_hyper={}):
    auc, acc, f1score = [], [], []  #metrics to keep track
    X, y, n_folds = data  #unpack data container into X, y and cv folds (n_folds)
    kf = KFold(n_splits=n_folds, shuffle = True, random_state= 111) #initiate cross validation object
    ret = {}  #initiate an empty dictionary for storing results
    for _, (train_index, test_index) in enumerate(kf.split(X, y)):
        clf = a_clf(**clf_hyper)  #unpack hyperparameters into clf if they exist
        clf.fit(X[train_index], y[train_index])  #train the model with the hyperparameters
        pred = clf.predict(X[test_index])  #predict test dataset
        #store AUC, F1 and accuracy scores of model prediction in three lists, auc, f1score and acc
        auc.append(roc_auc_score(y[test_index], pred))
        acc.append(accuracy_score(y[test_index], pred))
        f1score.append(f1_score(y[test_index], pred))
    #save model's average auc, accuracy and f1 scores in a dictionary
    ret[clf] = np.mean(auc), np.mean(acc), np.mean(f1score)
    return ret


#list of classifiers to train and their hyperparameters to tune
clfsList = [RandomForestClassifier, XGBClassifier]
clfDict = {
    RandomForestClassifier: {
        "criterion": ["gini", "entropy"],
        "min_samples_leaf": [1, 2, 3],
        "max_features": ["auto", "log2", 4]},
    XGBClassifier: {
        "max_depth": [3, 6, 9],
        "learning_rate": [0.1, 0.01, 0.001]}
        }

#tune hyperparameters and find the models with the best auc, acc and f1 scores
bestAuc, bestAcc, bestF1 = [], [], []
allResult = {}
for model in clfsList:
    result = {}
    if model in clfDict.keys():
        #expand hyperparameter dictionaries
        v1 = clfDict[model]
        #unpack the dictionary into two tuples, keys for the prarmeters and values for the lsits of values
        k1, v1 = zip(*v1.items())
        #iterate through cartesian products of v1
        for v in product(*v1):
            #combine the key and values (from v1 cartesian products) tuples into a dictionary as hyperparameters
            hyperparameter = dict(zip(k1, v))
            print(hyperparameter)
            #call run() function to train the model and store model metrics in the result dictionary
            result.update(run(model, data, clf_hyper=hyperparameter))
        #store results in allResult and find the best model for each classifier
        allResult.update(result)
        bestAuc.append(max(result.items(), key=lambda x: x[1][0]))
        bestAcc.append(max(result.items(), key=lambda x: x[1][1]))
        bestF1.append(max(result.items(), key=lambda x: x[1][2]))

#save the results
with open("result.csv", "w") as f:
    for key in allResult.keys():
        f.write("%s;%s;%s;%s\n" % (key, allResult[key][0], allResult[key][1], allResult[key][2]))
with open("bestAuc.csv", "w") as f:
    for i in range(len(clfsList)):
        f.write("%s;%s\n" % (bestAuc[i][0], bestAuc[i][1][0]))
with open("bestAcc.csv", "w") as f:
    for i in range(len(clfsList)):
        f.write("%s;%s\n" % (bestAcc[i][0], bestAcc[i][1][1]))
with open("bestF1.csv", "w") as f:
    for i in range(len(clfsList)):
        f.write("%s;%s\n" % (bestF1[i][0], bestF1[i][1][2]))

#plot the models with the best Auc, accuracy and F1 scores and save the plots
aucModel = list(zip(*bestAuc))[0]
aucList = list(zip(*bestAuc))[1]
auc = [aucList[i][0] for i in range(len(aucModel))]
plt.figure(figsize=[7, 7])
plt.barh(range(len(aucModel)), auc)
plt.yticks(range(len(aucModel)), aucModel)
plt.xlabel("AUC Score", fontsize=20)
plt.savefig("BestAUCModel.png", bbox_inches="tight")
plt.show()

accModel = list(zip(*bestAcc))[0]
accList = list(zip(*bestAcc))[1]
acc = [accList[i][0] for i in range(len(accModel))]
plt.figure(figsize=[7, 7])
plt.barh(range(len(accModel)), acc)
plt.yticks(range(len(accModel)), accModel)
plt.xlabel("Accuracy Score", fontsize=20)
plt.savefig("BestAccuracyModel.png", bbox_inches="tight")
plt.show()

f1Model = list(zip(*bestF1))[0]
f1List = list(zip(*bestF1))[1]
f1score = [f1List[i][0] for i in range(len(f1Model))]
plt.figure(figsize=[7, 7])
plt.barh(range(len(f1Model)), f1score)
plt.yticks(range(len(f1Model)), f1Model)
plt.xlabel("F1 Score", fontsize=20)
plt.savefig("BestF1_ScoreModel.png", bbox_inches="tight")
plt.show()

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
QUESTION 9: Best models' accuracy metrics 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

#We use the tuned hyperparameter to train the final models and identify the important features
train_X, test_X, train_y, test_y = train_test_split(raw_data, labels, test_size = 0.1)
 
RF_clf = RandomForestClassifier(criterion='entropy', min_samples_leaf=1, max_features='auto')
RF_clf.fit(train_X, train_y)
RF_pred = RF_clf.predict(test_X)

XGB_clf = XGBClassifier(learning_rate=0.1, max_depth=6)
XGB_clf.fit(train_X, train_y)
XGB_pred = XGB_clf.predict(test_X)

RF_auc = roc_auc_score(test_y, RF_pred)
print("\nThe Best RandomFoest model's AUC score: ", RF_auc)
XGB_auc = roc_auc_score(test_y, XGB_pred)
print("The best XGBoost model's AUC score: ", XGB_auc)

RF_acc = accuracy_score(test_y, RF_pred)
print("The Best RandomFoest model's accuracy score: ", RF_acc)
XGB_acc = accuracy_score(test_y, XGB_pred)
print("The best XGBoost model's accuracy score: ", XGB_acc)

RF_f1score = f1_score(test_y, RF_pred)
print("The Best RandomFoest model's f1 score: ", RF_f1score)
XGB_f1score = f1_score(test_y, XGB_pred)
print("The best XGBoost model's f1 score: ", XGB_f1score)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
QUESTION 10: Important features for predicting paid vs unpaid J-code claims 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
#Important features for the models
import pandas as pd
RF_feature_imp = pd.Series(RF_clf.feature_importances_, colnames).sort_values(ascending=False)
XGB_feature_imp = pd.Series(XGB_clf.feature_importances_, colnames).sort_values(ascending=False)

print('\nThe top features of the random forest model:\n', RF_feature_imp[:40])
print('\nThe top features of the xgboost model:\n', XGB_feature_imp[:40])

print("\nConlcusion: Denial Code is the top feature in both random forest and xgboost models")
