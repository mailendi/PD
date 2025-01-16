#!/usr/bin/env python
# coding: utf-8

# N26 - Retail Credit Risk Associate
# 
# Mailen Diaz
# 
# 

# In[1]:


#Import libraries
import pandas as pd
import numpy as np
from sklearn import linear_model 
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import StandardScaler


# In[2]:


train=pd.read_csv('C:/Users/54114/Documents/N26/train.csv')
train.info() #to transform into numeric
train.shape


# In[3]:


#Transform object into dummies X21 X43 X55
Dummies_train=   [pd.get_dummies(train['X21'], prefix='X21'),
                  pd.get_dummies(train['X43'], prefix='X43'),
                  pd.get_dummies(train['X55'], prefix='X55')]
Dummie_final = pd.concat(Dummies_train, axis=1)
print(Dummie_final)


# In[4]:


train2=pd.concat([train,Dummie_final],axis=1)


# In[5]:


train2=train2.drop(columns=['X21', 'X43', 'X55'])


# In[6]:


pd.options.display.max_columns = None
train.describe() 
#Check outliers


# In[7]:


#a briefly review of distributions
train2.hist(bins=5, figsize=(20, 40), layout=(16, 5), edgecolor='black')
plt.tight_layout()
plt.show()


# In[8]:


#I decided to drop this due to the min/max and std approach
[train2[train2['X27']==565940.000000],train2[train2['X27']==-158130.000000],
 train2[train2['X62']==451380.000000]]
            


# In[9]:



train3=train2.drop(train2[train2['ID'] == 'A5023'].index)


# In[10]:


train4=train3.drop(train3[train3['ID'] == 'A2714'].index)
train4['X27'].describe()


# In[11]:


train5=train4.drop(train4[train4['ID'] == 'A4169'].index)
train5['X62'].describe()


# In[12]:


#Check missings
pd.options.display.max_rows = None
train5.isnull().sum()


# In[13]:


#X37 has more than 40% missing
train5=train5.drop(columns=['X37'])


# In[14]:


#for the rest I use mean
for col in train5:
    if train5[col].isnull().any():  
        mean_value = train5[col].mean()  
        train5[col].fillna(mean_value, inplace=True) 

print(train5.isnull().sum())  


# Modeling

# In[15]:


#Test  selection using p values
X = train5.drop(['TARGET','ID'], axis=1)
y = train5['TARGET']


# In[16]:


#I use this technique to know Pvalues
X = sm.add_constant(X)

model_l = sm.OLS(y, X).fit()

print(model_l.summary())


# In[17]:


p_values = model_l.pvalues
threshold = 0.05
significant_features = p_values[p_values < threshold].index
print(significant_features)


# In[18]:


X_train_pvalues=train5[['X9', 'X19', 'X20', 'X22', 'X23', 'X25', 'X28', 'X31', 'X32',
       'X34', 'X35', 'X38', 'X39', 'X40', 'X44', 'X52', 'X54', 'X57',
       'X21_75%-80%', 'X21_80%-90%', 'X21_90%-100%', 'X21_<75%', 'X21_>100%',
       'X43_120-240 days', 'X43_30-60 days', 'X43_60-120 days',
       'X43_above 240 days', 'X43_less 1 month', 'X55_0 to 500',
       'X55_500 to 2000', 'X55_above 2000', 'X55_negative']]


# In[19]:


#to avoid multicollinearity
corr_matrix=X_train_pvalues.corr() 


# In[20]:


threshold = 0.8

correlated_features = set()
for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > threshold:
            colname = corr_matrix.columns[i]
            correlated_features.add(colname)


# In[21]:


X_train_pvalues_2 = X_train_pvalues.drop(columns=correlated_features)


# In[22]:


features_used=X_train_pvalues_2.columns


# In[23]:


from sklearn.linear_model import LogisticRegression
model_log= LogisticRegression(fit_intercept=True)


# In[24]:


from sklearn.model_selection import train_test_split

X_train_p, X_test_p, y_train_p, y_test_p = train_test_split(X_train_pvalues_2, y, test_size=0.3, random_state=42)


# In[25]:


#scaler
scaler = StandardScaler()
X_train_scaled2 = scaler.fit_transform(X_train_p)
X_test_scaled2 = scaler.fit_transform(X_test_p)


# Model Selected

# In[26]:


#data scaled = I used this
model_log.fit(X_train_scaled2, y_train_p)
y_pred_pvalues = model_log.predict(X_test_scaled2)
score_pvalues = model_log.score(X_test_scaled2, y_test_p)
print(score_pvalues) 
print("Accuracy P Values:",accuracy_score(y_test_p, y_pred_pvalues))


# In[27]:


model_sca= LogisticRegression(fit_intercept=True)
model_sca.fit(X_train_p, y_train_p)
y_pred_pv = model_sca.predict(X_test_p)
score_pv = model_sca.score(X_test_p, y_test_p)
print(score_pv) 
print("Accuracy w/o Scal:",accuracy_score(y_test_p, y_pred_pv)) 


# In[28]:


y_pred_proba = model_log.predict_proba(X_test_scaled2)
y_pred_default_proba = y_pred_proba[:, 1]

tot = pd.DataFrame({'PD': y_pred_default_proba, 'p': y_test_p})


# In[29]:


from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_p, y_pred_default_proba)
auc_train = roc_auc_score(y_train_p, model_log.predict_proba(X_train_scaled2)[:, 1])
auc = roc_auc_score(y_test_p, y_pred_default_proba)

print(f"AUC train: {auc_train}")
print(f"AUC test: {auc}")

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc:.2f})', color='green')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC 1')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()


# In[30]:


cm_pvalues = confusion_matrix(y_test_p, y_pred_pvalues)
disp = ConfusionMatrixDisplay(confusion_matrix=cm_pvalues, display_labels=model_log.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix 1')
plt.savefig('C:/Users/54114/Documents/N26/CM_m1.png')
plt.show()


# In[31]:


#Test with changes in umbral=Doesn't work

umbral = 0.8
y_pred_custom = (y_pred_proba >= umbral).astype(int)
y_pred_custom_m = np.argmax(y_pred_custom, axis=1)


# In[32]:


cm_pvalues_changes = confusion_matrix(y_test_p, y_pred_custom_m)
disp_changes = ConfusionMatrixDisplay(confusion_matrix=cm_pvalues_changes, display_labels=model_log.classes_)
disp_changes.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix draft')
plt.show()


# In[33]:


#review of discriminatory power
def ks_statistic(y_test_p, y_pred_default_proba):
    data = pd.DataFrame({'true': y_test_p, 'pred_prob': y_pred_default_proba})
    data = data.sort_values(by='pred_prob', ascending=False)
    data['cum_true_positives'] = (data['true'] == 1).cumsum() / (data['true'] == 1).sum()
    data['cum_false_positives'] = (data['true'] == 0).cumsum() / (data['true'] == 0).sum()

    data['ks_diff'] = data['cum_true_positives'] - data['cum_false_positives']

    ks_value = data['ks_diff'].max()
    return ks_value

ks_value = ks_statistic(y_test_p, y_pred_default_proba)
print(ks_value)


tot = pd.DataFrame({'PD': y_pred_default_proba, 'p': y_test_p})


# In[34]:


from sklearn.model_selection import cross_val_score
cv_scores = cross_val_score(model_log, X_train_scaled2, y_train_p, cv=5, scoring='roc_auc')
print(f'Cross-Validation AUC scores: {cv_scores}')
print(f'Mean CV AUC: {cv_scores.mean()}')


# In[35]:


print(f"Interc: {model_log.intercept_}, Coef: {X_train_pvalues_2.columns,model_log.coef_}")


# In[36]:


#Test other clasification

from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(X, y)
importances = model.feature_importances_


# In[37]:


threshold = 0.025 #I know is not too high.
important_features = X.columns[importances > threshold]
print(important_features)


# In[38]:


X2=X[['X22', 'X27', 'X34', 'X35', 'X39', 'X42', 'X46']]
X2=X[['X22', 'X27', 'X35', 'X39', 'X41', 'X42', 'X46']]
X2.corr()


# In[39]:


X_R=X[['X22', 'X27', 'X39', 'X41', 'X42', 'X46']]


# In[40]:


from sklearn.linear_model import LogisticRegression
model_r= LogisticRegression(fit_intercept=True)


# In[41]:


from sklearn.model_selection import train_test_split

X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X_R, y, test_size=0.3, random_state=42)


# In[42]:


#scaler
scaler = StandardScaler()
X_train_scaled3 = scaler.fit_transform(X_train_r)
X_test_scaled3 = scaler.fit_transform(X_test_r)
X_test_scaled3.shape


# In[43]:


model_r.fit(X_train_scaled3, y_train_r)
y_pred_r = model_r.predict(X_test_scaled3)
score_R = model_r.score(X_test_scaled3, y_test_r)
print(score_R) 
print("Accuracy model 2:",accuracy_score(y_test_r, y_pred_r))


# In[44]:


y_pred_proba_rf = model_r.predict_proba(X_test_scaled3)
y_pred_default_proba_rf = y_pred_proba_rf[:, 1]

tot_r = pd.DataFrame({'PD': y_pred_default_proba_rf, 'p': y_test_r})


# In[45]:


from sklearn.metrics import roc_curve, roc_auc_score
fpr, tpr, thresholds = roc_curve(y_test_r, y_pred_default_proba_rf)
auc_train2 = roc_auc_score(y_train_r, model_r.predict_proba(X_train_scaled3)[:, 1])
auc2 = roc_auc_score(y_test_r, y_pred_default_proba_rf)

print(f"AUC train: {auc_train2}")
print(f"AUC test: {auc2}")

plt.figure(figsize=(8,6))
plt.plot(fpr, tpr, label=f'ROC curve (AUC = {auc2:.2f})', color='green')
plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AUC 2')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()
plt.savefig('C:/Users/54114/Documents/N26/AUC_m2.png')


# In[46]:


cm_r = confusion_matrix(y_test_r, y_pred_r)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm_r, display_labels=model_r.classes_)
disp2.plot(cmap=plt.cm.Blues)
plt.title('Confusion Matrix 2')
plt.savefig('C:/Users/54114/Documents/N26/CM_m2.png')
plt.show()


# In[47]:


#Test 3 with decision Tree
from sklearn.tree import DecisionTreeClassifier
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X, y, test_size=0.3, random_state=42)
tree_model = DecisionTreeClassifier(max_depth=6, random_state=42)
tree_model.fit(X_train_d, y_train_d)


# In[48]:


#is not a good model or strategy
y_pred_d = tree_model.predict(X_test_d)

print("Accuracy:",accuracy_score(y_test_d, y_pred_d))
print("ROC AUC:", roc_auc_score(y_test_d, y_pred_d))


# Dataset Test

# In[49]:


test_df=pd.read_csv('C:/Users/54114/Documents/N26/test.csv')


# In[50]:


test_df.info() #to transform into numeric


# In[51]:


Dummies_test=   [pd.get_dummies(test_df['X21'], prefix='X21'),
                  pd.get_dummies(test_df['X43'], prefix='X43'),
                  pd.get_dummies(test_df['X55'], prefix='X55')]
Dummie_final_test = pd.concat(Dummies_test, axis=1)


# In[52]:


test_df2=pd.concat([test_df,Dummie_final_test],axis=1)


# In[53]:


test_df3=test_df2.drop(columns=['X21', 'X43', 'X55','ID']) #Drop ID as well for scaler


# In[54]:


for col in test_df3:
    if test_df3[col].isnull().any():  
        mean_value = test_df3[col].mean()  
        test_df3[col].fillna(mean_value, inplace=True) 


# In[55]:


test_df4=test_df3[features_used]#use the same features


# In[56]:


#scaler in test 
X_test_scaled_val = scaler.fit_transform(test_df4)


# In[57]:


y_pred_test = model_log.predict(X_test_scaled_val)
y_pred_proba_test = model_log.predict_proba(X_test_scaled_val)
y_pred_default_proba_test = y_pred_proba_test[:, 1]


# In[58]:


pred = pd.DataFrame(y_pred_default_proba_test , columns=['predictions'])
print(pred)


# In[59]:


test_df2['Pred']=pred['predictions'].values
print(test_df2)


# In[61]:


#export final predictions
pred_final=test_df2[['ID', 'Pred']]
pred_final.to_csv('C:/Users/54114/Documents/N26/submission_samplemd.csv', index=False)

