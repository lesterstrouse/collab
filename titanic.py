
# coding: utf-8

# In[1]:


# Dependencies and Setup
import matplotlib.pyplot as plt
import csv
import os
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.model_selection import cross_val_score
# File to Load (Remember to change these)
train_data_to_load=  "data/train.csv"
test_data_to_load =  "data/test.csv"


# Read the train and test Data
train_data = pd.read_csv("data/train.csv")
train_data
test_data = pd.read_csv("data/test.csv")
test_data.head()

test_data["Survived"]=""

# Combine the data into a single dataset
#combined_data = pd.merge(train_data, test_data, how="left", on=["Name","Name"])
Combined_data = train_data.append(test_data)

# Display the data table for preview
Combined_data.head()


# problem statement
# Complete the analysis of what sorts of people were likely to survive.
# In particular, to predict which passengers survived the Titanic tragedy.

# In[2]:


Combined_data.shape


# In[3]:


Combined_data.info()


# In[4]:


Combined_data.isnull().sum()


# In[5]:


Drop_data= Combined_data.drop(['Name','Ticket','Cabin'],axis =1)
Drop_data.head()


# In[6]:


Drop_data["Age"].fillna(Drop_data.groupby("Sex")["Age"].transform("mean"),inplace=True)


# In[7]:


Drop_data.isna().sum()


# In[8]:


Drop_data= Drop_data[~Drop_data["Fare"].isna()]
Drop_data= Drop_data[~Drop_data["Embarked"].isna()]


# In[9]:


Drop_data.info()


# In[10]:


mapping = {"female": 1, "male": 0}
Drop_data.Sex.replace(mapping, inplace=True)


# In[11]:


mapping = {'S':0,'C':1,'Q':2}
Drop_data.Embarked.replace(mapping, inplace=True)


# In[12]:


train = Drop_data[Drop_data["Survived"]!= '']


# In[13]:


test = Drop_data[Drop_data["Survived"]== '']


# In[15]:


surv = train['Survived'].values
sex = train['Sex'].values
corrsex,psex = stats.pearsonr(surv,sex)
fare=train["Fare"].values
corrfare,pfare = stats.pearsonr(surv,fare)
age=train['Age'].values
corrage,page = stats.pearsonr(surv,age)
par = train['Parch'].values
corrpar,ppar = stats.pearsonr(surv,par)
clas = train['Pclass'].values
corrclas,pclas = stats.pearsonr(surv,clas)
si = train['SibSp'].values
corrsi,psi = stats.pearsonr(surv, si)
em = train['Embarked'].values
correm,pem = stats.pearsonr(surv, em)
print('Correlations:')
#print(f'Sex-{corrsex}p={psex},Fare-{corrfare}p={pfare},Age-{corrage}p={page},Embarked-{correm}p={pem}')
#print(f'Parch-{corrpar}p={ppar},Pclass={corrclas}p={pclas},Sibsp-{corrsi}p={psi}')
prtcorr = pd.DataFrame({'':['r','p'],'Sex':[corrsex,psex],'Fare':[corrfare,pfare],'Age':[corrage,page],
                    'Parch':[corrpar,ppar],'Pclass':[corrclas,pclas],'SibSp':[corrsi,psi],'Embarked':[correm,pem]})
prtcorr


# In[16]:


#surv= list(train["Survived"])
#age = list(train['Age'])
#a_slope, a_int, a_r, a_p, a_std_err = stats.linregress(surv,age)


# In[17]:


train.shape, test.shape# survived to death comparison of overall population

Survived_count = train.groupby('Survived')
count_Survived = Survived_count['Survived'].count()
count_Survived
# Pie Chart

pies = ["Survive", "Death"]
pie_counts = [342, 549]
colors = ["green", "red"]
explode = [0.1, 0.1]

plt.pie(pie_counts, explode=explode, labels=pies, colors=colors,
       autopct="%1.2f%%", shadow=True, startangle=90)

plt.axis("equal")

# Save Figure
plt.savefig("Titanic_survived to death comparison of overall population.Png")


# In[18]:


#Age Analysis with respect to number of people boarded
train_hist= train.hist(column="Age", bins = 15, figsize = (10,8))
plt.title("No. of Boarded People vs Age")
plt.ylabel("Number of people")
plt.xlabel("Age")
plt.show()
# Save Figure
plt.savefig("Titanic_Number of People Vs Age.Png")


# In[19]:


# survival to death plot based on their age

a = sns.kdeplot(train.Age[train.Survived== 0], label ="Died")
b = sns.kdeplot(train.Age[train.Survived== 1], label ="Survived")
plt.title("survived to Death based on Age")
plt.xlabel("Distribution per Age")
plt.ylabel("mean(survived)")
plt.show()
# Save Figure
plt.savefig("Titanic_survived to Death based on Age.Png")


# In[20]:


#Age_Survival per Fare
ax = plt.subplot()

ax.scatter(train[train['Survived'] == 1]['Age'], train[train['Survived'] == 1]['Fare'], 
           c='green', s=train[train['Survived'] == 1]['Fare'])
ax.scatter(train[train['Survived'] == 0]['Age'], train[train['Survived'] == 0]['Fare'], 
           c='red', s=train[train['Survived'] == 0]['Fare']);
plt.title("Age-Survival Wrt Fare")
plt.xlabel("Age_Survival Distribution")
plt.ylabel("Fare")

plt.text(60, 70,"Note: Circle size correlates with Fare count per individual.")
# Save Figure
plt.savefig("Titanic_Age_Survival per Fare.Png")


# In[21]:


# Box plot fare between male vs female
from matplotlib import rcParams
sns.set(style="whitegrid")
rcParams['figure.figsize'] = 12,9
sns.boxplot( x=train["Sex"], y=train["Fare"],width=0.6, palette="Blues")

# Save Figure
plt.savefig("Titanic_fare between male vs female.Png")


# In[22]:


# survived to death comparison for men and women

sns.countplot(x=train["Sex"], hue=train["Survived"])
N=2
plt.title("survived to Death based sex")
index = np.arange(N)
bar_width = 0.35
plt.ylabel("number of people")
tick_locations = [value for value in index]
plt.xticks(tick_locations, ["men\ndead to survived","women\n to survived"])
plt.show()
# Save Figure
plt.savefig("Titanic_survived to death comparison for men and women.Png")


# In[23]:


#passenger class based on location of their seating arrplt.pie(part_class,labels=classes)

P1_class = (train['Pclass']==1).sum()
P2_class = (train['Pclass']==2).sum()
P3_class = (train['Pclass']==3).sum()
P1_class, P2_class, P3_class
classes = ['Higher and middle', 'middle', 'near to deck & Bottom']
part_class = [P1_class, P2_class, P3_class]
plt.pie(part_class,labels=classes)
plt.title("Location of Pclass in Ship")
colors = ["green", "red","blue"]
plt.pie(part_class,colors=colors,autopct="%1.2f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.show()
# Save Figure
plt.savefig("Titanic_Pclass Population Distribuction.Png")


# In[24]:


# survival to death plot based on their seating location

a = sns.kdeplot(train.Pclass[train.Survived== 0], label ="Died")
b = sns.kdeplot(train.Pclass[train.Survived== 1], label ="Survived")
plt.title("Ratio of survived to Death based on Class")
plt.xlabel("Class Distribution")
plt.ylabel("mean(survived)")
plt.show()
# Save Figure
plt.savefig("Titanic_survived to death based on Pclass.Png")


# In[25]:


train.describe()


# In[26]:


# Embarked station analysis
survived = train[train['Survived']==1].Embarked.value_counts()
dead = train[train['Survived']==0].Embarked.value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,5))
# Save Figure
plt.savefig("Titanic_Embarked station analysis.Png")


# In[27]:


# Sibling Spouse dependent Survival analysis
survived = train[train['Survived']==1].SibSp.value_counts()
dead = train[train['Survived']==0].SibSp.value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,5))
# Save Figure
plt.savefig("Titanic_Sibling Spouse dependent Survival Analysis.Png")


# In[28]:


# Parent Children dependent Survival analysis
survived = train[train['Survived']==1].Parch.value_counts()
dead = train[train['Survived']==0].Parch.value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,5))
# Save Figure
plt.savefig("Titanic_Parent Children dependent Survival analysis.Png")


# In[29]:


# regression analysis
X_test = test.drop("Survived",axis=1)
X_train = train.drop("Survived", axis=1)
Y_train = train["Survived"]
Y_train=Y_train.astype('int')
regres = LogisticRegression()
#print(regres)
regres.fit(X_train, Y_train)

Y_pred = regres.predict(X_test)

acc_log = round(regres.score(X_train, Y_train) * 100, 2)
print(acc_log)
print(Y_pred)


# In[30]:


print(regres.coef_)
# Those values, however, will show that the second parameter
# is more influential
#print(np.std(X_train)*regres.coef_)
#An alternative way to get a similar result is to examine the coefficients of the model fit on standardized parameters:
#regres.fit(X / np.std(X_train,0), Y_train)
#print(regres.coef_)


# In[31]:


#10-fold Cross-Validation
lr =  linear_model.SGDClassifier(n_iter=100)
scores = cross_val_score(lr, X_train, Y_train, cv=10, scoring = "accuracy")


# In[32]:


print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())


# In[33]:


#importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(logistic_regression.feature_importances_,3)})
#importances = importances.sort_values('importance',ascending=False).set_index('feature')
#importances.head(15)

