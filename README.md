# collab
1.Data Source: Kaggle (Titanic:Train and Test Data)
2.libraries imported: import matplotlib.pyplot as plt, csv, os, pandas as pd, numpy as np, seaborn as sns, scipy as stats, Sklearn as linear_model, 
sklearn.linear_model import LogisticRegression,from sklearn import metrics, from sklearn.model_selection import cross_val_score
3.Loading and reading of CSV files
4.Combining both train data and test data to do cleaing and filling missed data, where it is feasibly safe to fill without much alteration to the existing data
5.Converting string into integers to ease the analysis
6.Once data cleaned, Splitting up test and train data to do analysis
6.Overal Survival to death of whole population boarded chart is done
8.Overall Age distribution in Ship boarded details shown by histogram. Show thre is more population in age group 25-35, chart is done
9.In the line graph, we see the survival to age ratio, there is more percentage of survival in children, and there is more death in the peak age boarded as described 
in previous graph.
10. Age Survival per Fare analysis
In this we see, green color is survived and red is dead, bigger the circle higher the fare. We see that as the fare goes on increasing, the probbaility of survival 
increasees may be due to better seating arrangement or major priority in sending them out on first available lifeboats
11. Box Plot between male and female. - Box plot, though shows almost similar, says that more women have taken higher fare ticket in comparison to men, which 
explains the next graph
12. Survived to death based on sex analysis, shows more women were survived compared to men
13. Coming to the Passenger Class Location in the ship, prediction of survival is counted, shows the people located at top floors of ship has more chances of survival
compared to people located on basement and lower floors because, one of the major reason is the submerged ice caused crack in the ship from basement, where in water
started to come in and peple had less time to escape. This is shown in 2 graphs, one how many in each location, how many survived in each class
14. This may be coincidence, taking on the embarked stations, people pertaining to embarkment station S have made through survival compared to other 2 stations.
15. Based on SibSp or Parch, its similar to embarked stations graph. It says, chances are more to survive, if you are not alone. very delicate supporting to this 
analysis as well

Lester to Update:





Coding for all the above is given below,
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

Combined_data.shape

Combined_data.info()

Combined_data.isnull().sum()

Drop_data= Combined_data.drop(['Name','Ticket','Cabin'],axis =1)
Drop_data.head()

Drop_data["Age"].fillna(Drop_data.groupby("Sex")["Age"].transform("mean"),inplace=True)

Drop_data.isna().sum()

Drop_data= Drop_data[~Drop_data["Fare"].isna()]
Drop_data= Drop_data[~Drop_data["Embarked"].isna()]

Drop_data.info()

mapping = {"female": 1, "male": 0}
Drop_data.Sex.replace(mapping, inplace=True)

mapping = {'S':0,'C':1,'Q':2}
Drop_data.Embarked.replace(mapping, inplace=True)

train = Drop_data[Drop_data["Survived"]!= '']
test = Drop_data[Drop_data["Survived"]== '']

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

#surv= list(train["Survived"])
#age = list(train['Age'])
#a_slope, a_int, a_r, a_p, a_std_err = stats.linregress(surv,age)

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

#Age Analysis with respect to number of people boarded
train_hist= train.hist(column="Age", bins = 15, figsize = (10,8))
plt.title("No. of Boarded People vs Age")
plt.ylabel("Number of people")
plt.xlabel("Age")
#plt.show()
# Save Figure
plt.savefig("Titanic_Number of People Vs Age.Png")

# survival to death plot based on their age

a = sns.kdeplot(train.Age[train.Survived== 0], label ="Died")
b = sns.kdeplot(train.Age[train.Survived== 1], label ="Survived")
plt.title("survived to Death based on Age")
plt.xlabel("Distribution per Age")
plt.ylabel("mean(survived)")
#plt.show()
# Save Figure
plt.savefig("Titanic_survived to Death based on Age.Png")

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

# Box plot fare between male vs female
from matplotlib import rcParams
sns.set(style="whitegrid")
rcParams['figure.figsize'] = 12,9
sns.boxplot( x=train["Sex"], y=train["Fare"],width=0.6, palette="Blues")

# Save Figure
plt.savefig("Titanic_fare between male vs female.Png")

# survived to death comparison for men and women

sns.countplot(x=train["Sex"], hue=train["Survived"])
plt.title("survived to Death based sex")
bar_width = 0.35
plt.ylabel("number of people")
plt.xticks(tick_locations, ["men\ndead to survived","women\ndead to survived"])
plt.show()
# Save Figure
plt.savefig("Titanic_survived to death comparison for men and women.Png")

#passenger class based on location of their seating arrplt.pie(part_class,labels=classes)

P1_class = (train['Pclass']==1).sum()
P2_class = (train['Pclass']==2).sum()
P3_class = (train['Pclass']==3).sum()
P1_class, P2_class, P3_class
classes = ['Higher and middle', 'middle', 'near to deck & Bottom']
part_class = [P1_class, P2_class, P3_class]
plt.pie(part_class,labels=classes)
plt.title("Location of Pclass in Ship")
colors = ["green", "red","orange"]
plt.pie(part_class,colors=colors,autopct="%1.2f%%", shadow=True, startangle=90)
plt.axis("equal")
plt.show()
# Save Figure
plt.savefig("Titanic_Pclass Population Distribuction.Png")

# survival to death plot based on their seating location

a = sns.kdeplot(train.Pclass[train.Survived== 0], label ="Died")
b = sns.kdeplot(train.Pclass[train.Survived== 1], label ="Survived")
plt.title("Ratio of survived to Death based on Class")
plt.xlabel("Class Distribution")
plt.ylabel("mean(survived)")
#plt.show()
# Save Figure
plt.savefig("Titanic_survived to death based on Pclass.Png")

train.describe()

# Embarked station analysis
survived = train[train['Survived']==1].Embarked.value_counts()
dead = train[train['Survived']==0].Embarked.value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,5))
plt.title("Embarked Statrion Analysis")
plt.ylabel("Number of people")
plt.xlabel("Stations")
# Save Figure
plt.savefig("Titanic_Embarked station analysis.Png")

# Sibling Spouse dependent Survival analysis
survived = train[train['Survived']==1].SibSp.value_counts()
dead = train[train['Survived']==0].SibSp.value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,5))
plt.title("Sibling Spouse dependent Survival analysis")
plt.ylabel("Number of people")
plt.xlabel("No of Siblings")
# Save Figure
plt.savefig("Titanic_Sibling Spouse dependent Survival Analysis.Png")

# Parent Children dependent Survival analysis
survived = train[train['Survived']==1].Parch.value_counts()
dead = train[train['Survived']==0].Parch.value_counts()
df = pd.DataFrame([survived,dead])
df.index = ['Survived','Dead']
df.plot(kind='bar',stacked=True, figsize=(10,5))
plt.title("Parent Children dependent Survival analysis")
plt.ylabel("Number of people")
plt.xlabel("No of Parents/Children")
# Save Figure
plt.savefig("Titanic_Parent Children dependent Survival analysis.Png")

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

print(regres.coef_)
# Those values, however, will show that the second parameter
# is more influential
#print(np.std(X_train)*regres.coef_)
#An alternative way to get a similar result is to examine the coefficients of the model fit on standardized parameters:
#regres.fit(X / np.std(X_train,0), Y_train)
#print(regres.coef_)

#10-fold Cross-Validation
lr =  linear_model.SGDClassifier(n_iter=100)
scores = cross_val_score(lr, X_train, Y_train, cv=10, scoring = "accuracy")

print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

#importances = pd.DataFrame({'feature':X_train.columns,'importance':np.round(logistic_regression.feature_importances_,3)})
#importances = importances.sort_values('importance',ascending=False).set_index('feature')
#importances.head(15)

