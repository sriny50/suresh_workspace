import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.feature_selection import SelectFromModel  as sfm
from sklearn.ensemble import RandomForestClassifier as rfc
from sklearn.metrics import accuracy_score
from sklearn import metrics
import pickle

#Exploratory Data Analysis
missing_values = ["n/a","na","--","NA"]
df = pd.read_csv('Employeeattn.csv',na_values=missing_values)
temp=[]
for col in df.columns:
    if len(df[col].unique()) == 1:
        temp.append(col)
        df.drop(col,inplace=True,axis=2)
print(temp)
df[df.duplicated(keep=False)]
df.drop_duplicates(['EmployeeNumber'],keep='first')

'''search for missing/null values from the given csv file and discarding it'''
null_columns=df.columns[df.isnull().any()]
print(df[df["Education"].isnull()][null_columns])
df.dropna(inplace=True)
df.drop(['EmployeeNumber'], axis = 1, inplace = True)

#Correlation among the columns
corr=df.corr()
f,ax = plt.subplots(figsize=(18, 18))
sns.heatmap(corr, annot=True, linewidths=.10, fmt= '.1f',ax=ax,cmap='viridis')
corr.style.background_gradient().set_precision(1)

#Converting categorical to numerical
def transform(feature):
    le=LabelEncoder()
    df[feature]=le.fit_transform(df[feature])
    print(le.classes_)
df_catg=df.select_dtypes(include='object')
for col in df_catg.columns:
    transform(col)
df=df.astype(float)

#Scaling columns and target
target = df['Attrition']
features = df.drop('Attrition', axis = 1)
X_train, X_test, y_train, y_test = tts(features, target, test_size=0.3, random_state=10)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
sel = sfm(rfc(n_estimators = 870))
sel.fit(X_train, y_train)

#Featured columns selection
selected_features= sel.get_support(indices=True)
fc=features.columns.get_values()[selected_features]
print(fc)
df_fin=df[['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate',
       'MonthlyIncome', 'MonthlyRate', 'OverTime', 'PercentSalaryHike',
       'TotalWorkingYears', 'YearsAtCompany']].copy()

#Splitting records 60% Train , 20% Test , 20% Validate
X_train, X_test, X_validate = np.split(df_fin.sample(frac=1), [int(.6*len(df_fin)), int(.8*len(df_fin))])
y_train, y_test, y_validate = np.split(target.sample(frac=1), [int(.6*len(target)), int(.8*len(target))])
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Training model
from sklearn.ensemble import RandomForestClassifier
clf= RandomForestClassifier(n_estimators = 870)
clf.fit(X_train,y_train)
clf_trained=clf.fit(X_train,y_train)
print(clf_trained.score(X_train,y_train))

#Testing model
y_pred= clf.predict(X_test)
print(clf_trained.score(X_test,y_test))
testscore=('Accuracy: {:0.3f}'.format(accuracy_score(y_test,y_pred)))
print(testscore)
testpredprob= clf.predict_proba(X_test)
print(testpredprob)

#Score
y_targt=clf.predict(X_validate)
print(clf_trained.score(X_validate,y_validate))
valscore=('Accuracy: {:0.3f}'.format(accuracy_score(y_validate,y_targt)))
print(valscore)

#Confusionmatrix
from sklearn.metrics import confusion_matrix
desired = metrics.confusion_matrix(y_pred, y_test)
sns.heatmap(desired, annot=True, fmt='0.2f',xticklabels = ["left", "stayed"] , yticklabels = ["left", "stayed"],cmap="YlGnBu")
plt.ylabel('ACTUAL')
plt.xlabel('PREDICTED')
df_confusion= metrics.confusion_matrix(y_test,y_pred)
plt.savefig('testprediction.png')

#Retaining model
pickle.dump(clf,open('Emp_Att.pckl','wb'))


#Validate trained model with new data
inpt = pd.read_csv('Employee_newdata.csv',na_values=missing_values)
inptsc = pd.read_csv('Employee_newdata.csv')
inptsc=inptsc.astype(float)
sc = StandardScaler()
inptsc=sc.fit_transform(inptsc)
attnprob1=clf.predict_proba(inptsc)
inpt['Probability of Leaving'] = attnprob1[0:,1].tolist()
minim=inpt['Probability of Leaving'].min()
maxim=inpt['Probability of Leaving'].max()
tar=[minim,maxim]
bining=list(pd.cut(tar,10,retbins=True,labels=False))
ran=bining[1]
probrange=ran.tolist()
inpt['Rank']=pd.cut(inpt['Probability of Leaving'],probrange,labels=['10','9','8','7','6','5','4','3','2','1'])
print(inpt.head(5))
print(inpt.tail(5))

#Visuvalization
f, ax = plt.subplots(figsize=(15, 4))
sns.countplot(y='Rank',hue='Rank',data=inpt).set_title('Empattrition')
plt.ylabel('Probabilites of attrition rank')
plt.xlabel('Employee counts')
plt.plot()


#API Creation
import os.path
from flask import Flask, jsonify
app = Flask(__name__)
app.config.from_object(__name__)
@app.route('/', methods=['GET'])
def root_dir():
    return os.getcwd()
@app.route('/score', methods=['GET'])
def predict():
    return jsonify(inpt)

if __name__ == '__main__':
    app.run(port=8080,debug=True)


