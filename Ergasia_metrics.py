import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

filename = './telco_2023.csv'
df = pd.read_csv(filename)  # Replace 'yofrom sklearn.preprocessing import StandardScaler


df=df.drop(['region','custcat','marital','gender','ebill','confer','pager','forward'],axis=1)
X = df.drop('churn', axis=1)  # Features
y = df['churn']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kmetrics=pd.DataFrame(columns=['accuracy','recall','precision'])
dtmetrics=pd.DataFrame(columns=['accuracy','recall','precision'])
y_pred_all=[]
y_test_all=[]
for i in range(100):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    knn= KNeighborsClassifier(n_neighbors=10)
    
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test) #Προβλεψη στοχων
    nrow=pd.DataFrame([{'accuracy':accuracy_score(y_test, y_pred),'recall':recall_score(y_test, y_pred,pos_label=1),'precision':precision_score(y_test, y_pred,pos_label=1)}])
    kmetrics=pd.concat([kmetrics,nrow],ignore_index=True)
    y_pred_all.extend(y_pred)
    y_test_all.extend(y_test)
    
 
print("mean of metrics in 100 training cycles KNN")
print("Accuracy:", kmetrics["accuracy"].mean())
print("Recall:", kmetrics["recall"].mean())
print("Precision:", kmetrics["precision"].mean())  
 
cm = confusion_matrix(y_test_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Overal Knn Classifier Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

y_pred_all=[]
y_test_all=[]
for j in range(100):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    dt = DecisionTreeClassifier()  # Initialize the Decision Tree classifier
    dt.fit(X_train, y_train)  # Train the classifier using the training data
    y_pred=dt.predict(X_test)
    nrow=pd.DataFrame([{'accuracy':accuracy_score(y_test, y_pred),'recall':recall_score(y_test, y_pred, pos_label=1),'precision':precision_score(y_test, y_pred, pos_label=1)}])
    dtmetrics=pd.concat([dtmetrics,nrow],ignore_index=True)
    y_pred_all.extend(y_pred)
    y_test_all.extend(y_test)

print("mean of metrics in 100 training cycles of decision tree")    
print("Accuracy:", dtmetrics["accuracy"].mean())
print("Recall:", dtmetrics["recall"].mean())
print("Precision:", dtmetrics["precision"].mean())    

cm = confusion_matrix(y_test_all, y_pred_all)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Overal Decision Tree Classifier Confusion Matrix")
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()