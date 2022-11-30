# Importing libraries
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib

from sklearn.metrics import confusion_matrix
from sklearn import metrics
from sklearn.metrics import classification_report
import warnings
warnings.filterwarnings('ignore')


# Reading data set from url
data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
col_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age','Class']

dataframe=pd.read_csv(data_url,names=col_names)

# checking details about data
print(dataframe.info())
print(dataframe.describe().T)

# checking for null values
print(dataframe.isnull().sum())

# checking for zero values and replacing with mean values
dataframe_2 = dataframe.copy(deep=True)
dataframe_2[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = dataframe_2[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0,np.NaN)
print(dataframe_2.isnull().sum())

# imputing mean value of the column to each missing value within the particular column
dataframe_2['Glucose'].fillna(dataframe_2['Glucose'].mean(), inplace=True)
dataframe_2['BloodPressure'].fillna(dataframe_2['BloodPressure'].mean(), inplace=True)
dataframe_2['SkinThickness'].fillna(dataframe_2['SkinThickness'].mean(), inplace=True)
dataframe_2['Insulin'].fillna(dataframe_2['Insulin'].mean(), inplace=True)
dataframe_2['BMI'].fillna(dataframe_2['BMI'].mean(), inplace=True)

print(dataframe_2.isnull().sum())

array=dataframe_2.values

X1=array[:,0:8]
y=array[:,8]

scaler = StandardScaler()
model = scaler.fit(X1)
X = model.transform(X1)

print(X.head())

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=101)

model = LogisticRegression()
model.fit(X_train,y_train)

result =model.score(X_test,y_test)
print(result)