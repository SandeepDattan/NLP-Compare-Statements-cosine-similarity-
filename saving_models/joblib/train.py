import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import joblib

data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg','plas','pres','skin','test','mass','pedi','age','class']

dataframe = pandas.read_csv(data_url, names=names)

# print(dataframe)

array=dataframe.values

x=array[:,0:8]
y=array[:,8]

x_train, x_test, y_train, y_test=model_selection.train_test_split(x,y, test_size=0.2, random_state=101) 

model=LogisticRegression()
model.fit(x_train,y_train)

result=model.score(x_test,y_test)
print(result)

joblib.dump(model, 'final_model.pkl')
