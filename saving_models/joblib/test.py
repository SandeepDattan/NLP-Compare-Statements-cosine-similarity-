import joblib

model=joblib.load('final_model.pkl')
data=model.predict([[1,1,1,1,1,1,1,1]])

if data[0] == 0:
    print('Not Diabetic')
else:
    print('Diabetic')
