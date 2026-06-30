import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


os.makedirs('Machine Learning Projects/01_Placement_Prediction/data', exist_ok=True)
os.makedirs('Machine Learning Projects/01_Placement_Prediction/artifacts', exist_ok=True)
df = pd.read_csv(r"C:\\Users\\DELL\\Downloads\\placementdata.csv",usecols=[1,5,11])
placement_map = {'NotPlaced' : 0, 'Placed' : 1}
df["PlacementStatus"] = df["PlacementStatus"].map(placement_map)
print(df.info())
print(df.head())

sns.scatterplot(data=df,x='CGPA',y='AptitudeTestScore',hue='PlacementStatus')
pass

X = df[['CGPA','AptitudeTestScore']]
Y = df['PlacementStatus']
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

model = LogisticRegression()
model.fit(x_train,y_train)

y_prediction = model.predict(x_test)
print(y_prediction)

print(accuracy_score(y_test,y_prediction))
pickle.dump(scaler, open('Machine Learning Projects/01_Placement_Prediction/artifacts/scaler.pkl', 'wb'))
pickle.dump(model, open('Machine Learning Projects/01_Placement_Prediction/artifacts/model.pkl', 'wb'))

