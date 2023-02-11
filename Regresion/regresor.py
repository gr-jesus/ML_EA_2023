import pandas as pd 
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split
from joblib import dump
from scipy.stats import wilcoxon
import numpy as np

#--------------------------------------------------------------------------------------
# load the dataset 
data=pd.read_csv('experiments.log')
X=data[['learning_rate','beta_1','beta_2','epsilon']]
Y=data[['accuracy']]

#--------------------------------------------------------------------------------------
#data=pd.read_csv('interpretabilidad.csv')#.sort_values(by=['interpretability'])
#data=pd.read_csv('resultados_grid_search_intepretabilidad.csv')
#data=data[data["layer"]==4]
#X=data[['learning_rate','warmup_epochs','mask_period','mem','lr_reg','lambda_reg']]
#Y=data[['precision']]
#Y=data[['interpretability']]




#p_values=[]
mse_=[]

# build the regresor
reg=RandomForestRegressor(n_estimators=100)
#reg.fit(X.values,Y)
#dump(reg, 'reg_x-ray.joblib')


for i in range(10):
	X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
	#reg=RandomForestRegressor(n_estimators=100)
	reg=SVR(kernel='poly', degree=3)
	reg.fit(X_train.values,y_train)

	y_pred=[]
	y_true=[]
	for index, row in X_test.iterrows():
		y_pred.append(reg.predict([row.tolist()])[0])

	for index, row in y_test.iterrows():
		y_true.append(row.tolist()[0])

	mse_.append(mse(y_true, y_pred))

print(np.mean(np.array(mse_)),'+/-',np.std(np.array(mse_)))

