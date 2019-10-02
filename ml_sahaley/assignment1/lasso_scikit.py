'''
I selected Lasso because it is multi-feature, 
and there can be some dependence between this feature. 
So to remove this dependency used lasso insead of simple regression.
'''
#--------------------------------------------imports-------------------------------#
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import linear_model,preprocessing
from sklearn.metrics import mean_squared_error
#------------------------------------------reading data(updated, such that data is " " seperated)---------------------------#
df = pd.read_csv("PDBbind2015_refined-core.dat" , sep=" " , index_col=False)
df  = df.sample(frac=1).reset_index(drop=True)
#-------------------------------------------correlations--------------------------#
corr = df.corr()*100
#----------------------------------------------data split--------------------------#
xtraining,xtesting,ytraining,ytesting =  train_test_split(df.drop(columns=['affinity']) , df[['affinity']], test_size=0.25 , random_state=45)
#-------------------------------------------preprocessing---------------------------#
scaler = preprocessing.StandardScaler().fit(xtraining)
xtraining_transformed = scaler.transform(xtraining)
xtesting_transformed = scaler.transform(xtesting)
#--------------------------------------------Regression model-----------------------#
model = linear_model.Lasso(alpha=0.0001 , max_iter=1000000)
model.fit(xtraining_transformed , ytraining)
ytesting_predict = model.predict(xtesting_transformed)
print(model.coef_)#parameter vector (w in the cost function formula)
#---------------------------------------------------------error----------------------#
print("mean_squared_error = ",mean_squared_error(ytesting , ytesting_predict))