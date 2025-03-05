import scipy as sp
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression, ElasticNet, Lasso, Ridge, LinearRegression, LassoCV, ElasticNetCV, RidgeCV
from sklearn.ensemble import GradientBoostingRegressor as GB
from sklearn.metrics import r2_score, mean_squared_error,make_scorer
from sklearn.model_selection import train_test_split, cross_val_score 
import sweetviz as sv
from sklearn.ensemble import RandomForestRegressor
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, probplot
from scipy.stats import norm
from sklearn.model_selection import RandomizedSearchCV

import xgboost as xgb

df = pd.read_csv('kc_house_data.csv')
df['price'] = df['price']/1000000

print("Skewness: %f" % df['price'].skew())
print("Kurtosis: %f" % df['price'].kurt())

## Standardizing the data <==> Price
scaled_price = preprocessing.StandardScaler().fit_transform(df['price'].values.reshape(-1,1));
low_range = scaled_price[scaled_price[:,0].argsort()][:10]
high_range= scaled_price[scaled_price[:,0].argsort()][-10:]
print('outer range (low) of the distribution:')
print(low_range)
print('\nouter range (high) of the distribution:')
print(high_range)
df['price'] = np.log(df['price'])

# Standardizing the data
scaler = StandardScaler()
df['scaled_sqft_living'] = scaler.fit_transform(df[['sqft_living']])

# Selecting lowest and highest 10 values
sorted_values = np.sort(df['scaled_sqft_living'])
low_range = sorted_values[:10]
high_range = sorted_values[-10:]

print('Outer range (low) of the distribution:')
print(low_range)
print('\nOuter range (high) of the distribution:')
print(high_range)

# # going to log nature <==> sqft_living
df['sqft_living'] = np.log(df['sqft_living'])


# separating price from other features as the target
price = df.price.values

features_names = ['bedrooms', 'bathrooms', 'sqft_living', 'sqft_lot', 'floors','view', 'condition', 'grade', 'sqft_above',
                  'sqft_basement', 'yr_built', 'yr_renovated', 'zipcode', 'lat', 'long','sqft_living15', 'sqft_lot15']
features=df[features_names]

X_train, X_test, y_train, y_test = train_test_split(features, price, test_size=0.2, random_state=42)

lreg = LinearRegression()
lreg.fit(X_train,y_train)

r2 = r2_score(y_test,lreg.predict(X_test))

accuracy = lreg.score(X_test, y_test)
a = "Accuracy = {}%".format(int(round(accuracy * 100)))
print('R\u00b2' + "" + '=' + " " + str(r2))
print("")
print(a)
print("")
print('=========================================================')

for i in range(len(lreg.coef_)):
    print(features_names[i] + "" + '=' + "  " + str(lreg.coef_[i]))



def uncertainty(estimator, X_tr, X_te, y_tr, y_te):
    prediction_tr = estimator.predict(X_tr)
    
    print('')
    print('Estimator details')
    print('------------------------------------------')
    print(estimator)
    
    def get_score(prediction, lables):    
        print('R\u00b2: {}'.format(r2_score(prediction, lables)))
        print('RMSE: {}'.format(np.sqrt(mean_squared_error(prediction, lables))))
    
    
    
    print('')
    print("Error Values for Train Set")
    print('------------------------------------------')
    get_score(prediction_tr, y_tr)
    
    
    print('')
    print("Error Values for Test Set")
    print('------------------------------------------')
    prediction_test = estimator.predict(X_te)
    get_score(prediction_test, y_te)

# let's do default to see what will happen

Gradient_Boosting = GB(n_estimators=3000, learning_rate=0.05, max_depth=3, max_features='sqrt',
                                               min_samples_leaf=15, min_samples_split=10, loss='huber').fit(X_train, y_train)

uncertainty(Gradient_Boosting, X_train, X_test, y_train, y_test)

print('')
print('')
print("Gradient Boosting Model Accuracy")
print('------------------------------------------')
scores = cross_val_score(Gradient_Boosting, X_test, y_test, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
gb = GradientBoostingRegressor()

param_dist = {
    'n_estimators': np.arange(500, 5000, 500),
    'learning_rate': np.linspace(0.01, 0.2, 10),
    'max_depth': np.arange(3, 10, 1),
    'min_samples_split': np.arange(2, 20, 2),
    'min_samples_leaf': np.arange(1, 10, 2),
    'max_features': ['sqrt', 'log2', None],  
    'subsample': np.linspace(0.5, 1.0, 5)
}

random_search = RandomizedSearchCV(gb, param_dist, n_iter=50, cv=5, scoring='r2', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)

best_model = random_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test R² Score: {test_score:.4f}")


data_train = xgb.DMatrix(X_train, label=y_train)
data_train.save_binary('train.buffer')

data_test = xgb.DMatrix(X_test, label=y_test)
data_test.save_binary('test.buffer')
param = {'max_depth': 2, 'eta': 1, 'objective': 'reg:squarederror'}
evallist = [(data_train, 'train'), (data_test, 'eval')]
num_round = 1000
bst = xgb.train(param, data_train, num_round, evallist)
bst.save_model('RealEstateXGBosst.json')


y_pred = bst.predict(data_test)
print(f"{'Actual':<10}{'Prediction':<10}")
for i in range(10):
    print(f"{y_test.values[i]:<10.4f}{y_pred[i]:<10.4f}")
    

print('R\u00b2: {}'.format(r2_score(y_test, y_pred)))

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

param_grid = {
    'n_estimators': np.arange(100, 5000, 500),   
    'max_depth': np.arange(3, 10, 1),       
    'learning_rate': np.linspace(0.01, 0.2, 10),  
    'subsample': np.linspace(0.5, 1.0, 5),     
    'colsample_bytree': np.linspace(0.8, 1.0, 4)
}

random_search = RandomizedSearchCV(xgb_model, param_grid, n_iter=50, cv=5, scoring='r2', n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

print("Best Parameters:", random_search.best_params_)

best_model = random_search.best_estimator_
test_score = best_model.score(X_test, y_test)
print(f"Test R² Score: {test_score:.4f}")

