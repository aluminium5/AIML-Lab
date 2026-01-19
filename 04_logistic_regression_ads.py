# Dataset Link: https://www.kaggle.com/datasets/rakeshrau/social-network-ads
#tranformation is applied to raw data for cleaning data is called Data Preprocessing.

"""Pre processing refers to the transformations applied to the data before feeding it in the algorithm.Data preprocessing is technique that is used to convert the raw data into clean data set.In other words,whenever the data is gathered from sources it is collected in raw format which is not feasible for analysis."""



import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

try:
    df = pd.read_csv('Social_Network_Ads1.csv')
    print(df.head())
    X=df[['Age','EstimatedSalary']]
    y=df["Purchased"]

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
    print(X_train.shape)

    print(X_train.isna().sum())#toatl sum of true values

    imputer=SimpleImputer(missing_values=np.nan,strategy='mean')
    imputer.fit(X_train)
    X_train_imputed = imputer.transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    #Use feature Scaling to convert different scales to a standard scale to make it easier for Machine learning algorithms
    #(18-60)&(15k-1.5L)
    scalar=MinMaxScaler()
    scalar.fit(X_train_imputed)
    X_train_scaled=scalar.transform(X_train_imputed)
    X_test_scaled=scalar.transform(X_test_imputed)
    print(X_train_scaled)

    model=LogisticRegression()
    model.fit(X_train_scaled,y_train)

    print("Score:", model.score(X_test_scaled,y_test))
except FileNotFoundError:
    print("Social_Network_Ads1.csv not found.")
