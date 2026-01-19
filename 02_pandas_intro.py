# Dataset Link: https://www.kaggle.com/datasets/abcsds/pokemon


As compared to Excel, Pandas is lot flexible and through Pandas we can work with big data.
Pandas is an open-source library that is built on top of NumPy library.
It is a Python package that offers various data structures and operations for manipulating numerical data and time series.
It is mainly popular for importing and analyzing data much easier.
Pandas is fast and it has high-performance & productivity for users.
"""

#Pandas

#Type 1
from google.colab import files
# uploaded=files.upload # Commented out as this is Colab specific
#type 2 go in folder option and file will get uploaded

import pandas as pd
# df=pd.read_csv('pokemon_data.csv') # Assuming file needs to be present
# df

# print(df.shape)
# df.info()
# print(df.describe())
# print(df.head(5))
# print(df.tail(5))





import pandas as pd
# df=pd.read_csv('pokemon_data.csv')
# df

# print(df.shape)
# df.info()
# print(df.describe())
# print(df.head(5))
# print(df.tail(5))

# print(df.columns)



# print(df.describe())

# df = pd.read_csv('Ecommerce Purchases')
# display(df.head())

# df.sort_values(df['name','height','weight'],ascending=True)

# print(df[['name']])
# print(df.iloc[0:4])#integer location based function
# print(df.iloc[2,1])#i stands for index
# print(df.loc[df['Type 1']=='Fire'])

# pd.set_option('display.max_columns',13)
# pd.set_option('display.max_rows',800)
# df

# df['Total']=df['HP']+df['Attack']+df['Defense']
# print(df.head(5))
# df=df.drop(columns=['Total'])
# print(df)
# df['Total']=df.iloc[:,4:10].sum(axis=1)
# print(df)

# df.info()

# df.describe()#statistical data we get

# cols=list(df.columns)
# df=df[cols[0:4]+[cols[-1]]+cols[4:12]]
# print(df)
# df.to_csv('modified.csv',index=False)
# df.to_excel('modified.xlsx')
# df.to_csv('modified.txt',index=False,sep='\t')

# pd.set_option('display.max_columns',13)
# pd.set_option('display.max_rows',800)
# df

# df.head()
# df.head(10)
# df.tail(10)

# #Filtering Data
# #this will change the column value from fire to flamer
# df.loc[df["Type 1"]=='Fire','Type 1']='Flamer'
# df
# df.loc[df['Type 1']=='Flamer','Type 1']='Fire'
# df
# df.loc[df['Total']>500,["Generation","Legendary"]]=['Test1','Test2']
# df
# df=pd.read_csv('modified.csv')
# df

# df=pd.read_csv('modified.csv')
# df
# df_stats = df.groupby(['Type 1']).mean(numeric_only=True).sort_values('Defense',ascending=False)
# print(df_stats)

# df_attack_mean = df.groupby(['Type 1']).mean(numeric_only=True)['Attack']
# print(df_attack_mean)

# df.groupby(['Type 1']).count()
# df.groupby(['Type 1']).sum()
