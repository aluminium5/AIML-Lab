# Dataset Link: https://www.kaggle.com/datasets/jmmvutu/ecommerce-purchases
# perform the following task on economic purchase dta set
#1 load data set
#2. how many rows and columns are there.
#3. what is average purchase price
#4.what are the highest and lowest purchase price?
#5. HOW MANY PEOPLE HAS THE JOB TITLE AS LAWYER?
#6.HOW MANY PEOPLE MADE PURCHASE DURING THE AM AND PM?
#7. WHAT ARE THE 5 MOST COMMON JOB TITLES?
#8.WHAT IS THE EMAIL OF THE PERSON WITH THE FOLLOWING CREDIT CARD NUMBER 4926535242672853
#9.SOMEONE MADE A PURCHASE THAT CAME FROM LOT"90 WT",WHAT IS THE PURCHASE PRICE FOR THIS TRANSACTION
#10.HOW MANY PEOPLE HAVE AMERICAN EXPRESS AS THEIR CREDIT CARD PROVIDER AND MADE A PURCHASE ABOBE 95$



import pandas as pd
try:
    df = pd.read_csv('Ecommerce Purchases.csv')
    print(df.head())

    print("rows and columns:", df.shape)

    average = df['Purchase Price'].mean()
    print("Average Purchase Price:", average)

    highest = df['Purchase Price'].max()
    lowest = df['Purchase Price'].min()
    print("Highest Price:", highest)
    print("Lowest Price:", lowest)

    lawyer= df[df['Job'] == 'Lawyer'].count()
    print(" people with the job title 'Lawyer':", lawyer)

    am_pm_counts = df['AM or PM'].count()
    print(am_pm_counts)

    top_5_jobs = df['Job'].head(10).max()
    print("Top 5 most common job titles:")
    print(top_5_jobs)

    email = df[df['Credit Card'] == 4926535242672853]['Email'].iloc[0]
    print(f"Email for Credit Card 4926535242672853: {email}")

    lotprice = df[df['Lot'] == '90 WT']['Purchase Price'].iloc[0]
    print(f"Purchase Price for Lot '90 WT': {lotprice}")

    america = df[(df['CC Provider'] == 'American Express') & (df['Purchase Price'] > 95)]
    print(f"American Express users with purchase > $95: {america}")
except FileNotFoundError:
    print("Ecommerce Purchases.csv not found.")
