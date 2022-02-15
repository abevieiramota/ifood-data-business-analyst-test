import os
import pandas as pd
import re

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILEPATH = os.path.join(BASE_DIR, '../data/ml_project1_data.csv')
REPORTS_DIR = os.path.join(BASE_DIR, "../reports")


def iqr_outliers(v, kind='mild'):
    
    n = {'mild': 1.5, 'extreme': 3.0}[kind]
    
    v_desc = v.describe()
    
    q1 = v_desc["25%"]
    q3 = v_desc["75%"]
    iqr = q3 - q1
    
    upper_outliers = v >= q3 + n * iqr
    bottom_outliers = v <= q1 - n * iqr
    
    return v.loc[upper_outliers | bottom_outliers]


def read_data():

    df = pd.read_csv(DATA_FILEPATH, index_col="ID")

    return df


def apply_feature_engineering_1(df):

    previous_columns = df.columns

    # Dt_Customer
    df["Dt_Customer"] = df["Dt_Customer"].apply(lambda t: pd.to_datetime(t, format="%Y-%m-%d"))
    df["Year_Dt_Customer"] = df["Dt_Customer"].dt.year
    df["Mnt_Years_Relationship"] = df["Year_Dt_Customer"].max() + 1 - df["Year_Dt_Customer"]

    # Recency
    df["Is_Recency_gt_30"] = df["Recency"] > 30

    # Mnt*
    mnt_columns = ["MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts"]
    
    df["Total_Mnt"] = df.loc[:, mnt_columns].sum(axis=1)
    
    pct_mnt_columns = [f"Pct_{c}" for c in mnt_columns]
    df[pct_mnt_columns] = df[mnt_columns].div(df["Total_Mnt"], axis=0)

    df["Max_Mnt_Type"] = df[mnt_columns].idxmax(axis=1)

    # NumDealsPurchase
    df["Per_Year_NumDealsPurchases"] = df["NumDealsPurchases"].div(df["Mnt_Years_Relationship"], axis=0)

    # Num{sales_channel}Purchase
    num_sales_channel_purchase_columns = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    df["Total_Purchases"] = df.loc[:, num_sales_channel_purchase_columns].sum(axis=1)
    
    pct_num_sales_channel_purchase_columns = [f"Pct_{c}" for c in num_sales_channel_purchase_columns]
    df[pct_num_sales_channel_purchase_columns] = df[num_sales_channel_purchase_columns].div(df["Total_Purchases"], axis=0)

    df["Max_Purchase_Sales_Channel"] = df[num_sales_channel_purchase_columns].idxmax(axis=1)

    # NumDealsPurchase
    df["Pct_NumDealsPurchase"] = df["NumDealsPurchases"] / df["Total_Purchases"]

    # MntGoldProds
    df["Pct_MntGoldProds"] = df["MntGoldProds"] / df["Total_Purchases"]

    return df.drop(columns=previous_columns)