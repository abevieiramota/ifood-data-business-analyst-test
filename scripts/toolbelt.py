#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data")
DATA_FILEPATH = os.path.join(DATA_DIR, "ml_project1_data.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "../reports")
WARN_FILEPATH = os.path.join(REPORTS_DIR, "warns.csv")
CLEANED_DATA_FILEPATH = os.path.join(DATA_DIR, "cleaned_ml_project1_data.csv")
RFM_DATA_FILEPATH = os.path.join(DATA_DIR, "rfm_cleaned_ml_project1_data.csv")

if not os.path.isfile(WARN_FILEPATH):

    with open(WARN_FILEPATH, "w", encoding="UTF-8") as f:
        f.write("id,attr,type,action\n")


def register_warn(logs):

    with open(WARN_FILEPATH, "a", encoding="UTF-8") as f:
        for log in logs:
            f.write(f"{log['id']},{log['attr']},{log['type']},{log['action']}\n")


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

    # Dt_Customer
    df["Dt_Customer"] = df["Dt_Customer"].apply(lambda t: pd.to_datetime(t, format="%Y-%m-%d"))
    df["Year_Dt_Customer"] = df["Dt_Customer"].dt.year
    df["Total_Days_Relationship"] = (pd.to_datetime('01/01/2016', format='%d/%m/%Y') - df["Dt_Customer"]).dt.days

    # Recency
    df["Is_Recency_gt_30"] = df["Recency"] > 30

    # Mnt*
    mnt_columns = ["MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntWines"]
    
    df["Total_Mnt"] = df.loc[:, mnt_columns].sum(axis=1)

    df["Total_Mnt_Per_Total_Days_Relationship"] = df["Total_Mnt"].div(df["Total_Days_Relationship"], axis=0)
    
    pct_mnt_columns = [f"Pct_{c}" for c in mnt_columns]
    df[pct_mnt_columns] = df[mnt_columns].div(df["Total_Mnt"], axis=0)

    df["Max_Mnt_Type"] = df[mnt_columns].idxmax(axis=1)

    # NumDealsPurchase
    df["NumDealsPurchases_Per_Total_Days_Relationship"] = df["NumDealsPurchases"].div(df["Total_Days_Relationship"], axis=0)

    # Num{sales_channel}Purchase
    num_sales_channel_purchase_columns = ["NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
    df["Total_Purchases"] = df.loc[:, num_sales_channel_purchase_columns].sum(axis=1)
    df["Total_Purchases_Per_Total_Days_Relationship"] = df["Total_Purchases"].div(df["Total_Days_Relationship"], axis=0)
    
    pct_num_sales_channel_purchase_columns = [f"Pct_{c}" for c in num_sales_channel_purchase_columns]
    df[pct_num_sales_channel_purchase_columns] = df[num_sales_channel_purchase_columns].div(df["Total_Purchases"], axis=0)

    df["Max_Purchase_Sales_Channel"] = df[num_sales_channel_purchase_columns].idxmax(axis=1)

    # NumDealsPurchase
    df["Pct_NumDealsPurchase"] = df["NumDealsPurchases"] / df["Total_Purchases"]

    # MntGoldProds
    df["Pct_MntGoldProds"] = df["MntGoldProds"] / df["Total_Mnt"]

    return df


def clean_data_1(df):

    df = df.drop(index=
        [
            # Year_Birth outlier
            7829, 11004, 1150,
            # Income outlier
            9432
        ]
    )

    return df


def clean_data_2(df):

    df = df.drop(index=[
        # Pct_MntGoldProds outlier
        5255, 4246, 6237, 10311
    ])

    return df

def read_cleaned_data():

    df = pd.read_csv(CLEANED_DATA_FILEPATH, index_col="ID")

    return df