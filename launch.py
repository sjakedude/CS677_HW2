"""
Jake Stephens
Class: CS 677 - Summer 2
Date: 7/13/2021
Homework #1
Description: This program makes many different calculations
to analyze the stock data for 2 tickers, SUN and SPY.
"""
import pandas as pd
import os
import math

# Ticker file location
ticker_file = r"data\\SUN.csv"

# Function for computing True Labal
def true_label(row):
    if row["Return"] >= 0:
        return "+"
    else:
        return "-"

# ======================
# Main program execution
# ======================

# Reading csv into dataframe
df = pd.read_csv(ticker_file)

# Grabbing training subset
training_set = df.loc[df['Year'] < 2019]
training_set["True Label"] = training_set.apply (lambda row: true_label(row), axis=1)

def get_negative_probability(df, k):
    # Consecutive count
    count = 0

    # Counts of pos vs neg
    num_pos = 0
    num_neg = 0

    # Looping through each row
    for index, row in df.iterrows():
        if row["True Label"] == "-":
            if count == k:
                num_neg = num_neg + 1
                count = 0
            else:
                count = count + 1
        elif row["True Label"] == "+":
            if count == k:
                num_pos = num_pos + 1
                count = 0
            else:
                count = 0
    return num_pos, num_neg

for k in range(1,4):
    num_pos, num_neg = get_negative_probability(training_set, k)
    print("======================")
    print("For k=" + str(k))
    print("Probability = " + str(num_pos) + ":" + str(num_neg))




