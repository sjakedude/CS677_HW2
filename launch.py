"""
Jake Stephens
Class: CS 677 - Summer 2
Date: 7/13/2021
Homework #2
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

# Grabbing training subset of first 3 years
training_set = df.loc[df["Year"] < 2019]
training_set["True Label"] = training_set.apply(lambda row: true_label(row), axis=1)


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


def get_positive_probability(df, k):
    # Consecutive count
    count = 0

    # Counts of pos vs neg
    num_pos = 0
    num_neg = 0

    # Looping through each row
    for index, row in df.iterrows():
        if row["True Label"] == "+":
            if count == k:
                num_pos = num_pos + 1
                count = 0
            else:
                count = count + 1
        elif row["True Label"] == "-":
            if count == k:
                num_neg = num_neg + 1
                count = 0
            else:
                count = 0
    return num_pos, num_neg

chance_down = {
    1: {"Pos": None, "Neg": None},
    2: {"Pos": None, "Neg": None},
    3: {"Pos": None, "Neg": None},
}

print("Probability after seeing k consecutive 'down days'")
for k in range(1, 4):
    num_pos, num_neg = get_negative_probability(training_set, k)
    print("======================")
    print("For k=" + str(k))
    print("Probability = " + str(num_pos) + ":" + str(num_neg))
    chance_down[k]["Pos"] = num_pos
    chance_down[k]["Neg"] = num_neg

chance_up = {
    1: {"Pos": None, "Neg": None},
    2: {"Pos": None, "Neg": None},
    3: {"Pos": None, "Neg": None},
}

print("Probability after seeing k consecutive 'up days'")
for k in range(1, 4):
    num_pos, num_neg = get_positive_probability(training_set, k)
    print("======================")
    print("For k=" + str(k))
    print("Probability = " + str(num_pos) + ":" + str(num_neg))
    chance_up[k]["Pos"] = num_pos
    chance_up[k]["Neg"] = num_neg

# =================
# Predicting Labels
# =================
print("DOWN")
print(chance_down)

print("UP")
print(chance_up)

# Testing set of last 2 years
testing_set = df.loc[df["Year"] > 2018]
testing_set["True Label"] = testing_set.apply(lambda row: true_label(row), axis=1)

w = 3
d = 1

d_plus_one = None

s = []

index = testing_set.index[0]
num_correct = 0
num_incorrect = 0

while index < len(testing_set.index) + testing_set.index[0]:
    # preventing key error from viewing indexs outside dataset (in beginning 3)
    if d < w:
        d = d + 1
        index = index + 1
    else:
        # generating sequence of last 3 labels, including current day d
        s = [
            testing_set.loc[index]["True Label"],
            testing_set.loc[index - 1]["True Label"],
            testing_set.loc[index - 2]["True Label"],
        ]
        index = index + 1
        symbol = s[-1]
        num_consecutive = 1
        prediction = None
        # calculating the number of consecutive labels
        if s[-2] == symbol:
            num_consecutive = 2
            if s[-3] == symbol:
                num_consecutive = 3
        if symbol == "-":
            # compare to chance_down
            if chance_down[num_consecutive]["Pos"] >= chance_down[num_consecutive]["Neg"]:
                prediction = "+"
            elif chance_down[num_consecutive]["Pos"] < chance_down[num_consecutive]["Neg"]:
                prediction = "-"
        elif symbol == "+":
            # compare to chance_up
            if chance_up[num_consecutive]["Pos"] >= chance_up[num_consecutive]["Neg"]:
                prediction = "+"
            elif chance_up[num_consecutive]["Pos"] < chance_up[num_consecutive]["Neg"]:
                prediction = "-"
        # Preventing key error from trying to predict the next label outside the dataset
        if index + 1 < len(testing_set.index) + testing_set.index[0]:
            if prediction == testing_set.loc[index + 1]["True Label"]:
                num_correct = num_correct + 1
            else:
                num_incorrect = num_incorrect + 1

print(num_correct)
print(num_incorrect)
