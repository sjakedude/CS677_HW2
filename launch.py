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
import ast

# Ticker file location
ticker_file_sun = r"data\\SUN.csv"
ticker_file_spy = r"data\\SPY.csv"

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
df_sun = pd.read_csv(ticker_file_sun)
df_spy = pd.read_csv(ticker_file_spy)

# Grabbing SUN training subset of first 3 years
training_set_sun = df_sun.loc[df_sun["Year"] < 2019]
training_set_sun["True Label"] = training_set_sun.apply(
    lambda row: true_label(row), axis=1
)

# Grabbing SPY training subset of first 3 years
training_set_spy = df_spy.loc[df_spy["Year"] < 2018]
training_set_spy["True Label"] = training_set_spy.apply(
    lambda row: true_label(row), axis=1
)

# Function for computing the negative probability
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


# =======================================
# Populating the probability dictionaries
# =======================================

print("SUN")
print("======================")
print("Probability after seeing k consecutive 'down days'")
for k in range(1, 4):
    num_pos, num_neg = get_negative_probability(training_set_sun, k)
    print("======================")
    print("For k=" + str(k))
    print("Probability (pos:neg) = " + str(num_pos) + ":" + str(num_neg))

print("======================")
print("Probability after seeing k consecutive 'up days'")
for k in range(1, 4):
    num_pos, num_neg = get_positive_probability(training_set_sun, k)
    print("======================")
    print("For k=" + str(k))
    print("Probability (pos:neg) = " + str(num_pos) + ":" + str(num_neg))

print("SPY")
print("======================")
print("Probability after seeing k consecutive 'down days'")
for k in range(1, 4):
    num_pos, num_neg = get_negative_probability(training_set_spy, k)
    print("======================")
    print("For k=" + str(k))
    print("Probability (pos:neg) = " + str(num_pos) + ":" + str(num_neg))

print("======================")
print("Probability after seeing k consecutive 'up days'")
for k in range(1, 4):
    num_pos, num_neg = get_positive_probability(training_set_spy, k)
    print("======================")
    print("For k=" + str(k))
    print("Probability (pos:neg) = " + str(num_pos) + ":" + str(num_neg))

# =================
# Predicting Labels
# =================
print("=================================================")
print("Predicting Labels")
print("=================================================")

# Testing set of last 2 years for SUN and SPY
testing_set_sun = df_sun.loc[df_sun["Year"] > 2018]
testing_set_sun["True Label"] = testing_set_sun.apply(
    lambda row: true_label(row), axis=1
)

# Testing set of last 2 years for SUN and SPY
testing_set_spy = df_spy.loc[df_spy["Year"] > 2017]
testing_set_spy["True Label"] = testing_set_spy.apply(
    lambda row: true_label(row), axis=1
)

accuracy_dict = {"SUN": {}, "SPY": {}}


def get_training_dict(df, w):
    w_dict = {}
    w_reduced_dict = {}

    # Slide the window of size w
    for index, row in df.iterrows():
        # avoiding index error
        if index == len(df) - 1:
            continue
        if index < (w - 1):
            continue
        else:
            s = [None] * w
            for i in range(w):
                s[i] = df.loc[(index - (w - 1)) + i]["True Label"]
            # create key in w_dict if doesnt exist
            if w_dict.get(str(s)) == None:
                w_dict[str(s)] = []
            w_dict[str(s)].append(df.loc[index + 1]["True Label"])
    for key in w_dict.keys():
        num_pos = 0
        num_neg = 0
        items = ast.literal_eval(key)
        new_key = ""
        for i in items:
            new_key = new_key + i
        for value in w_dict[key]:
            if value == "+":
                num_pos = num_pos + 1
            else:
                num_neg = num_neg + 1
        if num_pos > num_neg:
            w_reduced_dict[new_key] = "+"
        elif num_neg > num_pos:
            w_reduced_dict[new_key] = "-"
        else:
            w_reduced_dict[new_key] = "+"

    return w_reduced_dict


# Function for predicting the next label
def predict_next_label(w, testing_df, training_df, ticker):

    training_w = get_training_dict(training_df, w)

    predictions = []
    d = 1
    index = testing_df.index[0]
    num_pos_correct = 0
    num_neg_correct = 0
    num_pos_incorrect = 0
    num_neg_incorrect = 0

    while index < len(testing_df) + testing_df.index[0]:
        # preventing key error from viewing indexs outside dataset (in beginning 3)
        if d < w:
            d = d + 1
            index = index + 1
        else:
            # generating sequence of last w labels, including current day d
            s = [None] * (w)
            for i in range(w):
                s[i] = testing_df.loc[index - i]["True Label"]
            index = index + 1
            key = ""
            for i in s:
                key = key + i
            prediction = training_w.get(key, "+")  # default to + if we cannot find key
            predictions.append(prediction)
            # Preventing key error from trying to predict the next label outside the dataset
            if index + 1 < len(testing_df.index) + testing_df.index[0]:
                if prediction == testing_df.loc[index + 1]["True Label"]:
                    if prediction == "+":
                        num_pos_correct = num_pos_correct + 1
                    else:
                        num_neg_correct = num_neg_correct + 1
                else:
                    if prediction == "+":
                        num_pos_incorrect = num_pos_incorrect + 1
                    else:
                        num_neg_incorrect = num_neg_incorrect + 1
    accuracy_overall = round(
        (
            (num_pos_correct + num_neg_correct)
            / (
                num_pos_correct
                + num_neg_correct
                + num_pos_incorrect
                + num_neg_incorrect
            )
        )
        * 100,
        2,
    )
    accuracy_pos = round(
        (num_pos_correct / (num_pos_correct + num_pos_incorrect)) * 100, 2
    )
    accuracy_neg = round(
        (num_neg_correct / (num_neg_correct + num_neg_incorrect)) * 100, 2
    )
    print("Accuracy Overall = " + str(accuracy_overall) + "%")
    print("Accuracy '+' = " + str(accuracy_pos) + "%")
    print("Accuracy '-' = " + str(accuracy_neg) + "%")
    accuracy_dict[ticker]["w=" + str(w)] = accuracy_overall
    return predictions


# ===========================
# Question #2
# ===========================

# ===========================
# TESTING WITH SUN
# ===========================

print("For SUN with w=2")
sun_w_2_predictions = predict_next_label(
    w=2, testing_df=testing_set_sun, training_df=training_set_sun, ticker="SUN"
)
print("------------------")

print("For SUN with w=3")
sun_w_3_predictions = predict_next_label(
    w=3, testing_df=testing_set_sun, training_df=training_set_sun, ticker="SUN"
)
print("------------------")

print("For SUN with w=4")
sun_w_4_predictions = predict_next_label(
    w=4, testing_df=testing_set_sun, training_df=training_set_sun, ticker="SUN"
)
print("------------------")

# ===========================
# TESTING WITH SPY
# ===========================

print("For SPY with w=2")
spy_w_2_predictions = predict_next_label(
    w=2, testing_df=testing_set_spy, training_df=training_set_spy, ticker="SPY"
)
print("------------------")

print("For SPY with w=3")
spy_w_3_predictions = predict_next_label(
    w=3, testing_df=testing_set_spy, training_df=training_set_spy, ticker="SPY"
)
print("------------------")

print("For SPY with w=4")
spy_w_4_predictions = predict_next_label(
    w=4, testing_df=testing_set_spy, training_df=training_set_spy, ticker="SPY"
)
print("------------------")

# ===========================
# Question #3
# ===========================

# Method for computing the most common label for w=2,3,4
def calculate_ensemble(row):
    return row.value_counts().idxmax()


def ensemble_stats(ensemble_df, testing_df, ticker):

    starting_day = 3

    num_pos_correct = 0
    num_neg_correct = 0
    num_pos_incorrect = 0
    num_neg_incorrect = 0

    for index, row in ensemble_df.iterrows():
        symbol = row["ensemble"]
        if symbol == "+":
            if symbol == testing_df.iloc[starting_day + index]["True Label"]:
                num_pos_correct = num_pos_correct + 1
            else:
                num_pos_incorrect = num_pos_incorrect + 1
        else:
            if symbol == testing_df.iloc[starting_day + index]["True Label"]:
                num_neg_correct = num_neg_correct + 1
            else:
                num_neg_incorrect = num_neg_incorrect + 1
    total = num_pos_correct + num_neg_correct + num_pos_incorrect + num_neg_incorrect
    accuracy = round(((num_pos_correct + num_neg_correct) / total) * 100, 2)
    accuracy_dict[ticker]["ensemble"] = accuracy
    print(
        "Correct    (pos:neg) -- " + str(num_pos_correct) + ":" + str(num_neg_correct)
    )
    print(
        "Incorrect  (pos:neg) -- "
        + str(num_pos_incorrect)
        + ":"
        + str(num_neg_incorrect)
    )
    print(
        "Accuracy (pos:neg) -- "
        + str(round((num_pos_correct / (num_pos_correct + num_pos_incorrect)) * 100, 2))
        + "%"
        + "  :  "
        + str(round((num_neg_correct / (num_neg_correct + num_neg_incorrect)) * 100, 2))
        + "%"
    )
    print("Accuracy (overall) -- " + str(accuracy) + "%")


# Creating a new DF to hold values for ensemble learning
# I have to subsplice the lists so they will have the same lengths
ensemble_df_sun = pd.DataFrame(
    {
        "w=2": sun_w_2_predictions[2:],
        "w=3": sun_w_3_predictions[1:],
        "w=4": sun_w_4_predictions,
    }
)

ensemble_df_sun["ensemble"] = ensemble_df_sun.apply(
    lambda row: calculate_ensemble(row), axis=1
)

ensemble_df_spy = pd.DataFrame(
    {
        "w=2": spy_w_2_predictions[2:],
        "w=3": spy_w_3_predictions[1:],
        "w=4": spy_w_4_predictions,
    }
)
ensemble_df_spy["ensemble"] = ensemble_df_spy.apply(
    lambda row: calculate_ensemble(row), axis=1
)

print("== Ensemble for SUN ==")
ensemble_stats(ensemble_df_sun, testing_set_sun, "SUN")
print("== Ensemble for SPY ==")
ensemble_stats(ensemble_df_spy, testing_set_spy, "SPY")

# ===========================
# Question #4
# ===========================


def calculate_stats(ensemble_df, testing_df, ticker):

    rows = ["w=2", "w=3", "w=4", "ensemble"]

    starting_day = 3

    tp = []
    fp = []
    tn = []
    fn = []
    tpr = []
    tnr = []

    num_pos_correct = 0
    num_neg_correct = 0
    num_pos_incorrect = 0
    num_neg_incorrect = 0

    for item in rows:
        for index, row in ensemble_df.iterrows():
            symbol = row[item]
            if symbol == "+":
                if symbol == testing_df.iloc[starting_day + index]["True Label"]:
                    num_pos_correct = num_pos_correct + 1
                else:
                    num_pos_incorrect = num_pos_incorrect + 1
            else:
                if symbol == testing_df.iloc[starting_day + index]["True Label"]:
                    num_neg_correct = num_neg_correct + 1
                else:
                    num_neg_incorrect = num_neg_incorrect + 1
        tp.append(num_pos_correct)
        fp.append(num_pos_incorrect)
        tn.append(num_neg_correct)
        fn.append(num_neg_incorrect)
        tpr.append(num_pos_correct / (num_neg_correct + num_pos_incorrect))
        tnr.append(num_neg_correct / (num_neg_correct + num_pos_incorrect))
        num_pos_correct = 0
        num_neg_correct = 0
        num_pos_incorrect = 0
        num_neg_incorrect = 0
    df = pd.DataFrame(
        {
            "TP": tp,
            "FP": fp,
            "TN": tn,
            "FN": fn,
            "accuracy": [
                accuracy_dict[ticker]["w=2"],
                accuracy_dict[ticker]["w=3"],
                accuracy_dict[ticker]["w=4"],
                accuracy_dict[ticker]["ensemble"],
            ],
            "TPR": tpr,
            "TNR": tnr,
        }
    )
    df.index = rows
    print(df)


# Final tables
print("========== SUN ==========")
calculate_stats(ensemble_df_sun, testing_set_sun, "SUN")
print("========== SPY ==========")
calculate_stats(ensemble_df_spy, testing_set_spy, "SPY")
