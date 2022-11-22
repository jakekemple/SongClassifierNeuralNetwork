"""
Jake Kemple (2022) UW-Bothell
CSS581 Machine Learning Project
Song Classifier Neural Network
"""

import os
import io
import sys
import numpy as np
import tables
import math
from statistics import mean
import pandas as pd
from scipy import stats
import h5py

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, auc
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics._plot.roc_curve import RocCurveDisplay

from sklearn.neural_network import MLPClassifier

from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

import warnings
warnings.filterwarnings('ignore')

from sklearn.impute import SimpleImputer

import csv
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


# Build Cross-Validated Classification Model
def cv_train(x, y):

  model = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 5), random_state=1)

  tot_prec, tot_recall, tot_f1score, tot_auc = [], [], [], []

  # Initialize ROC Plot Vars
  tprs = []
  aucs = []
  mean_fpr = np.linspace(0, 1, 100)
  fig, ax = plt.subplots()

  # Split data into folds
  kfolds = KFold(n_splits=10, shuffle=True, random_state=7)
  kfolds.get_n_splits(x, y)

  # Run training and testing on each fold
  for i, (train_index, test_index) in enumerate(kfolds.split(x,y)):
    # Normalize the data
    scaler = StandardScaler().fit(x.iloc[train_index])
    x_train = scaler.transform(x.iloc[train_index])
    x_test = scaler.transform(x.iloc[test_index])

    # Train the model
    model.fit(x_train, y[train_index])

    # Save ROC Curve Data for this fold
    viz = RocCurveDisplay.from_estimator(
        model,
        x_test,
        y[test_index],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)

    # Make predictions on test data
    y_pred = model.predict(x_test)

    tot_prec.append(precision_score(y[test_index], y_pred, average='binary', pos_label=1, zero_division=0))
    tot_recall.append(recall_score(y[test_index], y_pred, average='binary', pos_label=1, zero_division=0))
    tot_f1score.append(f1_score(y[test_index], y_pred, average='binary', pos_label=1, zero_division=0))
    try:
      tot_auc.append(roc_auc_score(y[test_index], model.predict_proba(x_test)[:, 1]))
    except ValueError:
      pass
    

  totals = {
      "prec": tot_prec,
      "rec": tot_recall,
      "f1score": tot_f1score,
      "auc": tot_auc,
  }

  roc = {
      "tprs": tprs,
      "aucs": aucs,
      "ax": ax,
      "mean_fpr": mean_fpr
  }

  return totals, roc

# Compute Metrics: Precision, Recall, F1 Score, AUC, ROC Curve
def compute_metrics(totals, roc):

  avg_prec = round(mean(totals['prec']), 2)
  avg_recall = round(mean(totals['rec']), 2)
  avg_f1score = round(mean(totals['f1score']), 2)
  avg_auc = round(mean(totals['auc']), 2)

  print("Average Precision: ", avg_prec)
  print("Average Recall: ", avg_recall)
  print("Average F1 Score: ", avg_f1score)
  print("Average AUC: ", avg_auc)

  ax = roc['ax']
  mean_fpr = roc['mean_fpr']

  # Setup for Plotting the ROC Curve
  ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
  mean_tpr = np.mean(roc['tprs'], axis=0)
  mean_tpr[-1] = 1.0
  mean_auc = auc(mean_fpr, mean_tpr)
  std_auc = np.std(roc['aucs'])
  ax.plot(
      mean_fpr,
      mean_tpr,
      color="b",
      label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
      lw=2,
      alpha=0.8,
  )
  std_tpr = np.std(roc['tprs'], axis=0)
  tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
  tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
  ax.fill_between(
      mean_fpr,
      tprs_lower,
      tprs_upper,
      color="grey",
      alpha=0.2,
      label=r"$\pm$ 1 std. dev.",
  )
  ax.set(
      xlim=[-0.05, 1.05],
      ylim=[-0.05, 1.05],
      title="Receiver Operating Characteristic Curve",
  )
  ax.legend(loc="lower right")
  plt.show()


if __name__ == '__main__':

    song_dataset = pd.read_csv('All_Songs.csv')

    # Feature Engineering
    x = song_dataset.drop(['track_uri', 'track_name', 'artist_name', 'artist_pop', 'artist_genres', 'album', 'track_pop', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'like'], axis=1)
    x[:] = SimpleImputer(strategy='mean').fit_transform(x)

    y = np.ravel(song_dataset['like'])

    # Train NN & Compute Metrics
    print("Results:")
    log_totals, log_roc = cv_train(x, y)
    compute_metrics(log_totals, log_roc)
    print('\n')
