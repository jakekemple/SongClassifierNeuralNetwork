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

from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show

import warnings
warnings.filterwarnings('ignore')

import csv
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

if __name__ == '__main__':

    song_dataset = pd.read_csv(io.BytesIO('Liked_Songs.csv'))

    # Feature Engineering
    x = song_dataset.drop(['track_uri', 'track_name', 'artist_name', 'artist_pop', 'artist_genres', 'album', 'track_pop', 'type', 'id', 'uri', 'track_href', 'analysis_url', 'like'], axis=1)
    y = np.ravel(song_dataset['like'])
    