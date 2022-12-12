"""
Jake Kemple (2022) UW-Bothell
CSS581 Machine Learning Project
Song Classifier Neural Network
"""

import numpy as np
from statistics import mean
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, auc
from sklearn.metrics._plot.roc_curve import RocCurveDisplay
from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def get_abs_mean_weights(model, features):
    mean_coefs = [np.abs(i.mean()) for i in model.coefs_[0]]
    feature_importance = pd.DataFrame({ "Features": features })
    feature_importance["Coef"] = mean_coefs
    return feature_importance

def scale_weights(feature_importance):
    # Scaling the weights to be between 0 and 1
    feature_importance["FeatImp"] = (
      (feature_importance["Coef"] - feature_importance["Coef"].min())
        / (feature_importance["Coef"].max() - feature_importance["Coef"].min())
      ).round(4)
    return feature_importance

def compute_feature_importance(model, features):
    weights = get_abs_mean_weights(model, features)
    weights = scale_weights(weights)
    feature_importance = weights.drop("Coef", axis=1).reset_index()
    feature_importance_dict = dict(
      zip(feature_importance["Features"], feature_importance["FeatImp"])
    )
    return feature_importance_dict

# Train Classification Model & Calculate Metrics
def cv_train(x, y):

  # Initialize ROC Plot Vars
  tprs = []
  aucs = []
  mean_fpr = np.linspace(0, 1, 100)
  fig, ax = plt.subplots()

  # Split the data up between train/test
  x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

  # Normalize the data
  scaler = StandardScaler().fit(x)
  x_train = scaler.transform(x_train)
  x_test = scaler.transform(x_test)

  # Train the model
  model.fit(x_train, y_train)

  # Save ROC Curve Data
  viz = RocCurveDisplay.from_estimator(
      model,
      x_test,
      y_test,
      name="ROC",
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

  prec = precision_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
  rec = recall_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
  f1 = f1_score(y_test, y_pred, average='binary', pos_label=1, zero_division=0)
  try:
    auc = roc_auc_score(y_test, model.predict_proba(x_test)[:, 1])
  except ValueError:
    auc = None

  metrics = {
      "prec": prec,
      "rec": rec,
      "f1score": f1,
      "auc": auc,
  }

  roc = {
      "tprs": tprs,
      "aucs": aucs,
      "ax": ax,
      "mean_fpr": mean_fpr
  }

  return metrics, roc

# Display Metrics: Precision, Recall, F1 Score, AUC, ROC Curve
def display_metrics(metrics, roc):

  print("Precision: ", metrics["prec"])
  print("Recall: ", metrics["rec"])
  print("F1 Score: ", metrics["f1score"])
  print("AUC: ", metrics["auc"])

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
    # unlabeled_songs = pd.read_csv('Unknown_Songs.csv')

    # Feature Engineering
    x = song_dataset.drop(
      ['track_uri', 'track_name', 'artist_name', 
      'artist_pop', 'artist_genres', 'album', 'track_pop', 'type', 'id', 'uri', 
      'track_href', 'analysis_url', 'like'], 
      axis=1
    )
    x[:] = SimpleImputer(strategy='mean').fit_transform(x)

    y = np.ravel(song_dataset['like'])

    model = MLPClassifier(
      activation='relu',
      solver='adam', #sgd
      alpha=1e-5, 
      hidden_layer_sizes=(50,), 
      random_state=7
    )

    # Train NN
    metrics, roc = cv_train(x, y)
    print("Results:")

    # Get Feature Importances
    features = x.columns.tolist()
    feature_importance = compute_feature_importance(model, features)
    print(feature_importance)

    # Compute Metrics
    display_metrics(metrics, roc)
    print('\n')

    # Predict on new/unknown songs
    # predictions = model.predict(unlabeled_songs)
    # Output predictions somehow