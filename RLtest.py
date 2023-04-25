import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import multiprocessing
import csv
import logging
from tabulate import tabulate

data = []
with open('DataSet Cardiotocographic data - Training.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        data.append(row)

df_train = pd.DataFrame(data)
df_train = df_train.astype(float)

data2 = []
with open('DataSet Cardiotocographic data - Test .csv', newline='') as csvfile:
    reader2 = csv.DictReader(csvfile)
    for row in reader2:
        data2.append(row)

df_test = pd.DataFrame(data2)
df_test = df_test.astype(float)

df = pd.concat([df_train, df_test], axis=0, ignore_index=True)

scaler = MinMaxScaler()
scaler.fit(df)
df_norm = pd.DataFrame(scaler.transform(df), columns=df.columns)
df = df_norm

train_data = df.iloc[:1998, :]  
test_data = df.iloc[1998:, :]  

x_train = train_data.iloc[:, :21]
x_train = x_train.to_numpy()
y_train = np.array(train_data['CLASE'])

x_test = test_data.iloc[:, :21]
x_test = x_test.to_numpy()

y_train2 = (y_train + 0.5)*2
y_train3 = y_train*2 

def listMean(bootstrap_list):
  # Transpose the list using zip
  transposed_lst = zip(*bootstrap_list)
  # Calculate the mean of each index
  mean_lst = [sum(col) / len(col) for col in transposed_lst]
  return mean_lst

def calculate(yv, y_pred):
  targets = np.unique(yv, return_counts=False)

  precision = []
  recall = []
  f1 = []
  roc_auc = []
  
  for t in targets:
      precision.append(precision_score(yv == t, y_pred == t, zero_division=1))
      recall.append(recall_score(yv == t, y_pred == t, zero_division=1))
      f1.append(f1_score(yv == t, y_pred == t, zero_division=1))
      roc_auc.append(roc_auc_score((yv == t), y_pred))
  
  return precision, recall, f1, roc_auc

def softmax(x, w):
    dot_products = np.dot(x, w.T)
    exps = np.exp(dot_products - np.max(dot_products, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    return probs

def RLtraining(x, y, epochs, alpha, lam=0.01, reg=0):
    # Add bias term to input data
    x = np.insert(x, 0, 1, axis=1)

    # Initialize weight matrix
    w = np.zeros((3, x.shape[1])) + 1e-12

    # Convert labels to one-hot encoding
    y_one_hot = np.zeros((y.shape[0], 3))
    y_one_hot[np.arange(y.shape[0]), y] = 1

    # Train the model for the specified number of epochs
    for i in range(epochs):
        # Compute predictions and error
        y_pred = softmax(x, w)
        error = y_one_hot - y_pred

        # Compute gradient with optional regularization
        if (reg == 1):
            dw = alpha * (np.dot(error.T, x) - lam * np.sign(w))
        elif (reg == 2):
            dw = alpha * (np.dot(error.T, x) - lam * np.sum(w**2))
        else:
            dw = alpha * np.dot(error.T, x)

        # Update weights
        w += dw

    # Return learned weight matrix
    return w

def RLprediction(x, w):
    # Add bias term to input data
    x = np.insert(x, 0, 1, axis=1)

    # Compute predictions and return argmax
    y_pred = softmax(x, w)
    return np.argmax(y_pred, axis=1)

def predict(model, xt, yt, xv, epochs, alpha, p1, p2):
  if model=="RL":
      w = RLtraining(xt,  yt,  epochs, alpha, p1, p2)
      y_pred = RLprediction(xv, w)
      return y_pred
  raise ValueError("Modelo inválido")


def bagging_with_cross_validation(x, y, model, epochs, alpha, p1, p2):
  # Create a K-fold cross-validator object
  n_splits=10

  kf = KFold(n_splits)
  kf_accs = np.empty(n_splits)
  kf_p = []
  kf_r = []
  kf_f = []
  kf_a = []

  j = 0
  cont = 0
  for train_indices, val_indices in kf.split(x):
      #Split the dataset into training and testing sets for this fold
      xt = x[train_indices]
      xv = x[val_indices]
      yt = y[train_indices]
      yv = y[val_indices]

      #iniciar bootstrap por cada k-fold
      # Set the number of bootstrap samples
      num_samples = 5

      # Create an empty array to store the bootstrap accuracies
      
      bootstrap_accs = np.empty(num_samples)
      bootstrap_p = []
      bootstrap_r = []
      bootstrap_f = []
      bootstrap_a = []

      for i in range(num_samples):
        # Generate a random sample with replacement
        np.random.seed(21)
        indices = np.random.choice(len(xt), size=len(xt), replace=True)
        x_sampled = xt[indices]
        y_sampled = yt[indices]

        #Training
        y_pred = predict(model, x_sampled, y_sampled, xv, epochs, alpha, p1, p2)

        #cont+=1
        #logging.info("Trainee = {}".format(cont))

        matching_elements = np.sum(y_pred ==  yv)
        accuracy = matching_elements/len(y_pred)

        precision, recall, f1, roc_auc = calculate(yv, y_pred)

        bootstrap_accs[i] = accuracy

        bootstrap_p.append(precision)
        bootstrap_r.append(recall)
        bootstrap_f.append(f1)
        bootstrap_a.append(roc_auc)

      kf_accs[j] = np.mean(bootstrap_accs)
      kf_p.append(listMean(bootstrap_p))
      kf_r.append(listMean(bootstrap_r))
      kf_f.append(listMean(bootstrap_f))
      kf_a.append(listMean(bootstrap_a))

      j+=1

  mean_acc = np.mean(kf_accs)
  mean_acc = round(mean_acc, 4)

  p = listMean(kf_p)
  r = listMean(kf_r)
  f = listMean(kf_f)
  a = listMean(kf_a)

  p = np.round(p,4)
  r = np.round(r,4)
  f = np.round(f,4)
  a = np.round(a,4)

  return mean_acc, p, r, f, a


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def my_function(args):
  proc = multiprocessing.current_process()
  logging.info(f"Running on processor {proc.name} (ID: {proc.pid})")

  reg = 1
  lam = 1
  alpha = 0.001
  epochs = 4823
 
  mean_acc, p, r, f, a = bagging_with_cross_validation(x_train, y_train3.astype(int), "RL", epochs, alpha, lam, reg)
  result = [alpha, epochs, reg, lam, mean_acc, p, r, f, a]
  logging.info(
        "LOGICAL REGRESION MODEL\nLearning rate = {}\nEpochs = {}\nRegularization = {}\nLambda = {}\nMean accuracy: {}\n"
        "Mean precision: {}\nMean recall: {}\nMean f1_score: {}\nMean auc: {}\n"
        .format(alpha, epochs, reg, lam, mean_acc, p, r, f, a)
  )
  return result

if __name__ == '__main__':
    list_of_args = [(None)]
    with multiprocessing.Pool(processes=5) as pool:
        results = pool.map(my_function, list_of_args)
    
    pool.join()
    headers = ["Alpha", "Epochs", "Reg", "Lambda", "Mean Accuracy", "Mean Precision", "Mean Recall", "Mean F1", "Mean AUC"]
    table = tabulate(results, headers=headers)

    print(table)
