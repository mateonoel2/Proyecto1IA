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

def KNNpredict(x, y, x_test, k):
        y_pred = []
        for i in range(len(x_test)):
            distances = []
            for j in range(len(x)):
                dist = np.sqrt(np.sum((x_test[i] - x[j])**2))
                distances.append((dist, y[j]))
            distances.sort()
            neighbors = distances[:k]
            labels = [neighbor[1] for neighbor in neighbors]
            label = max(set(labels), key=labels.count)
            y_pred.append(label)
        return np.array(y_pred)

def bagging_with_cross_validation(x, y, model, epochs, alpha, p1):
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
        y_pred = KNNpredict(x_sampled, y_sampled, xv, p1)

        cont+=1
        logging.info("Trainee = {}".format(cont))

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

logging.basicConfig(filename='output.log', level=logging.INFO)

def my_function(args):
  proc = multiprocessing.current_process()
  logging.info(f"Running on processor {proc.name} (ID: {proc.pid})")

  K = args

  mean_acc, p, r, f, a = bagging_with_cross_validation(x_train, y_train3, "DT", None, None, K)
  result = [K, mean_acc, p, r, f, a]
  logging.info(
        "KNN MODEL\nK = {}\nMean accuracy: {}\n"
        "Mean precision: {}\nMean recall: {}\nMean f1_score: {}\nMean auc: {}\n"
        .format(K, mean_acc, p, r, f, a)
  )
  return result

if __name__ == '__main__':
    list_of_args = [1, 2, 3, 4, 5, 6, 7, 8]
    with multiprocessing.Pool(processes=8) as pool:
        results = pool.map(my_function, list_of_args)
    
    pool.join()
    headers = ["K", "Mean Accuracy", "Mean Precision", "Mean Recall", "Mean F1", "Mean AUC"]
    table = tabulate(results, headers=headers)

    print(table)
