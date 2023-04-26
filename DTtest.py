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

class SplitNode:
    def __init__(self, feature, threshold, left, right):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

class DecisionTree:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        
    def fit(self, X, y):
        self.n_classes = len(np.unique(y))
        self.tree = self._grow_tree(X, y)
        
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = [np.sum(y == i) for i in range(self.n_classes)]
        most_common_label = np.argmax(n_labels)
        
        # Stopping criteria
        if depth == self.max_depth or n_samples < 2 or len(np.unique(y)) == 1:
            return most_common_label
        
        # Splitting criteria
        feature_idxs = np.random.choice(n_features, int(np.sqrt(n_features)), replace=False)
        best_feature, best_threshold = None, None
        best_gini = 1.0
        
        for idx in feature_idxs:
            thresholds = np.unique(X[:, idx])
            
            if len(thresholds) == 1:
                continue
                
            for threshold in thresholds:
                left_idx = X[:, idx] < threshold
                n_left = np.sum(left_idx)
                
                if n_left == 0:
                    continue
                    
                n_right = n_samples - n_left
                gini_left = self._gini(y[left_idx])
                gini_right = self._gini(y[~left_idx])
                gini = (n_left/n_samples) * gini_left + (n_right/n_samples) * gini_right
                
                if gini < best_gini:
                    best_gini = gini
                    best_feature = idx
                    best_threshold = threshold
        
        # Check if best_threshold is still None
        if best_threshold is None:
            return most_common_label
        
        # Recurse on children
        left_idx = X[:, best_feature] < best_threshold
        right_idx = X[:, best_feature] >= best_threshold
        left_tree = self._grow_tree(X[left_idx], y[left_idx], depth+1)
        right_tree = self._grow_tree(X[right_idx], y[right_idx], depth+1)
        
        return SplitNode(best_feature, best_threshold, left_tree, right_tree)

    
    def _gini(self, y):
        n_samples = len(y)
        _, counts = np.unique(y, return_counts=True)
        impurity = 1.0 - np.sum((counts / n_samples) ** 2)
        return impurity
    
    def predict(self, X):
        # Traverse the decision tree for each data point
        y_pred = np.array([self._traverse_tree(x, self.tree) for x in X])
        
        return y_pred

    
    def _traverse_tree(self, x, node):
        if (np.issubdtype(type(node), np.integer)):
          return node
        
        # Check if node is a leaf node
        if node.left is None and node.right is None:
            return node.label
        
        if x[node.feature] < node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


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
        dt = DecisionTree(max_depth=p1)
        dt.fit(x_sampled, y_sampled)
        y_pred = dt.predict(xv)

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

  Depth = args

  mean_acc, p, r, f, a = bagging_with_cross_validation(x_train, y_train3, "DT", None, None, Depth)
  result = [Depth, mean_acc, p, r, f, a]
  logging.info(
        "DT MODEL\nDepth = {}\nMean accuracy: {}\n"
        "Mean precision: {}\nMean recall: {}\nMean f1_score: {}\nMean auc: {}\n"
        .format(Depth, mean_acc, p, r, f, a)
  )
  return result

if __name__ == '__main__':
    list_of_args = [8, 14, 16, 20, 40, 60, 80, 100]
    with multiprocessing.Pool(processes=8) as pool:
        results = pool.map(my_function, list_of_args)
    
    pool.join()
    headers = ["Depth", "Mean Accuracy", "Mean Precision", "Mean Recall", "Mean F1", "Mean AUC"]
    table = tabulate(results, headers=headers)

    print(table)
