import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import multiprocessing
import csv
import logging
from tabulate import tabulate
import matplotlib.pyplot as plt

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

def RLtrainingwithValidation(x_train, y_train, x_val, y_val, epochs, alpha, lam, reg):
    # Add bias term to input data
    x_train = np.insert(x_train, 0, 1, axis=1)
    x_val = np.insert(x_val, 0, 1, axis=1)

    # Initialize weight matrix
    w = np.zeros((3, x_train.shape[1])) + 1e-12

    # Convert labels to one-hot encoding
    y_train_one_hot = np.zeros((y_train.shape[0], 3))
    y_train_one_hot[np.arange(y_train.shape[0]), y_train] = 1
    y_val_one_hot = np.zeros((y_val.shape[0], 3))
    y_val_one_hot[np.arange(y_val.shape[0]), y_val] = 1

    # Initialize lists for storing accuracy and loss values
    train_acc = []
    val_acc = []
    train_loss = []
    val_loss = []

    # Train the model for the specified number of epochs
    for i in range(epochs):
        # Compute predictions and error for training set
        y_train_pred = softmax(x_train, w)
        train_error = y_train_one_hot - y_train_pred

        # Compute predictions and error for validation set
        y_val_pred = softmax(x_val, w)
        val_error = y_val_one_hot - y_val_pred

        # Compute training and validation accuracy and loss
        train_acc.append(np.mean(np.argmax(y_train_pred, axis=1) == y_train))
        val_acc.append(np.mean(np.argmax(y_val_pred, axis=1) == y_val))
        train_loss.append(-np.mean(np.sum(y_train_one_hot * np.log(y_train_pred), axis=1)))
        val_loss.append(-np.mean(np.sum(y_val_one_hot * np.log(y_val_pred), axis=1)))

        # Compute gradient with optional regularization for training set
        if (reg == 1):
            dw = alpha * (np.dot(train_error.T, x_train) - lam * np.sign(w))
        elif (reg == 2):
            dw = alpha * (np.dot(train_error.T, x_train) - lam * np.sum(np.dot(w,w.T)))
        else:
            dw = alpha * np.dot(train_error.T, x_train)

        # Update weights
        w += dw

    # Return learned weight matrix and accuracy/loss values
    return w, train_acc, val_acc, train_loss, val_loss

def RLprediction(x, w):
    # Add bias term to input data
    x = np.insert(x, 0, 1, axis=1)

    # Compute predictions and return argmax
    y_pred = softmax(x, w)
    return np.argmax(y_pred, axis=1)


def bagging_with_cross_validation(x, y, model, epochs, alpha, p1, p2):
  # Create a K-fold cross-validator object
  n_splits=10

  kf = KFold(n_splits)
  kf_accs = np.empty(n_splits)
  kf_p = []
  kf_r = []
  kf_f = []
  kf_a = []
  kf_train_accs = []
  kf_val_acc = [] 
  kf_train_loss = [] 
  kf_val_loss = []

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
        w, train_acc, val_acc, train_loss, val_loss= RLtrainingwithValidation(x_sampled,  y_sampled,  xv, yv, epochs, alpha, p1, p2)
        y_pred = RLprediction(xv, w)

        cont+=1
        logging.info("Trainee = {}".format(cont))

        kf_train_accs.append(train_acc)
        kf_val_acc.append(val_acc) 
        kf_train_loss.append(train_loss) 
        kf_val_loss.append(val_loss)

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

  return mean_acc, p, r, f, a, kf_train_accs, kf_val_acc, kf_train_loss, kf_val_loss 


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

def my_function(args):
    proc = multiprocessing.current_process()
    logging.info(f"Running on processor {proc.name} (ID: {proc.pid})")

    lam = 1
    reg = 1
    alpha = 0.001
    epochs = args
    
    mean_acc, p, r, f, a, kf_train_accs, kf_val_acc, kf_train_loss, kf_val_loss = bagging_with_cross_validation(x_train, y_train3.astype(int), "RL", epochs, alpha, lam, reg)
    
    result = [alpha, epochs, reg, lam, mean_acc, p, r, f, a]
    logging.info(
            "LOGICAL REGRESION MODEL\nLearning rate = {}\nEpochs = {}\nRegularization = {}\nLambda = {}\nMean accuracy: {}\n"
            "Mean precision: {}\nMean recall: {}\nMean f1_score: {}\nMean auc: {}\n"
            .format(alpha, epochs, reg, lam, mean_acc, p, r, f, a)
    )

    fig = plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    
    plt.plot(kf_val_acc[0], label='Validation Accuracy', color='red')
    for val_acc in kf_val_acc[1:]:
        plt.plot(val_acc, color='red')

    plt.plot(kf_train_accs[0], label='Training Accuracy', color='blue')
    for train_acc in kf_train_accs[1:]:
        plt.plot(train_acc, color='blue')

    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    
    plt.plot(kf_val_loss[0], label='Validation Loss', color='red')
    for val_loss in kf_val_loss[1:]:
        plt.plot(val_loss, color='red')

    plt.plot(kf_train_loss[0], label='Training Loss', color='blue')
    for train_loss in kf_train_loss[1:]:
        plt.plot(train_loss, color='blue')

    my_list = kf_val_acc
    index = 250  # index to start checking from

    # initialize an empty list to store the averages
    averages = []

    # loop over the indices of the sub-lists
    for i in range(len(my_list[0])):
        if i >= index:
            total = 0
            for j in range(len(my_list)):
                total += my_list[j][i]
            averages.append(total / len(my_list))

    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    fig.savefig('figure_{}.png'.format(proc.name))
    plt.close(fig)

    lowest_avg = max(averages)
    lowest_avg_index = averages.index(lowest_avg) + index
    print("Best average", len(averages),"=", lowest_avg) 
    print("Best epoch", len(averages),"=", lowest_avg_index) 

    return result

    

if __name__ == '__main__':
    list_of_args = [(1000),(2000),(10000),(40000),(50000),(100000)]
    with multiprocessing.Pool(processes=6) as pool:
        results = pool.map(my_function, list_of_args)
    
    pool.join()
    headers = ["Alpha", "Epochs", "Reg", "Lambda", "Mean Accuracy", "Mean Precision", "Mean Recall", "Mean F1", "Mean AUC"]
    table = tabulate(results, headers=headers)
    print(table)


    