import numpy as np 
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import multiprocessing
import csv
import sys
import logging
from tabulate import tabulate
import matplotlib.pyplot as plt
from joblib import Parallel, delayed

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


def SVMtrain(x, y, val_x, val_y, epochs, alpha, C):
    # Initialize parameters
    w = np.zeros((x.shape[1], 3))
    bias = np.zeros(3)
    
    # Initialize lists for tracking training and validation metrics
    training_loss = [[] for _ in range(3)]
    validation_loss = [[] for _ in range(3)]
    validation_accuracy = [[] for _ in range(3)]
    training_accuracy = [[] for _ in range(3)]
    
    def train_svm(c):
        y_c = np.where(y == c+1, 1, -1)
        yv_c = np.where(val_y == c+1, 1, -1)
        for _ in range(epochs):
            for i, X in enumerate(x):
                if y_c[i] * (np.dot(X, w[:,c]) - bias[c]) >= 1:
                    w[:,c] -= alpha * (C * np.dot(w[:,c], X))
                else:
                    w[:,c] -= alpha * (C * np.dot(w[:,c], X) - np.dot(X, y_c[i]))
                    bias[c] -= alpha * y_c[i]
            
            # Calculate training loss and accuracy
            train_pred = np.sign(np.dot(x, w[:,c]) - bias[c])
            train_loss = np.mean(np.where(train_pred == y_c, 0, 1))
            train_acc = np.mean(train_pred == y_c)
            training_loss[c].append(train_loss)
            training_accuracy[c].append(train_acc)
            
            # Calculate validation loss and accuracy
            val_pred = np.sign(np.dot(val_x, w[:,c]) - bias[c])
            val_loss = np.mean(np.where(val_pred == yv_c, 0, 1))
            val_acc = np.mean(val_pred == yv_c)
            validation_loss[c].append(val_loss)
            validation_accuracy[c].append(val_acc)

        return w[:,c], bias[c], training_accuracy[c], validation_accuracy[c], training_loss[c], validation_loss[c]

    results = Parallel(n_jobs=3)(delayed(train_svm)(c) for c in range(3))

    for c in range(3):
        w[:,c], bias[c], training_accuracy[c], validation_accuracy[c], training_loss[c], validation_loss[c] = results[c]

                
    return w, bias, training_accuracy, validation_accuracy, training_loss, validation_loss

def SVMpredict(x, w, bias):
    predictions = []
    for X in x:
        scores = np.dot(X, w) - bias
        prediction = np.argmax(scores) + 1
        predictions.append(prediction)
    y_pred = np.array(predictions)
    return y_pred



def bagging_with_cross_validation(x, y, model, epochs, alpha, p1):
    # Create a K-fold cross-validator object
    n_splits=10

    kf = KFold(n_splits)
    kf_accs = np.empty(n_splits)
    kf_p = []
    kf_r = []
    kf_f = []
    kf_a = []
    training_loss = [[] for _ in range(3)]
    validation_loss = [[] for _ in range(3)]
    validation_accuracy = [[] for _ in range(3)]
    training_accuracy = [[] for _ in range(3)]

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
            w, bias, train_acc, val_acc, train_loss, val_loss= SVMtrain(x_sampled,  y_sampled,  xv, yv, epochs, alpha, p1)
            y_pred = SVMpredict(xv, w, bias)

            cont+=1
            logging.info("Trainee = {}".format(cont))

            training_accuracy[0].append(train_acc[0])
            training_accuracy[1].append(train_acc[1])
            training_accuracy[2].append(train_acc[2])
            validation_accuracy[0].append(val_acc[0])
            validation_accuracy[1].append(val_acc[1])
            validation_accuracy[2].append(val_acc[2])          
            training_loss[0].append(train_loss[0]) 
            training_loss[1].append(train_loss[1]) 
            training_loss[2].append(train_loss[2]) 
            validation_loss[0].append(val_loss[0])
            validation_loss[1].append(val_loss[1])
            validation_loss[2].append(val_loss[2])

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

    return mean_acc, p, r, f, a,  training_accuracy, validation_accuracy, training_loss, validation_loss


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)



C = 2
alpha = 0.01
epochs = 200

mean_acc, p, r, f, a, kf_train_accs, kf_val_acc, kf_train_loss, kf_val_loss = bagging_with_cross_validation(x_train, y_train2, "SVM", epochs, alpha, C)
result = [alpha, epochs, C, mean_acc, p, r, f, a]
logging.info(
        "SVM MODEL\nLearning rate = {}\nEpochs = {}\n C = {}\nMean accuracy: {}\n"
        "Mean precision: {}\nMean recall: {}\nMean f1_score: {}\nMean auc: {}\n"
        .format(alpha, epochs, C, mean_acc, p, r, f, a)
)

training_accuracy_c1 = kf_train_accs[0]
training_accuracy_c2 = kf_train_accs[1]
training_accuracy_c3 = kf_train_accs[2]

validation_accuracy_c1 = kf_val_acc[0]
validation_accuracy_c2 = kf_val_acc[1]
validation_accuracy_c3 = kf_val_acc[2]

training_loss_c1 = kf_train_loss[0]
training_loss_c2 = kf_train_loss[1]
training_loss_c3 = kf_train_loss[2]

validation_loss_c1 = kf_val_loss[0]
validation_loss_c2 = kf_val_loss[1]
validation_loss_c3 = kf_val_loss[2]

fig = plt.figure(figsize=(15, 10))

# Subplot for Class 1
plt.subplot(2, 3, 1)
for train_loss in training_loss_c1:
    plt.plot(train_loss, color= 'blue')
for val_loss in validation_loss_c1:
    plt.plot(val_loss, color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Class 1')


plt.subplot(2, 3, 4)
for train_acc in training_accuracy_c1:
    plt.plot(train_acc, color= 'blue')
for val_acc in validation_accuracy_c1:
    plt.plot(val_acc, color= 'red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Class 1')


# Subplot for Class 2
plt.subplot(2, 3, 2)
for train_loss in training_loss_c2:
    plt.plot(train_loss, color= 'blue')
for val_loss in validation_loss_c2:
    plt.plot(val_loss, color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Class 2')


plt.subplot(2, 3, 5)
for train_acc in training_accuracy_c2:
    plt.plot(train_acc, color= 'blue')
for val_acc in validation_accuracy_c2:
    plt.plot(val_acc, color= 'red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Class 2')


# Subplot for Class 3
plt.subplot(2, 3, 3)
for train_loss in training_loss_c3:
    plt.plot(train_loss, color= 'blue')
for val_loss in validation_loss_c3:
    plt.plot(val_loss, color='red')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Class 3')


plt.subplot(2, 3, 6)
for train_acc in training_accuracy_c3:
    plt.plot(train_acc, color= 'blue')
for val_acc in validation_accuracy_c3:
    plt.plot(val_acc, color= 'red')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Class 3')

fig.savefig('figure_{}.png'.format(epochs))
plt.close(fig)



