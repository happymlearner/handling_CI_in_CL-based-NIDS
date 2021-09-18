from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
import tensorflow as tf
import keras
import keras.backend as K
import random
from collections import Counter
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix, roc_curve, roc_auc_score
from keras.preprocessing import image
#import matplotlib.pyplot as plt
#from PIL import Image as im
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()

names = ['BENIGN', 'DoS Hulk', 'PortScan', 'DDoS', 'FTP-Patator', 'DoS slowloris', 'DoS Slowhttptest', 'SSH-Patator', 'Bot', 'Web Attack � Brute Force', 'DoS GoldenEye', 'Web Attack � XSS', 'Infiltration', 'Web Attack � Sql Injection', 'Heartbleed']

arrs = dict()
for name in names:
  name1 = name+'.npy'
  data = np.load("./CICIDS2017_Class-wise-normalized_datasets/"+name1,allow_pickle=True) # loads the individual data for each class
  arrs[name] = data

for i in names:
  print(arrs[i].shape)
  print(arrs[i][5][70])# printing random label

X_im = dict()
y_im = dict()
for i,name in enumerate(names):
  arr = arrs[name]
  X_im[name] = arr[:,:-1] #copying the features
  print(X_im[name].shape)
  y_im[name] = arr[:,-1]  #copying the labels
  print(y_im[name].shape)

X_image = dict()
for i,name in enumerate(names):
  arr_X = X_im[name]
  data_image = []
  for j in range(arr_X.shape[0]):
    data_image.append(arr_X[j].reshape(7,10,1))#data_image contains the image representation of whole samples for each class
  list_to_array = np.array(data_image)
  X_image[name] = list_to_array #Copying the image representations of the class into the dictionary
  print(X_image[name].shape)

train_set_x = dict()
train_set_y = dict()
test_set_x = dict()
test_set_y = dict()
val_set_x = dict()
val_set_y = dict()

for i,name in enumerate(names):
  arr_x = X_image[name]
  arr_y = y_im[name]
  class_length = arr_x.shape[0] # no.of instances per class
  print(class_length)
  train_split = int(0.70*class_length)    # I used 0.70 because I want 70% of the dataset to be train data, change this according to preference
  val_split = -int(0.002*class_length)  # Here I only want 0.2% of the dataset (around 6000 points) in my validation set, you can change this as well
                          # Whatever percentage is left forms the test dataset
  train_set_x[name] = arr_x[:train_split,:,:,:]
  test_set_x[name] = arr_x[train_split:val_split,:,:,:]
  val_set_x[name] = arr_x[val_split:,:,:,:]
  print('training dataset size of class',name,train_set_x[name].shape) 
  train_set_y[name] = arr_y[:train_split]
  test_set_y[name] = arr_y[train_split:val_split]
  val_set_y[name] = arr_y[val_split:]

'''
#May be useful in future
name_to_id = {'BENIGN':0, 'DoS Hulk':1, 'PortScan':2, 'DDoS':3, 'FTP-Patator':4, 'DoS slowloris':5, 'DoS Slowhttptest':6, 'SSH-Patator':7, 'Bot':8, 'Web Attack � Brute Force':9, 'DoS GoldenEye':10, 'Web Attack � XSS':11, 'Infiltration':12, 'Web Attack � Sql Injection':13, 'Heartbleed':14}
id_to_name = {name_to_id[i]:i for i in names}
print(id_to_name)
'''

for i,name in enumerate(names):
  X_temp, y_temp = test_set_x[name],test_set_y[name]
  X_temp1, y_temp1 = val_set_x[name],val_set_y[name]
  if i==0:
    X_test, y_test = X_temp, y_temp #ensures initialisation of X_test and y_test
    X_val, y_val = X_temp1, y_temp1
  else:
    X_test = np.concatenate((X_test,X_temp), axis=0)
    y_test = np.concatenate((y_test,y_temp), axis=0)
    X_val = np.concatenate((X_val,X_temp1), axis=0)
    y_val = np.concatenate((y_val,y_temp1), axis=0)
#print(X_test.shape, y_test.shape)

y_test = y_test.ravel()#flattens 2d to 1d
print(Counter(y_test))

y_val = y_val.ravel()
print(Counter(y_val))

for i in range(y_test.shape[0]):
  # For 2-Class setting use the below 4 lines
  if y_test[i] == 'BENIGN':
    y_test[i] = 0
  else:
    y_test[i] = 1
    
y_test = y_test.astype(float)
print(Counter(y_test))


for i in range(y_val.shape[0]):
  # For 2-Class setting use the below code
  if y_val[i] == 'BENIGN':
    y_val[i] = 0
  else:
    y_val[i] = 1
    
y_val = y_val.astype(float)
print(Counter(y_val))

'''
#These are the concerned task orders
Benign in task1
task_order = [('BENIGN', 'DDoS', 'PortScan' ),('Bot', 'Infiltration', 'Web Attack � Brute Force'),('Web Attack � XSS','Web Attack � Sql Injection','FTP-Patator'),('SSH-Patator','DoS slowloris','DoS Slowhttptest'),('DoS Hulk', 'DoS GoldenEye', 'Heartbleed')]

Benign in task2
task_order = [('PortScan', 'Infiltration','FTP-Patator' ), ('Web Attack � Brute Force','SSH-Patator','BENIGN'  ), ('DoS Hulk','DDoS', 'Web Attack � Sql Injection' ), ('Bot','Heartbleed','DoS GoldenEye'), ('DoS slowloris','DoS Slowhttptest', 'Web Attack � XSS' )]

Benign in task3
task_order = [('Heartbleed','SSH-Patator', 'DoS Hulk' ), ('Infiltration', 'Bot', 'Web Attack � Brute Force' ), ('BENIGN','DoS Slowhttptest','FTP-Patator'), ('Web Attack � Sql Injection', 'Web Attack � XSS', 'DoS slowloris'), ('PortScan','DoS GoldenEye', 'DDoS')]

Benign in task4
task_order = [('DDoS', 'Web Attack � Sql Injection', 'Infiltration' ),('Bot', 'DoS Hulk', 'PortScan'),('DoS slowloris','Web Attack � XSS','DoS Slowhttptest'),('DoS GoldenEye','FTP-Patator','BENIGN'),('SSH-Patator', 'Heartbleed', 'Web Attack � Brute Force')]

Benign in task5
task_order = [('DoS slowloris','DoS GoldenEye','Heartbleed' ), ('Bot','Web Attack � Brute Force','Web Attack � XSS' ), ('SSH-Patator', 'Infiltration','PortScan'), ('DDoS','DoS Hulk','FTP-Patator'), ('Web Attack � Sql Injection','DoS Slowhttptest','BENIGN')]

'''
task_order = [('BENIGN', 'DDoS', 'PortScan' ),('Bot', 'Infiltration', 'Web Attack � Brute Force'),('Web Attack � XSS','Web Attack � Sql Injection','FTP-Patator'),('SSH-Patator','DoS slowloris','DoS Slowhttptest'),('DoS Hulk', 'DoS GoldenEye', 'Heartbleed')]

tasks=[]
for task in task_order:
  for i,class_ in enumerate(task):
    print("class_name",class_)
    X_class, y_classname = train_set_x[class_], train_set_y[class_]  #Getting train set for class_ from our class wise train dictionary
    class_size = X_class.shape[0]     #Getting number of instances in the class
    #Now we manually encode our training labels in one-hot-encoding format
    #Use the below 5 lines for 2-class setting
    if class_ == 'BENIGN':
      class_encoding = [0]
    else:
      class_encoding = [1]
      #y_classname = np.array(['ATTACK']*class_size)   #We are changing all the attack class labels to 'ATTACK'

    #y_class now has the one-hot-encoded labels
    y_class = [class_encoding]*class_size 
    y_class = np.array(y_class)

    if i==0:
      X_task, y_task, yname_task = X_class, y_class, y_classname
    else:
      X_task = np.concatenate((X_task, X_class), axis=0)
      y_task = np.concatenate((y_task, y_class), axis=0)
      yname_task = np.concatenate((yname_task, y_classname), axis=0)

  #Shuffling all the different class instances in a task together
  shuffler = np.random.permutation(len(y_task))
  X_task = X_task[shuffler]
  y_task = y_task[shuffler]
  y_task = y_task.ravel()
  yname_task = yname_task[shuffler]

  # Now we append them all into our tasks array
  tasks.append((X_task, y_task, yname_task))

print('Total no.of tasks', len(tasks))
for x,y,z in tasks:
  print(y.shape, z.shape, x.shape)
  print(x.shape[0], Counter(y), Counter(z))



img_width, img_height = 7, 10
input_shape = (img_width, img_height,1)
hidden_size = 100
model = Sequential()
model.add(Conv2D(14, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(hidden_size, activation='relu'))
model.add(Dense(1, activation='sigmoid'))


optimizer = keras.optimizers.Adamax(lr=0.000001)      #Optimizer to use while training
loss_fn = keras.losses.BinaryCrossentropy()           #Loss function
train_acc_metric = keras.metrics.BinaryAccuracy()     #Accuracy metric for training - this metric will be displayed every few epochs in training
val_acc_metric = keras.metrics.BinaryAccuracy()       #Accuracy metric for validation dataset
print(model.summary())

train_losses=[]     #list that maintains training losses over epochs
val_losses=[]       #list that maintains validation losses over epochs

global_count, local_count = Counter(), Counter()
b = 5000            
classes_so_far = set()
full = set()
nc = 0
nb = 1
memory_size = 12000

# We are initialising memory with memory_size no.of instances in task 0
initial_X, initial_y, initial_yname = tasks[0]
mem = min(memory_size, initial_yname.shape[0]) # if task 0 has less than memory_size samples, then load all the samples in task 0 into memory
memory_X, memory_y, memory_y_name = initial_X[:mem,:,:,:], initial_y[:mem], initial_yname[:mem]

# local_count stores "class_name : no. of class_name instances in memory"
local_count = Counter()
for class_ in memory_y_name:
  local_count[class_]+=1

loop = 0
a = 0.3
batch_size =  128    #Set batch size 
epochs = 30        #Set number of epochs
X_val = X_val.astype(float)
y_val = y_val.astype(float)
val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)
norm_list1 = []
norm_list2 = []
norm_list3 = []
source_path_model = "./CodebaseAIMLSYS/CBRS/Mem12000/SimpleCNN/iacb/Benign1CNN/"
source_path_results_file = "./CodebaseAIMLSYS/CBRS/Mem12000/SimpleCNN/iacb/results.txt"
print(a)

# We loop in steps of b (batchsize) to simulate batch-wise reception of stream
for X,y,y_classname in tasks:

  p = np.array(model.layers[3].weights[0])
  q = np.array(model.layers[4].weights[0])
  r = np.array(model.layers[0].weights[0])
  #Calculates norm for the layer weights
  norm_list3.append(np.linalg.norm(p))
  norm_list2.append(np.linalg.norm(q))
  norm_list1.append(np.linalg.norm(r))
  loop+=1
  print("task number:",loop)
  task_size = X.shape[0]

  for i in range(0, task_size, b):
    print("till here", i+b)

    Xt, yt, ynamet = X[i:i+b,:,:,:], y[i:i+b], y_classname[i:i+b]

    print(local_count)
    # global_count stores "class_name : no. of class_name instances in the stream so far"
    for j in range(len(ynamet)):
      global_count[ynamet[j]]+=1
      if ynamet[j] not in classes_so_far:
        classes_so_far.add(ynamet[j])
        nc += 1  

    #Do weighted replay

    weights = []
    for x in range(memory_y_name.shape[0]):
      weights.append(1/local_count[memory_y_name[x]]) # weights are added for each instance, whose value equal to the inverse no.of instances present in memory per class it belongs to

    total_weight = sum(weights)
    probas = [weight/total_weight for weight in weights] #translates class_weigths to their probabilities

    numbers = np.random.choice(range(0,memory_y_name.shape[0]), b, replace=False, p=probas)

    replay_Xt, replay_yt, replay_yname = np.zeros((b,Xt.shape[1],Xt.shape[2],Xt.shape[3])), np.zeros(b), []

    for ind,rand in enumerate(numbers):
      replay_Xt[ind], replay_yt[ind] = memory_X[rand], memory_y[rand] 
      replay_yname.append(memory_y_name[rand])
      

    replay_Xt, Xt = replay_Xt.astype('float32'), Xt.astype('float32')
    print(Counter(replay_yname))

    #_______________________________________________________________________________#
    #                       CUSTOM TRAINING
    #_______________________________________________________________________________#

    stream_dataset = tf.data.Dataset.from_tensor_slices((Xt, yt))
    stream_dataset = stream_dataset.shuffle(buffer_size=1024).batch(batch_size//2)# need to add batch_size//2

    replay_dataset = tf.data.Dataset.from_tensor_slices((replay_Xt, replay_yt))
    replay_dataset = replay_dataset.shuffle(buffer_size=1024).batch(batch_size//2)# need to add batch_size//2

    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))

        # Iterate over the batches of the dataset.
        train_loss = 0
        count = 0
        for step, (stream, replay) in enumerate(zip(stream_dataset,replay_dataset)):

            x_stream_train, y_stream_train = stream
            x_replay_train, y_replay_train = replay

            # Open a GradientTape to record the operations run
            # during the forward pass, which enables auto-differentiation.
            with tf.GradientTape() as tape:

                # Run the forward pass of the layer. The operations that the layer applies
                # to its inputs are going to be recorded on the GradientTape.
                stream_logits = model(x_stream_train, training=True)  # Logits for this minibatch
                replay_logits = model(x_replay_train, training=True)

                # Compute the loss value for this minibatch.
                stream_loss = loss_fn(y_stream_train, stream_logits)
                replay_loss = loss_fn(y_replay_train, replay_logits)
                loss_value = stream_loss*a + replay_loss*(1-a)

            # Use the gradient tape to automatically retrieve
            # the gradients of the trainable variables with respect to the loss.
            grads = tape.gradient(loss_value, model.trainable_weights)

            count+=1
            train_loss += loss_value
            # Run one step of gradient descent by updating
            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, model.trainable_weights))

            # Update training metric.
            train_acc_metric.update_state(y_stream_train, stream_logits)
            train_acc_metric.update_state(y_replay_train, replay_logits)

            # Log every 200 batches.
            if step % 200 == 0:
                print( "Training loss (for one batch) at step %d: %.4f"% (step, float(loss_value)))
            #     print("Seen so far: %s samples" % ((step + 1) * 64))

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print("Training acc over epoch: %.4f" % (float(train_acc),))

        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        train_losses.append(train_loss/count)
        # Run a validation loop at the end of each epoch.
        val_loss = 0
        count=0
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val, training=False)
            count+=1
            val_loss += loss_fn(y_batch_val, val_logits)
            val_loss_value = loss_fn(y_batch_val, val_logits)

            val_acc_metric.update_state(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        val_losses.append(val_loss/count)

    #_______________________________________________________________________________#
    #               MEMORY POPULATION USING CBRS
    #_______________________________________________________________________________#
    
  if (loop == 1 and memory_y_name.shape[0]<memory_size and i+b>memory_y_name.shape[0]) or (loop != 1):
    for j in range(len(ynamet)):
      #if memory noot filled
      if (memory_y_name.shape[0]<memory_size):
        print(memory_X.shape,Xt[j].shape,Xt[j,:].shape)
        Xtreshape = Xt[j].reshape(1,7,10,1)
        ytnew = np.empty(1, dtype=object)
        ytnew[0] = yt[j]
        ynametnew = np.empty(1, dtype=object)
        ynametnew[0] = ynamet[j]
        print(ytnew, yt[j])
        print(ynametnew, ynamet[j])
        memory_X = np.concatenate((memory_X,Xtreshape), axis=0)
        memory_y = np.concatenate((memory_y,ytnew), axis=0)
        memory_y_name = np.concatenate((memory_y_name,ynametnew), axis=0)
        local_count[ynamet[j]]+=1
        print("class being added to memory not filled", ynamet[j])
        print("================================")
        print("memory.size",memory_y_name.shape[0])
        print("================================")

      # If the current instance doesnt belong to a full class
      elif ynamet[j] not in full:

        #Find the "largest" classes in the current iteration of memory population
        largest = set()
        for class_ in local_count.keys():
          if local_count[class_] == max(local_count.values()):
            largest.add(class_)
            full.add(class_)
          
        # We find the indices of the instances in memory buffer which are from one of the "largest" classes
        # and store these instances in an array "largest_members"
        largest_members = []
        for k in range(memory_size):
          if memory_y_name[k] in largest:
            largest_members.append(k)

        # We randomly pick one of the "largest" class members (which we stored in largest_members)
        rand = random.randint(0,len(largest_members)-1)
        rand_largest = largest_members[rand]

        # Update local_count, which stores the counts of each class present in the memory buffer
        local_count[memory_y_name[rand_largest]]-=1
        local_count[ynamet[j]]+=1

        # Replace the "largest" class member we picked earlier with the current instance
        memory_X[rand_largest] = Xt[j]
        memory_y[rand_largest] = yt[j]
        memory_y_name[rand_largest] = ynamet[j]

      # If the current instance belongs to a full class
      else:

        threshold = local_count[ynamet[j]] / global_count[ynamet[j]]
        u = random.uniform(0, 1)

        if u <= threshold:
          #We find the indices of the instances in memory which are from same class as current stream instance
          same_class = []
          for k in range(me):            
            if memory_y_name[k] == ynamet[j]:
               same_class.append(k)

          # We randomly pick one of the same class members (which we stored in same_class)
          rand = random.randint(0,len(same_class)-1)
          rand_same_class = same_class[rand]

          # NOTE: We don't need to update local_count since we are replacing a member from the same class
          # as the current instance, so overall count for that class stays same

          # Replace the chosen member with current instance
          memory_X[rand_same_class] = Xt[j]
          memory_y[rand_same_class] = yt[j]
          memory_y_name[rand_same_class] = ynamet[j]

    
  model_name = 'cbrs_binary2_benign1_adamax_train_loss' + str(loop)
  np.save(source_path_model+ model_name,train_losses)
  train_losses = []

  model_name = 'cbrs_binary2_benign1_adamax_val_loss' + str(loop)
  np.save(source_path_model + model_name,val_losses)
  val_losses = []

  model_name = 'cbrs_binary2_benign1_adamax' + str(loop)
  model.save(source_path_model+model_name)

  model_name = 'cbrs_binary2_benign1_adamax_memoryX' + str(loop)
  np.save(source_path_model+model_name,memory_X)

  model_name = 'cbrs_binary2_benign1_adamax_memoryy' + str(loop)
  np.save(source_path_model+model_name,memory_y)

  model_name = 'cbrs_binary2_benign1_adamax_memoryyname' + str(loop)
  np.save(source_path_model+ model_name,memory_y_name)

p = np.array(model.layers[3].weights[0])
q = np.array(model.layers[4].weights[0])
r = np.array(model.layers[0].weights[0])
  
norm_list3.append(np.linalg.norm(p))
norm_list2.append(np.linalg.norm(q))
norm_list1.append(np.linalg.norm(r))
np.save(source_path_model+"norm_layer_3_task",norm_list3)
np.save(source_path_model+"norm_layer_2_task",norm_list2)
np.save(source_path_model+"norm_layer_1_task",norm_list1)


print(X_test.shape)
X_test = X_test.astype(float)
yhat = model.predict(X_test)
# round probabilities to class labels
yhat = yhat.ravel()
yhat = yhat.round()

print("="*40)
print("accuracy    ", accuracy_score(y_test, yhat))
print("f1 score    ", f1_score(y_test, yhat))
print("precision    ", precision_score(y_test, yhat))
print("recall    ", recall_score(y_test, yhat))
print("="*40)
# Getting categorical encoding format ---- only use this if you're doing multiclass classification
# yhat = np.argmax(yhat,axis=1)
target_names = ['BENIGN', 'ATTACK']

# Printing out the confusion matrix
print(confusion_matrix(y_test, yhat))

from sklearn.metrics import classification_report
import sys

print(classification_report(y_test, yhat, target_names = target_names, digits = 6))

original_stdout = sys.stdout
with open(source_path_results_file, 'a') as f:
    sys.stdout = f # Change the standard output to the file we created.
    print("\n")
    print("Benign1CNN :\n")
    print(confusion_matrix(y_test, yhat))
    print(classification_report(y_test, yhat, target_names = target_names, digits = 6))

    sys.stdout = original_stdout