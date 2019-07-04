import pickle
import numpy
data = open("dataset.txt","r").readlines()[1:]
data_dir = "data/"
name = "npbracketing"

x_splits = {i:[] for i in range(10)}
y_splits = {i:[] for i in range(10)}
target_dict = {'R':0,'L':1}
for row in data:
  vals = row.strip().split()
  split = int(vals[0])
  target = target_dict[vals[-1]]
  x_splits[split].append( [word_pos.split('-')[0] for word_pos in vals[1].split('_') ] )
  y_splits[split].append(target)

for i in range(10):
  print(len(x_splits[i]), x_splits[i][0])
  print(len(y_splits[i]), y_splits[i][0])
  x_train = x_splits[i]
  y_train = y_splits[i]
  x_val = []
  y_val = []
  x_test = []
  y_test = []
  for j in range(10):
    if j!=i:
      x_test.extend(x_splits[j])
      y_test.extend(y_splits[j])
  pickle.dump(x_train,open(data_dir+name+"_"+"train_X"+str(i)+".pickle","wb"))
  pickle.dump(x_val,open(data_dir+name+"_"+"val_X" + str(i) + ".pickle","wb"))
  pickle.dump(x_test,open(data_dir+name+"_"+"test_X" + str(i) + ".pickle","wb"))
  pickle.dump(y_train,open(data_dir+name+"_"+"train_y"+ str(i) + ".pickle","wb"))
  pickle.dump(y_val,open(data_dir+name+"_"+"val_y"+ str(i) + ".pickle","wb"))
  pickle.dump(y_test,open(data_dir+name+"_"+"test_y" + str(i) + ".pickle","wb"))
