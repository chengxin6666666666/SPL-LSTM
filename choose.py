
from keras.models import Sequential
from keras.layers import LSTM
from keras import regularizers
from keras.layers import  Dropout
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from keras.losses import binary_crossentropy
import keras
from keras.layers import Dense
from keras.callbacks import TensorBoard

from keras.layers import LSTM, Masking
import tensorflow as tf
import keras.callbacks
import sys
import os
from keras import backend as K

import pandas as pd

import numpy as np
from keras.layers import LSTM
import keras

np.random.seed()

from sklearn.metrics import classification_report

df1 = pd.read_csv('38Weeks-WithPass-Data.csv', low_memory=False)

df1 = df1.sort_values(by=['id1', 'week_id'])
# df1 = df1.sort_values(by=[ 'week_id'])

# factorizing the final result....pass=0, fail=1
d = ['final_result']

for val in d:
    labels, levels = pd.factorize(df1[val])
    df1[val] = labels

df1.head()


############################# Same length window ########################
def make_frames1(df):
    trys = []
    big_flat = []
    for x in range(0, 10):

        trys.append(list(df.iloc[0:x + 1, 1:21].values[x]))
        flat_list = []
        for sublist in trys:
            for item in sublist:
                flat_list.append(item)

        big_flat.append(flat_list)

        # padding
    for i in range(0, 10):
        op = len(big_flat[i])
        for w in range(0, (200 - op)):
            big_flat[i].append(-1)

    data_a = pd.DataFrame({"bigflat": big_flat})
    return data_a


def make_labels(df):
    labels = []
    for x in range(0, 10):
        # print(1-df.iloc[0:x + 1, 1:2].values[x])
        labels.append(list(df.iloc[0:x + 1, 1:2].values[x]))
        # labels2.append(list( df.iloc[0:x + 1, 1:2].values[x]))
        # labels=np.concatenate((labels1,labels2),axis=1)
        # print(labels)

        # labels2.append(list(df.iloc[0:x + 1, 1:2].values[x]))

    return labels



df_label = pd.DataFrame({})

for num in df1['id1'].unique():
    t = make_labels(df1[df1['id1'] == num])  # make frame fr each unique id

    df_label = df_label.append(t)

    # df_label[1] = df_label.append(t1)# the total df that has all the row for each id
    # # 0-37 for 1 id, 38-75 for 2nd id...(75+38=113)....



df1 = df1.drop(['final_result'], axis=1)

# fr each unique id, it will create dataframes,
# appending 0-37 rows for each unique id
dfnew = pd.DataFrame({})
for num in df1['id1'].unique():
    t = make_frames1(df1[df1['id1'] == num])  # make frame fr each unique id
    # t=t.sort_values(by=['id1'])


    # print("t",t)
    dfnew = dfnew.append(t)  # the total df that has all the row for each id
    # 0-37 for 1 id, 38-75 for 2nd id...(75+38=113)....



df_col = pd.DataFrame(dfnew['bigflat'].values.tolist())
X_val= df_col.iloc[153249:178780:10,]
y_val= df_label.iloc[153249:178780:10,]

X_train= df_col.iloc[9:153240:10,]
y_train= df_label.iloc[9:153240:10,]
X_test= df_col.iloc[178789::10,]
y_test=df_label.iloc[178789::10,]
# X_train= df_col.iloc[0:383100,]
# y_train= df_label.iloc[0:383100,]
# X_val= df_col.iloc[383100:446950,]
# y_val= df_label.iloc[383100:446950,]
# X_test= df_col.iloc[446950:,]
# y_test=df_label.iloc[446950:,]
# X_train= df_col.iloc[0:446950,]
# y_train= df_label.iloc[0:446950,]
# X_test= df_col.iloc[446950:,]
# y_test=df_label.iloc[446950:,]
# X_train= df_col.iloc[0:100,]
# y_train= df_label.iloc[0:100,]
# X_test= df_col.iloc[0:100,]
# y_test=df_label.iloc[0:100,]
# X_train= df_col.iloc[0:392000,]
# y_train= df_label.iloc[0:392000,]
# X_test= df_col.iloc[392000:,]
# y_test=df_label.iloc[392000:,]



X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1]))
y_train = y_train.values.reshape((y_train.shape[0], 1))
X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1]))
y_test = y_test.values.reshape((y_test.shape[0], 1))
X_val = X_val.values.reshape((X_val.shape[0], X_val.shape[1]))
y_val = y_val.values.reshape((y_val.shape[0], 1))



X_train = X_train.reshape((X_train.shape[0], 10, 20))
X_test = X_test.reshape((X_test.shape[0], 10, 20))
X_val = X_val.reshape((X_val.shape[0], 10, 20))


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall))



# record history of training
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.accuracies = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.accuracies.append(logs.get('acc'))


total = len(sys.argv)
cmdargs = str(sys.argv)

print ("Script name: %s" % str(sys.argv[0]))
checkpoint = None
if len(sys.argv) == 2:
    if os.path.exists(str(sys.argv[1])):
        print ("Checkpoint : %s" % str(sys.argv[1]))
        checkpoint = str(sys.argv[1])
        print("check point")



#LSTM model
sequence_length=10
nb_features = X_train.shape[2] #20
nb_out = y_train.shape[1] #1

model = Sequential()

model.add(Masking(mask_value=-1, input_shape=(sequence_length, nb_features)))
model.add(LSTM(
         input_shape=(sequence_length, nb_features),
         units=300,
    dropout=0.5,
    recurrent_dropout=0.5,
        activation='tanh',
kernel_regularizer=regularizers.l2(0.01),
         return_sequences=True))
model.add(Dropout(0.5))
model.add(LSTM(
          units=200,
          activation='tanh',
    dropout=0.5,
    recurrent_dropout=0.5,
kernel_regularizer=regularizers.l2(0.01),
          return_sequences=True))
model.add(Dropout(0.5))

model.add(LSTM(
          units=100,
          activation='tanh',
    dropout=0.5,
    recurrent_dropout=0.5,
kernel_regularizer=regularizers.l2(0.01),
          return_sequences=False))
model.add(Dropout(0.3))# model.add(Dense(
#           units=1,activation='sigmoid'))
model.add(Dense(1,activation='sigmoid'))

# if checkpoint:
model.load_weights('LSTM-10week-best.h5')

file_name = os.path.basename(sys.argv[0]).split('.')[0]
check_cb = keras.callbacks.ModelCheckpoint('LSTM-try/'+ file_name + '.{epoch:02d}-{val_loss:.2f}.hdf5',
                                           monitor='val_loss',
                                           verbose=0, save_best_only=True, mode='min')

history = LossHistory()
# keras.optimizers.RMSprop(lr=0.0001)
model.summary()
tbCallBack = TensorBoard(log_dir="./model")
model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.01,decay=0.01), metrics=['accuracy',precision,recall,f1])

score=model.evaluate(X_test,y_test,batch_size=50,callbacks=[tbCallBack])
print(score)
y_pre=model.predict(X_train,batch_size=50)



loss1=binary_crossentropy(y_train,y_pre)
loss1=K.eval(loss1)

#
#
len1=len(loss1)
index=np.arange(0,len1,1)


d=dict(zip(index,loss1))
print(d)
d1=sorted(d.items(),key=lambda x:x[1])
print(d1)
for i in range(0,len1):
    index[i]=d1[i][0]


# print(X_train[0])
# print(X_train_new[0])
# print(X_train_new[0])
np.save('./filename-new10.npy',index)
# index=np.load('./filename.npy')
# print(index)

# for i in range(0,5):
#     up=(i+1)*0.2
#     len2=int(len1*up)
#     index1=index[0:len2]
#
#
#     model.compile(loss='binary_crossentropy', optimizer=keras.optimizers.Adam(learning_rate=0.002,decay=0.001), metrics=['accuracy',precision,recall,f1])
#
# # model.fit(X_train, y_train,  batch_size=1250,validation_split=0.14,
# #           epochs=12, shuffle=False, callbacks=[tbCallBack])
# score=model.evaluate(X_test,y_test,batch_size=500,callbacks=[tbCallBack])
# y_predict=model.predict(X_test,batch_size=500)
# loss1=binary_crossentropy(y_test,y_predict)
# loss1=K.eval(loss1)
# len1=len(loss1)
# index=np.arange(0,len1,1)
# d={}
# d=dict(zip(index,loss1))
# d1=sorted(d.items(),key=lambda x:x[1])
# print(d1)
# for i in range(0,len1):
#     index[i]=d1[i][0]



# loss1.sort()

# print(score)
# print(y_predict)
# print(loss1)
# print(loss1.shape)
# model.save('150(new)-weeksModel-WithPass.h5')
