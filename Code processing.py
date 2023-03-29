import pandas as pd
import numpy as np
from keras.layers import LSTM
import keras

np.random.seed()

df1 = pd.read_csv('38Weeks-WithPass-Data.csv', low_memory=False)

df1 = df1.sort_values(by=['id1', 'week_id'])

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
    for x in range(0, 25):

        trys.append(list(df.iloc[0:x + 1, 1:21].values[x]))
        flat_list = []
        for sublist in trys:
            for item in sublist:
                flat_list.append(item)

        big_flat.append(flat_list)

        # padding
    for i in range(0, 25):
        op = len(big_flat[i])
        for w in range(0, (500 - op)):
            big_flat[i].append(-1)

    data_a = pd.DataFrame({"bigflat": big_flat})
    return data_a


def make_labels(df):
    labels = []
    for x in range(0, 25):
        labels.append(list(df.iloc[0:x + 1, 1:2].values[x]))
    return labels


df_label = pd.DataFrame({})
for num in df1['id1'].unique():
    t = make_labels(df1[df1['id1'] == num])  # make frame fr each unique id

    # print("DF for one id:", t)
    df_label = df_label.append(t)  # the total df that has all the row for each id
    # 0-37 for 1 id, 38-75 for 2nd id...(75+38=113)....

# print("DF-labels: ", df_label)
#
# print(type(df_label))

df1 = df1.drop(['final_result'], axis=1)

# fr each unique id, it will create dataframes,
# appending 0-37 rows for each unique id
dfnew = pd.DataFrame({})
for num in df1['id1'].unique():
    t = make_frames1(df1[df1['id1'] == num])  # make frame fr each unique id
    # t=t.sort_values(by=['id1'])

    # print("DF for one id:", t)
    # print("t",t)
    dfnew = dfnew.append(t)  # the total df that has all the row for each id
    # 0-37 for 1 id, 38-75 for 2nd id...(75+38=113)....

# print("DF-Final: ", dfnew)
#
# print(dfnew.shape)
# print(type(dfnew))

df_col = pd.DataFrame(dfnew['bigflat'].values.tolist())
# print("df_col",df_col)
# print("df_LABEL",df_label)
all=len(df_label)
all3=all/25
all3=int(all3*0.7)
all3=all3*25
count1_7=0
count0_7=0
count1_3=0
count0_3=0
for i in range(0,all):
    if i>=all3:
        if df_label.iloc[i][0] == 1:
            count1_3 = count1_3 + 1
        else:
            count0_3 = count0_3 + 1

    if df_label.iloc[i][0]==1:
        count1_7=count1_7+1
    else:
        count0_7=count0_7+1
print('1:',count1_7/25)
print('0:',count0_7/25)
print('1:',count1_3/25)
print('0:',count0_3/25)
# X_test = df_col.iloc[0::10, ]
# y_test = df_label.iloc[0::10, ]
# X_train = df_col.iloc-X_test
# y_train = df_label.iloc-y_test
# X_train= df_col.iloc[0:400000,]
# y_train= df_label.iloc[0:400000,]
# X_test= df_col.iloc[400000:,]
# y_test=df_label.iloc[400000:,]

# In[111]:

#
# print(type(X_train))
# print(type(y_train))
# print(type(X_test))
# print(type(y_test))
# print("xtrain", X_train.shape)
# print("ytrain", y_train.shape)
# print("xtest", X_test.shape)
# print(y_test.shape)
# print(X_train.shape[1])
#
# X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1]))
# y_train = y_train.values.reshape((y_train.shape[0], 1))
# X_test = X_test.values.reshape((X_test.shape[0], X_test.shape[1]))
# y_test = y_test.values.reshape((y_test.shape[0], 1))
#
# print(X_train.shape)
# print("type xtrain", type(X_train))
# print(y_train.shape)
# print("type ytrain", type(y_train))
# print(X_test.shape)
# print(y_test.shape)
#
# print(X_train.shape[0])
# print(X_train.shape[1])
# print(X_test.shape[0])
#
# X_train = X_train.reshape((X_train.shape[0], 25, 20))
# X_test = X_test.reshape((X_test.shape[0], 25, 20))
#
# print(X_train.shape[0])
# print(X_train.shape[1])
# print("xtrain.shape[2]", X_train.shape[2])
# print(X_train.shape)
# print(X_test.shape[0])
# print(X_test.shape[1])



