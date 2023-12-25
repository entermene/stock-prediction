import numpy as np

print(1)
print(2)
print(1)
print(2)
print(1)
print(2)
import sys
sys.version
# !pip install TA-Lib
# !pip install --user TA-Lib
# !{sys.executable} -m pip install --user TA-Lib
import talib as ta
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler
df=pd.read_csv('stock.csv')
print(df)
df['Date']=pd.to_datetime(df['Date'])
print(df)
df['Date']=pd.to_datetime(df['Date'])
print(df)
df.index=df['Date']
print(df)
df=df.drop(["Date"],axis=1)
print(df)
df['MA']=ta.SMA(df.Close,90)
df=df.fillna(0)
df['RSI']=ta.RSI(df.Close,14)
df["STOCH_K"],df["STOCH_D"]=ta.STOCH(df["High"],df["Low"],df["Close"],fastk_period=14,slowk_period=3,slowk_matype=0)

df["MACD"],df["MACDSignal"],df["MACDHist"]=ta.MACD(df.Close,fastperiod=12,slowperiod=26,signalperiod=9)
df["WILLR"]=ta.WILLR(df["High"],df["Low"],df["Close"],timeperiod=14)
MA=df["MA"].rolling('5d').apply(lambda x: np.sign(x[-1]-x[0]),raw=False)
df["Trend"]=df["MA"].rolling('5d').apply(lambda x: np.sign(x[-1]-x[0]),raw=False)
def f(row):
    if row['Close'] > row['MA'] and row['Trend'] == 1:
        val = 1
    elif row['Close'] < row['MA'] and row['Trend'] == -1:
        val = 0
    else:
        val = 0
    return val
df['Trend'] = df.apply(f, axis=1, raw=False)

Close=df["Close"]
df["Tri"]=((Close-Close.rolling('3 d').min())/(Close.rolling('3 d').max()-Close.rolling('3 d').min()))* 0.5  + 0.5*df['Trend']
# df["DownTri"]=(Close-Close.rolling('3d').min())/(Close.rolling('3d').max()-Close.rolling('3d').min())* .5
Close=df["Close"]
df["Tri"]=((Close-Close.rolling('3 d').min())/(Close.rolling('3 d').max()-Close.rolling('3 d').min()))* 0.5  + 0.5*df['Trend']
# df["DownTri"]=(Close-Close.rolling('3d').min())/(Close.rolling('3d').max()-Close.rolling('3d').min())* .5
Close=df["Close"]
print(Close[14])
df["UpTrend"] = ((Close - Close.rolling('3 D').min())/(Close.rolling('3 D').max() - Close.rolling('3 D').min()) * .5) + .5
df["DownTrend"] = (Close - Close.rolling('3 D').min())/(Close.rolling('3 D').max() - Close.rolling('3 D').min()) * .5

def g(row):
    if row['Trend'] == 1:
        val = row['UpTrend']
    elif row['Trend'] == 0:
        val = row['DownTrend']
    else:
        val = 0
    return val
df['Trade_Signal'] = df.apply(g, axis=1, raw=False)
print(df['Trade_Signal'][50:100])
print(df['Tri'][50:100])
features=pd.concat([df['MA'],df['RSI'],df["STOCH_K"],df["STOCH_D"],df["MACD"],df["WILLR"]],axis=1)
print(features)
target=df['Tri']
print(target)
target=target.fillna(0.5)
features=features.fillna(0.5)
print(target[50:100])
print(features[50:100])
df2 = pd.concat([features, target], axis=1)
print(df2[50:100])

n=6
d=6

# ones_matrix=pd.DataFrame(data=1, index=range(n),columns=range(1))
initial_weights=np.random.uniform(low=0,high=1,size=(n,d+1))

train=df2.iloc[:1000,:]
test=df2.iloc[1000:,:]
target_column=['Tri']
# predictors=list(set(list(df2.columns))-set(target_column))
y=train['Tri']
X=train.iloc[:,0:6]
ONE=np.ones((len(X),1))
X=np.hstack((ONE,X))
# X: n*(d+1)
nparray=np.empty((len(X),n+1))
for i in range(len(X)):
    nparray[i]=np.array(X[i,:])
expanded_inputs=np.empty((len(X),n))
for i in range(len(X)):
    for j in range(n):
        array=np.mat(nparray[i])
        # print(array)
        # print(array.T.dot([initial_weights[j]]).shape)
        expanded_inputs[i,j]=np.dot(array, initial_weights[j])
        #array.dot([initial_weights[j]])
        # expanded_inputs[i,j]=X.iloc[i].dot(initial_weights[j])
#test_expanded_inputs=
print(expanded_inputs.shape)
print(expanded_inputs[80:100])

df3=df2.copy(deep=True)
print(df3)
df3.rename(columns={'Tri':'cx1','MA':'cx2','MACD':'cx3','RSI':'cx4','STOCH_D':'cx5','STOCH_K':'cx6'},inplace=True)
print(df3)
print(df3.columns)
df4=df3.drop(columns=['WILLR'])
print(df4)
for i in range(len(features)):
    df4['cx1'][i]=1
print(df4)    
# # df3['Date']=df['Date'].copy()
# # df3['cx1']=pd.DataFrame(expanded_inputs[:,0])
target_column = ['Tri'] 
predictors = list(set(list(df4.columns)))
# indices=df['Date']
print(expanded_inputs)
# df3.drop('MA',axis=1)
data_df = pd.DataFrame(expanded_inputs, index=df4.index, columns=predictors)

print(expanded_inputs[80:100])

# aa=features['MA'].copy()
# # print(aa)
# # print(aa.shape)
# for i in range(len(aa)):
#     aa[i]=1
# print(aa)
# print(features['MA'])
features['aa']=features['MA'].copy()
for i in range(len(features)):
    features['aa'][i]=1
print(features)
features1=features.drop(columns='aa')
# expanded_inputs=pd.concat([train.iloc[:,0:6],expanded_inputs],axis=1)
# expanded_inputs=pd.concat(features,expanded_inputs)
# ONE=np.ones((len(X),1))

# features1=np.hstack((ONE,features))

# one=[1]*966
# ONE=pd.DataFrame(one)
# print(ONE.shape)
# print(features.shape)
# features1=pd.DataFrame()
# print(features1.shape)
# features1=pd.concat([aa,features],axis=1)
# print(features1.shape)
print(expanded_inputs.shape)
expanded_inputs1=pd.DataFrame(expanded_inputs)
print(expanded_inputs1.shape)
print(expanded_inputs1)
expanded_inputs2=pd.concat([features1,data_df],axis=1)
print(expanded_inputs2.shape)
print(expanded_inputs2)
expanded_inputs3=expanded_inputs2.as_matrix()


#data_df


synaptic_weights = np.random.uniform(low=0, high=1, size=(n+d,1))
# synaptic_weights = np.append(synaptic_weights, 1)
print(synaptic_weights)




# train=df2.iloc[:1000,:]
# test=df2.iloc[1000:,:]
# target_column = ['Tri'] 
# predictors = list(set(list(df2.columns))-set(target_column))
# X= train[predictors]
# print('X:')
# print(X.iloc[0])
# print(len(X.iloc[0]))
# print(len(X))
# print(X[0:50])
# y = train[target_column]





# synaptic_weights = np.random.uniform(low=0, high=1, size=(6,1))
# synaptic_weights = np.append(synaptic_weights, 1)
# print('synaptic_weights:')
# print(synaptic_weights)
# print(len(synaptic_weights))
# feb=np.outer(X,synaptic_weights)
# print(feb)
# print(len(feb))
# print('feb:')
# print(len(feb[0]))
# print(feb[0])
# feb2=np.sum(feb,axis=1)
# print('feb2:')
# print(len(feb2))
# print(feb2)
# # feb3=np.sum(feb,axis=1)
# # print('feb3')
# # print(len(feb3))
# # print(feb3)
# expanded_inputs = np.tanh(feb2).reshape(len(X), -1)
# print(synaptic_weights)
# print('expanded_inputs:')
# print(len(expanded_inputs))
# print(len(expanded_inputs[0]))
# print(expanded_inputs[20:60])
# synaptic_weights=np.random.uniform(low=1,high=1,size=(6,1))
# # 在末尾加上w0
# synaptic_weights=np.append(synaptic_weights,1)

# # exband
# feb=np.outer(X,synaptic_weights)
# feb2=np.sum(feb.reshape(-1,7),axis=1)
# print(feb2)
# print(len(X))
# expanded_input=np.tanh(feb2).reshape(len(X),-1)
# print(len(expanded_inputs))
                                               
                        
        
expanded_inputs=expanded_inputs2.copy()
