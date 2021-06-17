import numpy as np
import pandas as pd
import keras
import tensorflow

from sklearn.ensemble import RandomForestRegressor
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input, BatchNormalization
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import Normalizer
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#regrRan = RandomForestRegressor(max_depth=50, n_estimators=150)
regr = LinearRegression()
score=[]
dimension=[]
variances =[]

### Preprocessing ###

df = pd.read_csv('C:\\github_repos\\Universal-Embeddings\\data\\geocoded-cities-master.csv')
columns = list(df.columns.values)

# Features list and removal of city, lat, lon
features = list(df.columns.values)
del features[:3]
del features[-1]

# y value list using last day of 7-month data
y = df.loc[:, ['pm25_2021_06_06']].values

# Normalize x values; save in data frame
x = df.loc[:, features].values
x = Normalizer().fit_transform(x)
dfx = pd.DataFrame(x)

# Training and test splits
train, dev, train_labels, dev_labels = train_test_split(dfx, y, test_size=0.20, random_state=10)
train, test, train_labels, test_labels = train_test_split(train, train_labels, test_size=0.20, random_state=10)
input_data = Input(shape=(191,))

x_train = train.loc[:, train.columns]
x_test = test.loc[:, test.columns]
x_dev = dev.loc[:, dev.columns]

encoded = Dense(4, activation='tanh', name='bottleneck')(input_data)
decoded = Dense(191, activation='tanh')(encoded)

autoencoder = Model(input_data, decoded)

# Compile the autoencoder
autoencoder.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.0001))
autoencoder.summary()

trained_model = autoencoder.fit(
    x_train, 
    x_train, 
    batch_size=128, 
    epochs=50, 
    verbose=1,
    validation_data=(x_dev, x_dev)
)

# bottleneck representation
encoder = Model(autoencoder.input, autoencoder.get_layer('bottleneck').output)
encoded_data = encoder.predict(x_train)  
decoded_output = autoencoder.predict(x_train)  
decoded_output_test = autoencoder.predict(x_test)

# Variance score explanation
variance = explained_variance_score(x_test, decoded_output_test, multioutput='uniform_average')
regr.fit(encoded_data, train_labels)

encoded_data_test = encoder.predict(x_test)
y_pred = regr.predict(encoded_data_test)

# R2 scores
r2score = regr.score(encoded_data_test, test_labels)
score.append(regr.score(encoded_data_test, test_labels))
dimension.append(10)

print('r2 score: {}', r2score)
print('variance: {}', variance)


# Printouts
print('Mean Absolute Error:', metrics.mean_absolute_error(test_labels, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(test_labels, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(test_labels, y_pred)))
print(variances.append(explained_variance_score(x_test, decoded_output_test, multioutput='uniform_average')))
print(explained_variance_score(x_test, decoded_output_test, multioutput='uniform_average'))



