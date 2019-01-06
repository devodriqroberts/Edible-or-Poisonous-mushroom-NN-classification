# Mushroom classification with Artifical Neural Network

#%%
# Import Libraries
import pandas as pd
import numpy as np 

# Load the data
data = pd.read_csv('/Users/devodriqroberts/Desktop/Software-Development/ArtificalNN/Classification/Mushroom_classification/mushrooms.csv')

# Split the data (features/label)
features = pd.DataFrame(data.iloc[:,1:])
label = list(data['class'])

# Get all columns in features set
features_cols = features.columns

# Onehot encode all features columns
from sklearn.preprocessing import LabelEncoder
def onehot_all_cols(cols):
    encoder = LabelEncoder()
    for col in cols:
        features[col] = encoder.fit_transform(features[col])

# Encode labels
def onehot_labels(label):
    return pd.get_dummies(label)

onehot_all_cols(features_cols)
label = onehot_labels(label)

# Build the model
from keras.models import Model
from keras.layers import Dense, Input

# Define input shape
inputs = Input(shape=(22,))

# Hidden layers
x = Dense(64, activation='relu')(inputs)
x = Dense(64, activation='relu')(x)
x = Dense(32, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs, predictions)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
model.summary()

# Fit the model
model.fit(features, label, batch_size=64, validation_split=0.2, epochs=10)

