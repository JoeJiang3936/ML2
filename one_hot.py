#https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/


'''

How to One Hot Encode Sequence Data in Python

Credit: Jason Brownlee
'''



#With sklearn
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
#define example
data = ['cold', 'cold', 'warm', 'cold', 'hot', 'hot', 'warm', 'cold', 'warm', 'hot']
values = array(data)
print(values)
#integer encode
label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(values)
print(integer_encoded)
#binary encode
onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
'''
['cold' 'cold' 'warm' 'cold' 'hot' 'hot' 'warm' 'cold' 'warm' 'hot']
[0 0 2 0 1 1 2 0 2 1]
'''
print(onehot_encoded)
#invert first example
inverted = label_encoder.inverse_transform([argmax(onehot_encoded[0, :])])
print(inverted)



#With Keras
data = [1, 3, 2, 0, 3, 2, 2, 1, 0, 1]
import tensorflow as tf
from numpy import array
from numpy import argmax
from keras.utils import to_categorical

#define example
data = [1, 3, 2, 0, 3, 2, 2, 1, 0, 1]
data = array(data)
print(data)
#one hot encode
encoded = to_categorical(data)
print(encoded)
#invert encoding
inverted = argmax(encoded[0])
print(inverted)
