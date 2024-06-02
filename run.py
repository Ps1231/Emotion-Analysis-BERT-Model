import time
from tensorflow.keras.models import load_model
import pandas as pd
import numpy as np

# preparing input to our model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from google.colab import files
# uploaded = files.upload()
tokenizer = Tokenizer()
max_seq_len = 500
class_names = ['joy', 'fear', 'anger', 'sadness', 'neutral']
# TensorFlow/Keras
model = load_model('cnn_w2v.h5')

message = ['delivery was hour late and my pizza was cold!']

seq = tokenizer.texts_to_sequences(message)
padded = pad_sequences(seq, maxlen=max_seq_len)

start_time = time.time()
pred = model.predict(padded)

print('Message: ' + str(message))
print('predicted: {} ({:.2f} seconds)'.format(
    class_names[np.argmax(pred)], (time.time() - start_time)))
