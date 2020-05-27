# imdb_lstm.py
# LSTM for sentiment analysis on the IMDB dataset
# Anaconda3 4.1.1 (Python 3.5.2) Keras 2.1.5 TensorFlow 1.7.0

import numpy as np
import keras as K
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main():
  # 0. get started
  print("IMDB sentiment analysis using Keras/TensorFlow")
  np.random.seed(1)
  tf.random.set_seed(1)

  # 1. load data into memory
  # 2. define and compile LSTM model
  # 3. train model
  # 4. evaluate model
  # 5. save model
  # 6. use model to make a prediction
if __name__=="__main__":
  main()

  # 1. load data
max_words = 200000
print("Loading data, max unique words = %d words\n" % max_words)
(train_x, train_y), (test_x, test_y) = \
  K.datasets.imdb.load_data(seed=1, num_words=max_words)

max_review_len = 80
train_x = K.preprocessing.sequence.pad_sequences(train_x,
  truncating='pre', padding='pre', maxlen=max_review_len)
test_x = K.preprocessing.sequence.pad_sequences(test_x,
  truncating='pre', padding='pre', maxlen=max_review_len)

# 2. define model
print("Creating LSTM model")
e_init = K.initializers.RandomUniform(-0.01, 0.01, seed=1)
init = K.initializers.glorot_uniform(seed=1)
simple_adam = K.optimizers.Adam()
embed_vec_len = 32  # values per word

model = K.models.Sequential()
model.add(K.layers.embeddings.Embedding(input_dim=max_words,
  output_dim=embed_vec_len, embeddings_initializer=e_init,
  mask_zero=True))
model.add(K.layers.LSTM(units=100, kernel_initializer=init,
  dropout=0.2, recurrent_dropout=0.2))  # 100 memory
model.add(K.layers.Dense(units=1, kernel_initializer=init,
  activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer=simple_adam,
  metrics=['acc'])
print(model.summary())

# 3. train model
bat_size = 32
max_epochs = 3
print("\nStarting training ")
model.fit(train_x, train_y, epochs=max_epochs,
  batch_size=bat_size, shuffle=True, verbose=1) 
print("Training complete \n")

# 4. evaluate model
loss_acc = model.evaluate(test_x, test_y, verbose=0)
print("Test data: loss = %0.6f  accuracy = %0.2f%% " % \
  (loss_acc[0], loss_acc[1]*100))

# 5. save model
print("Saving model to disk \n")
mp = "C:\\Users\\user\\Documents\\Assignment\\Level 6 Year 2\\imdb_model_v2.h5"
model.save(mp)

# 6. use model
print("New review: \'the movie was a great waste of my time\'")
d = K.datasets.imdb.get_word_index()
review = "the movie was a great waste of my time"

words = review.split()
review = []
for word in words:
  if word not in d: 
    review.append(2)
  else:
    review.append(d[word]+3)

review = K.preprocessing.sequence.pad_sequences([review],
  truncating='pre', padding='pre', maxlen=max_review_len)
prediction = model.predict(review)
print("Prediction (0 = negative, 1 = positive) = ", end="")
print("%0.4f" % prediction[0][0])
