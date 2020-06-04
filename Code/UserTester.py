import numpy as np
import keras as K
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import load_model


print("---Initializating---")
max_words = 40000
max_review_len = 1500
model = ""

typeofmodel = ""

print("---Loading word index---")
d = K.datasets.imdb.get_word_index()
print("---Done loading word index---")
  
def preapereText(review):
  words = review.split()
  review = []
  for word in words:
    if word not in d: 
        review.append(2)
    else:
        review.append(d[word]+3)
  return K.preprocessing.sequence.pad_sequences([review], truncating='pre', padding='pre', maxlen=max_review_len)

def getAnalysis(review):
  prediction = model.predict(review)
  return prediction[0][0]

print("---Initialization complete---")
while (True):
  typeofmodel = input("Choose version (v1/v2/v3): ")
  if (typeofmodel == "v1"):
    print("using v1")
    model = load_model("imdb_model.h5")
  elif (typeofmodel == "v2"):
    print("using v2")
    model = load_model("imdb_model_v2.h5")
  else:
    print("using v3")
    try:
      model = load_model("imdb_model_v3.h5")
    except:
      print("Could not find v3, using v1")
      model = load_model("imdb_model_v1.h5")



  review_texts = input("Enter your review")
    
  try:
    print(getAnalysis(preapereText(review_texts)))
  except:
    print("Error, an unrecognizable word was entered")
