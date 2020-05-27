import numpy as np
import keras as K
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.models import load_model

max_review_len = 1500
model = ""

typeofmodel = ""

typeofmodel = input("version")
if (typeofmodel == "v1"):
  print("using v1")
  model = load_model("imdb_model.h5")
else:
  print("using v2")
  model = load_model("imdb_model_v2.h5")
  
d = K.datasets.imdb.get_word_index()

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

review_texts = ["I did not like this at all, do not watch this.",
                "I really like this film! It's a must watch for everybody. I really loved the script of the film and all the emotions it brings, I cried when the dog died and I laughed when" +
                " I realized it was just plaiyng dead! Chris J really did a good job on this film, his acting was a masterpiece, second to none!",
                "This film is mediocre at best.",
                "This film is not good or bad.",
                "great great great great great great great great great great great great great great great great",
                "This makes me want to die",
                "bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad"]
review_scores = [0, 0, 0, 0, 0, 0, 0]
for i in range (0,7):
  review_scores[i] = getAnalysis(preapereText(review_texts[i]))
  print(review_texts[i])
  print(review_scores[i])


#review = "the movie was a great waste of my time"
#review = "Honestly, I enjoyed watching this movie. I loved every aspect of it, ranging from the great sound design to the intense lore moments. I will be recommending this movie to everyone I know, it's just that great!"
#print("New review: \'" + review + "\'")
#review = preapereText(review)
#score = getAnalysis(review)
#print("Prediction (0 = negative, 1 = positive) = ", end="")
#print("%0.4f" + review)

