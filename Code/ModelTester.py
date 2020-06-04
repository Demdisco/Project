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
  typeofmodel = input("version")
  if (typeofmodel == "v1"):
    print("using v1")
    model = load_model("imdb_model.h5")
  elif (typeofmodel == "v2"):
    print("using v2")
    model = load_model("imdb_model_v2.h5")
  else:
    print("using v3")
    model = load_model("imdb_model_v3.h5")

  review_texts = ["I did not like this at all, do not watch this.",
                  "I really like this film! It is a must watch for everybody. I really loved the script of the film and all the emotions it brings, I cried when the dog died, and I laughed when I realized it was just playing dead! Chris J really did a good job on this film, his acting was a masterpiece, second to none!",
                  "This film is mediocre at best.",
                  "This film is not good or bad.",
                  "great great great great great great great great great great great great great great great great",
                  "I’d rather have my eyes removed than be forced to watch this again",
                  "bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad bad",
                  "Fo sure the best film of 2019!! None better than this!",
                  "Th best film of 2020, I can not belive how good this is. Watch it!",
                  "This film is so bad, it makes the sound of ‘badom’ every time you watch it…",
                  "This film on the whole is great, but there are some flaws that one can does not ignore. Often, in the background, the cameras recording the film can be seen. It is not easy to spot them, but once you notice one, you start noticing more. The acting was not the best, but it was not that bad either. Overall, I still recommend you watch it."]

  review_scores = []
    
  for review in review_texts:
    print(review)
    try:
      print(getAnalysis(preapereText(review)))
    except:
      print("Error")
