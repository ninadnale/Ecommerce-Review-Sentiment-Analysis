import pandas as pd  
import numpy as np

my_df= pd.read_csv(r"/home/ninad/Downloads/Amazon Dataset/mobile_elec.tsv", delimiter = "\t", error_bad_lines=False)

df=my_df[['star_rating','review_headline','review_body']]

df['star_rating'].loc[df['star_rating']==1]=0
df['star_rating'].loc[df['star_rating']==2]=0
df['star_rating'].loc[df['star_rating']==4]=1
df['star_rating'].loc[df['star_rating']==5]=1
df=df[df.star_rating!=3]

df=df.rename(columns={
    'star_rating':'LABEL',
    'review_headline':'REVIEW_TITLE',
    'review_body':'REVIEW'
})

df.dropna(inplace=True)
df["review_Length"]= df["REVIEW"].str.len()

new = df[df.review_Length>60]
x = new.REVIEW
y = new.LABEL

# cross_validation changed to model_selection
from sklearn.model_selection import train_test_split
SEED = 2000
x_train, x_validation_and_test, y_train, y_validation_and_test = train_test_split(x, y, test_size=.02, random_state=SEED)
x_validation, x_test, y_validation, y_test = train_test_split(x_validation_and_test, y_validation_and_test, test_size=.5, random_state=SEED)

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils


def labelize_reviews_ug(reviews,label):
    result = []
    prefix = label
    for i, t in zip(reviews.index, reviews):
        result.append(TaggedDocument(t.split(), [prefix + '_%s' % i]))
    return result


all_x = pd.concat([x_train,x_validation,x_test])
all_x_w2v = labelize_reviews_ug(all_x, 'all')

cores = multiprocessing.cpu_count()
model_ug_cbow = Word2Vec(sg=0, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_cbow.build_vocab([x.words for x in tqdm(all_x_w2v)])

#%time
for epoch in range(30):
    model_ug_cbow.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_cbow.alpha -= 0.002
    model_ug_cbow.min_alpha = model_ug_cbow.alpha


model_ug_sg = Word2Vec(sg=1, size=100, negative=5, window=2, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_ug_sg.build_vocab([x.words for x in tqdm(all_x_w2v)])

#%time
for epoch in range(30):
    model_ug_sg.train(utils.shuffle([x.words for x in tqdm(all_x_w2v)]), total_examples=len(all_x_w2v), epochs=1)
    model_ug_sg.alpha -= 0.002
    model_ug_sg.min_alpha = model_ug_sg.alpha

model_ug_cbow.save('senti_w2v_model_ug_cbow.word2vec')
model_ug_sg.save('senti_w2v_model_ug_sg.word2vec')

from gensim.models import KeyedVectors
model_ug_cbow = KeyedVectors.load('senti_w2v_model_ug_cbow.word2vec')
model_ug_sg = KeyedVectors.load('senti_w2v_model_ug_sg.word2vec')

embeddings_index = {}
for w in model_ug_cbow.wv.vocab.keys():
    embeddings_index[w] = np.append(model_ug_cbow.wv[w],model_ug_sg.wv[w])


from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(x_train)
sequences = tokenizer.texts_to_sequences(x_train)

#for x in x_train:
#    length.append(len(x.split()))

x_train_seq = pad_sequences(sequences, maxlen=500)

sequences_val = tokenizer.texts_to_sequences(x_validation)
x_val_seq = pad_sequences(sequences_val, maxlen=500)

num_words = 60000
embedding_matrix = np.zeros((num_words, 200))
for word, i in tokenizer.word_index.items():
    if i >= num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector


seed = 7

from tensorflow.keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import Flatten
from keras.layers.embeddings import Embedding

import keras
num_classes=2
y_train_binary = keras.utils.to_categorical(y_train, num_classes)
y_val_binary = keras.utils.to_categorical(y_validation, num_classes)

model_ptw2v = Sequential()
e = Embedding(60000, 200, weights=[embedding_matrix], input_length=500, trainable=False)
model_ptw2v.add(e)
model_ptw2v.add(Flatten())
model_ptw2v.add(Dense(256, activation='relu'))
model_ptw2v.add(Dense(num_classes, activation='softmax'))
model_ptw2v.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_ptw2v.fit(x_train_seq, y_train_binary, validation_data=(x_val_seq, y_val_binary), epochs=5, batch_size=32, verbose=1)


predictions=model_ptw2v.predict_classes(x_train_seq[0:10])


from keras.layers import Conv1D, GlobalMaxPooling1D

structure_test = Sequential()
e = Embedding(60000, 200, input_length=210)
structure_test.add(e)
structure_test.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
structure_test.summary()

structure_test = Sequential()
e = Embedding(60000, 200, input_length=210)
structure_test.add(e)
structure_test.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
structure_test.add(GlobalMaxPooling1D())
structure_test.summary()

model_cnn_01 = Sequential()
e = Embedding(60000, 200, weights=[embedding_matrix], input_length=500, trainable=False)
model_cnn_01.add(e)
model_cnn_01.add(Conv1D(filters=100, kernel_size=2, padding='valid', activation='relu', strides=1))
model_cnn_01.add(GlobalMaxPooling1D())
model_cnn_01.add(Dense(256, activation='relu'))
model_cnn_01.add(Dense(1, activation='sigmoid'))
model_cnn_01.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model_cnn_01.fit(x_train_seq, y_train, validation_data=(x_val_seq, y_validation), epochs=12, batch_size=32, verbose=1)

import pickle
with open('sentiment_mobile_elec.pickle', 'wb') as file:
    pickle.dump(model_cnn_01, file)

#with open('sentiment_mobile_elec.pickle', 'rb') as file:
#    model=pickle.load(file)
#
#predictions=model.predict_classes(x_train_seq[0:10])
