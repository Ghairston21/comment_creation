import tensorflow as tf
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import numpy as np
import pandas as pd
vocab_size = 1000
tokenizer = Tokenizer(num_words = vocab_size)

comments = pd.read_csv(r"comments.csv")
comments.body=comments.body.astype(str)
comments = comments.body.to_numpy()
print(comments[2])
comments_list = []
print(comments[0])



comments = comments[0:500]
print(comments[5])
tokenizer.fit_on_texts(comments)
comments_list = []
for x in comments:
	comments_list.append(x.lower())


total_words = len(tokenizer.word_index) + 1


print("start")
input_sequences = []
for comment in comments:
	token_list = tokenizer.texts_to_sequences([comment])[0]
	for i in range(1, len(token_list)):
		n_gram_sequence = token_list[:i+1]
		input_sequences.append(n_gram_sequence)


max_sequence_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))


xs, labels = input_sequences[:,:-1],input_sequences[:,-1]

labels = tf.keras.utils.to_categorical(labels, num_classes=total_words)

checkpoint_path = "training/weights.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
weights_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=1)


model = Sequential()
model.add(Embedding(total_words, 100, input_length=max_sequence_len-1))
model.add(Bidirectional(LSTM(100)))
model.add(Dense(total_words, activation='softmax'))
adam = Adam(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])


#history = model.fit(xs, labels, epochs=50, verbose=2, callbacks=[weights_checkpoint])
print(model)

base = "what did you say to me"
next_words = 20

for _ in range(next_words):
	token_list = tokenizer.texts_to_sequences([base])[0]
	token_list = pad_sequences([token_list], maxlen=max_sequence_len - 1, padding='post')
	predicted = model.predict_classes(token_list, verbose=0)
	output_word = ""
	for word, index in tokenizer.word_index.items():
		if index == predicted:
			output_word = word
			break
	base += " " + output_word
print(base)