from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import numpy as np

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
import nltk

import time
import os

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'


def isNaN(string):
  return string != string


def list_to_dict(words_list):
  return dict([(word, True) for word in words_list])


# Function to convert  
def listToString(s): 
    
	# initialize an empty string
	str1 = "" 
    
	# traverse in the string  
	for ele in s: 
		str1 += str(ele) + " "
    
	# return string  
	return str1


def train():
	print("Training sedang berjalan...")
	factory = StemmerFactory() #untuk membuat driver stemmer
	stemmer = factory.create_stemmer() #pemanggil fungsi stemmer
	# dataset = pd.read_excel('dataset/training_dataset.xlsx')
	testing_data = pd.read_excel('dataset/testing_dataset.xlsx') #load pretrained(proses filtering) testing dataset

	tukang_sapu = pd.read_excel('tukang_sapu/slank_typo.xlsx') #load kamus slank_typo dataset, untuk memperbaiki kata-kata yang salah

	# tweets = list()
	new_tweets = list() #variabel penampung data_training
	new_testing = list() #varibel penampung data_testing

	# for (index, row) in dataset.iterrows():
	#   if not (isNaN(row["Label"])):
	#     tweets.append([index, row['tweet'], row['Label']])

	# tweets = pd.DataFrame(tweets, columns=['index', 'tweet', 'label'])

	tweets = pd.read_excel('dataset/final_training_set.xlsx') #load pretrained(proses filtering) training dataset -> untuk mempersingkat waktu proses olah data

	#perulangan untuk data_training
	for (index, tweet) in tweets.iterrows(): 
		temp = []
		t = tweet['tweet'].split(' ')

		for kata in t:
			if not (('#' in kata) or ('https://' in kata) or ('http://' in kata) or (kata == "") or ('@' in kata) or ('uaddown' in kata) or ('uadjahat' in kata) or ('https' in kata) or ('8tnqmbefvs' in kata)):
				kata_bersih = kata.lower()
				kata_bersih = kata_bersih.replace('.', '')
				kata_bersih = kata_bersih.replace(',', '')
				kata_bersih = kata_bersih.replace('!', '')

				for (index, kata) in tukang_sapu.iterrows():
					if kata_bersih == kata.slank_junk_typo:
						kata_bersih = kata.perbaikan
						break

				temp.append(kata_bersih)

		string_temp = listToString(temp) #merubah list menjadi string

		kalimat = stemmer.stem(string_temp)

		tuple_temp = tuple()
		tuple_temp += tuple([kalimat.split(' ')])
		tuple_temp += (tweet['label'], )

		new_tweets.append(tuple_temp) #variabel penampung hasil proses pengubahan kata

	#perulangan untuk data_testing
	for (index, tweet) in testing_data.iterrows():
		temp = []
		t = tweet['tweet'].split(' ')

		for kata in t:
			if not (('#' in kata) or ('https://' in kata) or ('http://' in kata) or (kata == "") or ('@' in kata)  or ('uaddown' in kata) or ('uadjahat' in kata) or ('https' in kata) or ('8tnqmbefvs' in kata)):
				kata = kata.lower()
				for (index, word) in tukang_sapu.iterrows():
					if kata == word.slank_junk_typo:
						kata = word.perbaikan
						break

				temp.append(kata)

		tuple_temp = tuple()
		tuple_temp += tuple([temp])
		tuple_temp += (tweet['Label'], )

		new_testing.append(tuple_temp)

	training_set_formatted = [(list_to_dict(element[0]), element[1]) for element in new_tweets] #transfromasi tipe data list ke dictioner data training

	numIterations = 100 #iterasi proses training

	algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[1]
	classifier = nltk.MaxentClassifier.train(training_set_formatted, algorithm, max_iter=numIterations)
	classifier.show_most_informative_features(10)

	test_set_formatted = [(list_to_dict(element[0]), element[1]) for element in new_testing] #transfromasi tipe data list ke dictioner data testing

	true_pred = 0
	total_test_set = len(test_set_formatted)

	# pos net neg
	result = [0, 0, 0]

	for review in test_set_formatted:
	  label = review[1]
	  text = review[0]
	  determined_label = classifier.classify(text)
	  print(determined_label, label)

	  if determined_label == label:
	    true_pred = true_pred + 1
	  
	  if determined_label == 'positif':
	    result[0] = result[0] + 1
	  elif determined_label == 'netral':
	    result[1] = result[1] + 1
	  else:
	    result[2] = result[2] + 1

	akurasi = true_pred / total_test_set

	sentiment = ['positif', 'netral', 'negatif']

	index = np.argmax(result)

	return (sentiment[index], akurasi)


@app.route('/', methods=['GET'])
def index():
	sentiment, akurasi = train()

	data = {"sentiment": sentiment, "accuracy": akurasi}

	return jsonify(data), 200


@app.route('/test', methods=['GET'])
def test():

	message = "Berhasil terhubung ke Aplikasi";

	data = {'message': message}

	print(data)

	return jsonify(data), 200


if __name__ == "__main__":
    app.debug = False
    app.run()