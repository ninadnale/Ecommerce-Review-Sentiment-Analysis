from flask import Flask, render_template, request, redirect
#from vsearch import search4letters

import product_recommendation as rcm
import detect_spam

import pandas as pd
import numpy as np
###################################################################################################################################################################
import pandas as pd

from tqdm import tqdm
tqdm.pandas(desc="progress-bar")
import gensim
from gensim.models.word2vec import Word2Vec
from gensim.models.doc2vec import TaggedDocument
import multiprocessing
from sklearn import utils

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

df= pd.read_csv(r"/home/ninad/Downloads/NewDataset/cell_phones_data_withoutNaN.csv", error_bad_lines=False)
x_test=df.reviewText[:]

tokenizer = Tokenizer(num_words=100000)
tokenizer.fit_on_texts(x_test)
sequences = tokenizer.texts_to_sequences(x_test)

x_test_seq = pad_sequences(sequences, maxlen=500)
'''
import pickle
with open('saved_model.pickle', 'rb') as file:
    model=pickle.load(file)

predictions=model.predict_classes(x_test_seq)
pred_list = predictions.tolist()
sent_list = []
for i in range(len(pred_list)):
    sent_list.append(pred_list[i][0])
df['SENTIMENT'] = sent_list
df.to_csv('Dataset/sentiment_data.csv')
'''
###################################################################################################################################################################

class sentiment:
	df = pd.read_csv('Dataset/sentiment_data.csv')
	
	def most_frequent(List): 
		return max(set(List), key = List.count) 

	def get_review_analysis(self) ->(int , int):
		#df = pd.read_csv('/home/ninad/Downloads/Amazon Dataset/amazon_sentiment.csv')
		pos = sentiment.df['SENTIMENT'][sentiment.df['SENTIMENT']==1].count()
		neg = sentiment.df['SENTIMENT'][sentiment.df['SENTIMENT']==0].count()
		return pos, neg
	def get_numberOfProducts(self) ->(int):
		nprod = len(sentiment.df['asin'].unique().tolist())
		return nprod
	def get_numberOfCategories(self) ->(int):
		ncat = len(sentiment.df['category'].unique().tolist())
		return ncat
	def get_numberOfReviews(self) ->(int):
		nrev = len(sentiment.df['reviewText'].unique().tolist())
		return nrev
	def get_numberOfBrands(self) ->(int):
		nbrnd = len(sentiment.df['brand'].unique().tolist())
		return nbrnd
	def get_detailAnalysis(self) ->(list):
		packing_pos = sentiment.df['reviewText'][(sentiment.df['reviewText'].str.contains('packing')) & (sentiment.df['SENTIMENT']==1)].count()
		packing_neg = sentiment.df['reviewText'][(sentiment.df['reviewText'].str.contains('packing')) & (sentiment.df['SENTIMENT']==0)].count()
		packing = packing_pos/(packing_pos+packing_neg)*100

		delivery_pos = sentiment.df['reviewText'][(sentiment.df['reviewText'].str.contains('delivery')) & (sentiment.df['SENTIMENT']==1)].count()
		delivery_neg = sentiment.df['reviewText'][(sentiment.df['reviewText'].str.contains('delivery')) & (sentiment.df['SENTIMENT']==0)].count()
		delivery = delivery_pos/(delivery_pos+delivery_neg)*100

		service_pos = sentiment.df['reviewText'][(sentiment.df['reviewText'].str.contains('service')) & (sentiment.df['SENTIMENT']==1)].count()
		service_neg = sentiment.df['reviewText'][(sentiment.df['reviewText'].str.contains('service')) & (sentiment.df['SENTIMENT']==0)].count()
		service = service_pos/(service_pos+service_neg)*100

		quality_pos = sentiment.df['reviewText'][(sentiment.df['reviewText'].str.contains('quality')) & (sentiment.df['SENTIMENT']==1)].count()
		quality_neg = sentiment.df['reviewText'][(sentiment.df['reviewText'].str.contains('quality')) & (sentiment.df['SENTIMENT']==0)].count()
		quality = quality_pos/(quality_pos+quality_neg)*100

		payment_pos = sentiment.df['reviewText'][(sentiment.df['reviewText'].str.contains('payment')) & (sentiment.df['SENTIMENT']==1)].count()
		payment_neg = sentiment.df['reviewText'][(sentiment.df['reviewText'].str.contains('payment')) & (sentiment.df['SENTIMENT']==0)].count()
		payment = payment_pos/(payment_pos+payment_neg)*100

		returns_pos = sentiment.df['reviewText'][(sentiment.df['reviewText'].str.contains('return')) & (sentiment.df['SENTIMENT']==1)].count()
		returns_neg = sentiment.df['reviewText'][(sentiment.df['reviewText'].str.contains('return')) & (sentiment.df['SENTIMENT']==0)].count()
		returns = returns_pos/(returns_pos+returns_neg)*100
		
		results = [packing, delivery, service, quality, payment, returns]
		return results
	def get_populars(self) ->(list):
		l = df['product_name'].tolist()
		prod = sentiment.most_frequent(l)
		l = df['category'].tolist()
		cat = sentiment.most_frequent(l)
		l = df['brand'].tolist()
		brnd = sentiment.most_frequent(l)
		populars = [prod, cat, brnd]
		return populars
###################################################################################################################################################################	
app = Flask(__name__)

sent = sentiment()
pos, neg = sent.get_review_analysis()
no_of_products = sent.get_numberOfProducts()
no_of_categories = sent.get_numberOfCategories()
no_of_reviews = sent.get_numberOfReviews()
no_of_brands = sent.get_numberOfBrands()
detail_analysis = sent.get_detailAnalysis()
populars = sent.get_populars()


@app.route('/')
@app.route('/sentiment_analysis')
def entry_page() -> 'html':
	count = detect_spam.get_spam_results()
	return render_template('index.html', spm_count=count, reviews=no_of_reviews, categories=no_of_categories, products=no_of_products, brands=no_of_brands )


@app.route('/viewDashboard')
def data_dashboard() -> 'html':
	return render_template('data_dashboard.html', positive=pos, negative=neg, reviews=no_of_reviews, packing=detail_analysis[0], delivery=detail_analysis[1], service=detail_analysis[2], quality=detail_analysis[3], payment=detail_analysis[4], returns=detail_analysis[5], pop_prod = populars[0], pop_cat = populars[1], pop_brnd = populars[2])


@app.route('/Recommendations', methods=['POST'])
def get_recommendations() -> 'html':
	userID = request.form.get('user_id')
	recommendations = rcm.get_rec(userID)
	userName = rcm.get_uname(userID)
	return render_template('recommendation_results.html', results=recommendations, userID=userID, userName=userName)


if __name__ == '__main__':
	app.run(debug=True)

