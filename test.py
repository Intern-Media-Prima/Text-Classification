from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from urllib.request import urlopen
from multiprocessing import Pool
from os import listdir
import pandas as pd
import pickle
import re

def flat(arr):
	items = []
	for item in arr:
		if isinstance(item, list): [items.append(i) for i in flat(item)]
		else: items.append(item)
	return items

def get_articles():
	return flat(map(lambda x: list(map(lambda y: 'articles/{}/{}'.format(x, y), listdir('articles/{}'.format(x)))), listdir('articles')))

stemmer = StemmerFactory().create_stemmer()
stop_words = set(urlopen('https://gist.githubusercontent.com/xxMrPHDxx/d7244906ff8159f6e030790162542524/raw/5ce745520a75af453eda44286ea75bf310a4702f/stopword-malay.txt').read().decode('utf-8').split('\n'))
articles = get_articles()
category_map = {corpus.replace('-', ' '): i for i, corpus in enumerate(listdir('articles'))}

df = pd.DataFrame()
rows = []
counter = 0
def load_article(filename):
	global rows
	global counter
	global articles
	_, Category, File_Name = filename.split('/')
	Category = Category.replace('-', ' ')
	try: Content = open(filename, 'r', encoding='utf-8').read()
	except Exception: Content = open(filename, 'r').read()

	# Special character cleaning
	Content_Parsed = Content.replace("\r", " ").replace("\n", " ").replace("    ", " ")

	# Double quotes removal
	Content_Parsed = Content_Parsed.replace('"', '')

	# Lowercasing all letters
	Content_Parsed = Content_Parsed.lower()

	# Removing punctuation signs
	punctuation_signs = list("?:!.,;")
	for punct_sign in punctuation_signs:
		Content_Parsed = Content_Parsed.replace(punct_sign, '')

	# Stemming/Lemmatization process
	Content_Parsed = stemmer.stem(Content_Parsed)

	# Removing stop words
	for stop_word in stop_words: Content_Parsed = Content_Parsed.replace(r'\b{}\b'.format(stop_word), '')

	# Replacing consequtive whitespaces more than 1 with a single space
	Content_Parsed = re.sub(r'\s\s+', ' ', Content_Parsed)

	Category_Code, News_length = category_map[Category], len(Content)
	Complete_Filename = filename

	rows.append((File_Name, Content, Content_Parsed, Category, Complete_Filename, Category_Code, News_length))
	counter += 1
	print('Done {}/{}'.format(counter, len(articles)))

if __name__ == '__main__':
	pool = Pool()
	articles = pool.map(load_article, articles)

	File_Name, Content, Content_Parsed, Category, Complete_Filename, Category_Code, News_length = [[] for _ in range(7)]
	for a, b, c, d, e, f, g in rows:
		File_Name.append(a)
		Content.append(b)
		Content_Parsed.append(c)
		Category.append(d)
		Complete_Filename.append(e)
		Category_Code.append(f)
		News_length.append(g)

	df['File_Name'] = File_Name
	df['Content'] = Content
	df['Content_Parsed'] = Content_Parsed
	df['Category'] = Category
	df['Complete_Filename'] = Complete_Filename
	df['Category_Code'] = Category_Code
	df['News_length'] = News_length

	print(df.head())
	with open('Pickle/df.pickle', 'wb') as file: file.write(pickle.dumps(df))