from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from urllib.request import urlopen
from os import listdir
import pandas as pd
import numpy as np
import pickle
import re

def parse(article):
	# Special character cleaning
	parsed = article.replace("\r", " ").replace("\n", " ").replace("    ", " ")

	# Double quotes removal
	parsed = parsed.replace('"', '')

	# Lowercasing all letters
	parsed = parsed.lower()

	# Removing punctuation signs
	punctuation_signs = list("?:!.,;")
	for punct_sign in punctuation_signs:
		parsed = parsed.replace(punct_sign, '')

	# Stemming/Lemmatization process
	parsed = stemmer.stem(parsed)

	# Removing stop words
	for stop_word in stop_words: parsed = parsed.replace(r'\b{}\b'.format(stop_word), '')

	# Replacing consequtive whitespaces more than 1 with a single space
	return re.sub(r'\s\s+', ' ', parsed)

category_to_code = {c: i for i, c in enumerate(listdir('articles'))}
code_to_category = {category_to_code[key]: key for key in category_to_code}
stemmer = StemmerFactory().create_stemmer()
stop_words = set(urlopen('https://gist.githubusercontent.com/xxMrPHDxx/d7244906ff8159f6e030790162542524/raw/5ce745520a75af453eda44286ea75bf310a4702f/stopword-malay.txt').read().decode('utf-8').split('\n'))
gbc = pickle.load(open('Models/best_gbc.pickle', 'rb'))

tfidf = TfidfVectorizer(encoding='utf-8',
                        ngram_range=(1, 2),
                        stop_words=None,
                        lowercase=False,
                        max_df=1.,
                        min_df=1,
                        max_features=300,
                        norm='l2',
                        sublinear_tf=True)

test = pd.Series([parse(article) for article in ['''
	JAKARTA: Kerajaan terpaksa membayar hampir RM14 bilion untuk faedah bagi hutang 1Malaysia Development Berhad (1MDB) sehingga tahun depan, kata Timbalan Menteri Kewangan, Datuk Amiruddin Hamzah.
	Beliau berkata, jumlah pembayaran faedah 1MDB itu sepatutnya dapat digunakan untuk pelbagai program bagi manfaat rakyat.
	“Kita berharap tidak lagi berlaku (tindakan menyeleweng) kepada mereka yang diberikan amanah untuk mentadbir negara,” katanya pada sesi ramah mesra bersama rakyat Malaysia di Kedutaan Besar Malaysia di Jakarta, malam ini.
	“Setakat ini, kita hanya membayar faedah hutang 1MDB, (hutang) pokoknya yang berjumlah RM36 bilion, masih belum dibayar,” katanya.
	Menteri Kewangan, Lim Guan Eng, sebelum ini berkata kerajaan mungkin perlu membayar sehingga RM43.9 bilion lagi bagi menyelesaikan hutang 1MDB itu. – BERNAMA
''', '''
	KOTA BHARU: Kerajaan Persekutuan sudah membayar faedah hampir RM9 bilion bagi hutang 1Malaysia Development Berhad (1MDB), kata Timbalan Menteri Kewangan, Datuk Amiruddin Hamzah.
	Bagaimanapun, beliau berkata, hutang pokok syarikat pelaburan strategik itu berjumlah RM36 bilion masih belum dibayar setakat ini.
	Katanya, kerajaan komited dan berusaha untuk menyelesaikan hutang berkenaan mengikut jadual ditetapkan bagi mengelak reputasi negara terjejas.
	“Setakat ini, kita hanya membayar faedah hutang, yang pokoknya berjumlah RM36 bilion masih belum disentuh (belum dibayar).
	“Pada masa sama, kita memerlukan perbelanjaan pembangunan untuk perkara lain juga. Jadi, kita akan melihat perkara yang menjadi keutamaan.
	“Apabila tiba masanya, kita akan bayar, namun kita juga akan memastikan projek pembangunan lain dapat diteruskan juga,” katanya selepas merasmikan Persidangan Akauntan Sektor Awam Kebangsaan Ke-29 di sini, hari ini.
	Turut hadir, Timbalan Akauntan Negara (Operasi), Dr Yacob Mustafa dan Timbalan Akauntan Negara (Korporat), Datuk Zamimi Awang.
	Menteri Kewangan, Lim Guan Eng ketika membentangkan Belanjawan 2019 pada tahun lalu dilaporkan berkata, kerajaan mendapati negara mungkin perlu membayar sehingga RM43.9 bilion lagi bagi menyelesaikan hutang 1MDB itu.
	Mengulas lanjut, Amiruddin berkata, jaminan kerajaan yang terpaksa membayar hutang 1MDB ini memberi kesan terhadap perbelanjaan pembangunan negara.
	“Alangkah baiknya, wang yang digunakan untuk membayar jaminan kerajaan ini diperuntukkan untuk pembinaan hospital, sekolah, jalan raya dan baik pulih sekolah daif yang akhirnya memberi manfaat kepada rakyat.
	“Saya ingat perkara ini kita dapat lakukan dengan lebih cepat jika tidak diganggu oleh pembayaran hutang yang tidak sepatutnya negara tanggung,” katanya.
'''][:1]]) # [0:1]

features = tfidf.fit_transform(test).toarray()
print(features[0])
if len(features[0]) != 300: 
	features = np.array([np.append(features[0], [0 for _ in range(300 - len(features[0]))])])
print(features.shape)

pred = gbc.predict(features)
print([code_to_category[val] for i, val in enumerate(pred)])