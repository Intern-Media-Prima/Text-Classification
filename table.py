from os import listdir
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from urllib.request import urlopen
from multiprocessing import Pool, Value
import pandas as pd
import pickle
import re

# Multi-threading stuff
import logging
import os
from queue import Queue
from threading import Thread
from time import time
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
# end #

stemmer = StemmerFactory().create_stemmer()
stop_words = set(urlopen('https://gist.githubusercontent.com/xxMrPHDxx/d7244906ff8159f6e030790162542524/raw/5ce745520a75af453eda44286ea75bf310a4702f/stopword-malay.txt').read().decode('utf-8').split('\n'))

class Time:
	def __init__(self, _time=None):
		t = time() if _time == None else _time # in seconds
		ms = int((t - int(t)) * 1000)
		t = int(t)
		s = t % 60
		m = int(t) // 60
		h = m // 60
		m = m % 60
		self.time = h, m, s, ms
	def __sub__(self, o):
		if isinstance(o, float) or isinstance(o, int): o = Time(o)
		if not isinstance(o, Time): raise Exception('Invalid time!')
		res = ()
		for i in range(4): res = (*res, self.time[i] - o.time[i])
		return res

def parse_article(filename):
	Complete_Filename = filename
	try: Content = open(filename, 'r', encoding='utf-8').read()
	except UnicodeDecodeError: Content = open(filename, 'r').read()
	News_length = len(Content)
	_, category, filename = filename.split('/')
	File_Name = filename
	Category = category.replace('-', ' ')
	Category_Code = category_map[category]
	# Special character cleaning
	Content_Parsed = Content.replace("\r", " ")
	Content_Parsed = Content_Parsed.replace("\n", " ")
	Content_Parsed = Content_Parsed.replace("    ", " ")

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
	return File_Name, Content, Content_Parsed, Category, Complete_Filename, Category_Code, News_length

folder = ('articles/{}'.format(i) for i in listdir('articles'))
files = []
for f in folder:
	for file in listdir(f):
		files.append('{}/{}'.format(f, file))
category_map = {c: i for i, c in enumerate(listdir('articles'))}

File_Name, Content, Content_Parsed, Category, Complete_Filename, Category_Code, News_length = [[] for _ in range(7)]
def insert_row(rows):
	global File_Name, Content, Content_Parsed, Category, Complete_Filename, Category_Code, News_length
	a, b, c, d, e, f, g = rows
	File_Name.append(a)
	Content.append(b)
	Content_Parsed.append(c)
	Category.append(d)
	Complete_Filename.append(e)
	Category_Code.append(f)
	News_length.append(g)
	print('Done processing {}/{}: {}'.format(len(File_Name), len(files), File_Name[-1]))

class Worker(Thread):
    def __init__(self, queue):
        Thread.__init__(self)
        self.queue = queue

    def run(self):
        while True:
            try: insert_row(self.queue.get())
            finally: self.queue.task_done()

def main():
    ts = time()

    # Create a queue to communicate with the worker threads
    queue = Queue()
    
    # Create 8 worker threads
    for x in range(8):
        worker = Worker(queue)
        # Setting daemon to True will let the main thread exit even though the workers are blocking
        worker.daemon = True
        worker.start()
    # Put the tasks into the queue as a tuple
    for filename in files:
        # logger.info('Queueing {}'.format(filename))
        queue.put(parse_article(filename))
    # Causes the main thread to wait for the queue to finish processing all the tasks
    queue.join()
    print('Done! Took %s', '{}h {}m {}s {}ms'.format(*Time(time() - ts).time))

if __name__ == '__main__':
	main()

	df = pd.DataFrame()
	df['File_Name'] = File_Name
	df['Content'] = Content
	df['Content_Parsed'] = Content_Parsed
	df['Category'] = Category
	df['Complete_Filename'] = Complete_Filename
	df['Category_Code'] = Category_Code
	df['News_length'] = News_length
	with open('df.pickle', 'wb') as file:
		file.write(pickle.dumps(df))