from urllib.request import urlopen, Request
from bs4 import BeautifulSoup as bs
from os import mkdir
from os.path import exists
from base64 import b64encode
import re

# Multi-threading stuff
from time import time
from queue import Queue
from threading import Thread

def open_url(url, headers={}, auth=None, encoding='utf-8'):
	if auth != None and isinstance(auth, str): headers['Authorization'] = auth
	headers['User-Agent'] = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.11 (KHTML, like Gecko) Chrome/23.0.1271.64 Safari/537.11'
	while True:
		try: return bs(urlopen(Request(url, headers=headers)).read().decode(encoding), 'html.parser')
		except Exception as e: print('Retrying "{}" in cause of error {}'.format(url, e.reason))

def basic_auth(username, password):
	return 'Basic {}'.format(str(b64encode(bytes('{}:{}'.format(username, password), 'utf-8')))[2:-1])

def get_article(url):
	url = 'https://staging.bharian.com.my{}'.format(url)
	return ' '.join(map(lambda x: x.text, filter(lambda x: len(x.text) > 30, open_url(url, auth=basic_auth('nstp', 'iamnstp')).find_all('p')[:-1])))

def has_key(key):
	def inner(obj):
		try: 
			if obj[key]: return True
		except Exception: return False
	return inner

def get_key(key):
	return lambda obj: obj[key] if has_key(key)(obj) else None

def text_matcher(regex):
	return lambda text: re.match(regex, text)

def make_path(path):
	cp = ''
	for i, folder in enumerate(path.split('/')):
		if i != 0: cp += '/'
		cp += folder
		if not exists(cp): mkdir(cp)

def save_article(url, title, full_path, filename):
	with open(full_path, 'w', encoding='utf-8') as file:
		article = get_article(url)
		file.write(article)
		return url, title, article, 1

article_link_regex = r'^/(.+?/.+?)/(\d{4}/\d{2})/(\d+?)/(.+)$'
def get_article_links(query, page=0):
	while page < 21: # 21 pages including page 0
		url = 'https://www.bharian.com.my/search?s={}{}'.format(query, '&page={}'.format(page) if page>0 else '')
		for url in set(filter(text_matcher(article_link_regex), map(get_key('href'), filter(has_key('href'), open_url(url).find_all('a'))))):
			corpus, date, id, title = text_matcher(article_link_regex)(url).groups()
			if len(corpus.split('/')) != 2: continue
			filepath = 'articles/{}'.format(corpus.replace('/', '-'))
			filename = '{}_{}_{}.txt'.format(date.replace('/', '-'), id, title)
			full_path = '{}/{}'.format(filepath, filename)
			if exists(full_path): 
				yield url, title, None, 0
				continue
			yield save_article(url, title, full_path, filename)
		page += 1

class Worker(Thread):
	def __init__(self, queue):
		Thread.__init__(self, daemon=True)
		self.queue = queue
		self.start()
	def run(self):
		while True:
			try: 
				for url, title, article, i in get_article_links(query=self.queue.get()):
					print('Got{} article: {}'.format('' if i==1 else ' existing', title))
			finally: self.queue.task_done()

def main():
	ts = time()
	queue = Queue()
	workers = [Worker(queue) for _ in range(8)]

	letters = 'abcdefghijklmnopqrstuvwxyz'
	for i in letters:
		for j in letters:
			for k in letters:
				queue.put(''.join([i, j, k]))
	queue.join()
	print('Done! Took {} seconds'.format(int(time()-ts)))

if __name__ == '__main__':
	main()