# Text Classification (Berita Harian)

This project used dataset scraped from [Berita Harian]('https://www.bharian.com.my') site.
About 32000 articles were scraped and splitted into training and testing datasets.

## Requirements
- Sastrawi
- scikit-learn (sklearn)
- pickle
- numpy
- matplotlib
- seaborn
- pandas

## Installation

Using pip: `pip install Sastrawi scikit-learn pickle numpy matplotlib seaborn pandas`
Using conda: `conda install Sastrawi scikit-learn pickle numpy matplotlib seaborn pandas`

Note: Additional module may be required for those who are using Jupyter Notebook (to display graphics) but I don't actually used it.

## Scrapping

Using search feature in Berita Harian's website, I search through all permutations of A-Z of length 3 as such:
```
letters = 'abcdefghijklmnopqrstuvwxyz'
for i in letters:
	for j in letters:
		for k in letters:
			crawl(query=''.join([i, j, k]))
```

I go through 21 pages for each query since the web apparently stop at a certain page and if proceeded will repeated the previous page.

Note: You could use multiple thread to fasten up the scrape process

## Model training

Model were create and 15% of at least 2000 articles was pre-trained and stored in the "Models" folder.
The model can be loaded using pickle and trained using `gbc.fit(features_train, labels_train)`.
You can see how it's done in 'train.py' Python script.