import nltk
from nltk.corpus import brown
##now I as going to get some of the categories sets##
news_set=brown.words(categories='news')
romance_set=brown.words(categories='romance')
religion_set=brown.words(categories='religion')
print(news_set)
print(romance_set)
print(religion_set)