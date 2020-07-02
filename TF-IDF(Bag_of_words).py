
#    Bag of words(with TF-IDF)
import nltk
 
para = """ Natural language processing (NLP) is a subfield of linguistics, computer science, information engineering,
and artificial intelligence concerned with the interactions between computers and human (natural) languages, 
in particular how to program computers to process and analyze large amounts of natural language data.
Challenges in natural language processing frequently involve speech recognition, natural language understanding,
and natural language generation."""

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
sentence = nltk.tokenize(para)
corpus = []
#Cleansing the data
for i in range(len(sentence)):
    review = re.sub('[^a-zA-Z]',' ',sentence[i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word)for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
#TF-IDF model
from sklearn.Feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()
x = tf.fit_transform(corpus).toarray()