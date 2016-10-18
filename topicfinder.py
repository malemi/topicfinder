import csv
import pickle
from pyspark import SparkContext, SparkConf
from pyspark.mllib.clustering import KMeans, KMeansModel
import word2vec
import numpy as np
import gensim as gs
import string, re
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer
import stop_words


synzero = np.load('data/wiki_iter.m.syn0.npy')
synoneneg = np.load('data/wiki_iter.m.syn1neg.npy')
table = np.load('data/wiki_iter.m.table.npy')

wiki_iter = np.load("data/wiki_iter.m")


with open("data/synzero.bin", 'wb') as zippo:
    pickle.dump(synzero, zippo)

wiki_iter.syn0 = synzero

specializzazioni = dict()
with open('data/REFERTI.csv', 'rb') as csvfile:
    r = csv.reader(csvfile, delimiter=';', quotechar='"')
    for row in r:
        k = row[40].lower()
        specializzazioni[k] = specializzazioni.get(k, 0) + 1


# len(specializzazioni)
#
#
# print(len(specializzazioni), specializzazioni)
#

diagnoses = []
with open('data/REFERTI.csv', 'rb') as csvfile:
    r = csv.reader(csvfile, delimiter=';', quotechar='"')
    for row in r:
        diagnoses.append(row[3])

exclude = set(string.punctuation)
table = string.maketrans("", "")
regex = re.compile('[%s]' % re.escape(string.punctuation))


word_dict = dict()
unigram_lists = []
for sentence in diagnoses:
    for word in sentence.split():
        unigrams = []
        if word not in word_dict:
            try:
                if len(word.split("'")) > 1:
                    for sub_word in len(word.split('')):
                        if len(word) > 2:
                            word = word.lower()
                            word = word.translate(table, string.punctuation).strip()
                            word_dict[word] = wiki_iter[word.lower()]                        
                else:
                    if len(word) > 2:
                        word = word.lower()
                        word = word.translate(table, string.punctuation).strip()
                        word_dict[word] = wiki_iter[word.lower()]
            except:
                continue
            unigrams.append(word)
    unigram_lists.append(unigrams)

def get_sentence_score(unigram_list, model_size, word_dict):
    """
    Compute the word2vec representation of a given sentence by summing the representation of each of its word
    If a word is not found in the word2vec vocabulary, its score is set to zero.
    If the word contains a punctuation mark and it does not exist in the vocabulary, punctuation is removed and the word2vec model is checked again before giving a score of zero
    :param sentence: The full sentence (a string) one would like to get the word2vec score of
    :param model: The word2vec model to use
    :param ngram_dict: A dictionary whose keys are the ngrams and values their word2vec representation
    :param N: The n-grams to use, by default we use unigrams
    :return: The score of input sentence as an array of size `model.vector_size`
    """
    score = np.zeros(model_size)
    for ngram in unigram_list:
        score += word_dict.get(ngram, np.zeros(model_size))
    #
    return score


sentence_scores = []
for unigram_list in unigram_lists:
    sentence_scores.append(get_sentence_score(unigram_list, 300, word_dict))



# Save Queries representation on a special file
with open("scratch/sentence_scores.txt", "w") as f:
    for score in sentence_scores:
        f.write(" ".join(score.astype("str"))+"\n")


conf = SparkConf().setMaster("local").setAppName("Italian_Clustering")
sc = SparkContext(conf=conf)


string_sentence_scores_rdd = sc.textFile("scratch/sentence_scores.txt")


sentence_scores_rdd = string_sentence_scores_rdd.map(lambda line: np.array([float(x) for x in line.split(' ')]))


clusters = KMeans.train(sentence_scores_rdd, 27, maxIterations=10, initializationMode="random")


# compute the memberships
memberships = sentence_scores_rdd.map(lambda x: clusters.predict(x)).collect()
memberships = np.array(memberships)


groups_members = Counter(memberships).most_common()


def get_common_ngrams(sentences, ngrams=(1, 2), n_keywords=3, stopwords=[]):
    """
    
    """
    
    count_vectorizer = CountVectorizer(analyzer="word",
                                       stop_words=stopwords,
                                       token_pattern="\\b[a-z][a-z]+\\b",
                                       strip_accents="ascii",
                                       lowercase=True,
                                       ngram_range=ngrams)
    # Builds a sparse matrix with docs and terms
    sentences_counts = count_vectorizer.fit_transform(sentences)
    # compute the ngrams frequencies
    vocab = count_vectorizer.get_feature_names()
    dist = np.sum(sentences_counts.toarray(), axis=0)
    ngram_freq = {}
    for tag, count in zip(vocab, dist):
        ngram_freq[tag] = count
    #
    total_counts = sum(ngram_freq.values())
    results = []
    for word, count in Counter(ngram_freq).most_common()[:n_keywords]:
        results.append((word, count / total_counts))
    return results


results = defaultdict(dict)
# compute the top ngrams
for i, (group, members) in enumerate(groups_members):
    mask = memberships == group
    # find the most common keywords per group
    sentences = np.array(diagnoses)[mask]
    ngrams_keywords = get_common_ngrams(sentences, stopwords=stop_words.get_stop_words('it'))
    results[group]["counts"] = len(sentences)
    results[group]["ngrams_keywords"] = ngrams_keywords
    results[group]["sentences"] = np.random.choice(sentences, size=10, replace=False)
    print(results[group])
    with open("/clustering_results.pickle", "wb") as f:
        pickle.dump(results, f)


