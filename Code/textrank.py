# **Download globe.6B from http://nlp.stanford.edu/data/glove.6B.zip
import nltk
nltk.download('stopwords')  # one time execution
nltk.download('averaged_perceptron_tagger') # one time execution
nltk.download('punkt')  # one time execution
nltk.download('stopwords')  # one time execution

import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize
#import nltk
# nltk.download('punkt')  # one time execution
# nltk.download('stopwords')  # one time execution
from rouge import Rouge
import json



def textrank2(uploaded=open('Data/tennis_articles_v4.csv', encoding='utf-8'), lines_in=10, idlSummary=open('Data/ideal-summary.txt').read()):

    # Read the CSV file
    df = pd.read_csv(uploaded)

    # split the the text in the articles into sentences
    sentences = []
    for s in df['article_text']:
        sentences.append(sent_tokenize(s))

    # flatten the list
    sentences = [y for x in sentences for y in x]

    # remove punctuations, numbers and special characters
    clean_sentences = pd.Series(sentences).str.replace(
        "[^a-zA-Z]", " ", regex=True)

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]

    stop_words = stopwords.words('english')

    # function to remove stopwords

    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new

    # remove stopwords from the sentences
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]

    # Extract word vectors
    word_embeddings = {}
    f = open('Data/glove.6B/glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,)))
                    for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)

    len(sentence_vectors)

    """The next step is to find similarities among the sentences. We will use cosine similarity to find similarity between a pair of sentences. Let's create an empty similarity matrix for this task and populate it with cosine similarities of the sentences."""

    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])

    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(
                    1, 100), sentence_vectors[j].reshape(1, 100))[0, 0]

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s)
                               for i, s in enumerate(sentences)), reverse=True)

    # Specify number of sentences to form the summary
    sn = lines_in
    summary = ""
    # Generate summary
    for i in range(sn):
        summary += ranked_sentences[i][1]
    r = Rouge()
    # ideal_summary = """Major players believe that too much tennis and insufficient rest will result from a major event in late November mixed with one in January before the Australian Open.With a victory, Nishikori, who is currently in ninth place, may get within 125 points of the cut for the eight-man event in London the following month.The world number three remarked at the Swiss Indoors competition where he will compete in Sunday's final against Romanian qualifier Marius Copil that given the absurdly short amount of time to make a decision, he opted out of any commitment.I felt that the Fed Cup or Olympic weeks, rather not necessarily during the competitions, were the finest weeks I had to get to know players when I was playing.Before going ahead 3-0 in the second set and winning on his first match point, he used his first break point to end the first set.The competition will take the place of the traditional home-and-away matches played four times a year for decades, and will feature 18 countries in the November 18–24 finals in Madrid. In the second set, Anderson was broken twice by the Spaniard, but in the third and final set, the South African's serve was not broken. The relaunched and compressed Davis Cup organisers reportedly gave Roger Federer three days to decide whether or not to participate in the contentious event. The majority of Davenport's success occurred in the late 1990s, and her third and final major championship victory occurred at the 2000 Australian Open. In the redesigned competition, Argentina and Britain will compete alongside the four semifinalists from 2018 and the 12 teams that advance from the qualifying rounds in February. """
    scores = r.get_scores(summary, idlSummary, avg=True)
    print(json.dumps(scores, indent=4, sort_keys=True))
    return summary, scores
