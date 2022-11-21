# /content/txt_csv.csv input file
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import gensim
from gensim import corpora
from rouge import Rouge
import json


def func(df=pd.read_csv('Data/txt_csv.csv', encoding='utf-8'), num_topics2=3, corpus2=None):

    # df = pd.read_csv(file_in)
    corpus = []
    if corpus2 is None:
        for i in range(0, len(df)):
            corpus.append(df.loc[i][0])
    else:
        corpus = corpus2
    # else:
    #     corpus = []
    #     for i in range(0, len(corpus2)):
    #         corpus.append(corpus2.loc[i][0])

    # Apply Preprocessing on the Corpus
    # nltk.download('stopwords') # one time execution
    # nltk.download('wordnet') # one time execution
    # nltk.download('omw-1.4') # one time execution

    # stop loss words
    stop = set(stopwords.words('english'))

    # punctuation
    exclude = set(string.punctuation)

    # lemmatization
    lemma = WordNetLemmatizer()

    # One function for all the steps:
    def clean(doc):

        # convert text into lower case + split into words
        stop_free = " ".join([i for i in doc.lower().split() if i not in stop])

        # remove any stop words present
        punc_free = ''.join(ch for ch in stop_free if ch not in exclude)

        # remove punctuations + normalize the text
        normalized = " ".join(lemma.lemmatize(word)
                              for word in punc_free.split())
        return normalized

    # clean data stored in a new list
    clean_corpus = [clean(doc).split() for doc in corpus]

    # Creating the term dictionary of our courpus that is of all the words (Sepcific to Genism syntax perspective),
    # where every unique term is assigned an index.

    dict_ = corpora.Dictionary(clean_corpus)

    # print(dict_)

    # for i in dict_.values():
    #     print(i)

    # Converting list of documents (corpus) into Document Term Matrix using the dictionary
    doc_term_matrix = [dict_.doc2bow(i) for i in clean_corpus]

    # Creating the object for LDA model using gensim library

    Lda = gensim.models.ldamodel.LdaModel

    # Running and Training LDA model on the document term matrix.

    ldamodel = Lda(doc_term_matrix, num_topics=num_topics2, id2word=dict_,
                   passes=1, random_state=0, eval_every=None)

    # Prints the topics with the indexes: 0,1,2 :

    topic_words = ldamodel.print_topics()

    # we need to manually check whethere the topics are different from one another or not

    # print(ldamodel.print_topics(num_topics=6, num_words=5))

    # num_topics mean: how many topics want to extract
    # num_words: the number of words that want per topic

    jip = []
    for i in topic_words:
        kim = i[1].split('+')
        lark = ''
        for t in kim:
            lark = lark+' '+t.split('*')[1].replace('"', '')
        jip.append(lark)

    # printing the topic associations with the documents
    count = 0
    arr = []
    best_topic = []
    for i in ldamodel[doc_term_matrix]:
        # print("doc : ",count,i)
        temp = []
        for b in range(len(i)):
            temp.append([i[b][1], i[b][0]])
        temp.sort(reverse=True)
        best_topic.append(temp[0][1])
        count += 1

    lis = []
    lis2 = []
    for i in range(len(corpus)):
        res = ''
        # print("        Topic Words for Document", i)
        # print()
        sli = jip[best_topic[i]].split(' ')
        for j in sli:
            res += j + ' '
            print()
        lis2.append(res)
        r = Rouge()
        scores = r.get_scores(jip[best_topic[i]], corpus[i], avg=True)
        lis.append(json.dumps(scores, indent=4, sort_keys=True))
    return lis2, lis


# print(func())
