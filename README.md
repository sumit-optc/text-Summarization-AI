# How to Run

```
pip install -r requirements.txt
streamlit run app.py
```

# [Text Summarizer](https://en.wikipedia.org/wiki/Automatic_summarization)

## Description

Text Summary tool - a project which was part of Artificial Intelligence course at BITS Pilani

## Algorithms

### 1. [TFIDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf)

This method of statistical extraction, known as term frequency-inverse document frequency, compares the frequency of words in one document with the inverse percentage frequency of that word in other documents. It means that if a term appears frequently in a document, the user may infer that the word is significant to the document. However, a word has no significance at all if it regularly appears in other papers as well.

A text document is used as the system's first input. Then, using regular expressions, text preprocessing is used to remove special characters and digits from the text. POS Only nouns and verbs are labeled using the approach's tagging system. The text is divided into a number of tokens during tokenization. Sentences serve as tokens for words, and paragraphs serve as tokens for sentences. Both verbs and nouns are used to determine the frequency score. The scores of sentences are now determined using the TF-IDF method. The most pertinent sentences for the generation of the summary are chosen based on those ratings. The precise summary is then created by combining a few selected sentences.

### 2. [Text Rank](https://ieeexplore.ieee.org/document/7726248)

The widely used method TextRank doesn't require any prior linguistic or subject-matter expertise. TextRank is an unsupervised method for extracting information from texts and creating summaries.

The system's input articles are first combined. Sentences are then separated from the text that was produced at this stage. After stopwords are eliminated during the text preparation stage, vector representation is applied for each and every phrase that is produced. Based on the overlapping content of the sentences, specific similarity metrics are employed to assess how similar the sentences are to one another. Cosine The TextRank technique is suggested using a similarity metric.

The similarity matrix uses similarity scores as storage. This method transforms the similarity matrix into graphs, where the edges of the graph stand in for the semantic connections connecting the phrases, and the nodes of the graph represent the sentences that are contained in the documents. The weighted edges in the graph are equivalent to the similarity between the nodes. Following the computation of the similarity scores, the top-ranked sentences are included in the final summary.

### 3. [LDA](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)

An unsupervised method based on a probabilistic algorithm, Latent Dirichlet Allocation is frequently used for topic modeling. The methods for comprehending, organizing, and producing summaries of lengthy publications are provided by topic modeling. Preprocessing the text after text data extraction from the documents is the initial stage in this methodology. Preprocessing involves cleaning up text data, removing stop words, and applying LDA for topic modeling as the following step. As it divides the text document into topic clusters that are based on probability distribution to indicate the relevance of the issue with reference to the document, LDA portrays the documents as the combination of topics.

The key sentences from the original text document are contained in the clusters that were created. Each cluster would be connected to the determined pertinent themes. In order to maximise coverage, the source document's sentences are categorised into many established subjects. The ability to choose topics from the paper to summarise improves the procedure. In order to provide a summary, the LDA approach employed in this paper isolates the five topics that are distinct from one another.
