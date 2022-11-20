from io import StringIO
import pandas as pd
import streamlit as st
import tfidf
import textrank
import LDA

streamlit_style = """
            <style>
            
            footer {visibility: hidden;}
            </style>
            """
st.markdown(streamlit_style, unsafe_allow_html=True)

st.title('AI Project')

tab1, tab2, tab3 = st.tabs(["TFIDF", "Text Rank", "LDA"])


def write_ldaSummary(lda_summary):
    for i, val in enumerate(lda_summary):
        wr = '**Summary for document ' + str(i+1) + '**'
        st.write(wr)
        st.write(val)


def write_ldaScores(lda_scores):
    for i, val in enumerate(lda_scores):
        wr = '**ROUGH for document ' + str(i+1) + '**'
        st.markdown(wr)
        st.json(val)


with tab1:
    st.header('TFIDF')
    uploaded_file = st.file_uploader(
        "**Upload Text File you want to be summariezed**", type='txt')
    percentage_in = -1
    percentage_in = st.number_input("**Percentage of information to retain**", min_value=0,
                                    max_value=100, value=50, format="%d", help="Enter a number from 0 to 100")
    tfidf_summary = ''
    tfidf_summary = st.text_area("Enter Manual Summary", key=0)
    if (uploaded_file is not None) & (percentage_in != -1) & (tfidf_summary != ''):
        tfidf_file = StringIO(uploaded_file.getvalue().decode("utf-8"))
        if (st.button('Get Summary & ROUGE score', key=1)):
            tfidf_hyps, tfidf_scores = tfidf.tfidf_func(
                tfidf_file, percentage_in, tfidf_summary)
            st.markdown('**Summary:**')
            st.write(tfidf_hyps)
            st.markdown('**ROUGE Scores:**')
            st.json(tfidf_scores)
    elif ((uploaded_file is not None) & (percentage_in != -1)):
        if st.button('Get Summary', key=2):
            tfidf_file = StringIO(uploaded_file.getvalue().decode("utf-8"))
            tfidf_hyps, tfidf_scores = tfidf.tfidf_func(
                tfidf_file, percentage_in)
            st.markdown('**Summary:**')
            st.write(tfidf_hyps)


with tab2:
    st.header('Text Rank')
    uploaded_file2 = st.file_uploader(
        "**Upload CSV File you want to be Test Ranked**", type='csv')
    lines_in = -1
    lines_in = st.number_input("**Specify number of sentences to form the summary**:",
                               min_value=0, value=10, format="%d", help="Enter a number")
    textrank_summary = ''
    textrank_summary = st.text_area("Enter Manual Summary", key=8)

    if (uploaded_file2 is not None) & (lines_in != -1) & (textrank_summary != ''):
        if (st.button('Get Summary & ROUGE score', key=3)):
            final_summ, textrank_scores = textrank.textrank2(
                uploaded_file2, lines_in, textrank_summary)
            st.markdown('**Summary:**')
            st.write(final_summ)
            st.markdown('**ROUGE Scores:**')
            st.json(textrank_scores)
    elif uploaded_file2 is not None:
        if st.button('Get Summary', key=4):
            final_summ, scores = textrank.textrank2(
                uploaded_file2, lines_in)
            st.markdown('**Summary:**')
            st.write(final_summ)


with tab3:
    st.header('LDA')
    uploaded_file3 = st.file_uploader(
        "**Upload CSV File for LDA**", type='csv')
    num_topics2 = -1
    num_topics2 = st.number_input(
        "**Number of Topics**", min_value=2, value=3, format="%d", help="Enter a number above 1")
    lda_sumFile = st.file_uploader(
        "**Upload CSV Summary File for LDA**", type='csv')

    if (uploaded_file3 is not None) & (lda_sumFile is not None) & (num_topics2 != -1):
        file_pd = pd.read_csv(uploaded_file3)
        file_pd2 = pd.read_csv(lda_sumFile)
        if len(file_pd) == len(file_pd2):
            if st.button('Get Summary and ROUGE Scores', key=5):
                corpus = []
                for i in range(0, len(file_pd2)):
                    corpus.append(file_pd2.loc[i][0])
                lda_summary, lda_scores = LDA.func(file_pd, corpus)
                write_ldaSummary(lda_summary)
                write_ldaScores(lda_scores)

        else:
            if st.button('Get Summary', key=6):
                lda_summary, lda_scores = LDA.func(file_pd)
                write_ldaSummary(lda_summary)
                st.error(
                    '**Error: Cant display ROUGE Score as number of rows in Summary csv file not same as original document file!**')

    elif (uploaded_file3 is not None) & (num_topics2 != -1):
        if (st.button('Get Summary', key=7)):
            file_pd = pd.read_csv(uploaded_file3)
            lda_summary, lda_scores = LDA.func(file_pd)
            write_ldaSummary(lda_summary)
            # for i, val in enumerate(lda_summary):
            #     wr = '**Summary for document ' + str(i+1) + '**'
            #     st.write(wr)
            #     st.write(val)
            # for i, val in enumerate(lda_scores):
            #     wr = '**ROUGH for document ' + str(i+1) + '**'
            #     st.markdown(wr)
            #     st.json(val, expanded=False)
