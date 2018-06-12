# function to plot the top k most frequent terms in each class

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns

def Top_K_ngrams(k, ngram, df, text_col, class_col):
    vect=TfidfVectorizer(ngram_range=(ngram,ngram))
    vect.fit(df[text_col])
    tfidf_df=pd.DataFrame(vect.transform(df[text_col]).toarray())
    tfidf_df.columns=vect.get_feature_names()
    top_K_ngrams_by_class=dict()
    for v in df[class_col].value_counts().index:
        freq_in_class=tfidf_df[df[class_col]==v].sum(axis=0).sort_values(ascending=False)
        frac_in_class=freq_in_class/freq_in_class.sum()
        top_K_ngrams_by_class[v]=frac_in_class[:k].index
        print ('the top {} frequent {}-gram terms for class {}:'.format(k,ngram, v))
        sns.barplot(y=frac_in_class[:k].index, x=frac_in_class[:k])
        plt.ylabel('{}-gram terms'.format(ngram))
        plt.xlabel('fraction')
        plt.show()
    return top_K_ngrams_by_class


# function to plot the top k most frequent nouns in each class
from nltk.tag.perceptron import PerceptronTagger

def Top_K_nouns(k, df, text_col, class_col, plot=False):
    vect=TfidfVectorizer()
    vect.fit(df[text_col])
    tfidf_df=pd.DataFrame(vect.transform(df[text_col]).toarray())
    tfidf_df.columns=vect.get_feature_names()
    tfidf_T=tfidf_df.transpose()
    tagger = PerceptronTagger()
    tfidf_T['pos']=tagger.tag(tfidf_T.index)
    tfidf_T=tfidf_T[tfidf_T['pos'].apply(lambda tup:tup[1] in ['NN','NNP','NNS','NNPS'])]
    tfidf_df=tfidf_T.drop(['pos'], axis=1).transpose()
    top_k_by_class=dict()
    for v in df[class_col].value_counts().index:
        freq_in_class=tfidf_df[df[class_col]==v].sum(axis=0).sort_values(ascending=False)
        frac_in_class=freq_in_class/freq_in_class.sum()
        top_k_by_class[v]=frac_in_class[:k].index
        
        if plot:
            print ('the top {} frequent nouns for class {}:'.format(k,v))
            plt.figure(figsize=(5, 10))
            sns.barplot(y=frac_in_class[:k].index, x=frac_in_class[:k])
            plt.xlabel('fraction')
            plt.show()

    return (top_k_by_class)       
    

from nltk.tag.perceptron import PerceptronTagger

def Top_K_verbs(k, df, text_col, class_col, plot=False):
    vect=TfidfVectorizer()
    vect.fit(df[text_col])
    tfidf_df=pd.DataFrame(vect.transform(df[text_col]).toarray())
    tfidf_df.columns=vect.get_feature_names()
    tfidf_T=tfidf_df.transpose()
    tagger = PerceptronTagger()
    tfidf_T['pos']=tagger.tag(tfidf_T.index)
    tfidf_T=tfidf_T[tfidf_T['pos'].apply(lambda tup:tup[1] in ['VB','VBD','VBG','VBN'])]
    tfidf_df=tfidf_T.drop(['pos'], axis=1).transpose()
    top_k_by_class=dict()
    for v in df[class_col].value_counts().index:
        freq_in_class=tfidf_df[df[class_col]==v].sum(axis=0).sort_values(ascending=False)
        frac_in_class=freq_in_class/freq_in_class.sum()
        top_k_by_class[v]=frac_in_class[:k].index
        
        if plot:
            print ('the top {} frequent nouns for class {}:'.format(k,v))
            plt.figure(figsize=(5, 10))
            sns.barplot(y=frac_in_class[:k].index, x=frac_in_class[:k])
            plt.xlabel('fraction')
            plt.show()

    return (top_k_by_class)   