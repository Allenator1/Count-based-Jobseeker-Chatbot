import os
import re
import pickle
import numpy as np
import pandas as pd
from string import punctuation, whitespace
from itertools import chain
from collections import defaultdict
from heapq import nlargest

from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from sklearn.metrics.pairwise import cosine_similarity

import nltk
from nltk import word_tokenize, sent_tokenize, edit_distance
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.collocations import *

from utils import get_synonyms, calc_vocab_size, merge_bigrams, retokenize_sents, join_spans

stopwords = set(stopwords.words('english'))
punctuation = set(punctuation)
whitespace = set(whitespace)

JOB_CATEGORY_GROUPS = {'science & technology': ['Information & Communication Technology', 'Science & Technology'],
                        'engineering': ['Engineering', 'Construction', 'Mining, Resources & Energy'],
                        'trade': ['Trade & Services', 'Construction', 'Mining, Resources & Energy'],
                        'office': ['Administration & Office Support', 'Accounting', 'Sales', 'Marketing & Communications', 
                                'Consulting and Strategy', 'Human Resources & Recruitment', 'CEO & General Management'],
                        'health': ['Healthcare & Medical'],
                        'finance': ['Banking & Financial Services', 'Accounting', 'Insurance & Superannuation'],
                        'creative': ['Advertising, Arts & Media', 'Design & Architecture'],
                        'community': ['Community Services & Development', 'Education & Training', 'Sport & Recreation'],
                        'retail': ['Retail & Consumer Products', 'Sales', 'Hospitality & Tourism', 'Call Centre & Customer Service'],
                        'legal': ['Legal'],
                        'civil': ['Government & Defense'],
                        'animals': ['Farming, Animals & Conservation']}


# To do: allow for multiple categories to be evaluated, choosing the best score between multiple tfidf matrices
def find_closest_jobs(user_input, cat_grp,  city=None, company_name=None, job_type=None, score_thresh=0.05):
    df_categories = JOB_CATEGORY_GROUPS[cat_grp]
    pruned_df = jobs_dataframe[jobs_dataframe.category.isin(df_categories)]
    
    if city:
        pruned_df = pruned_df[pruned_df.city.eq(city)]
    if company_name:
        pruned_df = pruned_df[pruned_df.company_name.eq(company_name)]
    if job_type:
        pruned_df = pruned_df[pruned_df.job_type.eq(job_type)]
    
    category_data = preprocessed_data[cat_grp]
    tfidf = category_data['tfidf']
    tfidf_matrix = category_data['tfidf_matrix']
    dictionary = category_data['dictionary']
    df_indices = category_data['df_indices']    # A list of data frame row indices corresponding to the docs in the corpus
    bigrams = category_data['bigrams']
    
    # Filter docs which are invalidated by city, and job_type
    valid_indices = [(i, df_ind) for i, df_ind in enumerate(df_indices) if df_ind in set(pruned_df.index.values)]
    corp_indices, df_indices = zip(*valid_indices)
    tfidf_matrix = [tfidf_matrix[i] for i in corp_indices]
    
    # Construct sparse embeddings
    embeddings = []
    for doc in tfidf_matrix:
        scores_dict = dict(doc)
        embeddings.append([scores_dict.get(id, 0) for id in range(len(dictionary))])
        
    input_tokens = process_doc(user_input, bigrams)
    input_BoW = dictionary.doc2bow(input_tokens)
    input_score_dict = dict(tfidf[input_BoW])
    input_embedding = [[input_score_dict.get(id, 0) for id in range(len(dictionary))]]
    cosine_scores = cosine_similarity(input_embedding, embeddings).flatten()
    job_listing_scores = filter(lambda x: x[1] > score_thresh, zip(df_indices, cosine_scores))
    job_listing_scores = sorted(job_listing_scores, key=lambda x: x[1], reverse=True)
    job_listing_scores = list(job_listing_scores)
        
    get_desc = lambda i: jobs_dataframe['job_description'].loc[i]
    top_score = job_listing_scores[0][1] if job_listing_scores else 0
    summaries = [(df_ind, summarize_description(get_desc(df_ind), bigrams, dictionary, tfidf)) for df_ind, _ in job_listing_scores]
    return summaries, top_score


def find_closest_columns(nlp, jobs, locations=[], companies=[], job_type=''):
    if jobs == []:
        raise ValueError('There must be at least one job description')
    
    out_data = {'categories': [], 'location': None, 'location_score': -1, 'company_name': None, 
                'company_score': -1, 'job_type': None}
    category_docs = [nlp(cat.lower()) for cat in JOB_CATEGORY_GROUPS.keys()]
    job_docs = [nlp(job) for job in jobs]
    category_scores = {cat: max([cat.similarity(job) for job in job_docs]) for cat in category_docs}
    best_cats = sorted(category_scores, key=category_scores.get, reverse=True)
    best_cat = best_cats[0].text
    out_data['category'] = best_cat
         
    if locations != []:
        locations = [loc.title() for loc in locations]
        city_scores = {city: min([edit_distance(loc, city) for loc in locations]) for city in set(jobs_dataframe['city'])}
        best_loc, best_score = sorted(city_scores.items(), key=lambda x: x[1])[0]
        if best_score == 0:
            out_data['location'] = best_loc
        out_data['location_score'] = best_score
    
    if companies != []:
        companies = [comp.title() for comp in companies]
        comp_scores = {df_comp: min([edit_distance(comp, df_comp) for comp in companies]) for df_comp in set(jobs_dataframe['city'])}
        best_company, best_score = sorted(comp_scores.items(), key=lambda x: x[1])[0]
        if best_score < 2:
            out_data['company_name'] = best_company
        out_data['company_score'] = best_score
    
    if job_type != '':
        job_type = job_type.title()
        possible_types = set(jobs_dataframe['job_type'])
        best_job_type = sorted(possible_types, key=lambda x: edit_distance(x, job_type))[0]
        out_data['job_type'] = best_job_type
    
    return out_data
        
    
def preprocess(csv_name, file_name):
    df = pd.read_csv(csv_name).drop_duplicates(subset='job_description')
    df = df[df['job_description'].notnull()]
    preprocessed_data = {}
    print('Preprocessing:\n')
    for i, (group_name, categories) in enumerate(JOB_CATEGORY_GROUPS.items()):
        job_listings = df[df.category.isin(categories)]
        print(f'=========> {i / len(categories) * 100:.2f} / 100 %', end='\r')
        preprocessed_data[group_name] = preprocess_descriptions(job_listings)
    pickle.dump(preprocessed_data, open(file_name, 'wb'))  


def preprocess_descriptions(job_listings, verbose=False):
    job_descriptions = job_listings['job_description']
    
    # Tokenization
    reg = re.compile('[^a-zA-Z0-9 !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]')
    corpus = [word_tokenize(re.sub(reg, ' ', job_desc).lower())
              for job_desc in job_descriptions]
    df_indices = list(job_descriptions.index)
    
    # Lemmatization
    lemmatizer = WordNetLemmatizer()
    lemmatized_corpus = [[lemmatizer.lemmatize(w) for w in doc] for doc in corpus]
    
    # Stemming
    stemmer = SnowballStemmer("english", ignore_stopwords=True)
    stemmed_corpus = [[stemmer.stem(w) for w in doc] for doc in corpus]
    
    # Obtaining collocations
    all_words = chain.from_iterable(lemmatized_corpus)
    bigram_finder = BigramCollocationFinder.from_words(all_words)
    bigram_measures = nltk.collocations.BigramAssocMeasures()
    
    def bigram_filter(*toks):
        if toks[0] in stopwords or toks[-1] in stopwords:
            return True
        if np.any([re.search('[^a-zA-Z0-9]', tok) for tok in toks]):
            return True
    
    bigram_finder.apply_ngram_filter(bigram_filter)
    bigrams = set(bigram_finder.above_score(bigram_measures.student_t, 4.5))
    
    # Removing punctuation
    for doc in lemmatized_corpus:
        for i, token in enumerate(doc):
            reg = re.compile('[^a-zA-Z0-9]')
            tokens = re.split(reg, token)[::-1]
            doc[i] = tokens[0]
            for tok in tokens[1:]:
                doc.insert(i, tok)
    
    # Further filtering
    lemmatized_corpus = [[w for w in doc if w.isalpha() and len(w) > 1] 
                       for doc in lemmatized_corpus]
    
    # Merge collocations in tokens
    corpus_merged_bigrams = []
    num_bigrams = 0
    for doc in lemmatized_corpus:
        merged_tokens, num_merged = merge_bigrams(doc, bigrams)
        corpus_merged_bigrams.append(merged_tokens)
        num_bigrams += num_merged     
    
    # Compute tfidf matrix
    dictionary = Dictionary()
    BoW_corpus = [dictionary.doc2bow(doc, allow_update=True) for doc in corpus_merged_bigrams]
    tfidf = TfidfModel(BoW_corpus, smartirs='ntc')
        
    if verbose:
        print(f'Initial vocabulary size is {calc_vocab_size(corpus)}')
        print(f'Vocabulary size after lemmatization is {calc_vocab_size(lemmatized_corpus)}')
        print(f'Vocabulary size after stemming is {calc_vocab_size(stemmed_corpus)}')
        print('Total number of merged bigrams:', num_bigrams)  
        print(f'Vocabulary size after merging collocations is {calc_vocab_size(corpus_merged_bigrams)}')
        
    return {'tfidf': tfidf, 'tfidf_matrix': tfidf[BoW_corpus], 'dictionary': dictionary, 'df_indices': df_indices, 'bigrams': bigrams}


def process_doc(input_str, bigrams):
    # Tokenization
    reg = re.compile('[^a-zA-Z0-9 !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]')
    tokens = word_tokenize(re.sub(reg, '', input_str).lower())
    
    # Lemmatization 
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(tok) for tok in tokens]
    
    # Removing punctuation
    for i, token in enumerate(tokens):
        reg = re.compile('[^a-zA-Z0-9]')
        new_toks = re.split(reg, token)[::-1]
        tokens[i] = new_toks[0]
        for tok in new_toks[1:]:
            tokens.insert(i, tok)
            
    # Alphanumeric filtering
    tokens = [tok for tok in tokens if tok.isalpha() and len(tok) > 1] 
    
    # Merging collocations
    tokens, _ = merge_bigrams(tokens, bigrams)
    return tokens


def summarize_description(description, bigrams, dictionary, tfidf):
    reg = re.compile('[^a-zA-Z0-9 !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]')
    description = re.sub(reg, '', description)
    
    word_weightings = {'role': 1.25, 'experience': 1.25, 'skill': 1.25, 'require': 1.5, 'contact': 2, 'email': 2.25}
    
    BoW = dictionary.doc2bow(process_doc(description, bigrams))
    tfidf_scores = dict(tfidf[BoW])
    word2score = {dictionary[key]: score for key, score in tfidf_scores.items()}
    sentences = sent_tokenize(description)
    sentences = retokenize_sents(sentences, max_sent_length=200)
    
    sentence_scores = defaultdict(int)
    for doc in sentences[1:-1]:
        tokens = process_doc(doc, bigrams)
        for tok in tokens:
            sentence_scores[doc] += word2score.get(tok, 0)
            for syn in get_synonyms(tok.lower()):
                sentence_scores[doc] *= word_weightings.get(syn, 1)
    
    summarized_spans = nlargest(3, sentence_scores, key=sentence_scores.get)
    if sentences[0] not in summarized_spans:
        summarized_spans.insert(0, sentences[0])
    if sentences[-1] not in summarized_spans:
        summarized_spans.append(sentences[-1])
    return join_spans(summarized_spans)
               

def test():
    category_data = preprocessed_data['Information & Communication Technology']
    tfidf = category_data['tfidf']
    tfidf_matrix = category_data['tfidf_matrix']
    dictionary = category_data['dictionary']
    df_indices = category_data['df_indices']
    bigrams = category_data['bigrams']
    
    embeddings = []
    for doc in tfidf_matrix:
        scores_dict = dict(doc)
        embeddings.append([scores_dict.get(id, 0) for id in range(len(dictionary))])
    
    input_str = 'Fintech data analyst with experience in cryptocurrency analysis'
    tokens = process_doc(input_str, bigrams)
    
    input_BoW = dictionary.doc2bow(tokens)
    input_score_dict = dict(tfidf[input_BoW])
    input_embedding = [[input_score_dict.get(id, 0) for id in range(len(dictionary))]]
    
    cosine_scores = cosine_similarity(input_embedding, embeddings).flatten()
    doc_scores = dict(zip(df_indices, cosine_scores))
    
    best_df_indices = sorted(doc_scores.keys(), key=doc_scores.get, reverse=True)[:50]
    for d in best_df_indices:
        print('----------Cosine Score------------', doc_scores[d])
        best_description = jobs_dataframe['job_description'].loc[d]
        print(summarize_description(best_description, bigrams, dictionary, tfidf))

pickle_file_name = 'preprocessed_category_groups.pkl'
if os.path.exists(pickle_file_name):
    jobs_dataframe = pd.read_csv('seek_australia.csv').drop_duplicates(subset='job_description')
    jobs_dataframe = jobs_dataframe[jobs_dataframe['job_description'].notnull()]
    preprocessed_data = pickle.load(open(pickle_file_name, 'rb'))
else:
    preprocess(csv_name='seek_australia.csv', file_name=pickle_file_name)