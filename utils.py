import re
from nltk import Tree, word_tokenize
from nltk.corpus import wordnet, stopwords
from string import punctuation, whitespace
from itertools import chain
punctuation = set(punctuation)
whitespace = set(whitespace)


# Recursive algorithm for generating an NLTK tree from a dependency parse
def to_nltk_tree(node):
    if node.n_lefts + node.n_rights > 0:
        return Tree(f'({node.text},{node.dep_},{node.pos_})', [to_nltk_tree(child) for child in node.children])
    else:
        return f'({node.text},{node.dep_},{node.pos_})'


# prints dependency trees for the sentences in doc
def print_deps(doc):
    [to_nltk_tree(sent.root).pretty_print() for sent in doc.sents]


# Obtains all synonyms of a string  
def get_synonyms(word): 
    synonyms = []
    for syn in wordnet.synsets(word):
        for l in syn.lemmas():
            synonyms.append(remove_punkt(l.name()))
    return set(synonyms)


# Removes punctuation from a string
def remove_punkt(text):
    reg = re.compile('[^a-zA-Z0-9 \n\.]')
    return re.sub(reg, ' ', text).lower()


# Removes stopwords from a string
def remove_stopwords(text):
    tokens = word_tokenize(text)
    return ' '.join([tok for tok in tokens if tok not in stopwords.words('english')])


# Calculates vocabulary size of a corpus
def calc_vocab_size(corpus):
    all_words = chain.from_iterable(corpus)
    reg = re.compile('[^a-zA-Z0-9]')
    vocab = [w for w in all_words if not re.search(reg, w)]
    print('Number of words:', len(vocab))
    return len(set(vocab))


# Returns a list of tokens where bigrams are merged
def merge_bigrams(tokens, bigrams):
    i = 0
    num_merged = 0
    merged_tokens = []
    while i < len(tokens) - 1:
        if (tokens[i], tokens[i + 1]) in bigrams:
            merged_tokens.append(tokens[i] + ' ' + tokens[i + 1])
            num_merged += 1
            i += 2
        else:
            merged_tokens.append(tokens[i])
            i += 1  
    return merged_tokens, num_merged


# Receives a list of sentences and splits sentences with length > max_sent_length
def retokenize_sents(sentences, max_sent_length):
    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sub_sentences = []
        start_j = 0
        for j in range(0, len(sent)):
            if j - start_j > max_sent_length and sent[j] in whitespace:
                sub_sentences.append(sent[start_j:j])
                start_j = j 
        sub_sentences.append(sent[start_j:])  
            
        sentences[i] = sub_sentences[-1]
        for sub_sent in sub_sentences[::-1][1:]:
            sentences.insert(i, sub_sent.strip())
        i += len(sub_sentences)
    return sentences


# Joins a list of sentences into a string
def join_spans(sentences):
    summary = ''
    for sent in sentences[:-1]:
        summary += sent
        if summary[-1] in set([',', '.', '!', '?']):
            summary += ' '
        else:
            summary += '; '
    summary += sentences[-1]
    return summary