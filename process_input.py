import spacy
from spacy.matcher import PhraseMatcher
from utils import get_synonyms, remove_punkt, print_deps

synonyms = {'job': get_synonyms('job'), 
            'experience': get_synonyms('experience')}             


def preprocess_input(doc, nlp):
    matcher = PhraseMatcher(nlp.vocab)
    with doc.retokenize() as retokenizer:
        # Merging full time and part time into single token
        def on_match(_, doc, id, matches):
            span = doc[matches[id][1]:matches[id][2]]
            employment_type = remove_punkt(span.text)
            attrs = {"TEXT": employment_type, "LEMMA": employment_type, 
                     'POS': 'ADJ', 'TAG': 'JJ', 'DEP': 'amod'}
            retokenizer.merge(span, attrs=attrs)
        
        terms = ['full-time', 'full time', 'part-time', 'part time']
        patterns = [nlp.make_doc(text) for text in terms]
        matcher.add("employment_types", patterns, on_match=on_match)
        matcher(doc)
        
    with doc.retokenize() as retokenizer:
        # Merging compound words into single token    
        for token in doc:
            j = token.i
            if j + 1 < len(doc):
                while doc[j].dep_ == "compound" and doc[j + 1].text not in synonyms['job']:
                    j += 1
                if j > token.i:
                    attrs = {"LEMMA": doc[token.i: j + 1].text}
                    try:
                        retokenizer.merge(doc[token.i: j + 1], attrs=attrs)
                    except ValueError as e:
                        print('Could not merge tokens: ', str(e))
      

def extract_intent(doc, user_prefs):
    for token in doc:
        # EXTRACTING EMPLOYMENT TYPE
        # E.g., I want to work part-time
        if token.lemma_ in ['full time', 'part time', 'casual', 'casually', 'contract']:
            user_prefs['JOB_TYPE'] = token.text
        
        # EXTRACTING DESIRED JOB ATTRIBUTES
        # E.g., I want a job in software engineering in Seattle
        if 'NN' in token.tag_:
            compound_attribute = token.dep_ == 'compound' and doc[token.i + 1].text in synonyms['job']  
            not_experience = 'experience' not in set(tok.lower_ for tok in token.ancestors)
            
            if token.dep_ == 'pobj' and not_experience:
                if token.ent_type_ == 'ORG' and doc[token.i - 1].text not in ['a', 'an']:
                    user_prefs['COMPANY'].append(token.text)
                elif token.ent_type_ == 'GPE' and doc[token.i - 1].text == 'in':
                    user_prefs['LOCATION'].append(token.text)
                elif token.head.text in ['for', 'in', 'to', 'as']:
                    job_words = [adj.text for adj in token.children if adj.dep_ == 'amod'] + [token.text]
                    user_prefs['JOB'].append(' '.join(job_words))
                    
            # E.g., I want a  software engineering job      
            elif compound_attribute or token.dep_ == 'npadvmod' or token.dep_ == 'attr':
                if token.ent_type_ == 'ORG':
                    user_prefs['COMPANY'].append(token.text)
                elif token.ent_type_ == 'GPE':
                    user_prefs['LOCATION'].append(token.text)
                else:
                    job_words = [adj.text for adj in token.children if adj.dep_ == 'amod'] + [token.text]
                    user_prefs['JOB'].append(' '.join(job_words))
        
        # EXTRACTING AMOUNT OF EXPERIENCE
        # E.g., I have 10 years of experience in bakery
        is_correct_head = token.text == 'years' or token.text == 'months' \
                            or token.text in synonyms['experience']
        if is_correct_head and token.dep_ == 'dobj':
            subtree = list(token.subtree)
            experiences = doc[subtree[0].i : subtree[-1].i + 1]
            user_prefs['EXPERIENCE'].append(experiences.text)
        
        # EXTRACTING LIST OF EXPERIENCES 
        # E.g., My experience includes sql, database, and c++
        if token.dep_ == 'nsubj' and token.text in synonyms['experience']:
            [dobj] = [child for child in token.head.rights if child.dep_ == 'dobj']
            subtree = list(dobj.subtree)
            experiences = doc[subtree[0].i : subtree[-1].i + 1]
            user_prefs['EXPERIENCE'].append(experiences.text)


if __name__ == '__main__':
    nlp = spacy.load('en_core_web_sm')
    doc = nlp('I have 10 years of experience')
    print_deps(doc)
    user_prefs = {
        'JOB': [],
        'COMPANY': [],
        'LOCATION': [],
        'EXPERIENCE': [],
        'JOB_TYPE': ''
    } 
    preprocess_input(doc, nlp)
    extract_intent(doc, user_prefs)
    print(user_prefs)
            