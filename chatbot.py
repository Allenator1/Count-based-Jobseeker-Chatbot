import re
import random
from telegram.ext import Updater, MessageHandler, Filters
from itertools import chain
from copy import deepcopy
from process_dataset import find_closest_columns, find_closest_jobs, jobs_dataframe
from process_input import preprocess_input, extract_intent
from utils import print_deps, get_synonyms
import spacy


nlp = spacy.load('en_core_web_md')
synonyms = {'hello': get_synonyms('hello')}
TOKEN = '5253730200:AAGWl7KNYtp5FoxjEC6JkpLCdbtuqWTSVnY'

base_user_prefs = {
    'JOB': [],
    'COMPANY': [],
    'LOCATION': [],
    'EXPERIENCE': [],
    'JOB_TYPE': ''
    }   


def utterance(update, context):
    global user_prefs; global best_match
    
    reg = reg = re.compile('[^a-zA-Z0-9 !"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]')
    msg = update.message.text
    re.sub(reg, msg, '')
    
    if msg.lower() in synonyms['hello']:
        update.message.reply_text("Hello! Welcome to a simple job seeking chat bot. Send a message if you would like to begin a query. " + 
                                    "If you would like to reset your current search query, send 'reset'. If you would like to undo your last message from " +
                                    "your current query, send 'undo'.")
        
    elif msg.lower() == 'reset':
        update.message.reply_text("You have reset your current search query. Send a new message to begin a new search.")
        reset_user_prefs()
        
    elif msg.lower() == 'view':
        if len([key for key in user_prefs.keys() if user_prefs[key]]) > 0:
            send_affirmation(update)
        else:
            update.message.reply_text("Your current query is empty and cannot be viewed. Please send a message to begin a new query.")
        
    elif msg.lower() == 'undo':
        if len(user_prefs_history) < 2:
            update.message.reply_text("Your current query is empty. Please send a message to begin a new query.")
        else:
            print(user_prefs_history)
            user_prefs = user_prefs_history.pop()
            update.message.reply_text("You have undone your previous message from the current query. This is the current state of your query:")
            send_affirmation(update)
            create_query(msg, update)
            
    elif msg.lower() == 'best match':
        send_best_match(update)
        
    elif msg.lower() == 'next':
        if len(matches) == 0:
            update.message.reply_text("There are no other matches remaining. Please send a message to begin a new query.")
        else:
            update.message.reply_text("You have selected to continue to the next best match")
            best_match = matches.pop(0)
            send_best_match(update)
            
    elif msg.lower() == 'stop':
        updater.stop()
        updater.is_idle = False
        
    else:
        if len(user_prefs_history) == 0 or user_prefs != user_prefs_history[-1]:
            user_prefs_history.append(deepcopy(user_prefs))
        create_query(msg, update)
            

def send_best_match(update):
    update.message.reply_text(f"The current best match is a {best_match['job_type']} position called '{best_match['job']}'. You will be working for " + 
                                  f"{best_match['company']}, in {best_match['city']}. Here is a summary of the job:\n\n{best_match['summary']}")
    update.message.reply_text("Please continue sending messages to refine your query. If you would like to view the next best match, send 'next'. " + 
                              "If you would like to start a new query, send 'reset'.")
    

def update_best_matches(closest_jobs):
    global matches; global best_match
    df_indices, top_summaries = zip(*closest_jobs)
    df_arr = jobs_dataframe[['city', 'company_name', 'job_title', 'job_type']].loc[list(df_indices)].to_numpy()
    matches = [{'city': city, 'company': company, 'job': job, 'job_type': job_type, 'summary': top_summaries[i]}
               for i, (city, company, job, job_type) in enumerate(df_arr)]
    best_match = matches.pop(0)


def reset_user_prefs():
    global user_prefs; global user_prefs_history
    user_prefs = deepcopy(base_user_prefs)
    user_prefs_history = []
    
    
    
def create_query(msg, update):
    doc = nlp(msg) 
    preprocess_input(doc, nlp)
    extract_intent(doc, user_prefs)
    unfilled_keys = [key for key in user_prefs.keys() if not user_prefs[key]]
    print(user_prefs)
    
    if len(unfilled_keys) == 5 or user_prefs == user_prefs_history[-1]:
        update.message.reply_text("Sorry, I could not understand your message. Please repeat with different wording.")
        return
        
    # Affirm the information the user has submitted
    send_affirmation(update)
    update.message.reply_text('Processing your query...')
    
    if 'JOB' in unfilled_keys:
        update.message.reply_text("Sorry, I could not decipher the particular job category or occupation name you are looking for. " + 
                                  "Please include this in your next message.")
        return
    
    # Create data to be processed
    user_input = ' '.join(chain(*user_prefs.values()))
    job_vals = [job for job in user_prefs['JOB']]
    loc_vals = [loc for loc in user_prefs['LOCATION']]
    comp_vals = [loc for loc in user_prefs['COMPANY']]
    
    column_data = find_closest_columns(nlp, job_vals, loc_vals, comp_vals, user_prefs['JOB_TYPE'])
    closest_jobs, top_cosine_score = find_closest_jobs(user_input, column_data['category'], column_data['location'], 
                                        column_data['company_name'], column_data['job_type'])
    if len(closest_jobs) > 0: 
        update_best_matches(closest_jobs)
    print('Identified job category:', column_data['category'])
    
    # Present results of the search
    if len(closest_jobs) == 0:
        update.message.reply_text("Sorry, I could not find any jobs that match your previous query. Your query has been reset. Please re-try your query.")
        reset_user_prefs()
        
    elif top_cosine_score < 0.20:
        update.message.reply_text(f"Good news! I have found {len(closest_jobs)} jobs that match your query. Please help refine the list by sending additional " + 
                                  "messages. If you would like to receive the current best match send 'best match'.")
        if column_data['location_score'] >= 2:
            update.message.reply_text(f"- Unfortunately, the locations you supplied do not match any of the jobs I have found. " + 
                                      "Please supply additional locations to narrow your search. Locations should be cities in Australia")
        if column_data['company_score'] >= 3:
            update.message.reply_text(f"- The companies you supplied do not match any of the jobs I have found. " + 
                                      "Please supply additional companies to narrow your search.")
        if unfilled_keys:
            update.message.reply_text("- The following information is missing from your query. Please expand your query to add these " + 
                                        f"attributes: {', '.join(unfilled_keys)}")
            
    else:
        update.message.reply_text("Good news! I have found a job that meets the sufficient similarity criteria.") 
        send_best_match(update)
        update.message.reply_text("Would you like to refine your query? If so, please continue sending messages to add to your query. " + 
                                    "If you would like to start a new query, send 'reset'.")
         
    
def send_affirmation(update):
    sentence_starters = ['it appears,', 'from my understanding,', 'from my predictions,', 'I see,', 'it seems']
    transition_words = ['also,', 'in addition,', 'furthermore,', 'on top of that,', 'as well as that,']
    sentences = []
    
    known_attributes = [key for key, val in user_prefs.items() if type(val) == list and val != []]
    random.shuffle(known_attributes)
    job_type = user_prefs['JOB_TYPE']
    
    for attr in known_attributes:
        if attr == 'JOB':
            for token in user_prefs['JOB']:
                job_phrase = job_type + ' ' + token if job_type else token
                if token.endswith('e') or token.endswith('ing'):
                    ling_pattern = random.choice(['pobj', 'compound', 'npadvmod'])
                    if ling_pattern == 'npadvmod':
                        job_str = f'you want a {job_phrase} based job'
                    else:
                        job_str = produce_base_sentence(ling_pattern, job_phrase, ['pobj', 'compound'])
                else:
                    ling_pattern = random.choice(['pobj', 'compound'])
                    if ling_pattern == 'pobj':
                        job_str = f'you want to work as a {job_phrase}'
                    else:
                        job_str = produce_base_sentence(ling_pattern, job_phrase, ['compound'])
                sentences.append(job_str)
                
        elif attr == 'LOCATION':
            for token in user_prefs['LOCATION']:
                ling_pattern = random.choice(['pobj', 'compound', 'npadvmod', 'attr'])
                if ling_pattern == 'attr':
                    loc_str = f'you want your location to be {token}'
                else:
                    loc_str = produce_base_sentence(ling_pattern, token, ['pobj', 'compound', 'npadvmod'])
                sentences.append(loc_str)
                
        elif attr == 'COMPANY':
            for token in user_prefs['COMPANY']:
                ling_pattern = random.choice(['pobj', 'compound', 'npadvmod'])
                if ling_pattern == 'pobj':
                    loc_str = f'you want to work for {token}'
                elif ling_pattern == 'attr':
                    loc_str = f'you want your company to be {token}'
                comp_str = produce_base_sentence(ling_pattern, token, ['compound'])
                sentences.append(comp_str)

        elif attr == 'EXPERIENCE':
            for token in user_prefs['EXPERIENCE']:
                exp_str = f'you have {token}'.strip('.')
                sentences.append(exp_str)
    
    for i in range(len(sentences)):
        starter = random.choice(sentence_starters)
        sentences[i] = starter + ' ' + sentences[i]
        if i > 0 and i < len(sentences) - 1:
            transition = random.choice(transition_words)
            sentences[i] = transition + ' ' + sentences[i]
        elif i == len(sentences) - 1 and len(sentences) > 3:
            sentences[i] = 'finally,' + ' ' + sentences[i]
        sentences[i] = sentences[i][0].upper() + sentences[i][1:]
        text = '. '.join(sentences) + '.'
    update.message.reply_text(text)
    

def produce_base_sentence(pattern, word, possible_patterns):
    if pattern == 'pobj' and 'pobj' in possible_patterns:
        s = f'you want to work in {word}'
    if pattern == 'compound' and 'compound' in possible_patterns:
        s = f'you want a {word} job'
    if pattern == 'npadvmod' and 'npadvmod' in possible_patterns:
        s = f'you want a {word} based job'
    if pattern == 'attr' and 'attr' in possible_patterns:
        s = f'you want to be a {word}'
    return s
         
            
if __name__ == '__main__': 
    reset_user_prefs()
    user_prefs_history = []
    
    updater = Updater(TOKEN, use_context=True)
    updater.dispatcher.add_handler(MessageHandler(Filters.text, utterance))
    updater.start_polling()
    print('Running...')
    updater.idle()