import re
from nltk import word_tokenize, FreqDist
from nltk.corpus import stopwords
import pandas


# Generate the histogram of job categories for the given city
def job_histogram(job_listings, city_name):
    job_listings = job_listings[job_listings['city'] == city_name.capitalize()]
    job_categories = set(job_listings['category'])
    histogram = {}
    for cat in job_categories:
        histogram[cat] = sum(job_listings['category'] == cat)
    return histogram  


# TODO: Construct a matrix showing the similarities of vocabularities between job categories
def find_similarity_matrix(job_listings):
    categories = set(job_listings['category'])
    pass
    

# Generate word frequencies for the given category
def category_vocabulary(job_listings, category):
    stops = set(stopwords.words('english'))
    reg = re.compile('[^a-zA-Z0-9 ]')
    job_listings = job_listings[job_listings['category'] == category]
    all_words = []
    for job_desc in job_listings['job_description']:
        if type(job_desc) == str:
            job_desc = re.sub(reg, ' ', job_desc).lower()
            job_words = [tok for tok in word_tokenize(job_desc) 
                         if tok not in stops]
            all_words += job_words
    all_words = map(lambda x: x.lower(), all_words)
    return FreqDist(all_words).most_common(30)
    

if __name__ == "__main__":
    df = pandas.read_csv('seek_australia.csv')
    print(df.info())
    print(job_histogram(df, 'Sydney'))
    print(category_vocabulary(df, 'Government & Defence'))
    
    