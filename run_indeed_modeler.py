from initial_analysis.src.indeed_modeler import IndeedModeler
# import numpy as np
# import pandas as pd
col_names = {'Current_Employee': 'Employment_Status',
             'Date': 'Date',
             'Job_Title': 'Title',
             'Location_City': 'City',
             'Location_State': 'State',
             'Rating': 'Rating',
             'Review': 'Review',
             'company': 'Company'}
indeed_model = IndeedModeler('data/all_reviews_new.csv', col_names=col_names,
                             remove_non_english=False)
indeed_model.resample_equal_reviews()
indeed_model.run_review_preprocessing(advanced=True)
indeed_model.compute_word_embeddings(n_comp=200, min_freq=2, min_sg=0)

import pickle
indeed_model = pickle.load(open('data/indeed_model_temp.pickle', 'rb'))

# indeed_model.compute_word_embed_nmf(n_comps=20)
# indeed_model.visualize_nmf_results(1, print_reviews=True)

from corex_topic.corextopic import corextopic as ct
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
def dummy(doc):
    return doc

vectorizer = CountVectorizer(preprocessor=dummy, tokenizer=dummy, token_pattern=None,
                             min_df=3)
doc_term_mat = vectorizer.fit_transform(indeed_model.docs_clean)
doc_term_mat = (doc_term_mat >= 1).astype(np.int_)

# Train the CorEx topic model
topic_model = ct.Corex(n_hidden=20)  # Define the number of latent (hidden) topics to use.
topic_model.fit(doc_term_mat, words=vectorizer.get_feature_names())
anchor_words = [['manager', '']]
topic_model.get_topics(topic=1, n_words=20)






