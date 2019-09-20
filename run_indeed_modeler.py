from src.indeed_modeler import IndeedModeler
from src.company_modeler import CompanyModeler
import numpy as np
import pandas as pd
import json

# col_names = {'Current_Employee': 'Employment_Status',
#              'Date': 'Date',
#              'Job_Title': 'Title',
#              'Location_City': 'City',
#              'Location_State': 'State',
#              'Rating': 'Rating',
#              'Review': 'Review',
#              'company': 'Company'}
# indeed_model = IndeedModeler('data/all_reviews.csv', col_names=col_names)

import pickle
indeed_model = pickle.load(open('data/indeed_model_temp.pickle', 'rb'))
# indeed_model.compute_word_embeddings()


# anchor_words = json.load(open('anchor_words.json', 'r'))
# anchor_words_list = list(anchor_words.values())
# topic_model = indeed_model.topic_model(20, anchor_words_list,
#                                        anchor_param=5)
topic_model = pickle.load(open('data/temp_topic_model.pickle', 'rb'))

company_model = CompanyModeler(topic_model.p_y_given_x,
                               indeed_model.reviews['Company'],
                               indeed_model.reviews['Rating'])

