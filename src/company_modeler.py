import pandas as pd
import numpy as np

class CompanyModeler:
    def __init__(self, topic_prob, company_labels, company_ratings):
        company_df, self.company_topic_weights = self.aggregate_topics_by_company(topic_prob,
                                                      company_labels,
                                                      company_ratings)
        self.company_df = self.merge_company_demographics(self.company_topic_weights)

    @staticmethod
    def aggregate_topics_by_company(topic_prob, company_labels, company_ratings):
        # Create dataframe of company labels and append topic probs
        topic_cols = ['Topic_{}'.format(i + 1) for i in range(topic_prob.shape[1])]
        topic_df = pd.DataFrame(topic_prob, columns=topic_cols)
        topic_df = pd.concat([company_labels, company_ratings, topic_df], axis=1)
        # Get companies' mean topic probs
        company_topic_weights = topic_df.groupby('Company')[topic_cols].agg('mean')
        company_topic_df_stack = company_topic_weights.stack().reset_index()  # convert indices to cols
        company_topic_df_stack.rename(columns={'level_1': 'Topic',
                                         0: 'Topic_Weight'}, inplace=True)
        # Get companies mean topic ratings
        avg_rating_dict = {}
        for topic in topic_cols:
            avg_rating_dict.update({
                topic: topic_df.groupby('Company').apply(
                    lambda x: np.average(x['Rating'], weights=x[topic]))
            })
        avg_rating_df = pd.DataFrame(avg_rating_dict)
        avg_rating_df_stack = avg_rating_df.stack().reset_index()
        avg_rating_df_stack.rename(columns={'level_1': 'Topic',
                                      0: 'Mean_Rating'}, inplace=True)
        # Merge together company weights and ratings
        final_company_df = company_topic_df_stack.merge(avg_rating_df_stack, how='left', left_on=['Company', 'Topic'],
                                                  right_on=['Company', 'Topic'])
        return final_company_df, company_topic_weights

    @staticmethod
    def merge_company_demographics(company_df):
        company_df.reset_index(inplace=True)
        import pdb; pdb.set_trace()
        fortune_1000 = pd.read_csv('Fortune1000_Demographics.csv', encoding="latin")
        company_df = company_df.merge(fortune_1000, how="left",
                                      left_on='Company',
                                      right_on="title")
        # Find missing merge links and drop
        temp_boolean = pd.isnull(company_df['rank'])
        company_df = company_df.loc[temp_boolean, :]
        return company_df


