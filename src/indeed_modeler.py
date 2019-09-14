import pandas as pd
import numpy as np
from nltk import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
from string import punctuation
from initial_analysis.src.word2vec_svd import Word2VecSVD
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.preprocessing import MinMaxScaler
from sklearn.mixture import GaussianMixture
from sklearn.feature_extraction.text import TfidfVectorizer
from initial_analysis.src.sea_NMF_sparse import SeaNMFL1
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Initialize lemmatizer
lem = WordNetLemmatizer()


class IndeedModeler:
    def __init__(self, input_csv, col_names=None, min_review=100,
                 remove_non_english=False):
        review_df = pd.read_csv(input_csv)
        # Drop Duplicate Rows
        # Unfortunately, the Indeed web scraper scrapes the same 'top' review
        # for every page, leading to duplicates, must drop identical rows
        # until this is fixed
        review_df = review_df.drop_duplicates()
        # We have standard column names, if your csv does not match these
        # standard names, you must supply a dictionary that translates
        if col_names is not None:
            if len(review_df.columns) < len(col_names):
                raise Exception('The number of column names supplied cannot'
                                ' exceed the number of columns in dataframe')
            else:
                review_df.rename(columns=col_names, inplace=True)
        # Remove companies with less than 'min_reviews' threshold
        company_counts = review_df['Company'].value_counts()
        company_thres = company_counts[company_counts >= min_review].index
        temp_boolean = review_df['Company'].apply(lambda x: x in company_thres)
        self.reviews = review_df.loc[temp_boolean, :]
        # If user specifies, detect non-english reviews, and remove
        # Wayyyy to slow right now
        if remove_non_english:
            print('Removing non-english reviews')
            # Using spacy and 'langdetect' package from spacy
            import spacy
            from spacy_langdetect import LanguageDetector
            # Disable extraneous components of spacy pipeline to speed up
            nlp = spacy.load('en_core_web_md', disable=['ner', 'tagger', 'textcat'])
            # Add language detector
            nlp.add_pipe(LanguageDetector(), name='language_detector', last=True)
            temp_boolean = review_df['Review'].apply(self.language_detect, nlp_pipe=nlp)
            # Remove non-english reviews
            self.reviews = self.reviews.loc[temp_boolean, :]

    def compute_dense_doc_term_matrix(self):
        # dummy function to trick sklearn 'CountVectorizer' to not preprocess
        def dummy(doc):
            return doc

        vocab = list(self.tok2indx.keys())
        vectorizer = TfidfVectorizer(vocabulary=vocab, preprocessor=dummy,
                                     tokenizer=dummy, token_pattern=None)
        doc_term_mat = vectorizer.fit_transform(self.docs_clean)
        doc_term_dense = doc_term_mat * self.word_vecs_norm
        return doc_term_dense

    def compute_word_embeddings(self, n_comp, min_freq, min_sg):
        print('Compute word embeddings from reviews')
        indeed_word2vec = Word2VecSVD(self.docs_clean, n_comp=n_comp,
                                      min_freq=min_freq, min_sg=min_sg)
        self.word_vecs_norm = indeed_word2vec.word_vecs_norm
        self.tok2indx = indeed_word2vec.tok2indx
        self.pmi_mat = indeed_word2vec.pmi_mat

    def cluster_reviews(self, n_clusters):
        print('Cluster reviews, supplemented with word embeddings')
        # Normalize document dense matrix with Max/Min scaler
        scaler = MinMaxScaler()
        doc_term_dense_scaled = scaler.fit_transform(self.doc_term_dense)
        # Cluster with GMM
        gmm = GaussianMixture(covariance_type='full',
                              n_components=n_clusters,
                              verbose=1)
        gmm.fit(doc_term_dense_scaled)
        self.cluster_results = gmm
        self.predicted_probs = gmm.predict_proba(doc_term_dense_scaled)
        # Append cluster probabilities to pandas dataframe
        self.reviews_resampled = pd.concat([self.reviews_resampled,
                                            pd.DataFrame(self.predicted_probs)],
                                           ignore_index=True, axis=1)

    @staticmethod
    def language_detect(text, nlp_pipe):
        # Detect whether language of text is english (or others)
        # If english, return 1, otherwise return 0
        processed_text = nlp_pipe(text)
        if processed_text._.language == 'en':
            return 1
        else:
            return 0

    @staticmethod
    def preprocess_minimal(row, unwanted_terms, stop_words):
        text = row['Review']
        # Provide some progress info
        if row.name % 10000 == 0:
            print('{} reviews preprocessed'.format(row.name))
        # Minimal preprocessing includes: tokenization, lowercase,
        # removal of unwanted terms, punctuation, and non-alphanumeric,
        # and lemmatization
        word_tokens = word_tokenize(text)
        text_clean = [lem.lemmatize(token.lower()) for token in word_tokens
                     if token.lower() not in unwanted_terms
                     if token is not punctuation
                     if token.lower() not in stop_words
                     if token.isalnum()]
        return text_clean

    @staticmethod
    def preprocess_spacy(row, nlp_pipe, unwanted_terms):
        text = row['Review']
        # Provide some progress info
        if row.name % 10000 == 0:
            print('{} reviews preprocessed'.format(row.name))
        # Advanced preprocessing includes: spacy tokenization,
        # lowercase, removal of unwanted terms, removal of nonsense
        # words (i.e. not in SpaCy vocabulary), stopwords, punctuation,
        # spaces and ascii  words
        word_tokens = nlp_pipe.tokenizer(text)
        text_clean = [str(token).lower()
                      for token in word_tokens
                      if token.lower_ not in unwanted_terms
                      if str(token) in nlp_pipe.vocab
                      if not token.is_punct
                      if not token.is_space
                      if token.is_ascii
                      if not token.is_stop]
        return text_clean

    def query_nearest_neighbors(self, query):
        if not hasattr(self, 'nbrs_model'):
            print('No nearest neighbor model detected, running initial model.'
                  'This may take a minute.')
            self.nbrs_model = NN(n_neighbors=10, algorithm='ball_tree').fit(
                self.word_vecs_norm)
        vocab = list(self.tok2indx.keys())
        if query in vocab:
            distances, indices = self.nbrs_model.kneighbors(
                self.word_vecs_norm[self.tok2indx[query], :].reshape(-1, 1).T)
            for dist, indx in zip(distances[0], indices[0]):
                print('{} : {} \n'.format(vocab[indx], dist))
        else:
            print('query token does not exist in vocabulary')

    def resample_equal_reviews(self):
        print('Re-sampling reviews to balance negative and positive reviews')
        reviews_resampled = pd.DataFrame()
        original_rating_counts = {}
        resampled_rating_counts = {}
        companies = self.reviews['Company'].unique()
        # Iterate through companies
        for company in companies:
            print(company)
            company_df = self.reviews.loc[self.reviews['Company'] ==
                                          company, :]
            # Compute number of reviews per rating
            original_rating_counts.update(
                {company: company_df.groupby('Rating').count()})
            # Compute what would be an equal sample size across the 5 ratings
            equal_sample = int(np.floor(company_df.shape[0] / 5))
            # Resample reviews based off an equal sample size across ratings
            company_df_resampled = company_df.groupby('Rating').apply(
                lambda s: s.sample(min(len(s), equal_sample)))
            # Reset index of resampled dataframe
            company_df_resampled.reset_index(drop=True, inplace=True)
            # Compute new number of reviews per rating
            resampled_rating_counts.update(
                {company: company_df_resampled.groupby('Rating').count()})
            # Concatenate resampled dataframe to new dataframe
            reviews_resampled = pd.concat([reviews_resampled, company_df_resampled], ignore_index=True)

        self.reviews_resampled = reviews_resampled
        self.original_rating_counts = original_rating_counts
        self.resampled_rating_counts = resampled_rating_counts

    def run_review_preprocessing(self, advanced=False, unwanted_terms=None):
        print('Preprocess reviews...')
        # If the users has resampled the dataframe to equalize review ratings,
        # use that instead of original reviews
        if hasattr(self, 'reviews_resampled'):
            df = self.reviews_resampled
        else:
            df = self.reviews
        # If no unwanted terms, set as empty list
        if unwanted_terms is None:
            unwanted_terms = []
        # Use SpaCy tokenizer and vocabulary to remove nonsense words
        # Gonna run a bit slower than the minimal pre-processing below
        if advanced:
            import spacy
            # Initialize SpaCy if advance preprocessing is called
            # Using the 'medium' sized english spacy model because
            # of its more extensive vocabulary
            nlp = spacy.load('en_core_web_md')
            doc_clean_all = df.apply(self.preprocess_spacy,
                                     nlp_pipe=nlp,
                                     unwanted_terms=unwanted_terms,
                                     axis=1)
        # If 'advanced' preprocessing is False, use quick
        # NLTK tools to tokenize, preprocess. Note, the tokenization
        # seems to perform poorly in a lot of cases
        else:
            stop_words = set(stopwords.words('english'))
            doc_clean_all = df.apply(self.preprocess_minimal,
                                     unwanted_terms=unwanted_terms,
                                     stop_words=stop_words,
                                     axis=1)

        self.docs_clean = doc_clean_all


    def visualize_cluster_results(self):
        # Get word embedding vocabulary
        vocab = list(self.tok2indx.keys())
        # Visualize Cluster Histogram
        n_clusters = len(self.cluster_results.weights_)
        plt.bar(range(1, n_clusters+1), self.cluster_results.weights_)
        plt.title('Cluster Weights')
        plt.show()
        # Visualize Centroid Cluster Weights with word clouds
        centroids = self.cluster_results.means_
        # Loop through each cluster
        for i in range(n_clusters):
            print('\n Cluster{}'.format(i + 1))
            centroid_cluster = centroids[i, :]
            corr_list = []
            # Loop through each word's vector and correlate with centroid
            for j in range(self.word_vecs_norm.shape[0]):
                corr_mat = np.corrcoef(centroid_cluster, self.word_vecs_norm[j, :])
                corr_list.append(corr_mat[0, 1])
            d = {}
            for a, x in zip(vocab, corr_list):
                d[a] = max(x, 0.0001)
            wordcloud = WordCloud()
            wordcloud.generate_from_frequencies(frequencies=d)
            plt.figure(num=None, figsize=(5, 5))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis("off")
            plt.title('Cluster ' + str(i + 1) + " Term Weights")
            plt.show()

    def visualize_nmf_results(self, comp_indx, print_reviews=False,
                              histogram=False):
        # Get word embedding vocabulary
        vocab = list(self.tok2indx.keys())
        # Visualize Cluster Histogram
        n_comps = self.nmf_res.W1.shape[1]
        if comp_indx > n_comps:
            raise Exception('The chosen component index is larger than'
                            'the number of components')
        elif comp_indx < 1:
            raise Exception('Not a valid index')
        # Visualize histogram
        if histogram:
            plt.hist(self.nmf_res.W1[:, comp_indx-1])
            plt.title('Component Weight Histograms')
            plt.show()
        if print_reviews:
            if hasattr(self, 'reviews_resampled'):
                self.top_topic_reviews(self.reviews_resampled,
                                       comp_indx)
            else:
                self.top_topic_reviews(self.reviews, comp_indx)

        # Visualize word weights with wordcloud
        d = {}
        for a, x in zip(vocab, self.nmf_res.W1[:, comp_indx-1]):
            d[a] = max(x, 0.0001)
        wordcloud = WordCloud()
        wordcloud.generate_from_frequencies(frequencies=d)
        plt.figure(num=None, figsize=(5, 5))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title('Cluster ' + str(comp_indx) + " Term Weights")
        plt.show()


