import pandas as pd
from multiprocessing import Pool
from initial_analysis.src.indeed_scraper import get_indeed_reviews
from initial_analysis.src.indeed_scraper import get_number_of_pages
hdr = {'User-Agent': 'Mozilla/5.0',
           'Cache-Control': 'no-cache'}


fortune_500_df = pd.read_csv('data/Fortune_500.csv')
fortune_500_df = fortune_500_df.iloc[42:, :]


def get_company_reviews(company_base_url):
    print(company_base_url)
    company_url = company_base_url + '/reviews'
    try:
        n_pages = get_number_of_pages(company_url, hdr, "all")
    except AttributeError:
        try:
            n_pages = get_number_of_pages(company_url, hdr, "all")
        except AttributeError:
            print('skipping {}'.format(company_base_url))
            return company_base_url

    if n_pages > 250:
            df_tuple = get_indeed_reviews(company_url, pages=250)
    else:
            df_tuple = get_indeed_reviews(company_url, pages="all")

    reviews_df = df_tuple[0]
    reviews_df['company'] = company_base_url
    return reviews_df


p = Pool(10)  # Pool tells how many at a time
company_urls = fortune_500_df.URL
records = p.map(get_company_reviews, company_urls)
p.terminate()
p.join()

