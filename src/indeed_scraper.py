"""
Indeed scraper
by Kevin Ghorbani
"""
import math

import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_number_of_pages(url_path, hdr, pages):
    """returns number of Indeed review pages"""

    if pages == "all":
        # Need to get first page to find max number of pages
        # Sort Indeed Pages by Date
        # Request URL
        url = url_path + "?start=0&sort=helpfulness"
        payload = requests.get(url, headers=hdr, timeout=5)

        # Organize HTML Code with BeautifulSoup
        page_content = BeautifulSoup(payload.content, "html.parser")

        # Find Maximum Number of Pages
        # Get the number of reviews from a HTML link on the top of the page
        click_num_of_reviews = page_content.find(
            "a", {"data-tn-element": "reviews-viewAllLink"})

        if click_num_of_reviews is None:
            click_num_of_reviews = page_content.find(
                attrs={'class': 'cmp-filter-result'})
            split_text = click_num_of_reviews.text.split(' ')
            num_of_reviews = [int(i) for i in split_text if i.isdigit()][0]
        else:
            split_text = click_num_of_reviews.text.split()
            num_of_reviews = int(split_text[2].replace(',', ''))

        # The pages have 20 reviews each, and this is reflected in the URL
        # link; use this to find num of pages
        num_of_pages = math.ceil(num_of_reviews / 20)
    else:
        try:
            num_of_pages = int(pages)
        except ValueError:
            print('"pages" must be an integer or "all"')
            return -1

    return num_of_pages


def _get_reviews_per_page(url_path, hdr, page_):
    # Get URL of this page
    url = url_path + "?start=" + str(page_ * 20) + "&sort=helpfulness"
    payload = requests.get(url, headers=hdr, timeout=5)
    # Organize HTML Code with BeautifulSoup
    page_content = BeautifulSoup(payload.content, "html.parser")
    # Get all Relevant Info: reviews, date, rating, location, job_title,
    # current_employee?
    review = page_content.findAll(itemprop="reviewBody")
    pros = page_content.findAll(attrs={'class': 'cmp-review-pros'})
    cons = page_content.findAll(attrs={'class': 'cmp-review-cons'})
    date = page_content.findAll(attrs={"class": "cmp-review-date-created"})
    rating = page_content.findAll(attrs={"class": "cmp-ratingNumber"})
    location = page_content.findAll(
        attrs={"class": "cmp-reviewer-job-location"})
    job_title = page_content.findAll(attrs={"class": "cmp-reviewer"})
    current_employee = page_content.findAll(
        attrs={"class": "cmp-reviewer-job-title"})
    # Create Dictionary From Scraped Info
    dict_ = {
        'Review': [x.text for x in review],
        'Rating': [int(float(x.text)) for x in rating],
        'Date': [x.text for x in date],
        'Location_City': [x.text.split(",")[0] for x in location],
        'Location_State': [x.text.split(",")[1]
                           if len(x.text.split(",")) > 1 else float('NaN')
                           for x in location],
        'Job_Title': [x.text for x in job_title],
        'Current_Employee': [x.text.split("(")[1].split(")")[0]
                             for x in current_employee]}

    d_pros = {'Pros': [x.text.split('Pros')[1]
                       for x in pros if len(x.text.split('Pros')) > 1]}
    d_cons = {'Cons': [x.text.split('Cons')[1]
                       for x in cons if len(x.text.split('Cons')) > 1]}

    return pd.DataFrame(dict_), pd.DataFrame(d_pros), pd.DataFrame(d_cons)


def get_review_data(url_path, num_of_pages, hdr):
    """
    Gets review data.
    Parameters
    ----------
    url_path : str
    num_of_pages : int
    hdr : dict
    Returns
    -------
    tuple of pandas.core.frame.DataFrame
        df_reviews_final, df_pros_final, df_cons_final
    """
    # Initialize Final DataFrames
    df_reviews_final = pd.DataFrame()
    df_pros_final = pd.DataFrame()
    df_cons_final = pd.DataFrame()
    # Loop through all selected pages

    if num_of_pages == 1:
        print('1 page will be loaded!')
    else:
        print(num_of_pages, 'pages will be loaded!')

    for page_ in range(1, num_of_pages + 1):

        # print('Loading page: ' + str(page_) + ' ... ',
        #       end='',
        #       flush=True)

        try:
            df_reviews_temp, df_pros_temp,\
            df_cons_temp = _get_reviews_per_page(
                url_path, hdr, page_)

            # Vertically Concatenate DataFrames
            df_reviews_final = df_reviews_final.append(
                df_reviews_temp, ignore_index=True)
            df_pros_final = df_pros_final.append(df_pros_temp,
                                                 ignore_index=True)
            df_cons_final = df_cons_final.append(df_cons_temp,
                                                 ignore_index=True)

            # print('Page loaded successfully!')
        except BaseException:
            # print('Page skipped!')
            pass

    return df_reviews_final, df_pros_final, df_cons_final


def get_indeed_reviews(url_path, pages="all"):
    """"
    This module scrapes reviews from the Indeed website.
    Parameters
    ----------
    url_path : str
        Indeed's main review page url
    pages : int or 'all', optional (default = 'all')
        number of review pages that needs to be scraped. Each page has
        20 reviews in it
        If 'all', all pages will be reviewed.
    Returns
    -------
    df_reviews_final : DataFrame
        DataFrame of the reviews
    df_pros_final : DataFrame
        DataFrame of pros in the reviews
    df_cons_final : DataFrame
        DataFrame of cons in the reviews
    """

    # header
    hdr = {'User-Agent': 'Mozilla/5.0',
           'Cache-Control': 'no-cache'}
    try:
        num_of_pages = get_number_of_pages(url_path, hdr, pages)
    # try one more time before failing
    except AttributeError:
        num_of_pages = get_number_of_pages(url_path, hdr, pages)

    if num_of_pages == -1:
        return -1

    df_reviews_final, df_pros_final, \
    df_cons_final = get_review_data(url_path,
                                    num_of_pages,
                                    hdr)

    print(len(df_pros_final), 'pros,', len(df_cons_final), 'cons, and',
          len(df_reviews_final), 'reviews loaded!')

    return df_reviews_final, df_pros_final, df_cons_final




