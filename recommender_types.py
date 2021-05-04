import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Reader, Dataset, SVD
from surprise.model_selection import cross_validate

# Load dataset
books = pd.read_csv('goodbooks-10k/books.csv')
ratings = pd.read_csv('goodbooks-10k/ratings.csv')
book_tags = pd.read_csv('goodbooks-10k/book_tags.csv')
tags = pd.read_csv('goodbooks-10k/tags.csv')

# Data preparation (content-based)
merge_tags = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how='inner')
books_columns = books[['book_id', 'title', 'goodreads_book_id', 'authors']]
books_and_tags = pd.merge(books_columns, merge_tags, left_on='goodreads_book_id', right_on='goodreads_book_id')
list_tags = books_and_tags.groupby(by='goodreads_book_id')['tag_name'].apply(set).apply(list)
books['tags'] = books['goodreads_book_id'].apply(lambda x: ' '.join(list_tags[x]))

# Data preparation (collaborative-filtering based)
newer_books = books[books['original_publication_year'] > 2000]
df_books = newer_books[['book_id', 'title']]
df_ratings = ratings[ratings.book_id.isin(df_books.book_id)]
user_counts = df_ratings['user_id'].value_counts()
df_ratings = df_ratings[df_ratings['user_id'].isin(user_counts[user_counts >= 100].index)]
df_titles = df_books[['book_id','title']]
df_titles.set_index('book_id', inplace=True)

# Algorithms
def imdb_weighted_rating(n=10):
    v = books['ratings_count']
    m = books['ratings_count'].quantile(0.55)
    R = books['average_rating']
    C = books['average_rating'].mean()
    books['weighted_rating'] = (R * v + C * m) / (v + m)
    qualified = books[books['ratings_count'] >= m]
    qualified = qualified.sort_values('weighted_rating', ascending=False).head(n)
    return qualified

def cosine_sim():
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(books['tags'])
    return cosine_similarity(tfidf_matrix, tfidf_matrix)

def singular_value_decomposition():
    reader = Reader()
    data = Dataset.load_from_df(df_ratings[['user_id', 'book_id', 'rating']], reader)
    svd = SVD()
    cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    trainset = data.build_full_trainset()
    svd.fit(trainset)
    return svd

# Popularity-based Recommender
def popularity_based_recommendation():
    top_popular_books = imdb_weighted_rating()
    popular_books = {'id': [], 'title': [], 'author': [], 'rating': [], 'image_url': [], 'year': []}
    url = 'https://www.goodreads.com/book/show/'
    for i in top_popular_books.index:
        popular_books['id'].append(url + str(top_popular_books.loc[i]['goodreads_book_id']))
        popular_books['title'].append(top_popular_books.loc[i]['title'])
        popular_books['author'].append(top_popular_books.loc[i]['authors'])
        popular_books['rating'].append(round(top_popular_books.loc[i]['weighted_rating'], 2))
        popular_books['image_url'].append(top_popular_books.loc[i]['image_url'])
        popular_books['year'].append(int(top_popular_books.loc[i]['original_publication_year']))
    return popular_books

# Content-based Recommender
def content_based_recommendation(title, n=10):
    similarities = cosine_sim()
    book_id = books.index[books['title'] == title][0]
    book_similarities = list(enumerate(similarities[book_id]))
    book_similarities = sorted(book_similarities, key=lambda x: x[1], reverse=True)
    most_similar_books = book_similarities[1:1 + n]
    similar = {'id': [], 'title': [], 'author': [], 'image_url': [], 'year': [], 'similarity': []}
    url = 'https://www.goodreads.com/book/show/'
    for i in most_similar_books:
        similar['id'].append(url + str(books.loc[i[0]]['goodreads_book_id']))
        similar['title'].append(books.loc[i[0]]['title'])
        similar['author'].append(books.loc[i[0]]['authors'])
        similar['image_url'].append(books.loc[i[0]]['image_url'])
        similar['year'].append(int(books.loc[i[0]]['original_publication_year']))
        similar['similarity'].append(round(i[1], 3))
    return similar

# Collaborative Filtering Recommender
def collaborative_filtering_recommendation(user_id, n=10):
    user = df_ratings[df_ratings['user_id'] == user_id]
    information = {'user_titles': [], 'user_ratings': [], 'est_titles': [], 'est_scores': []}
    user_book_ids = user['book_id'].tolist()
    user_ratings = user['rating'].tolist()
    books_rated_by_user = []
    for i in user_book_ids:
        books_rated_by_user.append(df_titles.loc[i]['title'])
    svd = singular_value_decomposition()
    user = df_books.copy()
    user = user.reset_index()
    already_read = df_ratings[df_ratings['user_id'] == user_id]['book_id'].unique()
    user = user[~user['book_id'].isin(already_read)]
    user['estimate_score'] = user['book_id'].apply(lambda x: svd.predict(user_id, x).est)
    user = user.drop('book_id', axis=1)
    user = user.sort_values('estimate_score', ascending=False)
    estimate_titles = user['title'].tolist()
    estimate_scores = user['estimate_score'].tolist()
    information['user_titles'] = books_rated_by_user
    information['user_ratings'] = user_ratings
    information['est_titles'] = estimate_titles[: n]
    information['est_scores'] = estimate_scores[: n]
    return information
