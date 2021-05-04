import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly_express as px

# Load dataset
books = pd.read_csv('goodbooks-10k/books.csv')
ratings = pd.read_csv('goodbooks-10k/ratings.csv')
book_tags = pd.read_csv('goodbooks-10k/book_tags.csv')
tags = pd.read_csv('goodbooks-10k/tags.csv')

'''

    Dataset Exploration

'''

# General information about the dataset
num_ratings = ratings['rating'].count()
num_users = ratings['user_id'].nunique()
num_books = ratings['book_id'].nunique()

pd.DataFrame([['# of Ratings', num_ratings],
             ['# of Users', num_users],
             ['# of Books', num_books]],
            columns=['Characteristic', 'Count'])

'''

    Books

'''

# Top Rated Books
top_rated_books = books.sort_values('average_rating', ascending=False)
top_rated = top_rated_books[:10]
fig = px.bar(top_rated, x='average_rating', y='title', title='Top Rated Books',
             orientation='h', color='title', width=1000, height=700)
fig.show()

# Top Popular Books
top_popular_books = books.sort_values('ratings_count', ascending=False)
top_popular = top_popular_books[:10]
fig = px.bar(top_popular, x='ratings_count', y='title', title='Top Popular Books',
             orientation='h', color='title', width=1000, height=700)
fig.show()

# Top Text-Reviewed Books
top_text_reviewed_books = books.sort_values('work_text_reviews_count', ascending=False)
top_text_reviewed = top_text_reviewed_books[:10]
fig = px.bar(top_text_reviewed, x='work_text_reviews_count', y='title', title='Top Text-reviewed Books',
             orientation='h', color='title', width=1000, height=700)
fig.show()

# Original Title vs Title
fig = px.scatter(books.head(20), x='original_title', y='title', title='Original Title vs Title',
             orientation='h', color='title', width=1000, height=700)
fig.show()

# Languages
languages = books['language_code'].value_counts().reset_index()
languages.columns = ['language_code', 'language_count']
fig = px.bar(languages, x='language_count', y='language_code', title='Languages',
             orientation='h', color='language_code', width=1000, height=700)
fig.show()

# Year of Publication (Part 1)
publication_year = books.groupby('original_publication_year')['title'].count().reset_index()
publication_year.columns = ['publication_year', 'books_count']
publication_year['publication_year'] = publication_year['publication_year'].astype(int)
fig = px.scatter(publication_year, x='books_count', y='publication_year', title='Year of Publication',
             orientation='h', color='books_count', width=1000, height=700)
fig.show()

# Year of Publication (Part 2)
old_years = books[['title', 'original_publication_year']].sort_values('original_publication_year').head(100)
fig = px.scatter(old_years, x='original_publication_year', y='title', title='Year of Publication (old years)', orientation='h', color='title',
             width=1500, height=700)
fig.show()


# Authors
top_author_counts = books['authors'].value_counts().reset_index()
top_author_counts.columns = ['author', 'book_count']
top_author_counts['author'] = top_author_counts['author']
top_author_counts = top_author_counts.sort_values('book_count', ascending=False)
fig = px.bar(top_author_counts.head(10), x='book_count', y='author', title='Top Authors', orientation='h', color='author',
             width=1000, height=700)
fig.show()

'''

    Ratings

'''

# Distribution of Ratings
plt.figure(figsize=(12,8))
sns.countplot(x='rating', data=ratings)
plt.show()

# Ratings per User
ratings_per_user = ratings.groupby('user_id')['user_id'].count().tolist()
plt.figure(figsize=(12,8))
plt.hist(ratings_per_user, bins='auto')
plt.grid(axis='y', alpha=0.75)
plt.xlabel('number of ratings per user')
plt.ylabel('count')
plt.show()

'''

    Tags

'''

# Top Tags
data = pd.merge(book_tags, tags, left_on='tag_id', right_on='tag_id', how='inner')
top_tags = data['tag_name'].value_counts(ascending=False).reset_index()
top_tags.columns = ['tag_name', 'count']
fig = px.bar(top_tags.head(15), x='count', y='tag_name', title='Top Tags', orientation='h', color='tag_name',
             width=1000, height=700)
fig.show()


'''

    Handling Missing Data

'''

# Check for NaN values
print("Are there Null values?"
    "\n in books.csv:",  books.isnull().values.any(),
    "\n in ratings.csv:", ratings.isnull().values.any(),
    "\n in tags.csv:", tags.isnull().values.any(),
    "\n in book_tags.csv:", book_tags.isnull().values.any())

null_counts = books.isnull().sum()
null_features = null_counts[null_counts > 0]
print(null_features)


