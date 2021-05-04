from flask import Flask, request, render_template
from recommender_types import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/popular_books')
def popularity_based():
    popular_books = popularity_based_recommendation()
    return render_template('popular_books.html', popular_books=popular_books, n=len(popular_books['id']))

@app.route('/content_based')
def content_based():
    titles = books['title'].tolist()
    return render_template('content_based.html', titles=titles, n=len(titles))

@app.route('/content_based_results', methods=['GET', 'POST'])
def content_based_results():
    if request.method == 'POST':
        title = request.form['title']
        similar_books = content_based_recommendation(title)
        return render_template('content_based_results.html', similar_books=similar_books, n=len(similar_books['id']), title=title)

@app.route('/collaborative_filtering')
def collaborative_filtering():
    user_ids = df_ratings['user_id'].unique().tolist()
    return render_template('collaborative_filtering.html', user_ids=user_ids, n=len(user_ids))

@app.route('/collaborative_filtering_results', methods=['GET', 'POST'])
def collaborative_filtering_results():
    if request.method == 'POST':
        user_id = int(request.form['user_id'])
        information = collaborative_filtering_recommendation(user_id)
        return render_template('collaborative_filtering_results.html', information=information, m=len(information['est_titles']), n=len(information['user_titles']), user_id=user_id)

if __name__ == '__main__':
    app.run()