"""
Enhanced Book Recommendation System — Flask Application

A hybrid book recommendation system combining best practices from:
- MainakRepositor/Book-Recommender: UI and friend-based recs
- nikunjsonule/Book-Recommendation-System: SVD with RMSE=1.63
- syedsharin/Book-Recommendation-System-Project: Surprise library
- mujtabaali02/Book-Recommendation-System: Content-based filtering
- fkemeth/book_collaborative_filtering: User-user CF

Enhancements over source repositories:
1. Multiple algorithm support (popularity, content, collaborative, hybrid)
2. Cold-start handling with popularity fallback
3. Friend-based recommendations
4. Modern responsive UI (Flask + HTML/CSS/JS)
5. Modular code architecture
"""

import os
import random

import pandas as pd
from flask import (
    Flask, render_template, request, session,
    redirect, url_for, flash, jsonify
)

from data_loader import DataLoader
from recommenders.popularity import PopularityRecommender
from recommenders.content_based import ContentBasedRecommender
from recommenders.collaborative import CollaborativeRecommender
from recommenders.hybrid import HybridRecommender

# ── App Setup ───────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "book-recommender-secret-2026")

# ── Load Data & Init Recommenders (runs once at startup) ───
print("📚 Loading data…")
loader = DataLoader()
books_df, users_df, ratings_df = loader.load_all()

if books_df is None or ratings_df is None:
    raise RuntimeError(
        "Failed to load data. Run `python data_loader.py` first to download the dataset."
    )

print("🧠 Initialising recommendation engines…")
recommender = HybridRecommender(books_df, ratings_df, users_df, loader=loader)

# Pre-compute active users list for demo login
_active_users = (
    ratings_df["user_id"]
    .value_counts()
    .loc[lambda s: s >= 5]
    .index.tolist()
)


# ── Helpers ─────────────────────────────────────────────────
def _get_stats() -> dict:
    """Return system-wide statistics for the sidebar."""
    return {
        "books": f"{len(books_df):,}",
        "ratings": f"{len(ratings_df):,}",
        "users": f"{ratings_df['user_id'].nunique():,}",
    }


def _ensure_session():
    """Seed session with a random active user + friends for demo."""
    if "user_id" not in session and _active_users:
        session["user_id"] = random.choice(_active_users[:100])
    if "friends" not in session:
        if len(_active_users) > 5:
            session["friends"] = random.sample(_active_users[:50], 4)
        else:
            session["friends"] = []


def _user_rated_count() -> int:
    uid = session.get("user_id")
    if uid is None:
        return 0
    return int((ratings_df["user_id"] == uid).sum())


def _df_to_dicts(df, n=10):
    """Convert a DataFrame (or None/empty) to a list of dicts for templates."""
    if df is None or df.empty:
        return []
    return df.head(n).to_dict("records")


def _add_avg_rating(df):
    """Merge avg_rating onto a recommendations DataFrame if missing."""
    if df is None or df.empty:
        return df
    if "avg_rating" not in df.columns:
        book_ratings = (
            ratings_df.groupby("book_id")["rating"]
            .mean()
            .reset_index()
            .rename(columns={"rating": "avg_rating"})
        )
        df = df.merge(book_ratings, on="book_id", how="left")
    return df


# ── Routes ──────────────────────────────────────────────────

@app.route("/")
def index():
    """Main dashboard page."""
    _ensure_session()
    popular_books = []
    if not session.get("user_id"):
        pop = recommender.popularity.get_best_books(n=10)
        popular_books = _df_to_dicts(pop)
    return render_template(
        "index.html",
        popular_books=popular_books,
        stats=_get_stats(),
        user_rated_count=_user_rated_count(),
    )


@app.route("/book/<int:book_id>")
def book_detail(book_id):
    """Book detail page."""
    _ensure_session()
    book_row = books_df[books_df["book_id"] == book_id]
    if book_row.empty:
        flash("Book not found.", "warning")
        return redirect(url_for("index"))

    book = book_row.iloc[0].to_dict()
    genres = loader.get_book_genres(book_id, top_n=8)

    similar = recommender.content.get_similar_to_book(book_id, n=10)
    similar_books = _df_to_dicts(similar)

    return render_template(
        "book_detail.html",
        book=book,
        genres=genres,
        similar_books=similar_books,
        stats=_get_stats(),
        user_rated_count=_user_rated_count(),
    )


@app.route("/search")
def search():
    """Search results page."""
    _ensure_session()
    query = request.args.get("q", "").strip()
    books = []
    if query:
        results = recommender.content.search_books(query, n=20)
        books = _df_to_dicts(results, n=20)
    return render_template(
        "search_results.html",
        query=query,
        books=books,
        stats=_get_stats(),
        user_rated_count=_user_rated_count(),
    )


@app.route("/login", methods=["POST"])
def login():
    """Log in as a user."""
    uid_str = request.form.get("user_id", "").strip()
    if not uid_str:
        flash("Please enter a User ID.", "warning")
        return redirect(url_for("index"))
    try:
        uid = int(uid_str)
    except ValueError:
        flash("User ID must be a number.", "error")
        return redirect(url_for("index"))

    if uid in ratings_df["user_id"].values:
        session["user_id"] = uid
        flash(f"Welcome back, User {uid}! 🎉", "success")
    else:
        flash("User not found.", "warning")
    return redirect(url_for("index"))


@app.route("/logout")
def logout():
    """Log out."""
    session.pop("user_id", None)
    flash("Logged out.", "info")
    return redirect(url_for("index"))


@app.route("/add_friend", methods=["POST"])
def add_friend():
    """Add a friend (AJAX)."""
    fid_str = request.form.get("friend_id", "").strip()
    if not fid_str:
        return jsonify(ok=False, message="Enter a User ID")
    try:
        fid = int(fid_str)
    except ValueError:
        return jsonify(ok=False, message="Invalid ID")
    if fid not in ratings_df["user_id"].values:
        return jsonify(ok=False, message="User not found")
    friends = session.get("friends", [])
    if fid in friends:
        return jsonify(ok=False, message="Already a friend")
    friends.append(fid)
    session["friends"] = friends
    return jsonify(ok=True)


# ── API: Tab Content (lazy-loaded) ──────────────────────────

@app.route("/api/recommendations/<tab>")
def api_recommendations(tab):
    """Return HTML fragments for each recommendation tab."""
    _ensure_session()
    uid = session.get("user_id")
    friends = session.get("friends", [])

    if uid and friends:
        recommender.add_friends(uid, friends)

    if tab == "foryou":
        return _render_foryou(uid)
    elif tab == "popular":
        return _render_popular()
    elif tab == "author":
        return _render_author(uid)
    elif tab == "friends":
        return _render_friends(uid, friends)
    elif tab == "personalized":
        return _render_personalized(uid)
    else:
        return '<div class="alert alert--warning">Unknown tab.</div>'


def _render_foryou(uid):
    h2 = '<h2 class="section-header">🔥 Recommended For You</h2>'
    sub = '<p class="section-sub">Personalized picks based on your reading history</p>'
    recs = recommender.get_recommendations(user_id=uid, n=10)
    recs = _add_avg_rating(recs)
    items = _df_to_dicts(recs)
    if not items:
        return h2 + '<div class="alert alert--info">Rate some books to get personalised recommendations!</div>'
    return h2 + sub + _book_grid_html(items)


def _render_popular():
    html = '<h2 class="section-header">⭐ Top Rated</h2>'
    top = recommender.popularity.get_top_rated(n=10, min_ratings=20)
    html += _book_grid_html(_df_to_dicts(top))

    html += '<hr class="divider">'
    html += '<h2 class="section-header">🔥 Most Reviewed</h2>'
    pop = recommender.popularity.get_most_popular(n=10)
    html += _book_grid_html(_df_to_dicts(pop))
    return html


def _render_author(uid):
    html = '<h2 class="section-header">📖 Explore by Author</h2>'
    user_ratings = ratings_df[ratings_df["user_id"] == uid]
    if user_ratings.empty:
        return html + '<div class="alert alert--info">Rate some books to discover authors you\'ll love!</div>'

    user_books = user_ratings.merge(books_df, on="book_id")
    if "author" not in user_books.columns:
        return html + '<div class="alert alert--info">No author data available.</div>'

    fav_authors = user_books.nlargest(5, "rating")["author"].unique()
    if len(fav_authors) == 0:
        return html + '<div class="alert alert--info">Rate some books to discover authors you\'ll love!</div>'

    html += '<p class="section-sub">Based on your favourite authors</p>'
    for author in fav_authors[:3]:
        if pd.notna(author):
            author_books = recommender.content.get_similar_by_author(author, n=5, exclude_book_id=None)
            if not author_books.empty:
                html += f'<div class="author-section"><div class="author-section__name">{author}</div>'
                html += '<div class="book-row">'
                for b in author_books.head(5).to_dict("records"):
                    html += _book_card_html(b)
                html += '</div></div>'
    return html


def _render_friends(uid, friends):
    html = '<h2 class="section-header">👥 What Your Friends Are Reading</h2>'
    if not friends:
        return html + '<div class="alert alert--info">Add friends to see what they\'re reading!</div>'

    html += f'<p class="section-sub">Based on {len(friends)} friends in your network</p>'
    recs = recommender.get_friend_recommendations(uid, n=10)
    items = _df_to_dicts(recs)
    if not items:
        return html + '<div class="alert alert--info">Your friends haven\'t rated any books yet.</div>'
    return html + _book_grid_html(items)


def _render_personalized(uid):
    html = '<h2 class="section-header">🎯 Because You Read…</h2>'
    because = recommender.get_because_you_read(uid, n=10)
    if not because:
        return html + '<div class="alert alert--info">Rate some books to get personalised recommendations!</div>'

    for source_title, recs_df in because.items():
        if recs_df is not None and not recs_df.empty:
            html += f'<div class="because-section">'
            html += f'<div class="because-section__label">Because you read <strong>{source_title}</strong></div>'
            html += '<div class="book-row">'
            for b in recs_df.head(5).to_dict("records"):
                html += _book_card_html(b)
            html += '</div></div>'
    return html


# ── HTML fragment builders ──────────────────────────────────

def _book_grid_html(items):
    """Return a <div class="book-grid"> with book cards."""
    if not items:
        return '<div class="empty-state"><div class="empty-state__icon">📚</div><div class="empty-state__text">No recommendations available</div></div>'
    html = '<div class="book-grid">'
    for b in items:
        html += _book_card_html(b)
    html += '</div>'
    return html


def _book_card_html(b):
    """Return a single book card <a> element."""
    book_id = b.get("book_id", "")
    img = b.get("image_url_m") or b.get("image_url_l") or "https://via.placeholder.com/180x240?text=No+Cover"
    if pd.isna(img):
        img = "https://via.placeholder.com/180x240?text=No+Cover"

    title = str(b.get("title", "Unknown"))
    title_display = title[:50] + "…" if len(title) > 50 else title

    author = str(b.get("author", "Unknown"))
    author_display = author[:35] + "…" if len(author) > 35 else author

    rating_html = ""
    rating = b.get("avg_rating") or b.get("rating")
    if rating and not pd.isna(rating):
        rating_html = f'<div class="book-card__rating">⭐ {float(rating):.1f}</div>'

    return (
        f'<a href="/book/{book_id}" class="book-card">'
        f'<img class="book-card__img" src="{img}" alt="{title_display}" loading="lazy" '
        f'onerror="this.src=\'https://via.placeholder.com/180x240?text=No+Cover\'">'
        f'<div class="book-card__body">'
        f'<div class="book-card__title">{title_display}</div>'
        f'<div class="book-card__author">{author_display}</div>'
        f'{rating_html}'
        f'</div></a>'
    )


# ── Run ─────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(debug=True, port=5000)
