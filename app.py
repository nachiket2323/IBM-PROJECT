"""
Enhanced Book Recommendation System - Streamlit Application

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
4. Modern responsive UI
5. Modular code architecture
"""

import streamlit as st
import pandas as pd
import random

# Configure page
st.set_page_config(
    page_title="📚 Book Recommender",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import our modules
from data_loader import DataLoader
from recommenders.popularity import PopularityRecommender
from recommenders.content_based import ContentBasedRecommender
from recommenders.collaborative import CollaborativeRecommender
from recommenders.hybrid import HybridRecommender
import templates as t


@st.cache_resource
def load_data():
    """Load and cache all datasets."""
    loader = DataLoader()
    books, users, ratings = loader.load_all()
    return books, users, ratings, loader


@st.cache_resource
def init_recommenders(_books, _ratings, _users, _loader):
    """Initialize all recommenders (cached)."""
    hybrid = HybridRecommender(_books, _ratings, _users, loader=_loader)
    return hybrid


def init_session_state(ratings_df):
    """Initialize session state variables."""
    if 'user_id' not in st.session_state:
        # Pick a random active user for demo
        active_users = ratings_df['user_id'].value_counts()
        active_users = active_users[active_users >= 5].index.tolist()
        if active_users:
            st.session_state['user_id'] = random.choice(active_users[:100])
    
    if 'friends' not in st.session_state:
        # Initialize with some random friends
        active_users = ratings_df['user_id'].value_counts()
        active_users = active_users[active_users >= 5].index.tolist()
        if len(active_users) > 5:
            st.session_state['friends'] = random.sample(active_users[:50], 4)
        else:
            st.session_state['friends'] = []
    
    if 'selected_book_id' not in st.session_state:
        st.session_state['selected_book_id'] = None
    
    if 'selected_genres' not in st.session_state:
        st.session_state['selected_genres'] = []


def render_header():
    """Render the main header."""
    st.title("📚 Enhanced Book Recommender")
    st.markdown("""
    *A hybrid recommendation system combining collaborative filtering, content-based, 
    and popularity-based approaches for personalized book suggestions.*
    """)
    st.divider()


def render_selected_book(books_df, ratings_df, recommender, loader):
    """Render the currently selected book with details and recommendations."""
    book_id = st.session_state.get('selected_book_id')
    
    if not book_id:
        return
    
    # Handle type mismatch if book_id is string/int
    if books_df['book_id'].dtype == 'int64':
        try:
            book_id = int(book_id)
        except:
            pass
    
    book = books_df[books_df['book_id'] == book_id]
    
    if book.empty:
        st.warning(f"Selected book {book_id} not found")
        st.session_state['selected_book_id'] = None
        return
    
    # Book details
    st.subheader("📖 Currently Viewing")
    t.book_details(book, ratings_df)
    
    # Tags/Genres
    genres = loader.get_book_genres(book_id, top_n=8)
    if genres:
        st.markdown("**Tags:** " + " • ".join([f"`{g}`" for g in genres]))
    
    # Rating Distribution
    col1, col2 = st.columns([1, 2])
    with col1:
        if ratings_df is not None:
            book_ratings = ratings_df[ratings_df['book_id'] == book_id]
            if not book_ratings.empty:
                st.markdown("**Rating Distribution**")
                dist = book_ratings['rating'].value_counts().sort_index()
                st.bar_chart(dist, height=150)
    
    # Similar books
    st.divider()
    similar = recommender.content.get_similar_to_book(book_id, n=10)
    if not similar.empty:
        t.book_grid(similar, title="📚 Similar Books", n_cols=5)
    
    # Clear selection button
    if st.button("✖ Clear Selection"):
        st.session_state['selected_book_id'] = None
        st.rerun()
    
    st.divider()


def render_recommendations(recommender, user_id, books_df, ratings_df, loader):
    """Render all recommendation sections."""
    
    # Add friends to recommender
    friends = st.session_state.get('friends', [])
    if friends:
        recommender.add_friends(user_id, friends)
    
    # Determine tags/genres selected
    selected_genres = st.session_state.get('selected_genres', [])
    
    tabs = [
        "🔥 For You",
        "📊 Popular",
        "📖 By Author", 
        "👥 Friends Reading",
        "🎯 Personalized"
    ]
    
    if selected_genres:
        tabs.insert(1, "🏷️ By Genre")
        
    st_tabs = st.tabs(tabs)
    
    # Tab logic helper
    def get_tab(name):
        return st_tabs[tabs.index(name)]
    
    with get_tab("🔥 For You"):
        st.subheader("🔥 Recommended For You")
        st.caption("*Personalized picks based on your reading history*")
        
        hybrid_recs = recommender.get_recommendations(
            user_id=user_id, 
            n=10, 
            selected_genres=selected_genres
        )
        if not hybrid_recs.empty:
            # Add rating info
            if 'avg_rating' not in hybrid_recs.columns:
                book_ratings = ratings_df.groupby('book_id')['rating'].mean().reset_index()
                book_ratings.columns = ['book_id', 'avg_rating']
                hybrid_recs = hybrid_recs.merge(book_ratings, on='book_id', how='left')
            t.book_grid(hybrid_recs, n_cols=5, show_rating=True)
        else:
            st.info("Rate some books to get personalized recommendations!")
            
    if selected_genres:
        with get_tab("🏷️ By Genre"):
            st.subheader(f"📚 Books in {', '.join(selected_genres[:3])}")
            
            # Find books with these tags
            # This is slow, so we cache or optimize
            tag_ids = loader.tags_df[loader.tags_df['tag_name'].isin(selected_genres)]['tag_id']
            goodreads_ids = loader.book_tags_df[loader.book_tags_df['tag_id'].isin(tag_ids)]['goodreads_book_id'].unique()
            
            genre_books = books_df[books_df['goodreads_book_id'].isin(goodreads_ids)].head(50)
            
            if not genre_books.empty:
                # Sort by rating
                t.book_grid(genre_books.sort_values('avg_rating', ascending=False).head(20), n_cols=5)
            else:
                st.info("No books found for these genres")
    
    with get_tab("📊 Popular"):
        st.subheader("📊 Most Popular Books")
        st.caption("*Trending and highly-rated across all users*")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**⭐ Top Rated**")
            top_rated = recommender.popularity.get_top_rated(n=5, min_ratings=20)
            t.book_grid(top_rated, n_cols=5, show_rating=True)
        
        with col2:
            st.markdown("**🔥 Most Reviewed**")
            most_popular = recommender.popularity.get_most_popular(n=5)
            t.book_grid(most_popular, n_cols=5, show_rating=True)
    
    with get_tab("📖 By Author"):
        st.subheader("📖 Explore by Author")
        
        # Get user's favorite authors
        user_ratings = ratings_df[ratings_df['user_id'] == user_id]
        if not user_ratings.empty:
            user_books = user_ratings.merge(books_df, on='book_id')
            if 'author' in user_books.columns:
                favorite_authors = user_books.nlargest(5, 'rating')['author'].unique()
                
                if len(favorite_authors) > 0:
                    st.caption("*Based on your favorite authors*")
                    
                    for author in favorite_authors[:3]:
                        if pd.notna(author):
                            author_books = recommender.content.get_similar_by_author(
                                author, n=5, 
                                exclude_book_id=None
                            )
                            if not author_books.empty:
                                st.markdown(f"**{author}**")
                                t.book_carousel(author_books, show_rating=False)
                else:
                    st.info("Rate some books to discover authors you'll love!")
        else:
            st.info("Rate some books to discover authors you'll love!")
    
    with get_tab("👥 Friends Reading"):
        st.subheader("👥 What Your Friends Are Reading")
        
        if friends:
            st.caption(f"*Based on {len(friends)} friends in your network*")
            friend_recs = recommender.get_friend_recommendations(user_id, n=10)
            
            if not friend_recs.empty:
                t.book_grid(friend_recs, n_cols=5, show_rating=True)
            else:
                st.info("Your friends haven't rated any books yet")
        else:
            st.info("Add friends to see what they're reading!")
    
    with get_tab("🎯 Personalized"):
        st.subheader("🎯 Because You Read...")
        
        because_recs = recommender.get_because_you_read(user_id, n=10)
        
        if because_recs:
            for source_title, recommendations in because_recs.items():
                if not recommendations.empty:
                    st.markdown(f"**Because you read {source_title}:**")
                    t.book_carousel(recommendations, show_rating=False)
        else:
            st.info("Rate some books to get personalized recommendations!")


def render_sidebar(users_df, ratings_df, content_recommender, loader):
    """Render the sidebar with user info and search."""
    
    # User login section
    t.sidebar_user_login(users_df, ratings_df)
    
    st.sidebar.divider()
    
    # Friends section
    t.sidebar_friends(ratings_df)
    
    st.sidebar.divider()
    
    # Genre Filter
    if loader and loader.tags_df is not None and loader.book_tags_df is not None:
        st.sidebar.header("🏷️ Filter Recommendations")
        
        # Get simplified list of popular genres
        # We'll use a cached list of top tags
        @st.cache_data
        def get_top_tags():
            top_tags = loader.book_tags_df.groupby('tag_id')['count'].sum().nlargest(200).index
            return loader.tags_df[loader.tags_df['tag_id'].isin(top_tags)]['tag_name'].tolist()
        
        genres = get_top_tags()
        # Filter out numbers/dates
        genres = [g for g in genres if not any(char.isdigit() for char in g)]
        
        selected = st.sidebar.multiselect("Select Genres", genres)
        if selected != st.session_state.get('selected_genres', []):
            st.session_state['selected_genres'] = selected
            st.rerun()
            
    st.sidebar.divider()
    
    # Stats
    st.sidebar.header("📊 System Stats")
    st.sidebar.metric("Total Books", f"{len(content_recommender.books_df):,}")
    st.sidebar.metric("Total Ratings", f"{len(ratings_df):,}")
    st.sidebar.metric("Active Users", f"{ratings_df['user_id'].nunique():,}")
    
    st.sidebar.divider()
    
    # About
    st.sidebar.header("ℹ️ About")
    st.sidebar.markdown("""
    This system combines:
    - 📊 **Popularity-based** filtering
    - 📚 **Content-based** recommendations
    - 👥 **Collaborative** filtering
    - 🔀 **Hybrid** ranking
    """)


def main():
    """Main application entry point."""
    
    # Apply custom CSS
    t.apply_custom_css()
    
    # Load data
    with st.spinner("📚 Loading book data..."):
        books, users, ratings, loader = load_data()
    
    if books is None or ratings is None:
        st.error("Failed to load data. Please check the data directory.")
        st.info("Run `python data_loader.py` to download the dataset first.")
        return
    
    # Initialize session state
    init_session_state(ratings)
    
    # Initialize recommenders
    with st.spinner("🧠 Initializing recommendation engines..."):
        recommender = init_recommenders(books, ratings, users, loader)
    
    # Render header
    render_header()
    
    # Render sidebar
    render_sidebar(users, ratings, recommender.content, loader)
    
    # 1. Search Bar (Separate from results)
    search_query = t.search_bar()
    
    # View Control Logic: Mutually exclusive views
    if search_query:
        # Case A: User is searching
        t.render_search_results(recommender.content, search_query)
        
    elif st.session_state.get('selected_book_id'):
        # Case B: User selected a book
        render_selected_book(books, ratings, recommender, loader)
        
    else:
        # Case C: Dashboard / Recommendations
        user_id = st.session_state.get('user_id')
        
        if user_id:
            render_recommendations(recommender, user_id, books, ratings, loader)
        else:
            st.info("👆 Log in with a User ID to get personalized recommendations!")
            
            # Show popular books for non-logged-in users
            st.subheader("📊 Popular Books")
            popular = recommender.popularity.get_best_books(n=10)
            t.book_grid(popular, n_cols=5, show_rating=True)


if __name__ == "__main__":
    main()
