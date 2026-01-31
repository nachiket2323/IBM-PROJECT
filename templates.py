"""
UI Templates for Book Recommendation System

Adapted from:
- MainakRepositor/Book-Recommender: template.py UI components

Enhanced with:
- Modern CSS styling
- Responsive grid layouts
- Interactive elements
"""

import streamlit as st
import random


def select_book(book_id):
    """Select a book to view details."""
    st.session_state['selected_book_id'] = book_id
    if 'search_query' in st.session_state:
        st.session_state['search_query'] = ''


def select_user(user_id):
    """Log in as a user."""
    st.session_state['user_id'] = user_id


def add_friend(friend_id):
    """Add a friend to the user's list."""
    if 'friends' not in st.session_state:
        st.session_state['friends'] = []
    if friend_id not in st.session_state['friends']:
        st.session_state['friends'].append(friend_id)
        st.success(f"Added friend {friend_id}!")
    else:
        st.info(f"Friend {friend_id} is already in your list")


def book_tile(column, book, show_rating=False):
    """
    Display a single book tile.
    
    Args:
        column: Streamlit column to render in
        book: Dict or Series with book data
        show_rating: Whether to show rating
    """
    with column:
        # Book image
        image_url = book.get('image_url_m', book.get('image_url_l', ''))
        if image_url and pd.notna(image_url):
            try:
                st.image(image_url, width=150)
            except:
                st.image("https://via.placeholder.com/150x200?text=No+Cover", 
                        use_container_width=True)
        else:
            st.image("https://via.placeholder.com/150x200?text=No+Cover", 
                    use_container_width=True)
        
        # Book title (truncated)
        title = book.get('title', 'Unknown Title')
        if len(str(title)) > 40:
            title = str(title)[:37] + "..."
        st.caption(f"**{title}**")
        
        # Author
        author = book.get('author', 'Unknown Author')
        if len(str(author)) > 30:
            author = str(author)[:27] + "..."
        st.caption(f"_{author}_")
        
        # Rating badge
        if show_rating:
            rating = book.get('avg_rating', book.get('rating', 0))
            if rating and pd.notna(rating):
                st.caption(f"⭐ {float(rating):.1f}")
        
        # Select button
        book_id = book.get('book_id', '')
        if book_id:
            st.button(
                "📖 View", 
                key=f"view_{book_id}_{random.random()}", 
                on_click=select_book, 
                args=(book_id,),
                use_container_width=True
            )


def book_grid(df, title="", n_cols=5, show_rating=True):
    """
    Display a grid of books.
    
    Args:
        df: DataFrame with book data
        title: Section title
        n_cols: Number of columns
        show_rating: Whether to show ratings
    """
    if df is None or df.empty:
        return
    
    if title:
        st.subheader(title)
    
    # Limit to avoid overflow
    items = df.head(n_cols * 2).to_dict('records')
    
    if not items:
        st.info("No recommendations available")
        return
    
    # Create rows of books
    for i in range(0, len(items), n_cols):
        row_items = items[i:i+n_cols]
        columns = st.columns(len(row_items))
        
        for col, item in zip(columns, row_items):
            book_tile(col, item, show_rating)


def book_carousel(df, title="", show_rating=True):
    """
    Display a horizontal carousel of books.
    
    Args:
        df: DataFrame with book data
        title: Section title
        show_rating: Whether to show ratings
    """
    if df is None or df.empty:
        return
    
    if title:
        st.subheader(title)
    
    # Convert to list of dicts
    items = df.head(10).to_dict('records')
    
    if not items:
        st.info("No recommendations available")
        return
    
    # Create horizontal scroll with columns
    n_cols = min(len(items), 5)
    columns = st.columns(n_cols)
    
    for i, item in enumerate(items[:n_cols]):
        book_tile(columns[i], item, show_rating)


def book_details(book_df, ratings_df=None):
    """
    Display detailed book information.
    
    Args:
        book_df: DataFrame/Series with single book data
        ratings_df: Optional ratings for this book
    """
    if book_df is None or book_df.empty:
        st.warning("Book not found")
        return
    
    book = book_df.iloc[0] if hasattr(book_df, 'iloc') else book_df
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        image_url = book.get('image_url_l', book.get('image_url_m', ''))
        if image_url and pd.notna(image_url):
            try:
                st.image(image_url, use_container_width=True)
            except:
                st.image("https://via.placeholder.com/200x300?text=No+Cover", 
                        use_container_width=True)
    
    with col2:
        st.title(book.get('title', 'Unknown Title'))
        st.markdown(f"**Author:** {book.get('author', 'Unknown')}")
        
        year = book.get('year', 0)
        publisher = book.get('publisher', 'Unknown')
        if year and year > 0:
            st.caption(f"📅 {year} | 📚 {publisher}")
        else:
            st.caption(f"📚 {publisher}")
        
        book_id = book.get('book_id', 'N/A')
        st.caption(f"book_id: {book_id}")
        
        # Show average rating if available
        if 'avg_rating' in book:
            rating = book['avg_rating']
            count = book.get('rating_count', 0)
            st.metric("Average Rating", f"⭐ {rating:.1f}", f"{int(count)} reviews")


def sidebar_user_login(users_df, ratings_df):
    """
    Render user login sidebar.
    
    Args:
        users_df: DataFrame with user data
        ratings_df: DataFrame with ratings
    """
    st.sidebar.header("👤 User Login")
    
    current_user = st.session_state.get('user_id', None)
    
    if current_user:
        st.sidebar.success(f"Logged in as: {current_user}")
        
        # User stats
        user_ratings = ratings_df[ratings_df['user_id'] == current_user]
        st.sidebar.metric("Books Rated", len(user_ratings))
        
        if st.sidebar.button("Log Out"):
            del st.session_state['user_id']
            st.rerun()
    else:
        # Login form
        user_id = st.sidebar.text_input(
            "User ID",
            placeholder="Enter your User ID"
        )
        
        if st.sidebar.button("Log In"):
            if user_id:
                try:
                    user_id_int = int(user_id)
                    if user_id_int in ratings_df['user_id'].values:
                        select_user(user_id_int)
                        st.sidebar.success("Welcome back! 🎉")
                        st.rerun()
                    else:
                        st.sidebar.warning("User not found")
                except ValueError:
                    st.sidebar.error("Please enter a valid User ID")


def sidebar_friends(ratings_df):
    """
    Render friends management sidebar.
    """
    st.sidebar.header("👥 Friends")
    
    friends = st.session_state.get('friends', [])
    
    if friends:
        st.sidebar.write(f"Your friends: {len(friends)}")
        for friend in friends[:5]:
            st.sidebar.caption(f"• User {friend}")
    
    friend_id = st.sidebar.text_input(
        "Add Friend",
        placeholder="Enter Friend's User ID"
    )
    
    if st.sidebar.button("Add"):
        if friend_id:
            try:
                friend_id_int = int(friend_id)
                if friend_id_int in ratings_df['user_id'].values:
                    add_friend(friend_id_int)
                else:
                    st.sidebar.warning("User not found")
            except ValueError:
                st.sidebar.error("Please enter a valid User ID")


def search_bar():
    """
    Render search bar and return query.
    """
    query = st.text_input(
        "🔍 Search Books",
        placeholder="Search by title or author...",
        key="search_query"
    )
    return query


def render_search_results(content_recommender, query):
    """
    Render search results.
    """
    if query:
        results = content_recommender.search_books(query, n=10)
        if not results.empty:
            st.subheader(f"Search Results for '{query}'")
            book_grid(results, n_cols=5, show_rating=False)
        else:
            st.info(f"No results found for '{query}'")


def message_box(message, type="info"):
    """Display styled message boxes."""
    if type == "success":
        st.success(message)
    elif type == "warning":
        st.warning(message)
    elif type == "error":
        st.error(message)
    else:
        st.info(message)


def loading_spinner(message="Loading..."):
    """Display loading spinner."""
    return st.spinner(message)


# Import pandas here to avoid issues
import pandas as pd


def apply_custom_css():
    """Apply custom CSS for enhanced styling."""
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    h1 {
        color: #1E3A8A;
        font-weight: 700;
    }
    
    h2, h3 {
        color: #3B82F6;
        border-bottom: 2px solid #93C5FD;
        padding-bottom: 0.5rem;
    }
    
    /* Book tile styling */
    .stImage {
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s;
    }
    
    .stImage:hover {
        transform: scale(1.05);
    }
    
    /* Button styling */
    .stButton button {
        background: linear-gradient(90deg, #3B82F6, #8B5CF6);
        color: white;
        border: none;
        border-radius: 20px;
        font-weight: 600;
        transition: all 0.3s;
    }
    
    .stButton button:hover {
        background: linear-gradient(90deg, #2563EB, #7C3AED);
        box-shadow: 0 4px 12px rgba(59, 130, 246, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1E3A8A, #3B82F6);
    }
    
    /* Metric styling */
    .css-1xarl3l {
        background: #F0F9FF;
        border-radius: 12px;
        padding: 1rem;
    }
    
    /* Caption styling */
    .stCaption {
        font-size: 0.85rem;
        line-height: 1.4;
    }
    </style>
    """, unsafe_allow_html=True)
