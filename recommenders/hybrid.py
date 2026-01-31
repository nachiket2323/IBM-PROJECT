"""
Hybrid Recommender

Combines multiple recommendation strategies for best results.

Adapted from:
- MainakRepositor/Book-Recommender: Friend-based filtering
- All repos: Combining content and collaborative approaches

Handles:
- Cold-start users: Falls back to popularity/content-based
- Warm users: Uses collaborative + content blending
- Context-aware switching between strategies
"""

import pandas as pd
import numpy as np

from .popularity import PopularityRecommender
from .content_based import ContentBasedRecommender
from .collaborative import CollaborativeRecommender


class HybridRecommender:
    """
    Hybrid recommendation system combining multiple strategies.
    
    Strategies:
    1. Popularity: For cold-start users
    2. Content-based: For users with limited history
    3. Collaborative: For users with sufficient history
    4. Combined: Weighted blend of all methods
    
    Enhancement: Friend-based filtering adapted from MainakRepositor
    """
    
    def __init__(self, books_df, ratings_df, users_df=None, loader=None):
        """
        Initialize hybrid recommender with all sub-recommenders.
        
        Args:
            books_df: DataFrame with book information
            ratings_df: DataFrame with user ratings
            users_df: Optional DataFrame with user information
            loader: DataLoader instance
        """
        self.books_df = books_df
        self.ratings_df = ratings_df
        self.users_df = users_df
        self.loader = loader
        
        # Initialize sub-recommenders
        print("\n" + "="*60)
        print("🚀 INITIALIZING HYBRID RECOMMENDER")
        print("="*60 + "\n")
        
        print("📊 Setting up Popularity Recommender...")
        self.popularity = PopularityRecommender(books_df, ratings_df)
        
        print("\n📚 Setting up Content-Based Recommender...")
        # Pass loader to content recommender
        self.content = ContentBasedRecommender(books_df, ratings_df, loader=loader)
        
        print("\n👥 Setting up Collaborative Recommender...")
        # Filter ratings for better collaborative filtering
        self.filtered_ratings = self._filter_ratings(ratings_df)
        self.collaborative = CollaborativeRecommender(
            self.filtered_ratings, 
            books_df, 
            users_df
        )
        
        # Friend relationships (simulated - in real app, would be from user data)
        self.friends_map = {}
        
        print("\n" + "="*60)
        print("✓ HYBRID RECOMMENDER READY")
        print("="*60 + "\n")
    
    def _filter_ratings(self, ratings_df, min_user_ratings=5, min_book_ratings=5):
        """Filter ratings to improve collaborative filtering quality."""
        df = ratings_df.copy()
        
        # Filter users with minimum ratings
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        df = df[df['user_id'].isin(valid_users)]
        
        # Filter books with minimum ratings
        book_counts = df['book_id'].value_counts()
        valid_books = book_counts[book_counts >= min_book_ratings].index
        df = df[df['book_id'].isin(valid_books)]
        
        print(f"  Filtered: {len(ratings_df):,} → {len(df):,} ratings")
        return df
    
    def get_user_history_count(self, user_id):
        """Get number of ratings for a user."""
        return len(self.ratings_df[self.ratings_df['user_id'] == user_id])
    
    def is_cold_start_user(self, user_id, threshold=3):
        """Check if user is cold-start (few or no ratings)."""
        return self.get_user_history_count(user_id) < threshold
    
    def add_friends(self, user_id, friend_ids):
        """
        Add friends for a user (for friend-based recommendations).
        Adapted from MainakRepositor's friend system.
        
        Args:
            user_id: User ID
            friend_ids: List of friend user IDs
        """
        self.friends_map[user_id] = list(friend_ids)
    
    def get_friend_recommendations(self, user_id, n=10):
        """
        Get recommendations based on friends' reading history.
        Adapted from MainakRepositor's "Trending among your friends".
        
        Args:
            user_id: User ID
            n: Number of recommendations
            
        Returns:
            DataFrame with friend-based recommendations
        """
        friends = self.friends_map.get(user_id, [])
        
        if not friends:
            return pd.DataFrame()
        
        # Get books rated by friends
        friend_ratings = self.ratings_df[
            self.ratings_df['user_id'].isin(friends)
        ]
        
        if friend_ratings.empty:
            return pd.DataFrame()
        
        # Aggregate friend ratings
        friend_books = friend_ratings.groupby('book_id').agg(
            avg_rating=('rating', 'mean'),
            friend_count=('user_id', 'nunique')
        ).reset_index()
        
        # Sort by number of friends who rated and average rating
        friend_books['score'] = (
            friend_books['avg_rating'] * 0.5 + 
            friend_books['friend_count'] * 0.5
        )
        friend_books = friend_books.sort_values('score', ascending=False)
        
        # Exclude books user has already rated
        user_rated = set(
            self.ratings_df[self.ratings_df['user_id'] == user_id]['book_id']
        )
        friend_books = friend_books[~friend_books['book_id'].isin(user_rated)]
        
        # Add book information
        result = friend_books.head(n).merge(self.books_df, on='book_id', how='left')
        
        return result
    
    def get_recommendations(self, user_id=None, book_id=None, n=10, weights=None, selected_genres=None):
        """
        Get hybrid recommendations combining multiple strategies.
        
        Args:
            user_id: Optional user ID for personalized recommendations
            book_id: Optional book_id for item-based recommendations
            n: Number of recommendations
            weights: Dict with strategy weights
            selected_genres: List of selected genres for cold-start filtering
            
        Returns:
            DataFrame with recommendations
        """
        if weights is None:
            weights = {
                'popularity': 0.2,
                'content': 0.3,
                'collaborative': 0.5
            }
        
        all_recommendations = []
        
        # Adjust weights based on user type
        if user_id is not None:
            if self.is_cold_start_user(user_id):
                # Cold-start: Favor popularity and content
                weights = {
                    'popularity': 0.5,
                    'content': 0.4,
                    'collaborative': 0.1
                }
            else:
                # Warm user: Favor collaborative
                weights = {
                    'popularity': 0.1,
                    'content': 0.2,
                    'collaborative': 0.7
                }
        
        # 1. Popularity-based recommendations
        if weights.get('popularity', 0) > 0:
            # If cold start and genres selected, filter popularity by genre
            if user_id is not None and self.is_cold_start_user(user_id) and selected_genres and self.loader:
                # Get more candidates to filter
                pop_recs = self.popularity.get_best_books(n=100)
                
                # Filter by genre
                try:
                    tag_ids = self.loader.tags_df[self.loader.tags_df['tag_name'].isin(selected_genres)]['tag_id']
                    goodreads_ids = self.loader.book_tags_df[self.loader.book_tags_df['tag_id'].isin(tag_ids)]['goodreads_book_id'].unique()
                    
                    # Filter pop_recs (which has book_id)
                    # We need to map book_id -> goodreads_book_id to check
                    # Or simpler: get the subset of books that are in the genre
                    
                    # Get IDs of books in these genres
                    genre_book_ids = self.books_df[self.books_df['goodreads_book_id'].isin(goodreads_ids)]['book_id']
                    
                    pop_recs = pop_recs[pop_recs['book_id'].isin(genre_book_ids)].head(n)
                except Exception as e:
                    print(f"Error filtering by genre: {e}")
                    pop_recs = pop_recs.head(n)
            else:
                pop_recs = self.popularity.get_best_books(n=n)
                
            if not pop_recs.empty:
                pop_recs = pop_recs.copy()
                pop_recs['strategy'] = 'popularity'
                pop_recs['weight'] = weights['popularity']
                all_recommendations.append(pop_recs)
        
        # 2. Content-based recommendations
        if weights.get('content', 0) > 0:
            if book_id:
                content_recs = self.content.get_similar_to_book(book_id, n=n)
            elif user_id:
                # Get user's top-rated book and find similar
                user_ratings = self.ratings_df[
                    self.ratings_df['user_id'] == user_id
                ].sort_values('rating', ascending=False)
                
                if not user_ratings.empty:
                    top_book_id = user_ratings.iloc[0]['book_id']
                    content_recs = self.content.get_similar_to_book(top_book_id, n=n)
                else:
                    content_recs = pd.DataFrame()
            else:
                content_recs = pd.DataFrame()
            
            if not content_recs.empty:
                content_recs = content_recs.copy()
                content_recs['strategy'] = 'content'
                content_recs['weight'] = weights['content']
                all_recommendations.append(content_recs)
        
        # 3. Collaborative recommendations
        if weights.get('collaborative', 0) > 0 and user_id is not None:
            collab_recs = self.collaborative.get_user_recommendations(user_id, n=n)
            if not collab_recs.empty:
                collab_recs = collab_recs.copy()
                collab_recs['strategy'] = 'collaborative'
                collab_recs['weight'] = weights['collaborative']
                all_recommendations.append(collab_recs)
        
        # 4. Friend recommendations (if available)
        if user_id is not None:
            friend_recs = self.get_friend_recommendations(user_id, n=n//2)
            if not friend_recs.empty:
                friend_recs = friend_recs.copy()
                friend_recs['strategy'] = 'friends'
                friend_recs['weight'] = 0.3
                all_recommendations.append(friend_recs)
        
        # Combine all recommendations
        if not all_recommendations:
            return self.popularity.get_best_books(n=n)
        
        combined = pd.concat(all_recommendations, ignore_index=True)
        
        # Calculate final score
        if 'weighted_rating' in combined.columns:
            combined['final_score'] = (
                combined['weight'] * 
                combined.get('weighted_rating', combined.get('avg_rating', 5))
            )
        elif 'avg_rating' in combined.columns:
            combined['final_score'] = combined['weight'] * combined['avg_rating']
        elif 'predicted_rating' in combined.columns:
            combined['final_score'] = combined['weight'] * combined['predicted_rating']
        else:
            combined['final_score'] = combined['weight']
        
        # Deduplicate by book_id, keeping highest score
        combined = combined.sort_values('final_score', ascending=False)
        combined = combined.drop_duplicates(subset='book_id', keep='first')
        
        # Select output columns
        output_cols = ['book_id', 'title', 'author', 'strategy', 'final_score']
        if 'image_url_m' in combined.columns:
            output_cols.append('image_url_m')
        
        available_cols = [c for c in output_cols if c in combined.columns]
        
        return combined.head(n)[available_cols]
    
    def get_because_you_read(self, user_id, n=10):
        """
        Get "Because you read X" style recommendations.
        Shows recommendations based on user's recent reads.
        
        Args:
            user_id: User ID
            n: Number of recommendations
            
        Returns:
            Dict with {source_book: [recommendations]}
        """
        user_ratings = self.ratings_df[
            self.ratings_df['user_id'] == user_id
        ].sort_values('rating', ascending=False)
        
        if user_ratings.empty:
            return {}
        
        # Get top 3 books the user rated highly
        top_books = user_ratings.head(3)
        
        result = {}
        for _, row in top_books.iterrows():
            book_id = row['book_id']
            book_info = self.books_df[self.books_df['book_id'] == book_id]
            
            if not book_info.empty:
                title = book_info.iloc[0].get('title', book_id)
                similar = self.content.get_similar_to_book(book_id, n=n//3)
                result[title] = similar
        
        return result
    
    def get_personalized_sections(self, user_id, friends=None):
        """
        Get all recommendation sections for a user's homepage.
        Adapted from MainakRepositor's multi-section layout.
        
        Args:
            user_id: User ID
            friends: Optional list of friend user IDs
            
        Returns:
            Dict with recommendation sections
        """
        if friends:
            self.add_friends(user_id, friends)
        
        sections = {}
        
        # 1. Top Picks (weighted popularity)
        sections['top_picks'] = self.popularity.get_best_books(n=10)
        
        # 2. Because You Read... (content-based)
        sections['because_you_read'] = self.get_because_you_read(user_id, n=10)
        
        # 3. Users Like You (collaborative)
        if not self.is_cold_start_user(user_id):
            sections['users_like_you'] = self.collaborative.get_user_recommendations(
                user_id, n=10
            )
        else:
            sections['users_like_you'] = pd.DataFrame()
        
        # 4. Friend Recommendations
        friend_recs = self.get_friend_recommendations(user_id, n=10)
        if not friend_recs.empty:
            sections['trending_with_friends'] = friend_recs
        
        # 5. Favorite Authors
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        if not user_ratings.empty:
            user_books = user_ratings.merge(self.books_df, on='book_id')
            if 'author' in user_books.columns:
                favorite_authors = user_books['author'].value_counts().head(3).index
                author_recs = []
                for author in favorite_authors:
                    author_books = self.content.get_similar_by_author(author, n=5)
                    author_recs.append(author_books)
                if author_recs:
                    sections['favorite_authors'] = pd.concat(author_recs, ignore_index=True)
        
        return sections
    
    def recommend(self, user_id=None, book_id=None, n=10):
        """
        Main recommendation method.
        
        Args:
            user_id: User ID for personalized recommendations
            book_id: book_id for item-based recommendations
            n: Number of recommendations
            
        Returns:
            DataFrame with recommendations
        """
        return self.get_recommendations(user_id=user_id, book_id=book_id, n=n)


if __name__ == '__main__':
    # Test hybrid recommender
    import sys
    sys.path.append('..')
    from data_loader import DataLoader
    
    loader = DataLoader()
    books, users, ratings = loader.load_all()
    
    if books is not None and ratings is not None:
        hybrid = HybridRecommender(books, ratings, users)
        
        # Get a test user
        test_user = ratings['user_id'].value_counts().index[0]
        
        print(f"\n🎯 Hybrid recommendations for user {test_user}:")
        print(hybrid.get_recommendations(user_id=test_user, n=10))
        
        # Test with friends
        friend_ids = ratings['user_id'].value_counts().index[1:5].tolist()
        hybrid.add_friends(test_user, friend_ids)
        
        print(f"\n👥 Friend recommendations for user {test_user}:")
        print(hybrid.get_friend_recommendations(test_user, n=5))
