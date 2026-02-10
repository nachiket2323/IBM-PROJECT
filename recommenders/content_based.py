"""
Content-Based Recommender

Adapted from:
- MainakRepositor/Book-Recommender: Author-based recommendations
- mujtabaali02/Book-Recommendation-System: Content-based filtering concepts

Uses book attributes (author, publisher, title) to find similar books.
Great for users who like specific authors or genres.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    """
    Recommends books based on content similarity.
    
    Strategies:
    1. Author similarity: Books by same/similar authors
    2. Title similarity: TF-IDF on book titles
    3. Combined: Weighted author + title similarity
    """
    
    def __init__(self, books_df, ratings_df=None, loader=None):
        """
        Initialize with books data.
        
        Args:
            books_df: DataFrame with book information
            ratings_df: Optional ratings for popularity weighting
            loader: DataLoader instance for accessing tags
        """
        self.books_df = books_df.copy()
        self.ratings_df = ratings_df
        self.loader = loader
        
        # TF-IDF vectorizer for title similarity
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        
        # Build similarity matrices
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data for content-based filtering."""
        if self.books_df is None or self.books_df.empty:
            print("Error: No books data available")
            return
        
        # Ensure we have required columns
        required_cols = ['book_id', 'title', 'author']
        missing = [col for col in required_cols if col not in self.books_df.columns]
        if missing:
            print(f"Warning: Missing columns {missing}")
            return
        
        # Clean and prepare title text
        self.books_df['clean_title'] = self.books_df['title'].fillna('').str.lower()
        self.books_df['clean_author'] = self.books_df['author'].fillna('Unknown').str.lower()
        
        # Enrich with tags if available
        self.books_df['tags_str'] = ''
        if self.loader and self.loader.book_tags_df is not None and self.loader.tags_df is not None:
            print("  Enriching content with tags...")
            try:
                # Merge book_tags with tags to get names
                bt = self.loader.book_tags_df.merge(self.loader.tags_df, on='tag_id')
                
                # Filter to top tags per book (top 15) to reduce noise
                bt = bt.sort_values('count', ascending=False).groupby('goodreads_book_id').head(15)
                
                # Aggregate tags into string
                book_tags_str = bt.groupby('goodreads_book_id')['tag_name'].apply(lambda x: ' '.join(x)).reset_index()
                book_tags_str.columns = ['goodreads_book_id', 'tags_content']
                
                # Merge back to books_df
                self.books_df = self.books_df.merge(book_tags_str, on='goodreads_book_id', how='left')
                self.books_df['tags_str'] = self.books_df['tags_content'].fillna('').str.lower()
                self.books_df = self.books_df.drop('tags_content', axis=1)
                print("  ✓ Tags integrated into content")
            except Exception as e:
                print(f"  ✗ Error integrating tags: {e}")
        
        # Build TF-IDF matrix for titles
        self._build_tfidf_matrix()
        
        print(f"✓ Content-based recommender ready with {len(self.books_df):,} books")
    
    def _build_tfidf_matrix(self):
        """Build TF-IDF matrix for book titles and tags."""
        try:
            # Combine title, author, and TAGS for richer content representation
            self.books_df['content'] = (
                self.books_df['clean_title'] + ' ' + 
                self.books_df['clean_author'] + ' ' +
                self.books_df.get('tags_str', '')
            )
            
            self.tfidf_vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=10000, # Increased features for tags
                ngram_range=(1, 2)
            )
            
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(
                self.books_df['content']
            )
            
            print(f"  ✓ Built TF-IDF matrix: {self.tfidf_matrix.shape}")
        except Exception as e:
            print(f"  ✗ Error building TF-IDF: {e}")
    
    def get_similar_by_author(self, author, n=10, exclude_book_id=None):
        """
        Get books by the same or similar authors.
        Adapted from MainakRepositor's author-based recommendations.
        
        Args:
            author: Author name to match
            n: Number of recommendations
            exclude_book_id: book_id to exclude (the source book)
            
        Returns:
            DataFrame with similar books
        """
        if self.books_df is None:
            return pd.DataFrame()
        
        # Find books by matching author
        author_lower = author.lower()
        author_books = self.books_df[
            self.books_df['clean_author'].str.contains(author_lower, na=False)
        ].copy()
        
        # Exclude source book if specified
        if exclude_book_id:
            author_books = author_books[author_books['book_id'] != exclude_book_id]
        
        # Sort by rating if available (from pre-computed or ratings data)
        if 'avg_rating' in author_books.columns:
            author_books = author_books.sort_values('avg_rating', ascending=False)
        elif self.ratings_df is not None:
            book_ratings = self.ratings_df.groupby('book_id').agg(
                computed_avg_rating=('rating', 'mean'),
                count=('rating', 'count')
            ).reset_index()
            
            author_books = author_books.merge(book_ratings, on='book_id', how='left')
            author_books['computed_avg_rating'] = author_books['computed_avg_rating'].fillna(0)
            author_books = author_books.sort_values('computed_avg_rating', ascending=False)
        
        result_cols = ['book_id', 'title', 'author', 'year', 'avg_rating', 'image_url_m']
        available_cols = [c for c in result_cols if c in author_books.columns]
        
        return author_books.head(n)[available_cols]
    
    def get_similar_by_title(self, book_id, n=10):
        """
        Get books with similar titles using TF-IDF cosine similarity.
        
        Args:
            book_id: book_id of the source book
            n: Number of recommendations
            
        Returns:
            DataFrame with similar books
        """
        if self.tfidf_matrix is None:
            print("TF-IDF matrix not built")
            return pd.DataFrame()
        
        # Find the index of the source book
        book_indices = self.books_df[self.books_df['book_id'] == book_id].index
        
        if len(book_indices) == 0:
            print(f"Book with book_id {book_id} not found")
            return pd.DataFrame()
        
        idx = book_indices[0]
        
        # Get the position in our TF-IDF matrix
        try:
            tfidf_idx = self.books_df.index.get_loc(idx)
        except:
            tfidf_idx = idx
        
        # Calculate similarity scores
        similarity_scores = cosine_similarity(
            self.tfidf_matrix[tfidf_idx:tfidf_idx+1], 
            self.tfidf_matrix
        ).flatten()
        
        # Get top similar books (excluding the source)
        similar_indices = similarity_scores.argsort()[::-1][1:n+1]
        
        similar_books = self.books_df.iloc[similar_indices].copy()
        similar_books['similarity_score'] = similarity_scores[similar_indices]
        
        result_cols = ['book_id', 'title', 'author', 'similarity_score', 'image_url_m']
        available_cols = [c for c in result_cols if c in similar_books.columns]
        
        return similar_books[available_cols]
    
    def get_similar_to_book(self, book_id, n=10):
        """
        Get comprehensive similar books using multiple strategies.
        Combines author and title similarity.
        
        Args:
            book_id: book_id of the source book
            n: Number of recommendations
            
        Returns:
            DataFrame with similar books
        """
        book_row = self.books_df[self.books_df['book_id'] == book_id]
        
        if book_row.empty:
            print(f"Book with book_id {book_id} not found")
            return pd.DataFrame()
        
        author = book_row['author'].values[0]
        
        # Get author-based recommendations
        author_recs = self.get_similar_by_author(author, n=n, exclude_book_id=book_id)
        author_recs['source'] = 'author'
        
        # Get title-based recommendations  
        title_recs = self.get_similar_by_title(book_id, n=n)
        title_recs['source'] = 'title'
        
        # Combine and deduplicate
        combined = pd.concat([author_recs, title_recs], ignore_index=True)
        combined = combined.drop_duplicates(subset='book_id', keep='first')
        
        return combined.head(n)
    
    def get_books_by_publisher(self, publisher, n=10, exclude_book_id=None):
        """
        Get books from the same publisher.
        
        Args:
            publisher: Publisher name to match
            n: Number of recommendations
            exclude_book_id: book_id to exclude
            
        Returns:
            DataFrame with publisher's books
        """
        if self.books_df is None or 'publisher' not in self.books_df.columns:
            return pd.DataFrame()
        
        publisher_lower = publisher.lower()
        publisher_books = self.books_df[
            self.books_df['publisher'].str.lower().str.contains(publisher_lower, na=False)
        ]
        
        if exclude_book_id:
            publisher_books = publisher_books[publisher_books['book_id'] != exclude_book_id]
        
        result_cols = ['book_id', 'title', 'author', 'publisher', 'year', 'image_url_m']
        available_cols = [c for c in result_cols if c in publisher_books.columns]
        
        return publisher_books.head(n)[available_cols]
    
    def search_books(self, query, n=10):
        """
        Search for books matching a query.
        
        Args:
            query: Search query (matches title, author)
            n: Number of results
            
        Returns:
            DataFrame with matching books
        """
        if self.books_df is None:
            return pd.DataFrame()
        
        # Ensure required columns exist
        required_cols = ['clean_title', 'clean_author']
        missing = [c for c in required_cols if c not in self.books_df.columns]
        if missing:
            print(f"Warning: Missing search columns {missing}")
            # Try to recover or return empty
            return pd.DataFrame()

        query_lower = query.lower()
        
        # Use regex=False to treat query as literal string, preventing errors with special chars like '['
        try:
            matches = self.books_df[
                (self.books_df['clean_title'].str.contains(query_lower, na=False, regex=False)) |
                (self.books_df['clean_author'].str.contains(query_lower, na=False, regex=False))
            ]
        except Exception as e:
            print(f"Search error: {e}")
            return pd.DataFrame()
        
        result_cols = ['book_id', 'title', 'author', 'year', 'publisher', 'image_url_m', 'image_url_l']
        available_cols = [c for c in result_cols if c in matches.columns]
        
        return matches.head(n)[available_cols]
    
    def recommend(self, book_id=None, author=None, n=10):
        """
        Get content-based recommendations.
        
        Args:
            book_id: Source book book_id
            author: Source author name  
            n: Number of recommendations
            
        Returns:
            DataFrame with recommendations
        """
        if book_id:
            return self.get_similar_to_book(book_id, n)
        elif author:
            return self.get_similar_by_author(author, n)
        else:
            # Return random diverse selection
            return self.books_df.sample(min(n, len(self.books_df)))[
                ['book_id', 'title', 'author', 'image_url_m']
            ]


if __name__ == '__main__':
    # Test with sample data
    from data_loader import DataLoader
    
    loader = DataLoader()
    books, users, ratings = loader.load_all()
    
    if books is not None:
        recommender = ContentBasedRecommender(books, ratings)
        
        print("\n📚 Search for 'Harry Potter':")
        results = recommender.search_books('harry potter', 5)
        print(results)
        
        if not results.empty:
            test_book_id = results.iloc[0]['book_id']
            print(f"\n📖 Books similar to {test_book_id}:")
            print(recommender.get_similar_to_book(test_book_id, 5))
