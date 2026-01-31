"""
Data Loader for Book Recommendation System

Supports the goodbooks-10k dataset from:
https://github.com/zygmuntz/goodbooks-10k

Dataset contains:
- 10,000 books with metadata (title, authors, average rating, cover images)
- ~6 million ratings from 53,424 users
- Ratings from 1-5 (no implicit 0 ratings)
- Book tags/genres

Adapted from:
- MainakRepositor/Book-Recommender: CSV loading patterns
- nikunjsonule/Book-Recommendation-System: Data preprocessing
- fkemeth/book_collaborative_filtering: Used same goodbooks-10k dataset
"""

import os
import pandas as pd
import numpy as np
import requests
from pathlib import Path

# Goodbooks-10k dataset URLs (from GitHub releases)
DATASET_URLS = {
    'books': 'https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/books.csv',
    'ratings': 'https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/ratings.csv',
    'book_tags': 'https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/book_tags.csv',
    'tags': 'https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/tags.csv',
    'to_read': 'https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/master/to_read.csv'
}

DATASET_FILES = {
    'books': 'books.csv',
    'ratings': 'ratings.csv',
    'book_tags': 'book_tags.csv',
    'tags': 'tags.csv',
    'to_read': 'to_read.csv'
}


class DataLoader:
    """
    Handles loading, preprocessing, and validation of goodbooks-10k dataset.
    
    Key features:
    - Automatic download from GitHub
    - Multiple encoding fallbacks for robust CSV reading
    - Genre/tag integration for content-based filtering
    - Memory-efficient loading options
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        self.books_df = None
        self.ratings_df = None
        self.tags_df = None
        self.book_tags_df = None
        
    def download_datasets(self, force=False):
        """
        Download goodbooks-10k dataset files from GitHub.
        
        Args:
            force: If True, re-download even if files exist
        """
        print("📚 Downloading goodbooks-10k dataset...")
        
        # Download core files (books and ratings)
        core_files = ['books', 'ratings']
        
        for name in core_files:
            url = DATASET_URLS[name]
            file_path = self.data_dir / DATASET_FILES[name]
            
            if file_path.exists() and not force:
                print(f"  ✓ {DATASET_FILES[name]} already exists")
                continue
                
            print(f"  ↓ Downloading {DATASET_FILES[name]}...")
            try:
                response = requests.get(url, timeout=120, stream=True)
                response.raise_for_status()
                
                # Stream to file for large files
                with open(file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                print(f"  ✓ Downloaded {DATASET_FILES[name]}")
            except Exception as e:
                print(f"  ✗ Failed to download {name}: {e}")
                return False
        
        # Download optional files (tags)
        optional_files = ['book_tags', 'tags']
        for name in optional_files:
            url = DATASET_URLS[name]
            file_path = self.data_dir / DATASET_FILES[name]
            
            if file_path.exists() and not force:
                continue
                
            try:
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                with open(file_path, 'wb') as f:
                    f.write(response.content)
                print(f"  ✓ Downloaded {DATASET_FILES[name]}")
            except:
                print(f"  ⚠ Optional file {name} not downloaded (tags will be unavailable)")
        
        return True
    
    def _read_csv_robust(self, file_path, **kwargs):
        """
        Read CSV with multiple encoding fallbacks.
        """
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                df = pd.read_csv(
                    file_path, 
                    encoding=encoding,
                    on_bad_lines='skip',
                    low_memory=False,
                    **kwargs
                )
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                print(f"  ✗ Error reading {file_path}: {e}")
                return None
        
        print(f"  ✗ Could not read {file_path} with any encoding")
        return None
    
    def load_books(self):
        """
        Load and preprocess books data.
        
        Goodbooks-10k columns:
        - book_id: Unique ID (1-10000)
        - goodreads_book_id, best_book_id, work_id: Goodreads IDs
        - books_count: Number of editions
        - isbn, isbn13: Book identifiers
        - authors: Author name(s)
        - original_publication_year: Year published
        - original_title, title: Book titles
        - language_code: Language
        - average_rating: Pre-computed average (1-5)
        - ratings_count: Total ratings
        - image_url, small_image_url: Cover images
        """
        file_path = self.data_dir / 'books.csv'
        
        if not file_path.exists():
            print(f"  ✗ {file_path} not found. Run download_datasets() first.")
            return None
        
        df = self._read_csv_robust(file_path)
        if df is None:
            return None
        
        print(f"  ✓ Loaded {file_path.name}")
        
        # Standardize column names to match our system
        df = df.rename(columns={
            'book_id': 'book_id',
            'authors': 'author',
            'title': 'title',
            'original_title': 'original_title',
            'original_publication_year': 'year',
            'average_rating': 'avg_rating',
            'ratings_count': 'rating_count',
            'image_url': 'image_url_m',
            'small_image_url': 'image_url_s',
            'isbn': 'isbn',
            'isbn13': 'isbn13',
            'language_code': 'language'
        })
        
        # Create large image URL (same as medium for goodreads)
        df['image_url_l'] = df['image_url_m']
        
        # Clean year column
        if 'year' in df.columns:
            df['year'] = pd.to_numeric(df['year'], errors='coerce')
            df['year'] = df['year'].fillna(0).astype(int)
        
        # Fill missing values
        df['author'] = df['author'].fillna('Unknown Author')
        df['title'] = df['title'].fillna(df.get('original_title', 'Unknown Title'))
        
        self.books_df = df
        print(f"  📖 Loaded {len(df):,} books")
        return df
    
    def load_ratings(self, sample_size=None):
        """
        Load and preprocess ratings data.
        
        Goodbooks-10k ratings columns:
        - user_id: User ID (1-53424)
        - book_id: Book ID (1-10000)  
        - rating: Rating (1-5)
        
        Args:
            sample_size: Optional limit on number of ratings (for faster testing)
        """
        file_path = self.data_dir / 'ratings.csv'
        
        if not file_path.exists():
            print(f"  ✗ {file_path} not found. Run download_datasets() first.")
            return None
        
        # For large files, optionally load a sample
        if sample_size:
            df = self._read_csv_robust(file_path, nrows=sample_size)
        else:
            df = self._read_csv_robust(file_path)
            
        if df is None:
            return None
        
        print(f"  ✓ Loaded {file_path.name}")
        
        # Ensure correct types
        df['user_id'] = df['user_id'].astype(int)
        df['book_id'] = df['book_id'].astype(int)
        df['rating'] = df['rating'].astype(float)
        
        self.ratings_df = df
        print(f"  ⭐ Loaded {len(df):,} ratings")
        return df
    
    def load_tags(self):
        """
        Load book tags for genre-based recommendations.
        """
        tags_path = self.data_dir / 'tags.csv'
        book_tags_path = self.data_dir / 'book_tags.csv'
        
        if not tags_path.exists() or not book_tags_path.exists():
            print("  ⚠ Tags files not found. Genre filtering will be unavailable.")
            return None, None
        
        self.tags_df = self._read_csv_robust(tags_path)
        self.book_tags_df = self._read_csv_robust(book_tags_path)
        
        if self.tags_df is not None and self.book_tags_df is not None:
            print(f"  🏷️ Loaded {len(self.tags_df):,} tags")
            return self.tags_df, self.book_tags_df
        
        return None, None
    
    def load_all(self, download_if_missing=True, sample_ratings=None):
        """
        Load all datasets and return them.
        
        Args:
            download_if_missing: Download datasets if not found
            sample_ratings: Optional limit on ratings for faster loading
            
        Returns:
            tuple: (books_df, None, ratings_df) - None is for users (not in this dataset)
        """
        print("\n" + "="*60)
        print("📚 GOODBOOKS-10K DATASET LOADER")
        print("="*60 + "\n")
        
        # Check if core data exists, download if needed
        books_exists = (self.data_dir / 'books.csv').exists()
        ratings_exists = (self.data_dir / 'ratings.csv').exists()
        
        if (not books_exists or not ratings_exists) and download_if_missing:
            self.download_datasets()
        
        # Load datasets
        print("\n📂 Loading datasets...")
        self.load_books()
        self.load_ratings(sample_size=sample_ratings)
        self.load_tags()
        
        # Print summary
        self._print_summary()
        
        # Return in expected format (books, users, ratings)
        # Users not in this dataset, but we can derive from ratings
        return self.books_df, None, self.ratings_df
    
    def _print_summary(self):
        """Print dataset summary statistics."""
        print("\n" + "-"*60)
        print("📊 DATASET SUMMARY")
        print("-"*60)
        
        if self.books_df is not None:
            print(f"\n📖 Books: {len(self.books_df):,} records")
            if 'author' in self.books_df.columns:
                print(f"   Unique authors: {self.books_df['author'].nunique():,}")
            if 'avg_rating' in self.books_df.columns:
                print(f"   Avg rating range: {self.books_df['avg_rating'].min():.2f} - {self.books_df['avg_rating'].max():.2f}")
        
        if self.ratings_df is not None:
            print(f"\n⭐ Ratings: {len(self.ratings_df):,} records")
            print(f"   Rating range: {self.ratings_df['rating'].min():.0f} - {self.ratings_df['rating'].max():.0f}")
            print(f"   Average rating: {self.ratings_df['rating'].mean():.2f}")
            print(f"   Unique users: {self.ratings_df['user_id'].nunique():,}")
            print(f"   Unique books rated: {self.ratings_df['book_id'].nunique():,}")
        
        if self.tags_df is not None:
            print(f"\n🏷️ Tags: {len(self.tags_df):,} unique tags")
        
        print("\n" + "="*60 + "\n")
    
    def get_book_by_id(self, book_id):
        """Get book information by book_id."""
        if self.books_df is None:
            return None
        return self.books_df[self.books_df['book_id'] == book_id]
    
    def get_book_genres(self, book_id, top_n=5):
        """
        Get top genres/tags for a book.
        
        Args:
            book_id: Internal Book ID (1-10000)
            top_n: Number of top tags to return
        """
        if self.book_tags_df is None or self.tags_df is None or self.books_df is None:
            return []
        
        # Map book_id to goodreads_book_id
        book_row = self.books_df[self.books_df['book_id'] == book_id]
        if book_row.empty:
            return []
            
        goodreads_id = book_row['goodreads_book_id'].iloc[0]
        
        # Get book's tag IDs with counts
        book_tag_ids = self.book_tags_df[
            self.book_tags_df['goodreads_book_id'] == goodreads_id
        ].nlargest(top_n, 'count')['tag_id']
        
        # Get tag names
        tags = self.tags_df[self.tags_df['tag_id'].isin(book_tag_ids)]
        return tags['tag_name'].tolist()
    
    def filter_by_interactions(self, min_user_ratings=5, min_book_ratings=5):
        """
        Filter to users and books with minimum number of interactions.
        
        Args:
            min_user_ratings: Minimum ratings per user
            min_book_ratings: Minimum ratings per book
        """
        if self.ratings_df is None:
            print("Please load ratings first")
            return None
        
        df = self.ratings_df.copy()
        original_len = len(df)
        
        # Filter users with minimum ratings
        user_counts = df['user_id'].value_counts()
        valid_users = user_counts[user_counts >= min_user_ratings].index
        df = df[df['user_id'].isin(valid_users)]
        
        # Filter books with minimum ratings
        book_counts = df['book_id'].value_counts()
        valid_books = book_counts[book_counts >= min_book_ratings].index
        df = df[df['book_id'].isin(valid_books)]
        
        print(f"Filtered by interactions: {original_len:,} → {len(df):,}")
        print(f"  Users: {len(valid_users):,} | Books: {len(valid_books):,}")
        
        return df


# Module-level functions for quick access
def load_data(data_dir='data', sample_ratings=None):
    """Quick function to load all datasets."""
    loader = DataLoader(data_dir)
    return loader.load_all(sample_ratings=sample_ratings)


if __name__ == '__main__':
    # Test the data loader
    loader = DataLoader()
    
    # Load with a sample for quick testing
    books, users, ratings = loader.load_all(sample_ratings=100000)
    
    if books is not None:
        print("\n📖 Sample books:")
        print(books[['book_id', 'title', 'author', 'avg_rating']].head())
    
    if ratings is not None:
        print("\n⭐ Sample ratings:")
        print(ratings.head())
        
        # Test filtering
        filtered = loader.filter_by_interactions(min_user_ratings=10, min_book_ratings=20)
