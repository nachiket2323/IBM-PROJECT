"""
Collaborative Filtering Recommender

Adapted from:
- nikunjsonule/Book-Recommendation-System: SVD with RMSE=1.63, cosine similarity
- syedsharin/Book-Recommendation-System-Project: Surprise library integration
- MainakRepositor/Book-Recommender: Jaccard distance for user overlap
- fkemeth/book_collaborative_filtering: User-user collaborative filtering

Implements:
- Memory-based: User-user and Item-item similarity (cosine)
- Model-based: SVD matrix factorization
- Jaccard distance for user overlap analysis
"""

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split

# Try to import Surprise library for SVD
try:
    from surprise import SVD, Dataset, Reader
    from surprise.model_selection import cross_validate
    from surprise import accuracy
    SURPRISE_AVAILABLE = True
except ImportError:
    SURPRISE_AVAILABLE = False
    print("Warning: scikit-surprise not installed. SVD will be unavailable.")
    print("Install with: pip install scikit-surprise")


class CollaborativeRecommender:
    """
    Collaborative filtering recommender using multiple approaches.
    
    Memory-based methods:
    - User-user similarity: Find users with similar rating patterns
    - Item-item similarity: Find books rated similarly by users
    
    Model-based methods:
    - SVD: Matrix factorization for latent factor discovery
    
    Enhancement over source repos:
    - Combined memory and model-based approaches
    - Optimized sparse matrix operations for large datasets
    - RMSE evaluation targeting < 2.0 (nikunjsonule achieved 1.63)
    """
    
    def __init__(self, ratings_df, books_df=None, users_df=None):
        """
        Initialize with ratings data.
        
        Args:
            ratings_df: DataFrame with user ratings (user_id, book_id, rating)
            books_df: Optional books info for displaying recommendations
            users_df: Optional users info
        """
        self.ratings_df = ratings_df.copy()
        self.books_df = books_df
        self.users_df = users_df
        
        # Matrix representations
        self.user_item_matrix = None
        self.item_user_matrix = None
        self.user_similarity_matrix = None
        self.item_similarity_matrix = None
        
        # SVD model
        self.svd_model = None
        self.svd_trainset = None
        
        # Mappings for matrix indices
        self.user_to_idx = {}
        self.idx_to_user = {}
        self.item_to_idx = {}
        self.idx_to_item = {}
        
        # Build matrices
        self._prepare_data()
    
    def _prepare_data(self):
        """Prepare data structures for collaborative filtering."""
        if self.ratings_df is None or self.ratings_df.empty:
            print("Error: No ratings data available")
            return
        
        print("Building collaborative filtering matrices...")
        
        # Create user and item mappings
        unique_users = self.ratings_df['user_id'].unique()
        unique_items = self.ratings_df['book_id'].unique()
        
        self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
        self.idx_to_user = {idx: user for user, idx in self.user_to_idx.items()}
        self.item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
        self.idx_to_item = {idx: item for item, idx in self.item_to_idx.items()}
        
        print(f"  Users: {len(unique_users):,} | Items: {len(unique_items):,}")
        
        # Build user-item matrix (sparse for memory efficiency)
        self._build_user_item_matrix()
        
        print(f"✓ Collaborative filtering ready")
    
    def _build_user_item_matrix(self):
        """Build sparse user-item rating matrix."""
        rows = self.ratings_df['user_id'].map(self.user_to_idx)
        cols = self.ratings_df['book_id'].map(self.item_to_idx)
        data = self.ratings_df['rating']
        
        # Drop any NaN values from mapping failures
        valid_mask = rows.notna() & cols.notna()
        rows = rows[valid_mask].astype(int)
        cols = cols[valid_mask].astype(int)
        data = data[valid_mask]
        
        n_users = len(self.user_to_idx)
        n_items = len(self.item_to_idx)
        
        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)),
            shape=(n_users, n_items)
        )
        
        self.item_user_matrix = self.user_item_matrix.T
        
        print(f"  ✓ User-Item matrix: {self.user_item_matrix.shape}")
    
    def compute_item_similarity(self, n_neighbors=50):
        """
        Compute item-item similarity matrix using cosine similarity.
        Adapted from nikunjsonule's item-based CF.
        
        Args:
            n_neighbors: Number of similar items to keep per item
        """
        if self.item_user_matrix is None:
            print("Please prepare data first")
            return
        
        print("Computing item-item similarity (this may take a while)...")
        
        # Compute cosine similarity between items
        # Using batched computation for memory efficiency
        n_items = self.item_user_matrix.shape[0]
        batch_size = 500
        
        similarity_data = []
        
        for i in range(0, n_items, batch_size):
            end_i = min(i + batch_size, n_items)
            batch_sim = cosine_similarity(
                self.item_user_matrix[i:end_i],
                self.item_user_matrix
            )
            
            # Keep only top n_neighbors for each item
            for j, row_idx in enumerate(range(i, end_i)):
                # Get indices of top similar items (excluding self)
                row = batch_sim[j]
                row[row_idx] = -1  # Exclude self
                top_indices = np.argsort(row)[-n_neighbors:]
                
                for idx in top_indices:
                    if row[idx] > 0:
                        similarity_data.append((row_idx, idx, row[idx]))
            
            if i % 1000 == 0 and i > 0:
                print(f"  Processed {i}/{n_items} items")
        
        print(f"  ✓ Computed {len(similarity_data):,} similarity pairs")
        
        # Store as sparse matrix
        if similarity_data:
            rows, cols, data = zip(*similarity_data)
            self.item_similarity_matrix = csr_matrix(
                (data, (rows, cols)),
                shape=(n_items, n_items)
            )
    
    def compute_user_similarity(self, n_neighbors=50):
        """
        Compute user-user similarity matrix.
        Adapted from fkemeth's user-user collaborative filtering.
        """
        if self.user_item_matrix is None:
            print("Please prepare data first")
            return
        
        print("Computing user-user similarity...")
        
        n_users = self.user_item_matrix.shape[0]
        
        # For large datasets, we compute on-demand instead of storing full matrix
        # Here we just validate the matrix is ready
        print(f"  ✓ User similarity ready for on-demand computation")
    
    def jaccard_distance(self, user_ids_a, user_ids_b):
        """
        Calculate Jaccard distance between two sets of users.
        Adapted from MainakRepositor's Jaccard-based recommendations.
        
        Args:
            user_ids_a: Set of user IDs who rated item A
            user_ids_b: Set of user IDs who rated item B
            
        Returns:
            float: Jaccard similarity score (0-1)
        """
        set_a = set(user_ids_a)
        set_b = set(user_ids_b)
        
        union = set_a.union(set_b)
        intersection = set_a.intersection(set_b)
        
        if len(union) == 0:
            return 0.0
        
        return len(intersection) / float(len(union))
    
    def get_similar_items_jaccard(self, book_id, n=10):
        """
        Find similar items using Jaccard distance on user overlap.
        Adapted from MainakRepositor/Book-Recommender.
        
        Args:
            book_id: Source book book_id
            n: Number of similar items
            
        Returns:
            DataFrame with similar books
        """
        # Get users who rated the source book
        source_users = set(
            self.ratings_df[self.ratings_df['book_id'] == book_id]['user_id']
        )
        
        if not source_users:
            print(f"No ratings found for book_id {book_id}")
            return pd.DataFrame()
        
        # Group ratings by book_id
        book_id_users = self.ratings_df.groupby('book_id')['user_id'].apply(set)
        
        # Calculate Jaccard similarity for each book
        similarities = []
        for other_book_id, users in book_id_users.items():
            if other_book_id != book_id:
                sim = self.jaccard_distance(source_users, users)
                if sim > 0:
                    similarities.append((other_book_id, sim))
        
        # Sort by similarity
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Create result DataFrame
        result_data = similarities[:n]
        if not result_data:
            return pd.DataFrame()
        
        result_df = pd.DataFrame(result_data, columns=['book_id', 'jaccard_similarity'])
        
        # Add book information if available
        if self.books_df is not None:
            result_df = result_df.merge(self.books_df, on='book_id', how='left')
        
        return result_df
    
    def get_user_recommendations(self, user_id, n=10):
        """
        Get recommendations for a user based on similar users' ratings.
        
        Args:
            user_id: Target user ID
            n: Number of recommendations
            
        Returns:
            DataFrame with recommended books
        """
        if user_id not in self.user_to_idx:
            print(f"User {user_id} not found in ratings data")
            return pd.DataFrame()
        
        user_idx = self.user_to_idx[user_id]
        user_ratings = self.user_item_matrix[user_idx].toarray().flatten()
        
        # Find items the user hasn't rated
        unrated_mask = user_ratings == 0
        
        # Get similar users
        user_vector = self.user_item_matrix[user_idx]
        similarities = cosine_similarity(user_vector, self.user_item_matrix).flatten()
        
        # Get top similar users (excluding self)
        similarities[user_idx] = -1
        similar_user_indices = np.argsort(similarities)[-20:]
        
        # Aggregate ratings from similar users for unrated items
        predicted_ratings = np.zeros(len(unrated_mask))
        total_similarity = 0
        
        for sim_idx in similar_user_indices:
            if similarities[sim_idx] > 0:
                sim_user_ratings = self.user_item_matrix[sim_idx].toarray().flatten()
                predicted_ratings += similarities[sim_idx] * sim_user_ratings
                total_similarity += similarities[sim_idx]
        
        if total_similarity > 0:
            predicted_ratings /= total_similarity
        
        # Only consider unrated items
        predicted_ratings[~unrated_mask] = -1
        
        # Get top recommendations
        top_indices = np.argsort(predicted_ratings)[-n:][::-1]
        
        recommendations = []
        for idx in top_indices:
            if predicted_ratings[idx] > 0:
                book_id = self.idx_to_item[idx]
                recommendations.append({
                    'book_id': book_id,
                    'predicted_rating': predicted_ratings[idx]
                })
        
        result_df = pd.DataFrame(recommendations)
        
        # Add book information
        if self.books_df is not None and not result_df.empty:
            result_df = result_df.merge(self.books_df, on='book_id', how='left')
        
        return result_df
    
    def train_svd(self, n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02):
        """
        Train SVD model using Surprise library.
        Adapted from nikunjsonule's model-based CF (achieved 1.63 RMSE).
        
        Args:
            n_factors: Number of latent factors
            n_epochs: Training epochs
            lr_all: Learning rate
            reg_all: Regularization factor
        """
        if not SURPRISE_AVAILABLE:
            print("Surprise library not available. Please install: pip install scikit-surprise")
            return None
        
        print("Training SVD model...")
        
        # Prepare data for Surprise
        reader = Reader(rating_scale=(1, 10))
        data = Dataset.load_from_df(
            self.ratings_df[['user_id', 'book_id', 'rating']],
            reader
        )
        
        # Train on full dataset
        self.svd_trainset = data.build_full_trainset()
        
        # Initialize and train SVD
        self.svd_model = SVD(
            n_factors=n_factors,
            n_epochs=n_epochs,
            lr_all=lr_all,
            reg_all=reg_all,
            verbose=True
        )
        
        self.svd_model.fit(self.svd_trainset)
        
        print("✓ SVD model trained")
        
        # Evaluate with cross-validation
        print("\nCross-validation results:")
        cv_results = cross_validate(
            SVD(n_factors=n_factors, n_epochs=n_epochs),
            data,
            measures=['RMSE', 'MAE'],
            cv=3,
            verbose=True
        )
        
        print(f"\nMean RMSE: {cv_results['test_rmse'].mean():.4f}")
        print(f"Mean MAE: {cv_results['test_mae'].mean():.4f}")
        
        return cv_results
    
    def predict_svd(self, user_id, book_id):
        """
        Predict rating using trained SVD model.
        
        Args:
            user_id: User ID
            book_id: Book book_id
            
        Returns:
            float: Predicted rating
        """
        if self.svd_model is None:
            print("Please train SVD model first")
            return None
        
        prediction = self.svd_model.predict(user_id, book_id)
        return prediction.est
    
    def get_svd_recommendations(self, user_id, n=10):
        """
        Get recommendations using SVD predictions.
        
        Args:
            user_id: Target user ID
            n: Number of recommendations
            
        Returns:
            DataFrame with recommended books
        """
        if self.svd_model is None:
            print("Please train SVD model first")
            return pd.DataFrame()
        
        # Get items the user hasn't rated
        user_rated = set(
            self.ratings_df[self.ratings_df['user_id'] == user_id]['book_id']
        )
        
        all_items = set(self.ratings_df['book_id'].unique())
        unrated_items = all_items - user_rated
        
        # Predict ratings for unrated items
        predictions = []
        for book_id in unrated_items:
            pred = self.predict_svd(user_id, book_id)
            predictions.append((book_id, pred))
        
        # Sort by predicted rating
        predictions.sort(key=lambda x: x[1], reverse=True)
        
        # Create result DataFrame
        result_data = predictions[:n]
        result_df = pd.DataFrame(result_data, columns=['book_id', 'predicted_rating'])
        
        # Add book information
        if self.books_df is not None:
            result_df = result_df.merge(self.books_df, on='book_id', how='left')
        
        return result_df
    
    def recommend(self, user_id=None, book_id=None, n=10, method='auto'):
        """
        Get collaborative filtering recommendations.
        
        Args:
            user_id: User ID for user-based recommendations
            book_id: book_id for item-based recommendations
            n: Number of recommendations
            method: 'user', 'item_jaccard', 'svd', or 'auto'
            
        Returns:
            DataFrame with recommendations
        """
        if method == 'auto':
            # Choose based on available data
            if self.svd_model is not None and user_id is not None:
                method = 'svd'
            elif user_id is not None:
                method = 'user'
            elif book_id is not None:
                method = 'item_jaccard'
            else:
                method = 'user'
        
        if method == 'svd' and user_id is not None:
            return self.get_svd_recommendations(user_id, n)
        elif method == 'user' and user_id is not None:
            return self.get_user_recommendations(user_id, n)
        elif method == 'item_jaccard' and book_id is not None:
            return self.get_similar_items_jaccard(book_id, n)
        else:
            print(f"Cannot get recommendations with method={method}, user_id={user_id}, book_id={book_id}")
            return pd.DataFrame()


if __name__ == '__main__':
    # Test collaborative filtering
    from data_loader import DataLoader
    
    loader = DataLoader()
    books, users, ratings = loader.load_all()
    
    if ratings is not None:
        # Filter to users/books with enough interactions
        filtered_ratings = loader.filter_by_interactions(
            min_user_ratings=10,
            min_book_ratings=20
        )
        
        if filtered_ratings is not None and len(filtered_ratings) > 0:
            recommender = CollaborativeRecommender(filtered_ratings, books, users)
            
            # Test Jaccard similarity
            test_book_id = filtered_ratings['book_id'].value_counts().index[0]
            print(f"\n📖 Similar books to {test_book_id} (Jaccard):")
            print(recommender.get_similar_items_jaccard(test_book_id, 5))
            
            # Test user recommendations
            test_user = filtered_ratings['user_id'].value_counts().index[0]
            print(f"\n👤 Recommendations for user {test_user}:")
            print(recommender.get_user_recommendations(test_user, 5))
            
            # Train and test SVD (if Surprise is available)
            if SURPRISE_AVAILABLE:
                print("\n🎯 Training SVD model...")
                recommender.train_svd(n_factors=50, n_epochs=10)
                
                print(f"\n⭐ SVD recommendations for user {test_user}:")
                print(recommender.get_svd_recommendations(test_user, 5))
