# 📚 Enhanced Book Recommendation System

A hybrid book recommendation system combining best practices from 5 open-source repositories. Uses the **goodbooks-10k** dataset with 10,000 books and 6 million ratings.

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

## ✨ Features

- **📊 Popularity-Based**: IMDB-style weighted ratings for trending books
- **📖 Content-Based**: TF-IDF title similarity + author matching
- **👥 Collaborative Filtering**: User-user similarity + Jaccard distance
- **🔀 Hybrid Ranking**: Smart blending with cold-start handling
- **🤝 Friend Recommendations**: Social filtering based on friend network
- **🏷️ Genre Tags**: 34,000+ book tags for enhanced recommendations

## 📁 Project Structure

```
├── app.py                    # Streamlit web interface
├── data_loader.py            # Dataset loading (auto-downloads)
├── templates.py              # UI components
├── requirements.txt          # Dependencies
└── recommenders/
    ├── popularity.py         # Trending/top-rated books
    ├── content_based.py      # Similar books by content
    ├── collaborative.py      # User-based recommendations
    └── hybrid.py             # Combined approach
```

## 📊 Dataset

Uses [goodbooks-10k](https://github.com/zygmuntz/goodbooks-10k):
- **10,000** books with metadata and cover images
- **~6 million** ratings from 53,424 users
- **34,252** genre tags
- Rating scale: 1-5 stars

Dataset downloads automatically on first run.

## 🔧 Source Repositories

Enhanced from:
1. [MainakRepositor/Book-Recommender](https://github.com/MainakRepositor/Book-Recommender) - Streamlit UI, Jaccard similarity
2. [fkemeth/book_collaborative_filtering](https://github.com/fkemeth/book_collaborative_filtering) - User-user CF
3. [nikunjsonule/Book-Recommendation-System](https://github.com/nikunjsonule/Book-Recommendation-System) - SVD, RMSE=1.63
4. [syedsharin/Book-Recommendation-System-Project](https://github.com/syedsharin/Book-Recommendation-System-Project) - Surprise library
5. [mujtabaali02/Book-Recommendation-System](https://github.com/mujtabaali02/Book-Recommendation-System) - Hybrid approach

## 🎯 Improvements Made

| Enhancement | Description |
|-------------|-------------|
| Modular architecture | Separated recommenders into distinct modules |
| Multiple algorithms | Combined popularity, content, collaborative filtering |
| Cold-start handling | Automatic fallback to popularity for new users |
| Friend system | Social recommendations |
| Auto-download | Dataset downloads from GitHub automatically |
| Modern UI | Multi-tab Streamlit interface with book covers |

## 📄 License
MIT License - See source repositories for their respective licenses.
