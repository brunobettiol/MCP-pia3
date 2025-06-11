import json
import os
from typing import List, Dict, Optional
import re
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from app.models.blog import BlogEntry, BlogList, BlogResponse
from app.core.config import settings


class BlogService:
    def __init__(self):
        """Initialize the blog service with TF-IDF indexing"""
        self.blogs: List[BlogEntry] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.processed_content = []
        self._load_blogs()
        self._build_search_index()
    
    def _parse_firestore_date(self, date_obj) -> Optional[datetime]:
        """Parse Firestore date format with __time__ field"""
        if isinstance(date_obj, dict) and "__time__" in date_obj:
            try:
                return datetime.fromisoformat(date_obj["__time__"].replace("Z", "+00:00"))
            except:
                return None
        elif isinstance(date_obj, str):
            try:
                return datetime.fromisoformat(date_obj.replace("Z", "+00:00"))
            except:
                return None
        return None
    
    def _load_blogs(self):
        """Load all blog entries from Firestore export JSON files"""
        blogs_dir = os.path.join(settings.BASE_DIR, "data", "blogs")
        if not os.path.exists(blogs_dir):
            return
        
        for filename in os.listdir(blogs_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(blogs_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        firestore_data = json.load(f)
                        
                        # Handle Firestore export format
                        if 'data' in firestore_data:
                            # This is a Firestore export
                            for blog_id, blog_data in firestore_data['data'].items():
                                try:
                                    # Clean and process the blog data
                                    processed_data = {
                                        'id': blog_data.get('id', blog_id),
                                        'title': blog_data.get('title', ''),
                                        'content': self._strip_html(blog_data.get('content', '')),
                                        'author': blog_data.get('author', ''),
                                        'category': blog_data.get('category', ''),
                                        'subcategory': blog_data.get('subcategory'),
                                        'summary': blog_data.get('summary'),
                                        'tags': blog_data.get('tags', []),
                                        'featured': blog_data.get('featured', False),
                                        'contentType': blog_data.get('contentType'),
                                        'createdAt': self._parse_firestore_date(blog_data.get('createdAt')) or datetime.now(),
                                        'updatedAt': self._parse_firestore_date(blog_data.get('updatedAt')),
                                        'publishedDate': self._parse_firestore_date(blog_data.get('publishedDate'))
                                    }
                                    
                                    blog = BlogEntry(**processed_data)
                                    self.blogs.append(blog)
                                except Exception as e:
                                    print(f"Error processing blog {blog_id}: {e}")
                        else:
                            # Handle old format (array of blog objects)
                            if isinstance(firestore_data, list):
                                for item in firestore_data:
                                    blog = BlogEntry(**item)
                                    self.blogs.append(blog)
                            else:
                                # Single blog object
                                blog = BlogEntry(**firestore_data)
                                self.blogs.append(blog)
                                
                except Exception as e:
                    print(f"Error loading {filename}: {e}")

    def _strip_html(self, html_content: str) -> str:
        """Strip HTML tags from content and return clean text"""
        if not html_content:
            return ""
        
        # Remove HTML tags using regex
        clean_text = re.sub(r'<[^>]+>', ' ', html_content)
        
        # Replace common HTML entities
        clean_text = clean_text.replace('&nbsp;', ' ')
        clean_text = clean_text.replace('&amp;', '&')
        clean_text = clean_text.replace('&lt;', '<')
        clean_text = clean_text.replace('&gt;', '>')
        clean_text = clean_text.replace('&quot;', '"')
        clean_text = clean_text.replace('&#39;', "'")
        
        # Normalize whitespace
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()
        
        return clean_text

    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess text for better matching"""
        if not text:
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _build_search_index(self):
        """Build TF-IDF index for all blog content"""
        if not self.blogs:
            return
        
        # Combine title, content, tags, summary, category, and subcategory for each blog
        self.processed_content = []
        for blog in self.blogs:
            # Combine all text fields with appropriate weights
            # Title gets the highest weight (3x), tags and summary get medium weight (2x), content gets base weight
            combined_text = f"{blog.title} {blog.title} {blog.title} " + \
                           f"{blog.category} {blog.category} " + \
                           f"{blog.subcategory or ''} {blog.subcategory or ''} " + \
                           f"{blog.summary or ''} {blog.summary or ''} " + \
                           f"{' '.join(blog.tags)} {' '.join(blog.tags)} " + \
                           f"{blog.content}"
            
            processed = self._preprocess_text(combined_text)
            self.processed_content.append(processed)
        
        # Initialize TF-IDF vectorizer with optimized parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,  # Limit vocabulary size
            stop_words='english',  # Remove common English stop words
            ngram_range=(1, 2),  # Use both unigrams and bigrams
            min_df=1,  # Minimum document frequency
            max_df=0.9,  # Maximum document frequency (filter out very common terms)
            sublinear_tf=True,  # Apply sublinear tf scaling
            norm='l2'  # Normalize vectors
        )
        
        # Build TF-IDF matrix
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_content)

    def get_all_blogs(self) -> BlogList:
        """Get all blogs"""
        return BlogList(blogs=self.blogs, total=len(self.blogs))

    def search_blogs(self, query: str, limit: int = 10) -> BlogList:
        """Search blogs using TF-IDF similarity"""
        if not query or not self.blogs or self.tfidf_matrix is None:
            return BlogList(blogs=[], total=0)
        
        # Preprocess query
        processed_query = self._preprocess_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get indices sorted by similarity (descending)
        sorted_indices = similarities.argsort()[::-1]
        
        # Filter results with minimum similarity threshold
        min_similarity = 0.4  # Extremely high threshold for search
        filtered_results = []
        
        for idx in sorted_indices:
            if similarities[idx] >= min_similarity and len(filtered_results) < limit:
                filtered_results.append(self.blogs[idx])
            elif len(filtered_results) >= limit:
                break
        
        return BlogList(blogs=filtered_results, total=len(filtered_results))

    def get_blogs_by_category(self, category: str) -> BlogList:
        """Get blogs filtered by category"""
        filtered_blogs = [blog for blog in self.blogs if blog.category.lower() == category.lower()]
        return BlogList(blogs=filtered_blogs, total=len(filtered_blogs))

    def get_blogs_by_subcategory(self, subcategory: str) -> BlogList:
        """Get blogs filtered by subcategory"""
        filtered_blogs = [blog for blog in self.blogs 
                         if blog.subcategory and blog.subcategory.lower() == subcategory.lower()]
        return BlogList(blogs=filtered_blogs, total=len(filtered_blogs))

    def get_blogs_by_tag(self, tag: str) -> BlogList:
        """Get blogs filtered by tag"""
        filtered_blogs = [blog for blog in self.blogs 
                         if tag.lower() in [t.lower() for t in blog.tags]]
        return BlogList(blogs=filtered_blogs, total=len(filtered_blogs))

    def get_blog_by_source_id(self, source_id: str) -> Optional[BlogEntry]:
        """Get a specific blog by source ID"""
        for blog in self.blogs:
            if blog.source_id == source_id:
                return blog
        return None

    def get_recommendations(self, query: str, limit: int = 5) -> BlogList:
        """Get blog recommendations based on query using TF-IDF similarity"""
        if not query or not self.blogs or self.tfidf_matrix is None:
            return BlogList(blogs=[], total=0)
        
        # Preprocess query
        processed_query = self._preprocess_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Get indices sorted by similarity (descending)
        sorted_indices = similarities.argsort()[::-1]
        
        # Filter results with minimum similarity threshold
        min_similarity = 0.5  # Extremely high threshold for recommendations
        recommended_blogs = []
        
        for idx in sorted_indices:
            if similarities[idx] >= min_similarity and len(recommended_blogs) < limit:
                recommended_blogs.append(self.blogs[idx])
            elif len(recommended_blogs) >= limit:
                break
        
        return BlogList(blogs=recommended_blogs, total=len(recommended_blogs))

    def recommend_best_blog_with_score(self, query: str) -> tuple[Optional[BlogEntry], float]:
        """Get the best blog recommendation with TF-IDF similarity score"""
        if not query or not self.blogs or self.tfidf_matrix is None:
            return None, 0.0
        
        # Preprocess query
        processed_query = self._preprocess_text(query)
        
        # Transform query to TF-IDF vector
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Find the best match
        best_idx = similarities.argmax()
        best_similarity = similarities[best_idx]
        
        # Convert cosine similarity to a more intuitive score (0-10)
        score = best_similarity * 10
        
        # Set minimum relevance threshold (adjust as needed)
        min_relevance_score = 4.0  # Equivalent to 40% cosine similarity - extremely high threshold
        
        if score >= min_relevance_score:
            return self.blogs[best_idx], score
        
        return None, 0.0

    def get_best_recommendation(self, query: str) -> Optional[str]:
        """Get the single best blog recommendation source_id based on query with threshold"""
        blog, score = self.recommend_best_blog_with_score(query)
        
        if blog and score >= 4.0:  # Minimum threshold - extremely high for best quality
            return blog.source_id
        
        return None

    def get_categories(self) -> List[str]:
        """Get all unique categories"""
        categories = set(blog.category for blog in self.blogs)
        return sorted(list(categories))
    
    def get_subcategories(self) -> List[str]:
        """Get all unique subcategories"""
        subcategories = set(blog.subcategory for blog in self.blogs if blog.subcategory)
        return sorted(list(subcategories))

    def get_statistics(self) -> Dict:
        """Get blog statistics"""
        categories = {}
        subcategories = {}
        
        for blog in self.blogs:
            categories[blog.category] = categories.get(blog.category, 0) + 1
            if blog.subcategory:
                subcategories[blog.subcategory] = subcategories.get(blog.subcategory, 0) + 1
        
        return {
            "total_blogs": len(self.blogs),
            "categories": categories,
            "subcategories": subcategories,
            "featured_count": len([blog for blog in self.blogs if blog.featured]),
            "authors": list(set(blog.author for blog in self.blogs))
        }

    def recommend_blogs_for_product(self, product_handle: str, limit: int = 3) -> BlogList:
        """
        Recommend blogs related to a specific product using TF-IDF similarity.
        
        Args:
            product_handle: The product handle.
            limit: Maximum number of blogs to return.
            
        Returns:
            BlogList with recommended blogs.
        """
        try:
            from app.services.shopify_service import ShopifyService
            
            # Get product information
            shopify_service = ShopifyService()
            # Note: This would need to be adapted based on your Shopify service implementation
            # For now, using the product_handle as search query
            search_query = product_handle.replace('-', ' ')
            
            # Find relevant blogs using TF-IDF
            return self.get_recommendations(search_query, limit)
            
        except Exception as e:
            print(f"Error recommending blogs for product: {str(e)}")
            return BlogList(blogs=[], total=0) 