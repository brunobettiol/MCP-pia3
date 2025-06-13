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
        """Initialize the blog service with intelligent category-based matching"""
        self.blogs: List[BlogEntry] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.processed_content = []
        self._load_blogs()
        self._build_search_index()
        
        # Domain-specific keyword mappings for better blog category matching
        self.category_keywords = {
            'safe': [
                'safety', 'home safety', 'fall prevention', 'grab bars', 'handrails',
                'lighting', 'lights', 'bathroom safety', 'kitchen safety', 'stairs',
                'ramps', 'modification', 'accessibility', 'home modification',
                'safety equipment', 'non-slip', 'secure', 'protection', 'hazard',
                'accident prevention', 'emergency', 'fire safety', 'carbon monoxide',
                'smoke detector', 'security', 'locks', 'doorbell', 'motion sensor',
                'stair lift', 'shower seat', 'toilet safety', 'walker', 'cane'
            ],
            'healthy': [
                'health', 'medical', 'medication', 'doctor', 'healthcare', 'chronic',
                'condition', 'disease', 'treatment', 'therapy', 'nutrition', 'diet',
                'exercise', 'fitness', 'wellness', 'mental health', 'depression',
                'anxiety', 'sleep', 'pain management', 'hospice', 'palliative',
                'end of life', 'terminal', 'comfort care', 'symptom management',
                'quality of life', 'medical equipment', 'oxygen', 'CPAP', 'diabetes',
                'heart disease', 'stroke', 'dementia', 'alzheimer', 'memory'
            ],
            'prepared': [
                'emergency', 'planning', 'preparation', 'disaster', 'evacuation',
                'emergency kit', 'supplies', 'legal', 'documents', 'will', 'trust',
                'power of attorney', 'advance directive', 'financial planning',
                'insurance', 'medicare', 'medicaid', 'benefits', 'estate planning',
                'guardianship', 'conservatorship', 'elder law', 'attorney', 'lawyer',
                'legal advice', 'probate', 'inheritance', 'tax planning', 'retirement',
                'social security', 'pension', 'investment', 'savings', 'budget'
            ],
            'caregiver_care': [
                'caregiver', 'caregiving', 'caregiver stress', 'caregiver burnout',
                'respite care', 'support', 'self-care', 'caregiver support',
                'family caregiver', 'caring for elderly', 'caregiver resources',
                'caregiver health', 'caregiver wellness', 'support groups',
                'counseling', 'therapy', 'grief', 'bereavement', 'loss',
                'emotional support', 'mental health', 'stress management',
                'work-life balance', 'caregiver tips', 'caregiver advice'
            ]
        }
    
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

    def _calculate_category_score(self, query: str, blog: BlogEntry) -> float:
        """Calculate category relevance score based on domain keywords"""
        query_lower = query.lower()
        max_score = 0.0
        
        # Check each category and its keywords
        for category, keywords in self.category_keywords.items():
            if blog.category.lower() == category or category in blog.category.lower():
                # Calculate keyword match score for this category
                keyword_matches = sum(1 for keyword in keywords if keyword in query_lower)
                if keyword_matches > 0:
                    # Score based on percentage of keywords matched and relevance
                    category_score = (keyword_matches / len(keywords)) * 10
                    max_score = max(max_score, category_score)
        
        return max_score

    def _calculate_direct_keyword_score(self, query: str, blog: BlogEntry) -> float:
        """Calculate direct keyword matching score"""
        query_lower = query.lower()
        score = 0.0
        
        # Check title for direct matches (highest weight)
        if any(word in blog.title.lower() for word in query_lower.split()):
            score += 5.0
        
        # Check tags for matches
        for tag in blog.tags:
            if tag.lower() in query_lower:
                score += 3.0
        
        # Check summary for matches
        if blog.summary and any(word in blog.summary.lower() for word in query_lower.split()):
            score += 2.0
        
        # Key phrases that should boost relevance
        key_phrases = [
            'safety', 'lighting', 'grab bars', 'handrails', 'home modification',
            'health', 'medication', 'caregiver', 'emergency planning', 'legal'
        ]
        
        blog_text = f"{blog.title} {blog.summary or ''} {' '.join(blog.tags)}".lower()
        for phrase in key_phrases:
            if phrase in query_lower and phrase in blog_text:
                score += 2.0
        
        return min(score, 10.0)  # Cap at 10

    def _build_search_index(self):
        """Build TF-IDF index for all blog content with category emphasis"""
        if not self.blogs:
            return
        
        # Combine title, content, tags, summary, category, and subcategory for each blog
        self.processed_content = []
        for blog in self.blogs:
            # Emphasize category and title much more
            combined_text = f"{blog.category} " * 8 + \
                           f"{blog.subcategory or ''} " * 5 + \
                           f"{blog.title} " * 5 + \
                           f"{blog.summary or ''} " * 3 + \
                           f"{' '.join(blog.tags)} " * 3 + \
                           f"{blog.content}"
            
            processed = self._preprocess_text(combined_text)
            self.processed_content.append(processed)
        
        # Initialize TF-IDF vectorizer with optimized parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,  # Smaller vocabulary for better focus
            stop_words='english',  # Remove common English stop words
            ngram_range=(1, 3),  # Include trigrams for better phrase matching
            min_df=1,  # Minimum document frequency
            max_df=0.8,  # Maximum document frequency
            sublinear_tf=True,  # Apply sublinear tf scaling
            norm='l2'  # Normalize vectors
        )
        
        # Build TF-IDF matrix
        if self.processed_content:
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_content)
            except Exception as e:
                print(f"Error building TF-IDF matrix: {e}")
                # Fallback to simpler configuration
                self.tfidf_vectorizer = TfidfVectorizer(
                    stop_words=None,
                    ngram_range=(1, 1),
                    min_df=1,
                    max_df=1.0
                )
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_content)

    def get_all_blogs(self) -> BlogList:
        """Get all blogs"""
        return BlogList(blogs=self.blogs, total=len(self.blogs))

    def search_blogs(self, query: str, limit: int = 10) -> BlogList:
        """Search blogs using hybrid scoring (category + keywords + TF-IDF)"""
        if not query or not self.blogs:
            return BlogList(blogs=[], total=0)
        
        scored_blogs = []
        
        for i, blog in enumerate(self.blogs):
            # Calculate multiple scores
            category_score = self._calculate_category_score(query, blog)
            direct_keyword_score = self._calculate_direct_keyword_score(query, blog)
            
            # TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 10  # Scale to 0-10
            
            # Combined score with weights
            # Category matching is most important, then direct keywords, then TF-IDF
            combined_score = (category_score * 0.5) + (direct_keyword_score * 0.3) + (tfidf_score * 0.2)
            
            if combined_score > 0.5:  # Minimum threshold
                scored_blogs.append((blog, combined_score))
        
        # Sort by score and return top results
        scored_blogs.sort(key=lambda x: x[1], reverse=True)
        filtered_results = [blog for blog, score in scored_blogs[:limit]]
        
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
        """Get blog recommendations based on query using hybrid scoring"""
        if not query or not self.blogs:
            return BlogList(blogs=[], total=0)
        
        scored_blogs = []
        
        for i, blog in enumerate(self.blogs):
            # Calculate multiple scores
            category_score = self._calculate_category_score(query, blog)
            direct_keyword_score = self._calculate_direct_keyword_score(query, blog)
            
            # TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 10  # Scale to 0-10
            
            # Combined score with weights favoring category matching
            combined_score = (category_score * 0.6) + (direct_keyword_score * 0.3) + (tfidf_score * 0.1)
            
            if combined_score > 1.0:  # Higher threshold for recommendations
                scored_blogs.append((blog, combined_score))
        
        # Sort by score and return top results
        scored_blogs.sort(key=lambda x: x[1], reverse=True)
        recommended_blogs = [blog for blog, score in scored_blogs[:limit]]
        
        return BlogList(blogs=recommended_blogs, total=len(recommended_blogs))

    def recommend_best_blog_with_score(self, query: str) -> tuple[Optional[BlogEntry], float]:
        """Get the best blog recommendation with hybrid scoring"""
        if not query or not self.blogs:
            return None, 0.0
        
        best_blog = None
        best_score = 0.0
        
        for i, blog in enumerate(self.blogs):
            # Calculate multiple scores
            category_score = self._calculate_category_score(query, blog)
            direct_keyword_score = self._calculate_direct_keyword_score(query, blog)
            
            # TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 10  # Scale to 0-10
            
            # Combined score with heavy emphasis on category matching
            combined_score = (category_score * 0.7) + (direct_keyword_score * 0.2) + (tfidf_score * 0.1)
            
            if combined_score > best_score:
                best_score = combined_score
                best_blog = blog
        
        return best_blog, best_score

    def get_best_recommendation(self, query: str) -> Optional[str]:
        """Get the single best blog recommendation source_id based on query with threshold"""
        blog, score = self.recommend_best_blog_with_score(query)
        
        # Set minimum relevance threshold - higher for quality
        min_relevance_score = 2.0  # Require meaningful category or keyword matching
        
        if blog and score >= min_relevance_score:
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