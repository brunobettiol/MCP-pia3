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
        """Initialize the blog service optimized for eldercare queries"""
        self.blogs: List[BlogEntry] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.processed_content = []
        
        # Eldercare-focused keyword mappings based on your question categories
        self.eldercare_keywords = {
            # SAFE Category - Home Safety & Fall Prevention
            'home_safety': [
                'home safety', 'safety', 'safe', 'fall prevention', 'falls', 'fall',
                'accident prevention', 'injury prevention', 'home modification',
                'safety assessment', 'hazard removal', 'risk reduction', 'secure',
                'protection', 'safe environment', 'safety measures', 'safety tips',
                'elder safety', 'senior safety'
            ],
            'stairs_safety': [
                'stairs', 'stair', 'handrails', 'handrail', 'stair rail', 'stairway',
                'steps', 'step', 'railing', 'banister', 'stair safety', 'staircase'
            ],
            'bathroom_safety': [
                'bathroom safety', 'grab bars', 'grab bar', 'bathroom', 'shower',
                'toilet', 'bath', 'shower safety', 'bath safety', 'toilet safety',
                'bathroom modification', 'shower grab bar', 'bath rail', 'tub rail'
            ],
            'lighting_safety': [
                'lighting', 'lights', 'light', 'illumination', 'adequate lighting',
                'hallway lighting', 'stair lighting', 'night light', 'motion sensor',
                'LED light', 'bright light', 'pathway light', 'emergency lighting'
            ],
            'mobility_safety': [
                'mobility', 'mobility aid', 'walker', 'cane', 'wheelchair',
                'rollator', 'walking aid', 'mobility device', 'mobility scooter',
                'walking stick', 'crutches', 'transport chair'
            ],
            'emergency_preparedness': [
                'emergency', 'emergency preparedness', 'emergency planning',
                'personal emergency response', 'medical alert', 'emergency system',
                'emergency kit', 'disaster preparedness', 'emergency contacts',
                'emergency response', 'alert system', 'help button'
            ],
            
            # HEALTHY Category - Health & Medical Management
            'health_management': [
                'health', 'healthcare', 'medical', 'wellness', 'health management',
                'medical care', 'health services', 'preventive care', 'health screening',
                'health monitoring', 'medical appointments', 'doctor visits',
                'health check', 'medical checkup', 'health assessment'
            ],
            'chronic_conditions': [
                'chronic conditions', 'chronic', 'diabetes', 'arthritis', 'heart disease',
                'COPD', 'hypertension', 'chronic pain', 'disease management',
                'condition management', 'chronic illness', 'medical conditions'
            ],
            'medication_management': [
                'medication', 'medication management', 'pill', 'prescription',
                'medicine', 'drug', 'medication safety', 'pill organization',
                'medication adherence', 'prescription management', 'dosage',
                'medication reminders', 'pill dispenser', 'medication errors',
                'drug interactions', 'pharmacy', 'medication review'
            ],
            'nutrition_health': [
                'nutrition', 'diet', 'eating', 'meals', 'food', 'balanced meals',
                'healthy eating', 'meal planning', 'grocery shopping', 'cooking',
                'meal preparation', 'dietary needs', 'nutritional needs'
            ],
            'exercise_fitness': [
                'exercise', 'physical activity', 'fitness', 'activity', 'movement',
                'exercise routine', 'physical therapy', 'rehabilitation', 'therapy',
                'strength training', 'balance training', 'flexibility', 'workout'
            ],
            'mental_health': [
                'mental health', 'depression', 'anxiety', 'emotional health',
                'cognitive problems', 'memory issues', 'dementia', 'alzheimer',
                'mood', 'psychological', 'mental wellness', 'emotional wellness'
            ],
            'sleep_health': [
                'sleep', 'sleep quality', 'sleeping', 'insomnia', 'sleep problems',
                'sleep disorders', 'rest', 'sleep hygiene', 'sleep patterns'
            ],
            
            # PREPARED Category - Planning & Legal
            'advance_planning': [
                'advance directives', 'living will', 'healthcare proxy', 'power of attorney',
                'durable power of attorney', 'estate planning', 'will', 'trust',
                'end-of-life care', 'end-of-life planning', 'legal documents',
                'advance planning', 'future planning', 'legal preparation'
            ],
            'financial_planning': [
                'financial planning', 'insurance', 'medicare', 'medicaid',
                'long-term care insurance', 'financial plan', 'retirement planning',
                'financial security', 'insurance coverage', 'supplemental insurance',
                'financial assistance', 'benefits', 'social security'
            ],
            'care_planning': [
                'care planning', 'long-term care', 'care needs', 'care options',
                'living arrangements', 'independent living', 'assisted living',
                'nursing home', 'home care', 'care decisions', 'care coordination'
            ],
            'document_organization': [
                'important documents', 'document organization', 'medical records',
                'financial records', 'legal documents', 'paperwork', 'records',
                'documentation', 'file organization', 'document management'
            ],
            
            # Caregiver Support
            'caregiver_support': [
                'caregiver', 'caregiver support', 'caregiver stress', 'caregiver burnout',
                'family caregiver', 'caregiving', 'caregiver health', 'caregiver wellness',
                'respite care', 'caregiver resources', 'caregiver education',
                'caregiver training', 'support groups', 'caregiver assistance'
            ],
            
            # Functional Abilities
            'daily_living': [
                'daily living', 'activities of daily living', 'ADL', 'independence',
                'bathing', 'dressing', 'toileting', 'eating', 'mobility',
                'personal care', 'self-care', 'functional ability', 'assistance'
            ]
        }
        
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
        
        # Remove punctuation but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace and normalize
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text

    def _calculate_eldercare_relevance_score(self, query: str, blog: BlogEntry) -> float:
        """Calculate eldercare relevance score based on domain-specific keywords"""
        query_lower = query.lower()
        blog_text = f"{blog.title} {blog.summary or ''} {' '.join(blog.tags)} {blog.category}".lower()
        
        total_score = 0.0
        
        # Check each eldercare category
        for category, keywords in self.eldercare_keywords.items():
            query_matches = sum(1 for keyword in keywords if keyword in query_lower)
            blog_matches = sum(1 for keyword in keywords if keyword in blog_text)
            
            if query_matches > 0 and blog_matches > 0:
                # Score based on relevance strength
                category_score = min(query_matches * blog_matches * 2.0, 12.0)
                total_score += category_score
        
        # Bonus for category matching
        if blog.category:
            category_keywords = {
                'safe': ['safety', 'fall', 'home', 'bathroom', 'lighting', 'mobility', 'emergency'],
                'healthy': ['health', 'medical', 'medication', 'chronic', 'nutrition', 'exercise', 'mental'],
                'prepared': ['planning', 'legal', 'financial', 'insurance', 'documents', 'care'],
                'caregiver': ['caregiver', 'support', 'stress', 'burnout', 'respite']
            }
            
            blog_category = blog.category.lower()
            if blog_category in category_keywords:
                category_words = category_keywords[blog_category]
                if any(word in query_lower for word in category_words):
                    total_score += 5.0
        
        return total_score

    def _calculate_direct_keyword_score(self, query: str, blog: BlogEntry) -> float:
        """Calculate direct keyword matching score"""
        query_lower = query.lower()
        score = 0.0
        
        # Split query into meaningful words (filter out very short words)
        query_words = [word.strip() for word in query_lower.split() if len(word.strip()) > 2]
        
        if not query_words:
            return 0.0
        
        # Check title for matches (highest weight)
        title_words = blog.title.lower().split()
        title_matches = 0
        for word in query_words:
            if any(word in title_word for title_word in title_words):
                title_matches += 1
        
        if title_matches > 0:
            score += (title_matches / len(query_words)) * 10.0
        
        # Check tags for matches
        tag_matches = 0
        for tag in blog.tags:
            tag_lower = tag.lower()
            for word in query_words:
                if word in tag_lower:
                    tag_matches += 1
        
        if tag_matches > 0:
            score += min(tag_matches / len(query_words), 1.0) * 8.0
        
        # Check summary for matches
        if blog.summary:
            summary_words = blog.summary.lower().split()
            summary_matches = 0
            for word in query_words:
                if any(word in summary_word for summary_word in summary_words):
                    summary_matches += 1
            
            if summary_matches > 0:
                score += (summary_matches / len(query_words)) * 6.0
        
        # Check category for matches
        if blog.category:
            category_lower = blog.category.lower()
            for word in query_words:
                if word in category_lower:
                    score += 4.0
        
        return score

    def _build_search_index(self):
        """Build TF-IDF index for all blog content"""
        if not self.blogs:
            return
        
        # Combine title, summary, tags, category for each blog with strategic emphasis
        self.processed_content = []
        for blog in self.blogs:
            # Strategic emphasis: category and title get highest weight
            category_text = blog.category.replace('_', ' ') if blog.category else ''
            subcategory_text = blog.subcategory.replace('_', ' ') if blog.subcategory else ''
            
            # Build content with strategic repetition for TF-IDF
            combined_text = f"{category_text} " * 8 + \
                           f"{subcategory_text} " * 6 + \
                           f"{blog.title} " * 6 + \
                           f"{blog.summary or ''} " * 4 + \
                           f"{' '.join(blog.tags)} " * 3 + \
                           f"{blog.content[:300]}"  # Limit content to avoid overwhelming
            
            processed = self._preprocess_text(combined_text)
            self.processed_content.append(processed)
        
        # Handle small datasets
        if len(self.processed_content) < 2:
            print(f"Skipping TF-IDF for small dataset ({len(self.processed_content)} blogs)")
            self.tfidf_matrix = None
            self.tfidf_vectorizer = None
            return
        
        # Initialize TF-IDF vectorizer with reasonable parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=3000,
            stop_words='english',
            ngram_range=(1, 3),
            min_df=1,
            max_df=0.85,
            sublinear_tf=True,
            norm='l2',
            use_idf=True,
            smooth_idf=True
        )
        
        # Build TF-IDF matrix
        if self.processed_content:
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_content)
                print(f"Built TF-IDF matrix for {len(self.processed_content)} blogs with {self.tfidf_matrix.shape[1]} features")
            except Exception as e:
                print(f"Error building TF-IDF matrix: {e}")
                # Fallback to simpler configuration
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words='english',
                    ngram_range=(1, 2),
                    min_df=1,
                    max_df=0.95
                )
                try:
                    self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_content)
                    print(f"Built fallback TF-IDF matrix for {len(self.processed_content)} blogs")
                except Exception as e2:
                    print(f"Failed to build TF-IDF matrix: {e2}")
                    self.tfidf_matrix = None
                    self.tfidf_vectorizer = None

    def get_all_blogs(self) -> BlogList:
        """Get all blogs"""
        return BlogList(blogs=self.blogs, total=len(self.blogs))

    def search_blogs(self, query: str, limit: int = 10) -> BlogList:
        """Search blogs using optimized scoring"""
        if not query or not self.blogs:
            return BlogList(blogs=[], total=0)
        
        scored_blogs = []
        
        for i, blog in enumerate(self.blogs):
            # Calculate scores
            eldercare_score = self._calculate_eldercare_relevance_score(query, blog)
            direct_keyword_score = self._calculate_direct_keyword_score(query, blog)
            
            # TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                try:
                    processed_query = self._preprocess_text(query)
                    if processed_query:
                        query_vector = self.tfidf_vectorizer.transform([processed_query])
                        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                        tfidf_score = similarities[i] * 12  # Scale for better differentiation
                except Exception as e:
                    print(f"Error calculating TF-IDF score: {e}")
                    tfidf_score = 0.0
            
            # Combined scoring with balanced weights
            if tfidf_score > 0.3:  # If TF-IDF found meaningful similarity
                combined_score = (eldercare_score * 0.4) + (direct_keyword_score * 0.3) + (tfidf_score * 0.3)
            else:  # Fall back to eldercare and keyword matching
                combined_score = (eldercare_score * 0.6) + (direct_keyword_score * 0.4)
            
            # Lower threshold for search to return more results
            if combined_score > 0.5:  # Much lower threshold
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
        """Get blog recommendations with optimized scoring"""
        if not query or not self.blogs:
            return BlogList(blogs=[], total=0)
        
        scored_blogs = []
        
        for i, blog in enumerate(self.blogs):
            # Calculate scores
            eldercare_score = self._calculate_eldercare_relevance_score(query, blog)
            direct_keyword_score = self._calculate_direct_keyword_score(query, blog)
            
            # TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                try:
                    processed_query = self._preprocess_text(query)
                    if processed_query:
                        query_vector = self.tfidf_vectorizer.transform([processed_query])
                        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                        tfidf_score = similarities[i] * 12  # Consistent scaling
                except Exception as e:
                    print(f"Error calculating TF-IDF score for recommendations: {e}")
                    tfidf_score = 0.0
            
            # Combined scoring with emphasis on eldercare matching for recommendations
            if tfidf_score > 0.5:  # Higher TF-IDF threshold for recommendations
                combined_score = (eldercare_score * 0.5) + (direct_keyword_score * 0.3) + (tfidf_score * 0.2)
            else:
                combined_score = (eldercare_score * 0.7) + (direct_keyword_score * 0.3)
            
            # Lower threshold for recommendations to ensure results
            if combined_score > 2.0:  # Much lower than previous 6.0
                scored_blogs.append((blog, combined_score))
        
        # Sort by score and return top results
        scored_blogs.sort(key=lambda x: x[1], reverse=True)
        recommended_blogs = [blog for blog, score in scored_blogs[:limit]]
        
        return BlogList(blogs=recommended_blogs, total=len(recommended_blogs))

    def recommend_best_blog_with_score(self, query: str) -> tuple[Optional[BlogEntry], float]:
        """Get the best blog recommendation with optimized scoring"""
        if not query or not self.blogs:
            return None, 0.0
        
        best_blog = None
        best_score = 0.0
        
        for i, blog in enumerate(self.blogs):
            # Calculate scores
            eldercare_score = self._calculate_eldercare_relevance_score(query, blog)
            direct_keyword_score = self._calculate_direct_keyword_score(query, blog)
            
            # TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                try:
                    processed_query = self._preprocess_text(query)
                    if processed_query:
                        query_vector = self.tfidf_vectorizer.transform([processed_query])
                        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                        tfidf_score = similarities[i] * 12  # Consistent scaling
                except Exception as e:
                    print(f"Error calculating TF-IDF score for best recommendation: {e}")
                    tfidf_score = 0.0
            
            # Combined scoring with heavy emphasis on eldercare matching
            if tfidf_score > 0.5:  # Meaningful TF-IDF similarity
                combined_score = (eldercare_score * 0.6) + (direct_keyword_score * 0.2) + (tfidf_score * 0.2)
            else:
                combined_score = (eldercare_score * 0.8) + (direct_keyword_score * 0.2)
            
            if combined_score > best_score:
                best_score = combined_score
                best_blog = blog
        
        return best_blog, best_score

    def get_best_recommendation(self, query: str) -> Optional[str]:
        """Get the single best blog recommendation source_id with reasonable threshold"""
        blog, score = self.recommend_best_blog_with_score(query)
        
        # Much more reasonable threshold for eldercare queries
        min_relevance_score = 2.0  # Lowered from 7.0 to actually return results
        
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
        Blog recommendations for products using optimized eldercare matching.
        
        Args:
            product_handle: The product handle.
            limit: Maximum number of blogs to return.
            
        Returns:
            BlogList with recommended blogs.
        """
        try:
            # Convert product handle to search query
            search_query = product_handle.replace('-', ' ')
            
            # Find relevant blogs using optimized recommendation system
            return self.get_recommendations(search_query, limit)
            
        except Exception as e:
            print(f"Error recommending blogs for product: {str(e)}")
            return BlogList(blogs=[], total=0) 