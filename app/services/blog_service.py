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
        """Initialize the blog service with enhanced TF-IDF semantic matching"""
        self.blogs: List[BlogEntry] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.processed_content = []
        
        # Enhanced stop words for better content filtering
        self.custom_stop_words = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
            'had', 'what', 'said', 'each', 'which', 'their', 'time', 'if',
            'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
            'would', 'make', 'like', 'into', 'him', 'two', 'more', 'go', 'no',
            'way', 'could', 'my', 'than', 'first', 'been', 'call', 'who',
            'oil', 'sit', 'now', 'find', 'down', 'day', 'did', 'get', 'come',
            'made', 'may', 'part'
        }
        
        # Enhanced domain-specific keyword mappings for better blog category matching
        self.category_keywords = {
            'safe': [
                'safety', 'lighting', 'lights', 'grab bars', 'handrails', 'fall prevention',
                'home safety', 'bathroom safety', 'stair safety', 'mobility safety',
                'emergency preparedness', 'safety equipment', 'accident prevention',
                'home modification', 'safety assessment', 'hazard removal', 'secure',
                'protection', 'safe environment', 'risk reduction', 'injury prevention',
                'safety measures', 'safety tips', 'home security', 'elder safety'
            ],
            'healthy': [
                'health', 'medical', 'medication', 'doctor', 'healthcare', 'wellness',
                'nutrition', 'exercise', 'fitness', 'mental health', 'physical health',
                'chronic conditions', 'disease management', 'preventive care',
                'health monitoring', 'medical appointments', 'health screening',
                'treatment', 'therapy', 'rehabilitation', 'recovery', 'symptoms',
                'diagnosis', 'medical care', 'health services', 'clinical care'
            ],
            'prepared': [
                'emergency planning', 'disaster preparedness', 'emergency kit',
                'legal documents', 'advance directives', 'power of attorney',
                'estate planning', 'financial planning', 'insurance', 'will',
                'emergency contacts', 'medical information', 'important documents',
                'preparation', 'planning ahead', 'future planning', 'readiness',
                'contingency planning', 'documentation', 'legal preparation'
            ],
            'caregiver': [
                'caregiver support', 'caregiver stress', 'caregiver burnout',
                'respite care', 'caregiver resources', 'family caregiver',
                'caregiver health', 'support groups', 'caregiver tips',
                'caring for caregiver', 'caregiver wellness', 'caregiver education',
                'caregiving challenges', 'caregiver assistance', 'caregiver guidance',
                'caregiver training', 'caregiver relief', 'caregiver community'
            ],
            'medication_management': [
                'medication management', 'pill organization', 'medication safety',
                'prescription management', 'medication adherence', 'drug interactions',
                'medication reminders', 'pill dispensers', 'medication errors',
                'pharmacy services', 'medication review', 'prescription drugs',
                'medication side effects', 'medication storage', 'pill splitting',
                'dosage management', 'medication scheduling', 'prescription tracking',
                'medication compliance', 'drug safety', 'pharmaceutical care'
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
        """Enhanced text preprocessing for better TF-IDF performance"""
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
        
        # Remove custom stop words
        words = text.split()
        filtered_words = [word for word in words if word not in self.custom_stop_words and len(word) > 2]
        
        return ' '.join(filtered_words)

    def _calculate_category_score(self, query: str, blog: BlogEntry) -> float:
        """Enhanced category relevance score with semantic understanding"""
        query_lower = query.lower()
        max_score = 0.0
        
        # Check each category and its keywords
        for category, keywords in self.category_keywords.items():
            if blog.category.lower() == category or category in blog.category.lower():
                # Calculate keyword match score for this category
                keyword_matches = sum(1 for keyword in keywords if keyword in query_lower)
                if keyword_matches > 0:
                    # Enhanced scoring with diminishing returns for multiple matches
                    base_score = min(keyword_matches / len(keywords), 0.8) * 12  # Increased max score
                    
                    # Bonus for exact category match in query
                    if category.replace('_', ' ') in query_lower:
                        base_score += 3.0
                    
                    # Bonus for subcategory relevance
                    if blog.subcategory and any(word in query_lower for word in blog.subcategory.lower().split()):
                        base_score += 2.0
                    
                    max_score = max(max_score, base_score)
        
        return max_score

    def _calculate_direct_keyword_score(self, query: str, blog: BlogEntry) -> float:
        """Enhanced direct keyword matching with semantic context"""
        query_lower = query.lower()
        score = 0.0
        
        # Preprocess query for better matching
        query_words = [word for word in query_lower.split() if word not in self.custom_stop_words and len(word) > 2]
        
        if not query_words:
            return 0.0
        
        # Check title for direct matches (highest weight) with enhanced context awareness
        title_words = self._preprocess_text(blog.title).split()
        meaningful_title_matches = 0
        
        for word in query_words:
            if any(word in title_word for title_word in title_words):
                if self._is_meaningful_blog_match(word, query_lower, blog):
                    meaningful_title_matches += 1
        
        if meaningful_title_matches > 0:
            title_score = (meaningful_title_matches / len(query_words)) * 8.0  # Increased weight
            score += title_score
        
        # Check tags for matches with enhanced weighting
        tag_matches = 0
        for tag in blog.tags:
            tag_lower = tag.lower()
            for word in query_words:
                if word in tag_lower and self._is_meaningful_blog_match(word, query_lower, blog):
                    tag_matches += 1
        
        if tag_matches > 0:
            tag_score = min(tag_matches / len(query_words), 1.0) * 6.0  # Enhanced tag scoring
            score += tag_score
        
        # Check summary for matches with improved context
        if blog.summary:
            summary_words = self._preprocess_text(blog.summary).split()
            meaningful_summary_matches = 0
            
            for word in query_words:
                if any(word in summary_word for summary_word in summary_words):
                    if self._is_meaningful_blog_match(word, query_lower, blog):
                        meaningful_summary_matches += 1
            
            if meaningful_summary_matches > 0:
                summary_score = (meaningful_summary_matches / len(query_words)) * 4.0  # Increased weight
                score += summary_score
        
        # Enhanced phrase matching with better context
        relevant_phrases = self._get_relevant_blog_phrases_for_query(query_lower)
        blog_text = f"{blog.title} {blog.summary or ''} {' '.join(blog.tags)}".lower()
        
        for phrase in relevant_phrases:
            if phrase in query_lower and phrase in blog_text:
                score += 3.0  # Increased phrase bonus
        
        # Improved negative scoring with more nuanced penalties
        medication_keywords = ['medication', 'pill', 'prescription', 'drug', 'medicine', 'dosage', 'pharmacy']
        query_is_medication = any(keyword in query_lower for keyword in medication_keywords)
        
        if query_is_medication:
            # Check if blog is actually medication-related
            blog_content = f"{blog.title} {blog.summary or ''} {' '.join(blog.tags)} {blog.category}".lower()
            medication_indicators = ['medication', 'pill', 'prescription', 'drug', 'medicine', 'pharmacy', 'dosage', 'medical']
            
            if not any(indicator in blog_content for indicator in medication_indicators):
                # Apply graduated penalty based on how off-topic the blog is
                if blog.category in ['caregiver']:
                    score -= 3.0  # Light penalty for related but not specific content
                else:
                    score -= 6.0  # Heavier penalty for unrelated content
        
        return max(score, 0.0)  # Ensure non-negative scores
    
    def _is_meaningful_blog_match(self, word: str, query: str, blog: BlogEntry) -> bool:
        """Enhanced semantic meaningfulness check for blog matches"""
        # Skip very short words and stop words
        if len(word) <= 2 or word in self.custom_stop_words:
            return False
        
        # Context-specific checks with improved logic
        if word == 'management':
            # Check for relevant management contexts
            blog_content = f"{blog.title} {blog.summary or ''} {' '.join(blog.tags)} {blog.category}".lower()
            management_contexts = ['medication', 'pill', 'prescription', 'care', 'health', 'chronic', 'condition']
            return any(context in blog_content for context in management_contexts)
        
        if word == 'system':
            # More nuanced system context checking
            blog_content = f"{blog.title} {blog.summary or ''} {' '.join(blog.tags)}".lower()
            system_contexts = ['medication', 'pill', 'safety', 'health', 'care', 'emergency', 'support', 'organization']
            return any(context in blog_content for context in system_contexts)
        
        if word in ['establish', 'create', 'develop', 'implement']:
            # These action words need specific context to be meaningful
            return len([w for w in query.split() if w not in self.custom_stop_words]) > 2
        
        # Enhanced medication-related word validation
        medication_words = ['medication', 'pill', 'prescription', 'drug', 'medicine', 'dosage', 'pharmacy']
        if word in medication_words:
            # Check if blog has medication-related content
            blog_content = f"{blog.title} {blog.summary or ''} {' '.join(blog.tags)} {blog.category}".lower()
            medication_indicators = ['medication', 'pill', 'prescription', 'drug', 'medicine', 'pharmacy', 'dosage', 'medical', 'health']
            return any(indicator in blog_content for indicator in medication_indicators)
        
        return True  # Default to meaningful for other words
    
    def _get_relevant_blog_phrases_for_query(self, query: str) -> List[str]:
        """Enhanced phrase relevance detection for blog queries"""
        all_phrases = [
            'home safety', 'fall prevention', 'medication management', 'pill organization',
            'caregiver support', 'emergency planning', 'health monitoring', 'chronic conditions',
            'safety equipment', 'lighting installation', 'grab bars', 'handrails',
            'prescription management', 'medication safety', 'health screening',
            'advance directives', 'estate planning', 'financial planning',
            'caregiver burnout', 'respite care', 'support groups'
        ]
        
        # Enhanced phrase matching with partial word matching
        relevant_phrases = []
        query_words = set(query.split())
        
        for phrase in all_phrases:
            phrase_words = set(phrase.split())
            # Include phrase if there's significant word overlap
            if len(phrase_words.intersection(query_words)) >= min(2, len(phrase_words)):
                relevant_phrases.append(phrase)
        
        return relevant_phrases

    def _build_search_index(self):
        """Enhanced TF-IDF index building with optimized parameters"""
        if not self.blogs:
            return
        
        # Enhanced content combination with strategic emphasis
        self.processed_content = []
        for blog in self.blogs:
            # Strategic emphasis: category and title get highest weight, then summary and tags
            category_text = blog.category.replace('_', ' ')
            subcategory_text = blog.subcategory.replace('_', ' ') if blog.subcategory else ''
            
            # Build content with strategic repetition for TF-IDF
            combined_text = f"{category_text} " * 12 + \
                           f"{subcategory_text} " * 8 + \
                           f"{blog.title} " * 10 + \
                           f"{blog.summary or ''} " * 6 + \
                           f"{' '.join(blog.tags)} " * 4 + \
                           f"{blog.content[:500]}"  # Limit content to avoid overwhelming
            
            processed = self._preprocess_text(combined_text)
            self.processed_content.append(processed)
        
        # Enhanced TF-IDF vectorizer with optimized parameters
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,  # Increased vocabulary size
            stop_words=None,  # We handle stop words in preprocessing
            ngram_range=(1, 3),  # Include trigrams for better phrase matching
            min_df=1,  # Include all terms that appear at least once
            max_df=0.85,  # Exclude terms that appear in >85% of documents
            sublinear_tf=True,  # Apply sublinear tf scaling
            norm='l2',  # L2 normalization
            use_idf=True,  # Use inverse document frequency
            smooth_idf=True,  # Smooth IDF weights
            token_pattern=r'\b\w{3,}\b'  # Only include words with 3+ characters
        )
        
        # Build TF-IDF matrix with error handling
        if self.processed_content:
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_content)
                print(f"Built enhanced TF-IDF matrix for {len(self.processed_content)} blogs with {self.tfidf_matrix.shape[1]} features")
            except Exception as e:
                print(f"Error building TF-IDF matrix: {e}")
                # Fallback to simpler configuration
                self.tfidf_vectorizer = TfidfVectorizer(
                    max_features=1000,
                    stop_words=None,
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
        """Enhanced blog search using hybrid scoring with improved TF-IDF"""
        if not query or not self.blogs:
            return BlogList(blogs=[], total=0)
        
        scored_blogs = []
        
        for i, blog in enumerate(self.blogs):
            # Calculate enhanced scores
            category_score = self._calculate_category_score(query, blog)
            direct_keyword_score = self._calculate_direct_keyword_score(query, blog)
            
            # Enhanced TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                try:
                    processed_query = self._preprocess_text(query)
                    if processed_query:  # Only proceed if query has meaningful content
                        query_vector = self.tfidf_vectorizer.transform([processed_query])
                        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                        tfidf_score = similarities[i] * 15  # Increased scaling for better differentiation
                except Exception as e:
                    print(f"Error calculating TF-IDF score: {e}")
                    tfidf_score = 0.0
            
            # Enhanced combined scoring with adaptive weights
            if tfidf_score > 0.5:  # If TF-IDF found meaningful similarity
                combined_score = (category_score * 0.4) + (direct_keyword_score * 0.3) + (tfidf_score * 0.3)
            else:  # Fall back to category and keyword matching
                combined_score = (category_score * 0.6) + (direct_keyword_score * 0.4)
            
            if combined_score > 1.0:  # Lowered threshold for search
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
        """Enhanced blog recommendations with improved scoring and higher thresholds"""
        if not query or not self.blogs:
            return BlogList(blogs=[], total=0)
        
        scored_blogs = []
        
        for i, blog in enumerate(self.blogs):
            # Calculate enhanced scores
            category_score = self._calculate_category_score(query, blog)
            direct_keyword_score = self._calculate_direct_keyword_score(query, blog)
            
            # Enhanced TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                try:
                    processed_query = self._preprocess_text(query)
                    if processed_query:
                        query_vector = self.tfidf_vectorizer.transform([processed_query])
                        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                        tfidf_score = similarities[i] * 15  # Consistent scaling
                except Exception as e:
                    print(f"Error calculating TF-IDF score for recommendations: {e}")
                    tfidf_score = 0.0
            
            # Enhanced combined scoring with emphasis on category matching for recommendations
            if tfidf_score > 1.0:  # Higher TF-IDF threshold for recommendations
                combined_score = (category_score * 0.5) + (direct_keyword_score * 0.3) + (tfidf_score * 0.2)
            else:
                combined_score = (category_score * 0.7) + (direct_keyword_score * 0.3)
            
            if combined_score > 6.0:  # Higher threshold for recommendations to ensure quality
                scored_blogs.append((blog, combined_score))
        
        # Sort by score and return top results
        scored_blogs.sort(key=lambda x: x[1], reverse=True)
        recommended_blogs = [blog for blog, score in scored_blogs[:limit]]
        
        return BlogList(blogs=recommended_blogs, total=len(recommended_blogs))

    def recommend_best_blog_with_score(self, query: str) -> tuple[Optional[BlogEntry], float]:
        """Enhanced best blog recommendation with improved hybrid scoring"""
        if not query or not self.blogs:
            return None, 0.0
        
        best_blog = None
        best_score = 0.0
        
        for i, blog in enumerate(self.blogs):
            # Calculate enhanced scores
            category_score = self._calculate_category_score(query, blog)
            direct_keyword_score = self._calculate_direct_keyword_score(query, blog)
            
            # Enhanced TF-IDF score
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                try:
                    processed_query = self._preprocess_text(query)
                    if processed_query:
                        query_vector = self.tfidf_vectorizer.transform([processed_query])
                        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                        tfidf_score = similarities[i] * 15  # Consistent scaling
                except Exception as e:
                    print(f"Error calculating TF-IDF score for best recommendation: {e}")
                    tfidf_score = 0.0
            
            # Enhanced combined scoring with heavy emphasis on category and semantic matching
            if tfidf_score > 1.0:  # Meaningful TF-IDF similarity
                combined_score = (category_score * 0.6) + (direct_keyword_score * 0.2) + (tfidf_score * 0.2)
            else:
                combined_score = (category_score * 0.8) + (direct_keyword_score * 0.2)
            
            if combined_score > best_score:
                best_score = combined_score
                best_blog = blog
        
        return best_blog, best_score

    def get_best_recommendation(self, query: str) -> Optional[str]:
        """Get the single best blog recommendation source_id with enhanced quality threshold"""
        blog, score = self.recommend_best_blog_with_score(query)
        
        # Enhanced minimum relevance threshold for quality assurance
        min_relevance_score = 7.0  # Increased threshold to ensure high-quality matches
        
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
        Enhanced blog recommendations for products using improved TF-IDF similarity.
        
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
            # Convert product handle to search query
            search_query = product_handle.replace('-', ' ')
            
            # Find relevant blogs using enhanced recommendation system
            return self.get_recommendations(search_query, limit)
            
        except Exception as e:
            print(f"Error recommending blogs for product: {str(e)}")
            return BlogList(blogs=[], total=0) 