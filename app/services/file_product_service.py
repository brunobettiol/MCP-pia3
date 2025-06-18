import csv
import json
import os
from typing import List, Dict, Optional, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from app.models.product import Product, ProductImage, ProductPrice, ProductVariant
from app.core.config import settings


class FileProductService:
    def __init__(self):
        """Initialize the file-based product service optimized for the 35 specific eldercare questions"""
        self.products: List[Product] = []
        self.tfidf_vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix = None
        self.processed_content = []
        self._loaded = False
        
        # Manual mapping for the 35 specific questions to exact product categories/keywords
        self.question_product_mapping = {
            # SAFE Category Products
            'handrails_stairs': [
                'handrail', 'stair rail', 'banister', 'safety rail', 'grab rail',
                'stair safety', 'railing system', 'handrail kit', 'stair grab bar'
            ],
            'grab_bars_bathroom': [
                'grab bar', 'bathroom safety', 'shower bar', 'toilet safety',
                'bath rail', 'safety bar', 'grab handle', 'shower grab bar',
                'bathroom grab bar', 'toilet grab bar', 'bath grab bar'
            ],
            'lighting_improvement': [
                'LED light', 'motion sensor light', 'night light', 'pathway light',
                'stair light', 'hallway light', 'bright light', 'lighting system',
                'motion light', 'sensor light', 'automatic light'
            ],
            'fall_prevention': [
                'fall prevention', 'safety mat', 'non-slip', 'balance aid',
                'stability aid', 'fall alarm', 'safety sensor', 'anti-slip',
                'fall detector', 'balance trainer', 'stability trainer'
            ],
            'mobility_aids': [
                'walker', 'cane', 'wheelchair', 'rollator', 'mobility scooter',
                'walking stick', 'mobility aid', 'walking frame', 'walking aid',
                'mobility device', 'transport chair', 'walking support'
            ],
            'tripping_hazards': [
                'rug gripper', 'cord organizer', 'cable management', 'safety tape',
                'non-slip pad', 'floor safety', 'rug pad', 'cord cover'
            ],
            'bedroom_accessibility': [
                'bed rail', 'bed assist', 'bedroom safety', 'bed mobility',
                'adjustable bed', 'bedroom modification', 'bed handle', 'bed support'
            ],
            'emergency_response': [
                'medical alert', 'emergency button', 'help button', 'alert system',
                'personal alarm', 'emergency device', 'SOS button', 'panic button',
                'medical alarm', 'emergency pendant', 'alert pendant'
            ],
            
            # HEALTHY Category Products
            'medication_management': [
                'pill organizer', 'medication dispenser', 'pill box', 'med organizer',
                'pill reminder', 'medication alarm', 'pill tracker', 'dose dispenser',
                'medication box', 'pill container', 'medication reminder', 'med box'
            ],
            'health_monitoring': [
                'blood pressure monitor', 'glucose monitor', 'pulse oximeter',
                'thermometer', 'health tracker', 'vital signs monitor', 'BP monitor',
                'blood glucose meter', 'oxygen monitor', 'heart rate monitor'
            ],
            'nutrition_aids': [
                'jar opener', 'can opener', 'bottle opener', 'kitchen aid',
                'ergonomic utensils', 'adaptive utensils', 'meal prep aid',
                'kitchen tools', 'eating aids', 'dining aids', 'food prep'
            ],
            'exercise_equipment': [
                'resistance band', 'exercise bike', 'balance pad', 'pedometer',
                'fitness tracker', 'yoga mat', 'therapy ball', 'exercise equipment',
                'physical therapy', 'rehabilitation', 'fitness aid'
            ],
            'cognitive_aids': [
                'memory aid', 'reminder device', 'cognitive games', 'brain training',
                'memory book', 'alert system', 'cognitive support', 'memory game',
                'brain game', 'mental stimulation', 'cognitive trainer'
            ],
            'sleep_aids': [
                'sleep tracker', 'white noise machine', 'sound machine', 'sleep mask',
                'comfort pillow', 'mattress pad', 'sleep aid', 'bedtime comfort',
                'sleep support', 'rest aid', 'sleep improvement'
            ],
            
            # Functional Ability Products
            'bathing_assistance': [
                'shower chair', 'bath bench', 'shower seat', 'bathing aid',
                'shower safety', 'bath safety', 'shower stool', 'bath stool',
                'shower bench', 'bathing equipment', 'shower aid'
            ],
            'dressing_assistance': [
                'dressing aid', 'clothing aid', 'sock aid', 'shoe horn',
                'button hook', 'zipper pull', 'dressing stick', 'reacher',
                'grabber', 'dressing tool', 'adaptive clothing'
            ],
            'transfer_mobility': [
                'transfer board', 'slide board', 'transfer belt', 'gait belt',
                'transfer aid', 'mobility aid', 'lift aid', 'transfer disc',
                'pivot disc', 'transfer cushion', 'lift cushion'
            ],
            'toilet_assistance': [
                'toilet seat', 'commode', 'toilet aid', 'toilet safety',
                'raised toilet seat', 'toilet frame', 'bedside commode',
                'toilet support', 'toilet rail', 'bathroom aid'
            ],
            
            # PREPARED Category Products  
            'document_organization': [
                'file organizer', 'document folder', 'filing system', 'paper organizer',
                'document storage', 'file box', 'organization system', 'filing cabinet',
                'document holder', 'file folder', 'organizer'
            ],
            'financial_planning': [
                'financial organizer', 'budget planner', 'bill organizer',
                'expense tracker', 'financial calculator', 'planning guide',
                'money organizer', 'budget tracker', 'financial planner'
            ],
            'communication_aids': [
                'tablet', 'communication device', 'video phone', 'large button phone',
                'amplified phone', 'communication board', 'phone amplifier',
                'hearing aid', 'assistive technology', 'communication aid'
            ],
            
            # Incontinence Products (major category from dependabledaughter)
            'incontinence_products': [
                'brief', 'diaper', 'underwear', 'pad', 'liner', 'underpad',
                'incontinence', 'adult diaper', 'adult brief', 'protective underwear',
                'absorbent', 'disposable', 'washable', 'reusable', 'overnight',
                'bariatric', 'heavy duty', 'maximum', 'super', 'ultra', 'premium'
            ],
            
            # Personal Care Products
            'personal_care': [
                'wipes', 'cleansing', 'skin care', 'barrier cream', 'moisturizer',
                'body wash', 'shampoo', 'personal hygiene', 'cleansing wipes',
                'body care', 'hygiene', 'cleaning', 'wash', 'care'
            ]
        }
        
        # Keyword mappings for eldercare relevance
        self.eldercare_keywords = {
            'safety_equipment': [
                'grab bar', 'handrail', 'safety rail', 'non-slip', 'fall prevention',
                'bathroom safety', 'shower safety', 'mobility aid', 'walker', 'cane'
            ],
            'health_monitoring': [
                'blood pressure', 'glucose monitor', 'thermometer', 'pulse oximeter',
                'health tracker', 'vital signs', 'medical device', 'health monitor'
            ],
            'medication_management': [
                'pill organizer', 'medication dispenser', 'pill box', 'med reminder',
                'dose tracker', 'medication alarm', 'pill dispenser', 'med organizer'
            ],
            'daily_living_aids': [
                'jar opener', 'can opener', 'ergonomic', 'adaptive', 'easy grip',
                'kitchen aid', 'utensils', 'daily living', 'independence'
            ],
            'emergency_safety': [
                'medical alert', 'emergency button', 'help button', 'alert system',
                'personal alarm', 'emergency device', 'SOS', 'safety alarm'
            ],
            'comfort_wellness': [
                'comfort', 'support', 'cushion', 'pillow', 'mattress', 'sleep',
                'relaxation', 'therapy', 'wellness', 'pain relief'
            ],
            'incontinence_care': [
                'incontinence', 'brief', 'diaper', 'underwear', 'pad', 'liner',
                'absorbent', 'protection', 'overnight', 'adult', 'disposable'
            ],
            'personal_hygiene': [
                'wipes', 'cleansing', 'hygiene', 'personal care', 'skin care',
                'body wash', 'cleaning', 'bathing', 'shower', 'wash'
            ]
        }
        
        self._load_products()
    
    def _load_products(self):
        """Load products from both CSV files"""
        try:
            products_data = []
            
            # Load WooCommerce products (dependabledaughter)
            woo_file = os.path.join(settings.BASE_DIR, "data", "products", "dependabledaughter_products.csv")
            if os.path.exists(woo_file):
                products_data.extend(self._load_woocommerce_products(woo_file))
            
            # Load Shopify products (harmonyhomemedical)
            shopify_file = os.path.join(settings.BASE_DIR, "data", "products", "harmonyhomemedical_all.csv")
            if os.path.exists(shopify_file):
                products_data.extend(self._load_shopify_products(shopify_file))
            
            self.products = products_data
            print(f"Loaded {len(self.products)} total products from CSV files")
            
            # Build search index after loading
            if self.products:
                self._build_search_index()
                self._loaded = True
                
        except Exception as e:
            print(f"Error loading products from CSV files: {e}")
            self.products = []
    
    def _load_woocommerce_products(self, file_path: str) -> List[Product]:
        """Load products from WooCommerce CSV"""
        products = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                
                for row in reader:
                    try:
                        # Skip variations, only process main products
                        if row.get('Type') == 'variation':
                            continue
                        
                        # Parse images from comma-separated URLs
                        images = []
                        if row.get('Images'):
                            image_urls = [url.strip() for url in row['Images'].split(',') if url.strip()]
                            images = [ProductImage(url=url) for url in image_urls]
                        
                        # Parse tags from comma-separated string
                        tags = []
                        if row.get('Tags'):
                            tags = [tag.strip() for tag in row['Tags'].split(',') if tag.strip()]
                        
                        # Add categories as tags
                        if row.get('Categories'):
                            categories = [cat.strip() for cat in row['Categories'].split(',') if cat.strip()]
                            tags.extend(categories)
                        
                        # Create variants from the main product
                        price = float(row.get('Regular price', 0)) if row.get('Regular price') else 0.0
                        variant_price = ProductPrice(amount=price, currency_code='USD')
                        variants = [ProductVariant(
                            id=row.get('ID', ''),
                            title='Default',
                            price=variant_price,
                            available=row.get('In stock?', '1') == '1'
                        )]
                        
                        # Create unique handle from ID for WooCommerce
                        product_id = row.get('ID', '')
                        handle = f"woo-{product_id}"
                        
                        product = Product(
                            id=product_id,
                            title=row.get('Name', ''),
                            handle=handle,
                            description=row.get('Description', ''),
                            price=price,
                            currency='USD',
                            images=images,
                            variants=variants,
                            available=row.get('In stock?', '1') == '1',
                            tags=tags
                        )
                        
                        products.append(product)
                        
                    except Exception as e:
                        print(f"Error parsing WooCommerce product row: {e}")
                        continue
                
                print(f"Loaded {len(products)} products from WooCommerce CSV")
                
        except Exception as e:
            print(f"Error loading WooCommerce products: {e}")
        
        return products
    
    def _load_shopify_products(self, file_path: str) -> List[Product]:
        """Load products from Shopify CSV"""
        products = []
        
        try:
            # Use utf-8-sig to handle BOM (Byte Order Mark) in the CSV file
            with open(file_path, 'r', encoding='utf-8-sig') as file:
                reader = csv.DictReader(file)
                product_dict = {}
                row_count = 0
                
                for row in reader:
                    row_count += 1
                    
                    try:
                        handle = row.get('Handle', '').strip()
                        if not handle:
                            continue
                        
                        # Group variants by handle
                        if handle not in product_dict:
                            # Parse tags from comma-separated string
                            tags = []
                            if row.get('Tags'):
                                tags = [tag.strip() for tag in row['Tags'].split(',') if tag.strip()]
                            
                            product_dict[handle] = {
                                'id': handle,
                                'title': row.get('Title', ''),
                                'handle': handle,
                                'description': row.get('Body (HTML)', ''),
                                'tags': tags,
                                'vendor': row.get('Vendor', ''),
                                'images': [],
                                'variants': [],
                                'published': row.get('Published', 'TRUE').upper() == 'TRUE'
                            }
                        
                        # Add image if present
                        if row.get('Image Src'):
                            image = ProductImage(
                                url=row['Image Src'],
                                alt_text=row.get('Image Alt Text')
                            )
                            if image not in product_dict[handle]['images']:
                                product_dict[handle]['images'].append(image)
                        
                        # Add variant if present
                        if row.get('Variant Price'):
                            try:
                                price = float(row['Variant Price'])
                                variant_price = ProductPrice(amount=price, currency_code='USD')
                                
                                # Create variant title from options
                                variant_title = 'Default'
                                option_parts = []
                                if row.get('Option1 Value'):
                                    option_parts.append(row['Option1 Value'])
                                if row.get('Option2 Value'):
                                    option_parts.append(row['Option2 Value'])
                                if row.get('Option3 Value'):
                                    option_parts.append(row['Option3 Value'])
                                if option_parts:
                                    variant_title = ' / '.join(option_parts)
                                
                                # Parse inventory safely
                                inventory_qty = 0
                                try:
                                    inventory_qty = int(row.get('Variant Inventory Qty', 0) or 0)
                                except (ValueError, TypeError):
                                    inventory_qty = 0
                                
                                variant = ProductVariant(
                                    id=row.get('Variant SKU', handle),
                                    title=variant_title,
                                    price=variant_price,
                                    sku=row.get('Variant SKU'),
                                    available=row.get('Variant Inventory Policy', 'deny') != 'deny' or inventory_qty > 0
                                )
                                
                                product_dict[handle]['variants'].append(variant)
                                
                            except (ValueError, TypeError):
                                continue
                        
                    except Exception as e:
                        print(f"Error parsing Shopify product row {row_count}: {e}")
                        continue
                
                print(f"Processed {row_count} rows from Shopify CSV")
                print(f"Found {len(product_dict)} unique products")
                
                # Convert dictionary to Product objects
                for handle, product_data in product_dict.items():
                    try:
                        # Skip unpublished products
                        if not product_data['published']:
                            continue
                        
                        # Get main price from first variant
                        main_price = 0.0
                        available = False
                        if product_data['variants']:
                            main_price = product_data['variants'][0].price.amount
                            available = any(v.available for v in product_data['variants'])
                        else:
                            # If no variants, create a default one with price 0
                            available = True
                        
                        # Add vendor as tag
                        if product_data['vendor']:
                            product_data['tags'].append(product_data['vendor'])
                        
                        product = Product(
                            id=product_data['id'],
                            title=product_data['title'],
                            handle=product_data['handle'],
                            description=product_data['description'],
                            price=main_price,
                            currency='USD',
                            images=product_data['images'],
                            variants=product_data['variants'] if product_data['variants'] else [
                                ProductVariant(
                                    id=handle,
                                    title='Default',
                                    price=ProductPrice(amount=main_price, currency_code='USD'),
                                    available=available
                                )
                            ],
                            available=available,
                            tags=product_data['tags']
                        )
                        
                        products.append(product)
                        
                    except Exception as e:
                        print(f"Error creating Shopify product {handle}: {e}")
                        continue
                
                print(f"Loaded {len(products)} products from Shopify CSV")
                
        except Exception as e:
            print(f"Error loading Shopify products: {e}")
        
        return products
    
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

    def _calculate_exact_question_match_score(self, query: str, product: Product) -> float:
        """Calculate exact match score for the 35 specific questions"""
        query_lower = self._preprocess_text(query)
        product_text = f"{product.title} {self._strip_html(product.description or '')} {' '.join(product.tags)}".lower()
        
        # Check for direct question mapping
        for question_key, product_keywords in self.question_product_mapping.items():
            # Calculate similarity between query and question key
            question_words = set(question_key.replace('_', ' ').split())
            query_words = set(query_lower.split())
            
            # Calculate overlap
            overlap = len(question_words.intersection(query_words))
            if overlap > 0:
                similarity = overlap / max(len(question_words), len(query_words))
                if similarity > 0.3:  # Threshold for considering it a match
                    # Check if product contains any of the mapped keywords
                    keyword_matches = sum(1 for keyword in product_keywords if keyword.lower() in product_text)
                    if keyword_matches > 0:
                        return 70.0 + (similarity * 30.0) + (keyword_matches * 5.0)  # Score 70-100+
        
        return 0.0

    def _calculate_eldercare_relevance_score(self, query: str, product: Product) -> float:
        """Calculate eldercare relevance score based on domain-specific keywords"""
        query_lower = query.lower()
        product_text = f"{product.title} {self._strip_html(product.description or '')} {' '.join(product.tags)}".lower()
        
        total_score = 0.0
        
        # Check each eldercare category
        for category, keywords in self.eldercare_keywords.items():
            query_matches = sum(1 for keyword in keywords if keyword in query_lower)
            product_matches = sum(1 for keyword in keywords if keyword in product_text)
            
            if query_matches > 0 and product_matches > 0:
                # Score based on relevance strength
                category_score = min(query_matches * product_matches * 4.0, 20.0)
                total_score += category_score
        
        return total_score

    def _calculate_direct_keyword_score(self, query: str, product: Product) -> float:
        """Calculate direct keyword matching score"""
        query_lower = query.lower()
        score = 0.0
        
        # Split query into meaningful words (filter out very short words)
        query_words = [word.strip() for word in query_lower.split() if len(word.strip()) > 2]
        
        if not query_words:
            return 0.0
        
        # Check title for matches (highest weight)
        title_words = product.title.lower().split()
        title_matches = 0
        for word in query_words:
            if any(word in title_word for title_word in title_words):
                title_matches += 1
        
        if title_matches > 0:
            score += (title_matches / len(query_words)) * 20.0
        
        # Check tags for matches
        tag_matches = 0
        for tag in product.tags:
            tag_lower = tag.lower()
            for word in query_words:
                if word in tag_lower:
                    tag_matches += 1
        
        if tag_matches > 0:
            score += min(tag_matches / len(query_words), 1.0) * 15.0
        
        # Check description for matches
        if product.description:
            description_text = self._strip_html(product.description).lower()
            desc_matches = 0
            for word in query_words:
                if word in description_text:
                    desc_matches += 1
            
            if desc_matches > 0:
                score += (desc_matches / len(query_words)) * 10.0
        
        return score

    def _build_search_index(self):
        """Build TF-IDF index for all product content"""
        if not self.products:
            return
        
        # Combine title, description, tags for each product with emphasis on title and tags
        self.processed_content = []
        for product in self.products:
            # Emphasize title and tags more than description
            combined_text = f"{product.title} " * 3 + \
                           f"{' '.join(product.tags)} " * 2 + \
                           f"{self._strip_html(product.description or '')}"
            
            processed = self._preprocess_text(combined_text)
            self.processed_content.append(processed)
        
        # Handle small datasets
        if len(self.processed_content) < 2:
            print(f"Skipping TF-IDF for small dataset ({len(self.processed_content)} products)")
            self.tfidf_matrix = None
            self.tfidf_vectorizer = None
            return
        
        # Initialize TF-IDF vectorizer
        num_docs = len(self.processed_content)
        max_features = min(5000, num_docs * 50)
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features if max_features > 0 else None,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95,
            sublinear_tf=True,
            norm='l2'
        )
        
        # Build TF-IDF matrix
        if self.processed_content:
            try:
                self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_content)
                print(f"Built TF-IDF matrix for {len(self.processed_content)} products")
            except Exception as e:
                print(f"Error building TF-IDF matrix: {e}")
                self.tfidf_matrix = None
                self.tfidf_vectorizer = None

    def get_all_products(self) -> List[Product]:
        """Get all products"""
        return self.products

    def search_products(self, query: str, limit: int = 10) -> List[Product]:
        """Search products with TF-IDF semantic similarity"""
        if not query or not self.products:
            return []
        
        scored_products = []
        
        for i, product in enumerate(self.products):
            # Calculate multiple scores
            exact_question_score = self._calculate_exact_question_match_score(query, product)
            eldercare_score = self._calculate_eldercare_relevance_score(query, product)
            direct_keyword_score = self._calculate_direct_keyword_score(query, product)
            
            # TF-IDF score (only if matrix exists)
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 15  # Scale to 0-15
            
            # Combined score with emphasis on exact question matching
            if exact_question_score > 0:
                combined_score = exact_question_score + (eldercare_score * 0.3) + (direct_keyword_score * 0.2)
            elif self.tfidf_matrix is not None and tfidf_score > 0.1:
                combined_score = (eldercare_score * 0.3) + (direct_keyword_score * 0.4) + (tfidf_score * 0.3)
            else:
                combined_score = (eldercare_score * 0.4) + (direct_keyword_score * 0.6)
            
            # Threshold for search results
            if combined_score > 0.1:
                scored_products.append((product, combined_score))
        
        # Sort by score and return top results
        scored_products.sort(key=lambda x: x[1], reverse=True)
        return [product for product, score in scored_products[:limit]]

    def get_recommendations(self, query: str, limit: int = 5) -> List[Product]:
        """Get product recommendations with higher threshold"""
        if not query or not self.products:
            return []
        
        scored_products = []
        
        for i, product in enumerate(self.products):
            # Calculate multiple scores
            exact_question_score = self._calculate_exact_question_match_score(query, product)
            eldercare_score = self._calculate_eldercare_relevance_score(query, product)
            direct_keyword_score = self._calculate_direct_keyword_score(query, product)
            
            # TF-IDF score (only if matrix exists)
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 15  # Scale to 0-15
            
            # Combined score with heavy emphasis on exact question matching
            if exact_question_score > 0:
                combined_score = exact_question_score + (eldercare_score * 0.2) + (direct_keyword_score * 0.1)
            elif self.tfidf_matrix is not None and tfidf_score > 0.15:
                combined_score = (eldercare_score * 0.4) + (direct_keyword_score * 0.4) + (tfidf_score * 0.2)
            else:
                combined_score = (eldercare_score * 0.5) + (direct_keyword_score * 0.5)
            
            # Higher threshold for recommendations
            if combined_score > 0.15:
                scored_products.append((product, combined_score))
        
        # Sort by score and return top results
        scored_products.sort(key=lambda x: x[1], reverse=True)
        return [product for product, score in scored_products[:limit]]

    def get_best_recommendation(self, query: str) -> Tuple[Optional[Product], float]:
        """Get the single best product recommendation with score"""
        if not query or not self.products:
            return None, 0.0
        
        best_product = None
        best_score = 0.0
        
        for i, product in enumerate(self.products):
            # Calculate multiple scores
            exact_question_score = self._calculate_exact_question_match_score(query, product)
            eldercare_score = self._calculate_eldercare_relevance_score(query, product)
            direct_keyword_score = self._calculate_direct_keyword_score(query, product)
            
            # TF-IDF score (only if matrix exists)
            tfidf_score = 0.0
            if self.tfidf_matrix is not None and self.tfidf_vectorizer is not None:
                processed_query = self._preprocess_text(query)
                query_vector = self.tfidf_vectorizer.transform([processed_query])
                similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
                tfidf_score = similarities[i] * 15  # Scale to 0-15
            
            # Combined score with heavy emphasis on exact question matching
            if exact_question_score > 0:
                combined_score = exact_question_score + (eldercare_score * 0.2) + (direct_keyword_score * 0.1)
            elif self.tfidf_matrix is not None and tfidf_score > 0.15:
                combined_score = (eldercare_score * 0.5) + (direct_keyword_score * 0.3) + (tfidf_score * 0.2)
            else:
                combined_score = (eldercare_score * 0.6) + (direct_keyword_score * 0.4)
            
            if combined_score > best_score:
                best_score = combined_score
                best_product = product
        
        return best_product, best_score

    def get_product_by_handle(self, handle: str) -> Optional[Product]:
        """Get a product by its handle"""
        for product in self.products:
            if product.handle == handle:
                return product
        return None

    def get_product_url(self, product: Product) -> str:
        """Generate the correct URL for a product based on its source"""
        if product.handle.startswith('woo-'):
            # WooCommerce product - extract ID from handle
            product_id = product.handle.replace('woo-', '')
            return f"https://dependabledaughter.com/?p={product_id}&post_type=product"
        else:
            # Shopify product - use handle
            return f"https://harmonyhomemedical.com/products/{product.handle}"

    def get_statistics(self) -> Dict:
        """Get product statistics"""
        if not self.products:
            return {
                "total_products": 0,
                "available_products": 0,
                "average_price": 0,
                "currencies": [],
                "total_tags": 0,
                "woocommerce_products": 0,
                "shopify_products": 0
            }
        
        available_count = sum(1 for p in self.products if p.available)
        prices = [p.price for p in self.products if p.price > 0]
        avg_price = sum(prices) / len(prices) if prices else 0
        currencies = list(set(p.currency for p in self.products))
        all_tags = set()
        for product in self.products:
            all_tags.update(product.tags)
        
        # Count by source
        woo_count = sum(1 for p in self.products if p.handle.startswith('woo-'))
        shopify_count = len(self.products) - woo_count
        
        return {
            "total_products": len(self.products),
            "available_products": available_count,
            "average_price": round(avg_price, 2),
            "currencies": currencies,
            "total_tags": len(all_tags),
            "woocommerce_products": woo_count,
            "shopify_products": shopify_count
        } 