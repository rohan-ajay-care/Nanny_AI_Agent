"""
Safe RAG Chatbot - Pure keyword-based with Gemini API (No sentence-transformers during queries)
"""
import os
import pickle
import faiss
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai
import re
from collections import Counter

load_dotenv()

class SafeRAGChatbot:
    def __init__(self, vector_db_dir='vector_db', top_k=5):
        """Initialize the Safe RAG Chatbot - loads embeddings once, uses keyword search."""
        self.vector_db_dir = vector_db_dir
        self.top_k = top_k
        
        # Load pre-computed embeddings and metadata (no encoding during queries)
        print("Loading vector database...")
        faiss_path = os.path.join(vector_db_dir, 'faiss_index.index')
        self.index = faiss.read_index(faiss_path)
        
        metadata_path = os.path.join(vector_db_dir, 'metadata.pkl')
        with open(metadata_path, 'rb') as f:
            self.metadata = pickle.load(f)
        
        # Load embedding model ONLY for query encoding (lightweight use)
        model_name_path = os.path.join(vector_db_dir, 'model_name.txt')
        with open(model_name_path, 'r') as f:
            self.embedding_model_name = f.read().strip()
        
        # Lazy load embedding model - only when needed
        self.embedding_model = None
        self._embedding_loaded = False
        
        # Configure Gemini API
        gemini_api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
        if not gemini_api_key:
            raise ValueError("GEMINI_API_KEY not found in .env file")
        
        genai.configure(api_key=gemini_api_key)
        # Try gemini-2.5-flash first (latest, fastest), fallback to others
        self.model = None
        try:
            # Try gemini-2.5-flash first
            try:
                self.model = genai.GenerativeModel('gemini-2.5-flash')
                print("✅ Gemini API configured (using gemini-2.5-flash)")
            except:
                # Fallback to gemini-2.0-flash
                try:
                    self.model = genai.GenerativeModel('gemini-2.0-flash')
                    print("✅ Gemini API configured (using gemini-2.0-flash)")
                except:
                    # List available models as last resort
                    available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                    if available_models:
                        model_name = available_models[0].split('/')[-1]
                        self.model = genai.GenerativeModel(model_name)
                        print(f"✅ Gemini API configured (using {model_name})")
                    else:
                        print("⚠️ No Gemini models available, will use extraction fallback")
        except Exception as e:
            print(f"Warning: Could not initialize Gemini model: {e}")
            print("Will use extraction-based fallback only")
            self.model = None
        
        # Keywords for off-topic detection
        self.relevant_keywords = [
            'nanny', 'babysitter', 'caregiver', 'child care', 'childcare',
            'hiring', 'interview', 'background check', 'contract', 'pay',
            'salary', 'benefits', 'tax', 'reference', 'trial', 'screening',
            'live-in', 'live-out', 'part-time', 'full-time', 'overtime',
            'vacation', 'sick days', 'holiday', 'payroll', 'employment'
        ]
        
        # Stop words
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should', 'could'
        }
        
        print("✅ Chatbot initialized (safe mode)")
    
    def _load_embedding_model(self):
        """Lazy load embedding model only when needed."""
        if self._embedding_loaded:
            return
        
        try:
            from sentence_transformers import SentenceTransformer
            print("Loading embedding model for query encoding...")
            self.embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device='cpu'
            )
            self._embedding_loaded = True
            print("✅ Embedding model loaded")
        except Exception as e:
            print(f"Warning: Could not load embedding model: {e}")
            self.embedding_model = None
    
    def _keyword_search(self, query, top_k=None):
        """Keyword-based search with fuzzy matching."""
        if top_k is None:
            top_k = self.top_k
        
        query_words = set(query.lower().split()) - self.stop_words
        query_words = {w for w in query_words if len(w) > 2}
        
        scored_docs = []
        for idx, doc_meta in enumerate(self.metadata):
            # Search in question, category, and article (more of the article for better matching)
            search_text = f"{doc_meta.get('question', '')} {doc_meta.get('category', '')} {doc_meta.get('article', '')[:1000]}".lower()
            search_words = set(search_text.split()) - self.stop_words
            
            # Calculate overlap
            overlap = len(query_words.intersection(search_words))
            
            # Also check for partial word matches (e.g., "hourly" matches "hour")
            partial_matches = 0
            for q_word in query_words:
                for s_word in search_words:
                    if len(q_word) > 3 and len(s_word) > 3:
                        if q_word in s_word or s_word in q_word:
                            partial_matches += 0.5
                            break
            
            total_score = overlap + partial_matches
            
            if total_score > 0:
                score = total_score / max(len(query_words), 1)
                scored_docs.append((score, idx))
        
        # Sort and return top k
        scored_docs.sort(reverse=True, key=lambda x: x[0])
        results = []
        for score, idx in scored_docs[:top_k]:
            results.append({
                'metadata': self.metadata[idx],
                'distance': 1.0 - score,  # Convert score to distance-like metric
                'score': score
            })
        
        return results
    
    def retrieve_documents(self, query, top_k=None):
        """Retrieve relevant documents - uses keyword search with LLM query expansion."""
        if top_k is None:
            top_k = self.top_k
        
        # Expand query using LLM for better matching
        expanded_query = self._expand_query_with_llm(query)
        
        # Use keyword search with expanded query
        return self._keyword_search(expanded_query, top_k)
        
        # Embedding-based search (disabled for stability)
        # Uncomment below if you want to try embeddings (may cause crashes)
        """
        try:
            self._load_embedding_model()
            if self.embedding_model:
                query_embedding = self.embedding_model.encode([query], show_progress_bar=False, convert_to_numpy=True)
                query_embedding = np.array(query_embedding).astype('float32')
                
                distances, indices = self.index.search(query_embedding, top_k)
                
                results = []
                for idx, dist in zip(indices[0], distances[0]):
                    if idx < len(self.metadata):
                        results.append({
                            'metadata': self.metadata[idx],
                            'distance': float(dist),
                            'score': float(1 / (1 + dist))
                        })
                
                if results:
                    return results
        except Exception as e:
            print(f"Embedding search failed, using keyword search: {e}")
        
        return self._keyword_search(query, top_k)
        """
    
    def _expand_query_with_llm(self, query):
        """Use LLM to understand query intent and generate better search terms."""
        if self.model is None:
            return query  # Fallback to original query
        
        try:
            expansion_prompt = f"""Someone asked about nanny hiring: "{query}"

Rephrase this question to include common related terms and concepts that might appear in nanny hiring guides. Keep it under 20 words.

For example:
- "how much do nannies charge hourly?" → "nanny hourly rate cost pay salary wages per hour"
- "what questions to ask a nanny?" → "nanny interview questions ask hiring screening"

Just give me the expanded search terms, nothing else:"""
            
            response = self.model.generate_content(
                expansion_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.3,
                    max_output_tokens=50,
                ),
            )
            
            if hasattr(response, 'text') and response.text:
                expanded = response.text.strip()
                print(f"Query expanded: '{query}' → '{expanded}'")
                return expanded
        except Exception as e:
            print(f"Query expansion failed: {e}")
        
        return query  # Fallback to original
    
    def is_relevant_question(self, question):
        """Check if the question is relevant to nanny hiring using LLM."""
        # First, quick keyword check
        question_lower = question.lower()
        has_keywords = any(keyword in question_lower for keyword in self.relevant_keywords)
        
        # If obvious keywords found, it's relevant
        if has_keywords:
            return True
        
        # Otherwise, use LLM to check if it's related to nanny hiring
        if self.model is None:
            return False  # Conservative fallback
        
        try:
            relevance_prompt = f"""Is this question about hiring a nanny or childcare? Answer ONLY "yes" or "no".

Question: "{question}"

Answer (yes/no):"""
            
            response = self.model.generate_content(
                relevance_prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.1,
                    max_output_tokens=10,
                ),
            )
            
            if hasattr(response, 'text') and response.text:
                answer = response.text.strip().lower()
                return 'yes' in answer
        except Exception as e:
            print(f"Relevance check failed: {e}")
        
        return False  # Conservative fallback
    
    def _clean_article_text(self, article_text):
        """Remove author names, dates, and metadata from article text."""
        if not article_text:
            return ""
        
        import re
        
        # Remove common metadata patterns more aggressively
        # Remove "by Author Name" patterns (including variations)
        article_text = re.sub(r'\bby\s+[A-Z][a-z]+(?:\s+[A-Z][a-z-]+)*\s*', '', article_text, flags=re.IGNORECASE)
        article_text = re.sub(r'\bwritten\s+by\s+[A-Z][a-z]+(?:\s+[A-Z][a-z-]+)*\s*', '', article_text, flags=re.IGNORECASE)
        article_text = re.sub(r'\barticle\s+by\s+[A-Z][a-z]+(?:\s+[A-Z][a-z-]+)*\s*', '', article_text, flags=re.IGNORECASE)
        
        # Remove author names that appear standalone (e.g., "Nicole Fabian-Weber")
        article_text = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*-\s*[A-Z][a-z]+)?\s*', '', article_text)
        
        # Remove "Updated on: [date]" patterns
        article_text = re.sub(r'Updated\s+on:\s*[^\n\.]+', '', article_text, flags=re.IGNORECASE)
        article_text = re.sub(r'Updated\s+on\s+[^\n\.]+', '', article_text, flags=re.IGNORECASE)
        
        # Remove "X min read" patterns
        article_text = re.sub(r'\d+\s*min\s*read', '', article_text, flags=re.IGNORECASE)
        
        # Remove date patterns (e.g., "November 21, 20257" or "September 19")
        article_text = re.sub(r'\b(?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s*\d{0,5}\b', '', article_text, flags=re.IGNORECASE)
        
        # Remove "Avatar photo" or similar image references
        article_text = re.sub(r'Avatar\s+photo\s*:?', '', article_text, flags=re.IGNORECASE)
        
        # Split into lines and filter out metadata lines
        lines = article_text.split('\n')
        clean_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip lines that are clearly metadata
            line_lower = line.lower()
            if any(pattern in line_lower for pattern in [
                'by ', 'updated on', 'min read', 
                'article by', 'written by', 'author:',
                'published', 'last updated', 'avatar photo',
                'september', 'october', 'november', 'december',
                'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august'
            ]):
                # Check if it's actually content (longer lines might be OK)
                if len(line) < 50:
                    continue
            
            # Skip very short lines that might be metadata
            if len(line) < 30 and any(word in line_lower for word in ['by', 'updated', 'read', 'photo']):
                continue
            
            # Skip lines that are just author names (pattern: FirstName LastName or FirstName-LastName)
            if re.match(r'^[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*-\s*[A-Z][a-z]+)?\s*$', line):
                continue
            
            # Skip lines that start with "Article X:" if they're just titles
            if re.match(r'^Article\s+\d+:\s*$', line, re.IGNORECASE):
                continue
            
            clean_lines.append(line)
        
        # Join and clean up extra whitespace
        cleaned = ' '.join(clean_lines)
        cleaned = re.sub(r'\s+', ' ', cleaned)  # Multiple spaces to single space
        cleaned = re.sub(r'\bArticle\s+\d+:\s*', '', cleaned, flags=re.IGNORECASE)  # Remove "Article X:" prefixes
        cleaned = cleaned.strip()
        
        return cleaned
    
    def generate_response(self, query, retrieved_docs):
        """Generate a response using Gemini API with safe error handling."""
        # Build context - clean up article text thoroughly
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            article_text = doc['metadata'].get('article', '')
            
            # Clean the article text
            article_text = self._clean_article_text(article_text)
            
            # Truncate if too long (keep most relevant content)
            if len(article_text) > 2500:
                article_text = article_text[:2500] + "..."
            
            # Only add if we have meaningful content
            if len(article_text.strip()) > 50:
                context_parts.append(f"Article {i}:\n{article_text}")
        
        if not context_parts:
            return "Hmm, having trouble pulling info from the articles. Check out the links below for all the details!"
        
        context = "\n\n".join(context_parts)
        
        # Create improved prompt that emphasizes content over metadata
        # Simplified prompt to avoid safety filters
        prompt = f"""Hey! Someone's asking: "{query}"

Here's the scoop from our guides:
{context}

Answer like you're texting a friend who needs advice - super casual and helpful. Use "you" and "your", throw in contractions (like "it's", "you'll", "don't"), and keep it short and sweet (3-5 sentences max). No corporate speak or fancy words. Just straight-up helpful info in a friendly way."""
        
        try:
            # Skip Gemini if model not available
            if self.model is None:
                raise ValueError("Gemini model not available")
            
            # Generate with Gemini - with safe error handling
            # Use safety settings to allow more content
            safety_settings = [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
            
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=0.7,  # Even higher for natural, conversational tone
                    max_output_tokens=300,  # Increased for better summaries
                    top_p=0.95,
                    top_k=60
                ),
                safety_settings=safety_settings
            )
            
            # Handle Gemini response safely
            try:
                if hasattr(response, 'text') and response.text:
                    answer = response.text.strip()
                elif hasattr(response, 'candidates') and response.candidates:
                    # Check if response was blocked or filtered
                    candidate = response.candidates[0]
                    finish_reason = getattr(candidate, 'finish_reason', None)
                    
                    # finish_reason 2 = SAFETY, 3 = RECITATION, etc.
                    if finish_reason and finish_reason != 1:  # 1 = STOP (normal)
                        # Try to get partial content if available
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text]
                            if text_parts:
                                answer = ' '.join(text_parts).strip()
                            else:
                                raise ValueError(f"Response blocked (finish_reason: {finish_reason})")
                        else:
                            raise ValueError(f"Response blocked (finish_reason: {finish_reason})")
                    else:
                        # Try to get text from parts
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            text_parts = [part.text for part in candidate.content.parts if hasattr(part, 'text') and part.text]
                            if text_parts:
                                answer = ' '.join(text_parts).strip()
                            else:
                                raise ValueError("No text content in response")
                        else:
                            raise ValueError("Empty response from Gemini")
                else:
                    raise ValueError("Empty response from Gemini")
            except Exception as api_error:
                # If we got partial content, use it; otherwise raise
                if 'answer' in locals() and answer:
                    pass  # Use the partial answer
                else:
                    raise api_error
            
            # Clean up answer - remove any author mentions that might have slipped through
            answer = answer.replace('\n', ' ')
            answer = ' '.join(answer.split())
            
            # Remove any remaining author mentions and metadata
            import re
            # Remove author patterns
            answer = re.sub(r'\b(?:by|written by|article by)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z-]+)*\s*', '', answer, flags=re.IGNORECASE)
            # Remove date/update patterns
            answer = re.sub(r'\b(?:Updated|Published|Last updated).*?\.', '', answer, flags=re.IGNORECASE)
            # Remove standalone author names
            answer = re.sub(r'\b[A-Z][a-z]+\s+[A-Z][a-z]+(?:\s*-\s*[A-Z][a-z]+)?\s*', '', answer)
            # Remove "Article X:" prefixes
            answer = re.sub(r'\bArticle\s+\d+:\s*', '', answer, flags=re.IGNORECASE)
            # Remove colons that are artifacts
            answer = re.sub(r'^\s*:\s*', '', answer)
            answer = re.sub(r'\s+:\s+', ' ', answer)
            
            # Ensure 3-5 sentences - split properly
            # Split by sentence endings, but keep the periods
            import re
            sentence_pattern = r'[^.!?]*[.!?]'
            sentences = re.findall(sentence_pattern, answer)
            sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 15]
            
            # Filter out sentences that mention authors or dates
            filtered_sentences = []
            for s in sentences:
                s_lower = s.lower()
                # Skip sentences that are mostly about authors/dates
                if not any(pattern in s_lower for pattern in ['by ', 'updated on', 'min read', 'article by']):
                    filtered_sentences.append(s)
            
            if not filtered_sentences:
                filtered_sentences = sentences  # Fallback if all were filtered
            
            # Strictly limit to 3-5 sentences
            if len(filtered_sentences) > 5:
                filtered_sentences = filtered_sentences[:5]
            elif len(filtered_sentences) < 3 and len(sentences) > len(filtered_sentences):
                # Try to get at least 3 sentences (use original if filtered removed too many)
                for s in sentences:
                    if s not in filtered_sentences and len(s) > 20:
                        filtered_sentences.append(s)
                        if len(filtered_sentences) >= 3:
                            break
            
            # Join sentences (they already have periods)
            if filtered_sentences:
                answer = ' '.join(filtered_sentences)
            else:
                answer = answer  # Keep original if filtering removed everything
            
            return answer.strip()
            
        except Exception as e:
            print(f"Gemini API error: {e}")
            # Fallback extraction
            return self._extract_fallback(query, context)
    
    def _extract_fallback(self, query, context):
        """Fallback extraction method - creates a better summary without metadata."""
        # Clean context first
        context = self._clean_article_text(context)
        
        query_words = set(query.lower().split()) - self.stop_words
        query_words = {w for w in query_words if len(w) > 2}
        
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', context)
        
        # Score sentences based on query relevance
        scored = []
        for s in sentences:
            s_clean = s.strip()
            if len(s_clean) < 30 or len(s_clean) > 500:  # Skip too short or too long
                continue
                
            s_lower = s_clean.lower()
            # Skip sentences with metadata
            if any(pattern in s_lower for pattern in ['by ', 'updated on', 'min read', 'article by', 'avatar photo']):
                continue
            
            # Skip questions (they're usually not answers)
            if s_clean.strip().endswith('?'):
                continue
            
            # Skip sentences that are just titles or headers
            if s_clean.isupper() or (len(s_clean) < 50 and not any(c.islower() for c in s_clean)):
                continue
            
            s_words = set(s_lower.split()) - self.stop_words
            s_words = {w for w in s_words if len(w) > 2}
            
            # Calculate relevance score
            overlap = len(query_words.intersection(s_words))
            if overlap > 0:
                # Normalize by query length
                score = overlap / max(len(query_words), 1)
                # Prefer longer sentences that are more informative (but not too long)
                length_bonus = min(len(s_clean) / 300, 0.2)
                # Prefer sentences that start with capital letters (more likely to be content)
                structure_bonus = 0.1 if s_clean[0].isupper() else 0
                total_score = score + length_bonus + structure_bonus
                scored.append((total_score, s_clean))
        
        # Sort by score and select top 3-5 sentences
        scored.sort(reverse=True, key=lambda x: x[0])
        selected = []
        seen_content = set()
        
        # Target 3-5 sentences
        target_count = 4  # Aim for middle of range
        for score, sentence in scored[:10]:  # Look at top 10, then filter
            if score > 0.1:  # Minimum relevance threshold
                # Avoid duplicates (check first 50 chars)
                sentence_key = sentence[:50].lower()
                if sentence_key not in seen_content:
                    selected.append(sentence)
                    seen_content.add(sentence_key)
                    if len(selected) >= target_count:
                        break
        
        if selected:
            # Clean up the sentences - remove artifacts
            cleaned_selected = []
            for s in selected:
                # Remove leading colons or artifacts
                s = re.sub(r'^\s*[:•]\s*', '', s)
                # Remove "Article X:" if present
                s = re.sub(r'^\s*Article\s+\d+:\s*', '', s, flags=re.IGNORECASE)
                s = s.strip()
                if len(s) > 20:  # Only keep substantial sentences
                    cleaned_selected.append(s)
            
            if cleaned_selected:
                # Strictly limit to 3-5 sentences
                if len(cleaned_selected) > 5:
                    cleaned_selected = cleaned_selected[:5]
                elif len(cleaned_selected) < 3 and len(scored) > len(cleaned_selected):
                    # Try to get at least 3 sentences
                    for score, sentence in scored[len(cleaned_selected):]:
                        s_clean = sentence.strip()
                        s_clean = re.sub(r'^\s*[:•]\s*', '', s_clean)
                        s_clean = re.sub(r'^\s*Article\s+\d+:\s*', '', s_clean, flags=re.IGNORECASE)
                        s_clean = s_clean.strip()
                        if len(s_clean) > 20 and s_clean not in cleaned_selected:
                            cleaned_selected.append(s_clean)
                            if len(cleaned_selected) >= 3:
                                break
                
                # Ensure each sentence ends with punctuation, then join
                final_sentences = []
                for s in cleaned_selected[:5]:  # Max 5
                    s = s.strip()
                    # Ensure sentence ends with punctuation
                    if not s.rstrip().endswith(('.', '!', '?')):
                        s += '.'
                    final_sentences.append(s)
                
                # Join with single space (sentences already have periods)
                answer = ' '.join(final_sentences)
                
                # Clean up any double spaces or artifacts
                answer = re.sub(r'\s+', ' ', answer)
                answer = re.sub(r'\s+:\s+', ' ', answer)
                # Remove double periods
                answer = re.sub(r'\.\s*\.', '.', answer)
                return answer.strip()
        
        # If no good sentences found, return a generic response
        return "Check out the articles below for the full details - they've got everything you need!"
    
    def chat(self, query):
        """Main chat function with comprehensive error handling."""
        try:
            # Check relevance
            if not self.is_relevant_question(query):
                return {
                    'answer': "Hey! I'm all about helping with nanny hiring stuff - that's my specialty. Your question seems a bit outside that area. Check out our guides at https://www.care.com/c/guides/hiring-a-nanny-guide/ or browse around Care.com for other topics!",
                    'links': [],
                    'sources': [],
                    'primary_link': None
                }
            
            # Retrieve documents
            retrieved_docs = self.retrieve_documents(query)
            
            if not retrieved_docs:
                return {
                    'answer': "Hmm, I'm not finding anything on that in our guides right now. Try checking out https://www.care.com/c/guides/hiring-a-nanny-guide/ - there's tons of info there!",
                    'links': [],
                    'sources': [],
                    'primary_link': None
                }
            
            # Generate response
            answer = self.generate_response(query, retrieved_docs)
            
            # Sort retrieved_docs by score (highest first) to get most relevant first
            sorted_docs = sorted(retrieved_docs, key=lambda x: x.get('score', 0), reverse=True)
            
            # Extract links and sources - sorted by relevance
            links = []
            sources = []
            seen_links = set()
            
            for doc in sorted_docs:
                link = doc['metadata'].get('link', '')
                if link and link not in seen_links:
                    links.append(link)
                    seen_links.add(link)
                
                sources.append({
                    'question': doc['metadata'].get('question', 'N/A'),
                    'category': doc['metadata'].get('category', 'N/A'),
                    'link': doc['metadata'].get('link', ''),
                    'score': doc.get('score', 0)
                })
            
            # Use the most relevant (first/highest score) link as primary
            primary_link = links[0] if links else None
            
            # Add primary link and sources to answer
            if primary_link:
                answer += f"\n\n**For more detailed information:** [{primary_link}]({primary_link})"
            
            # Add sources section directly in answer
            if sources:
                answer += "\n\n**Related Articles:**"
                for i, source in enumerate(sources[:5], 1):  # Show top 5 sources
                    if source.get('link'):
                        answer += f"\n{i}. [{source.get('question', 'N/A')}]({source['link']})"
                        if source.get('category'):
                            answer += f" *({source['category']})*"
            
            return {
                'answer': answer,
                'links': links,
                'sources': sources,
                'primary_link': primary_link
            }
            
        except Exception as e:
            print(f"Error in chat: {e}")
            import traceback
            traceback.print_exc()
            return {
                'answer': f"Oops, something went wrong on my end. Give it another shot?",
                'links': [],
                'sources': [],
                'primary_link': None
            }

