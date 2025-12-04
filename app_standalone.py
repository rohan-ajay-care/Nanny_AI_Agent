"""
Care.com Nanny Hiring Assistant - Standalone Version
All-in-one file for easy Streamlit Cloud deployment
"""

import streamlit as st
import os
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pickle
import re
from typing import List, Dict
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Care.com Nanny Hiring Assistant",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with Care.com branding
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Commissioner:wght@400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Commissioner', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    :root {
        --care-green: #00A86B;
        --care-green-dark: #025747;
        --care-green-light: #E6F5F0;
        --care-text: #000000;
        --care-text-dark-green: #025747;
        --care-white: #FFFFFF;
        --care-border: #E5E7EB;
        --care-bg-light: #F9FAFB;
    }
    
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    section[data-testid="stAppViewContainer"],
    .main,
    main {
        background-color: #FFFFFF !important;
    }
    
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: var(--care-bg-light) !important;
    }
    
    .main .block-container {
        padding: 1rem 2rem !important;
        background-color: #FFFFFF;
        max-width: 1200px;
    }
    
    /* Hide sidebar collapse button */
    [data-testid="collapsedControl"] {
        display: none !important;
    }
    
    /* Header styling */
    .care-header {
        text-align: left;
        padding: 1rem 0;
        margin-bottom: 1rem;
    }
    
    .care-logo-text {
        font-size: 2rem;
        font-weight: 800;
        color: var(--care-green-dark);
        margin-bottom: 0.5rem;
    }
    
    h1 {
        color: var(--care-text-dark-green) !important;
        font-size: 2.25rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Chat input - transparent with dark green border */
    .stChatInputContainer,
    [data-testid="stChatInputContainer"],
    div[data-testid="stChatInput"] {
        background: transparent !important;
        border-radius: 8px !important;
        border: 3px solid #025747 !important;
        padding: 0.75rem !important;
    }
    
    .stChatInputContainer input,
    .stChatInputContainer textarea {
        background: transparent !important;
        color: #025747 !important;
        font-weight: 700 !important;
        border: none !important;
    }
    
    .stChatInputContainer input::placeholder,
    .stChatInputContainer textarea::placeholder {
        color: #025747 !important;
        opacity: 0.6 !important;
        font-weight: 600 !important;
    }
    
    /* Chat messages */
    .stChatMessage {
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    
    .stChatMessage:has([alt="👤"]) {
        background: #f7f5f0 !important;
        border: 1px solid #e8e4db !important;
    }
    
    .stChatMessage p {
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: transparent !important;
        color: var(--care-text-dark-green) !important;
        border: 2px solid var(--care-green) !important;
        border-radius: 6px !important;
        padding: 0.5rem 0.75rem !important;
        font-weight: 700 !important;
        width: 100% !important;
        text-align: left !important;
        margin-bottom: 0.25rem !important;
    }
    
    .stButton > button:hover {
        background-color: var(--care-green-light) !important;
        border-color: var(--care-green-dark) !important;
    }
</style>
""", unsafe_allow_html=True)


# ==================== RAG CHATBOT CLASS ====================
class SafeRAGChatbot:
    def __init__(self, csv_path="Child_Care_Articles.csv", vector_db_path="vector_db"):
        self.csv_path = csv_path
        self.vector_db_path = vector_db_path
        self.embedding_model = None
        self.index = None
        self.metadata = []
        self.model = None
        
        # Initialize Gemini
        api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", "")
        if api_key:
            genai.configure(api_key=api_key)
            try:
                self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            except:
                try:
                    self.model = genai.GenerativeModel('gemini-pro')
                except Exception as e:
                    st.error(f"Failed to initialize Gemini: {e}")
        
        # Load or build vector database
        if not self._load_vector_db():
            st.info("Building vector database for the first time...")
            self._build_vector_db()
    
    def _build_vector_db(self):
        """Build FAISS vector database from CSV"""
        try:
            df = pd.read_csv(self.csv_path)
            
            # Load embedding model
            if self.embedding_model is None:
                self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Combine text for embedding
            texts = []
            for _, row in df.iterrows():
                text = f"{row['question']} {row['category']} {row['article'][:500]}"
                texts.append(text)
                self.metadata.append({
                    'question': row['question'],
                    'category': row['category'],
                    'article': row['article'],
                    'link': row['link']
                })
            
            # Create embeddings
            embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
            
            # Build FAISS index
            dimension = embeddings.shape[1]
            self.index = faiss.IndexFlatL2(dimension)
            self.index.add(embeddings.astype('float32'))
            
            # Save
            os.makedirs(self.vector_db_path, exist_ok=True)
            faiss.write_index(self.index, f"{self.vector_db_path}/faiss_index.index")
            with open(f"{self.vector_db_path}/metadata.pkl", 'wb') as f:
                pickle.dump(self.metadata, f)
            
            st.success("✅ Vector database built successfully!")
            return True
        except Exception as e:
            st.error(f"Error building vector database: {e}")
            return False
    
    def _load_vector_db(self):
        """Load existing vector database"""
        try:
            if os.path.exists(f"{self.vector_db_path}/faiss_index.index"):
                self.index = faiss.read_index(f"{self.vector_db_path}/faiss_index.index")
                with open(f"{self.vector_db_path}/metadata.pkl", 'rb') as f:
                    self.metadata = pickle.load(f)
                return True
            return False
        except:
            return False
    
    def is_relevant_question(self, query: str) -> bool:
        """Check if question is about nanny hiring"""
        keywords = ['nanny', 'babysitter', 'caregiver', 'childcare', 'child care', 
                   'hire', 'hiring', 'interview', 'pay', 'cost', 'contract', 'background check']
        return any(keyword in query.lower() for keyword in keywords)
    
    def retrieve_documents(self, query: str, k: int = 5) -> List[Dict]:
        """Retrieve relevant documents using keyword and semantic search"""
        # Keyword search
        results = []
        query_lower = query.lower()
        for i, doc in enumerate(self.metadata):
            score = 0
            if any(word in doc['question'].lower() for word in query_lower.split()):
                score += 2
            if any(word in doc['article'].lower() for word in query_lower.split()):
                score += 1
            if score > 0:
                results.append({'metadata': doc, 'score': score / 3.0})
        
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
    
    def _clean_article_text(self, article_text):
        """Remove author names and metadata"""
        if not article_text:
            return ""
        
        # Remove metadata patterns
        article_text = re.sub(r'\bby\s+[A-Z][a-z]+(?:\s+[A-Z][a-z-]+)*\s*', '', article_text, flags=re.IGNORECASE)
        article_text = re.sub(r'Updated\s+on:\s*[^\n\.]+', '', article_text, flags=re.IGNORECASE)
        article_text = re.sub(r'\d+\s*min\s*read', '', article_text, flags=re.IGNORECASE)
        
        lines = [line.strip() for line in article_text.split('\n') if line.strip() and len(line) > 30]
        return ' '.join(lines)
    
    def generate_response(self, query, retrieved_docs):
        """Generate response using Gemini"""
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            article_text = self._clean_article_text(doc['metadata'].get('article', ''))
            if len(article_text) > 2000:
                article_text = article_text[:2000] + "..."
            context_parts.append(f"Article {i}:\n{article_text}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""You are a helpful assistant that answers questions about hiring a nanny based on Care.com's guides.

IMPORTANT: Only use information from the provided context. Do not make up information.

Articles:
{context}

Provide a clear answer in 3-5 sentences using only the information from the articles above. Focus on the content, not who wrote it."""
        
        try:
            if self.model:
                response = self.model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.2,
                        max_output_tokens=300,
                    )
                )
                answer = response.text.strip()
                
                # Clean and format
                answer = re.sub(r'\b(?:by|written by)\s+[A-Z][a-z]+(?:\s+[A-Z][a-z-]+)*\s*', '', answer, flags=re.IGNORECASE)
                sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', answer) if s.strip() and len(s) > 15]
                
                if len(sentences) > 5:
                    answer = '. '.join(sentences[:5]) + '.'
                elif sentences:
                    answer = '. '.join(sentences) + '.'
                
                return answer
        except Exception as e:
            st.error(f"Error generating response: {e}")
        
        # Fallback extraction
        sentences = []
        for doc in retrieved_docs[:2]:
            article = self._clean_article_text(doc['metadata'].get('article', ''))
            article_sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', article) if len(s) > 40]
            sentences.extend(article_sentences[:2])
        
        return '. '.join(sentences[:3]) + '.' if sentences else "I couldn't find specific information about that."
    
    def chat(self, query):
        """Main chat function"""
        try:
            if not self.is_relevant_question(query):
                return {
                    'answer': "I'm sorry, but I'm specifically designed to help with questions about hiring a nanny. Your question seems to be outside this scope. Please review our comprehensive guides at https://www.care.com/c/guides/hiring-a-nanny-guide/",
                    'links': [],
                    'sources': []
                }
            
            retrieved_docs = self.retrieve_documents(query)
            
            if not retrieved_docs:
                return {
                    'answer': "I couldn't find relevant information in our guides. Please check https://www.care.com/c/guides/hiring-a-nanny-guide/ for more information.",
                    'links': [],
                    'sources': []
                }
            
            answer = self.generate_response(query, retrieved_docs)
            
            sorted_docs = sorted(retrieved_docs, key=lambda x: x.get('score', 0), reverse=True)
            
            primary_link = sorted_docs[0]['metadata'].get('link', '') if sorted_docs else ''
            
            sources = [
                {
                    'question': doc['metadata'].get('question', 'N/A'),
                    'category': doc['metadata'].get('category', 'N/A'),
                    'link': doc['metadata'].get('link', ''),
                    'score': doc.get('score', 0)
                }
                for doc in sorted_docs
            ]
            
            if primary_link:
                answer += f"\n\n**For more detailed information:** [{primary_link}]({primary_link})"
            
            if sources:
                answer += "\n\n**Related Articles:**"
                for i, source in enumerate(sources[:5], 1):
                    answer += f"\n{i}. [{source['question']}]({source['link']}) *({source['category']})*"
            
            return {
                'answer': answer,
                'links': [s['link'] for s in sources if s['link']],
                'sources': sources
            }
        except Exception as e:
            return {
                'answer': f"I encountered an error: {str(e)}. Please try again.",
                'links': [],
                'sources': []
            }


# ==================== STREAMLIT APP ====================

# Initialize chatbot
@st.cache_resource
def get_chatbot():
    return SafeRAGChatbot()

try:
    chatbot = get_chatbot()
except Exception as e:
    st.error(f"Error initializing chatbot: {e}")
    st.stop()

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Header
st.markdown("""
<div class="care-header">
    <div class="care-logo-text">Care.com 💚</div>
    <h1>We're here to help</h1>
    <p style="font-size: 1.1rem; color: #6B7280;">Ask me anything about hiring a nanny. I can help you find the right caregiver for your family.</p>
</div>
""", unsafe_allow_html=True)

# Hero section
st.markdown("""
<div style="text-align: center; margin: 1rem 0;">
    <div style="font-size: 4rem; margin-bottom: 0.5rem;">👶</div>
    <div style="font-size: 1.5rem; font-weight: 700; color: #025747;">Nanny Hiring Assistant</div>
    <div style="color: #6B7280; margin-top: 0.5rem;">Get expert guidance on finding, interviewing, and hiring the perfect nanny for your family</div>
</div>
<hr style="margin: 1.5rem 0; border: 1px solid #E5E7EB;">
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 1rem;">
        <div style="font-size: 3rem;">👶</div>
        <h2 style="color: #025747; margin: 0.5rem 0; font-size: 1.5rem;">Care.com</h2>
        <h3 style="color: #025747; margin: 0; font-size: 1.1rem;">Nanny Hiring Assistant</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div style="margin-bottom: 1rem;">
        <h3 style="margin-bottom: 0.5rem; color: #025747;">💡 What I can help with:</h3>
        <ul style="text-align: left; margin-top: 0.5rem;">
            <li><strong>Finding the right nanny</strong></li>
            <li><strong>Understanding costs and payment</strong></li>
            <li><strong>Interview questions and screening</strong></li>
            <li><strong>Safety checks and background checks</strong></li>
            <li><strong>Contracts and employment benefits</strong></li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 💡 Example Questions")
    example_questions = [
        "What's the difference between a nanny and a babysitter?",
        "How much does a nanny cost?",
        "What questions should I ask when interviewing a nanny?",
        "Do nannies get overtime pay?",
        "What should be included in a nanny contract?",
        "How do I screen a nanny before hiring?",
    ]
    
    for q in example_questions:
        if st.button(f"💬 {q}", key=f"example_{hash(q)}", use_container_width=True):
            st.session_state.user_input = q
            st.rerun()
    
    st.markdown("---")
    
    st.markdown("""
    <div style="text-align: center; padding: 0.75rem; background: #E6F5F0; border-radius: 6px;">
        <p style="color: #025747; font-weight: 700; margin-bottom: 0.5rem;">Need more help?</p>
        <a href="https://www.care.com/c/guides/hiring-a-nanny-guide/" target="_blank" style="color: #00A86B; font-weight: 600; text-decoration: none;">
            View Full Guides →
        </a>
    </div>
    """, unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    avatar = "🎧" if message["role"] == "assistant" else "👤"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])

# User input
user_input = st.chat_input("Search our help articles...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    
    # Get response
    with st.chat_message("assistant", avatar="🎧"):
        with st.spinner("Thinking..."):
            response = chatbot.chat(user_input)
            
            if response and 'answer' in response:
                st.markdown(response['answer'])
                
                # Save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response['answer']
                })
            else:
                error_msg = "I encountered an error. Please try again."
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_msg
                })

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; padding: 1rem; color: #6B7280;">
    <p style="margin: 0;">
        <strong style="color: #025747;">Care.com</strong> Nanny Hiring Assistant | 
        Powered by Google Gemini API
    </p>
    <p style="margin: 0.5rem 0 0 0; font-size: 0.9rem;">
        Built with ❤️ using Streamlit, Google Gemini, and FAISS
    </p>
</div>
""", unsafe_allow_html=True)

