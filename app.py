"""
Care.com Nanny Hiring Assistant - Official Care.com Design
"""
import streamlit as st
import os
import base64
from dotenv import load_dotenv
from rag_chatbot_safe import SafeRAGChatbot
from PIL import Image

load_dotenv()

# Load images
logo_path = "Care_new_logo.png"
child_agent_path = "child_agent.png"

care_logo = None
child_agent_img = None

if os.path.exists(logo_path):
    care_logo = Image.open(logo_path)

if os.path.exists(child_agent_path):
    child_agent_img = Image.open(child_agent_path)
    if child_agent_img.mode != 'RGBA':
        child_agent_img = child_agent_img.convert('RGBA')
    child_agent_avatar = child_agent_img.resize((64, 64), Image.Resampling.LANCZOS)

# Page configuration with child agent as favicon
st.set_page_config(
    page_title="Care.com Nanny Hiring Assistant",
    page_icon=child_agent_path if os.path.exists(child_agent_path) else "🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Care.com branding with Commissioner font
# Add JavaScript to remove keyboard text
st.markdown("""
<script>
(function() {
    function removeKeyboardText() {
        const collapseBtn = document.querySelector('[data-testid="collapsedControl"] button');
        if (collapseBtn) {
            // Remove all text content
            collapseBtn.innerHTML = '';
            collapseBtn.textContent = '';
            // Remove all child nodes
            while (collapseBtn.firstChild) {
                collapseBtn.removeChild(collapseBtn.firstChild);
            }
            // Add hamburger icon
            const icon = document.createElement('span');
            icon.textContent = '☰';
            icon.style.cssText = 'font-size: 1.4rem; color: white; display: block; line-height: 1;';
            collapseBtn.appendChild(icon);
        }
    }
    // Run immediately and on DOM changes
    removeKeyboardText();
    setTimeout(removeKeyboardText, 100);
    setTimeout(removeKeyboardText, 500);
    setTimeout(removeKeyboardText, 1000);
    // Watch for changes
    const observer = new MutationObserver(removeKeyboardText);
    observer.observe(document.body, { childList: true, subtree: true });
})();
</script>
""", unsafe_allow_html=True)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Commissioner:wght@400;500;600;700;800&display=swap');
    
    /* Apply Commissioner font globally */
    * {
        font-family: 'Commissioner', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    /* Care.com Colors */
    :root {
        --care-green: #00A86B;
        --care-green-dark: #025747;
        --care-green-light: #E6F5F0;
        --care-text: #000000;
        --care-text-dark-green: #025747;
        --care-text-light: #6B7280;
        --care-white: #FFFFFF;
        --care-border: #E5E7EB;
        --care-bg-light: #F9FAFB;
    }
    
    /* Global light theme for entire app */
    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"],
    section[data-testid="stAppViewContainer"],
    section[data-testid="stAppViewContainer"] > div,
    .main,
    main {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
    }
    
    /* Bottom area where chat input is */
    .stBottom,
    [data-testid="stBottom"],
    section.main > div:last-child,
    div[class*="st-emotion-cache"] {
        background-color: #FFFFFF !important;
        background: #FFFFFF !important;
    }
    
    /* Sidebar - light background */
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div {
        background-color: var(--care-bg-light) !important;
    }
    
    /* Main container - reduced padding, left aligned */
    .main .block-container {
        padding-top: 0.25rem;
        padding-bottom: 0.5rem;
        padding-left: 2rem;
        padding-right: 2rem;
        background-color: #FFFFFF;
        max-width: 1200px;
        text-align: left !important;
    }
    
    /* Ensure all content is left-aligned */
    .main, .main *, .main p, .main div, .main h1, .main h2, .main h3 {
        text-align: left !important;
    }
    
    /* Header - left aligned */
    .care-header {
        text-align: left !important;
    }
    
    /* Hero section - left aligned */
    .hero-section {
        text-align: left !important;
    }
    
    .child-agent-img {
        justify-content: flex-start !important;
    }
    
    /* Increase main section font size for readability */
    .main p, .main div, .main span, .main li {
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
    }
    
    .main h1 {
        font-size: 2.25rem !important;
    }
    
    .main h2 {
        font-size: 1.75rem !important;
    }
    
    .main h3 {
        font-size: 1.5rem !important;
    }
    
    .stApp {
        background-color: #FFFFFF;
    }
    
    /* Make all text readable - black or dark green */
    p, div, span, li, label {
        color: var(--care-text) !important;
    }
    
    /* Headers - dark green */
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Commissioner', sans-serif;
        font-weight: 700;
        color: var(--care-text-dark-green) !important;
    }
    
    /* Header styling - compact */
    .care-header {
        background: #FFFFFF;
        padding: 1rem 1rem;
        border-radius: 12px;
        margin-bottom: 0.5rem;
        text-align: center;
        border: none;
    }
    
    .care-header h1 {
        color: var(--care-text-dark-green) !important;
        font-size: 2rem;
        margin-bottom: 0.5rem;
        font-weight: 700;
    }
    
    .care-header p {
        color: var(--care-text) !important;
        font-size: 1rem;
        margin-top: 0.25rem;
        font-weight: 400;
    }
    
    /* Logo styling */
    .care-logo {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
        margin-bottom: 0.75rem;
    }
    
    .care-logo img {
        height: 40px;
        width: auto;
    }
    
    /* Hero section - compact, left aligned */
    .hero-section {
        text-align: left !important;
        padding: 1rem 1rem;
        background: #FFFFFF;
        border-radius: 12px;
        margin-bottom: 0.5rem;
        border: none;
    }
    
    .hero-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--care-text-dark-green) !important;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1rem;
        color: var(--care-text) !important;
        margin-bottom: 0.5rem;
        font-weight: 400;
    }
    
    /* Child agent image styling */
    .child-agent-img {
        margin: 0.5rem auto;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    
    .child-agent-img img {
        width: 100px;
        height: auto;
    }
    
    /* Chat input box - ALWAYS visible dark green border #025747 */
    .stChatInputContainer,
    [data-testid="stChatInputContainer"],
    .stChatFloatingInputContainer,
    [data-testid="stChatFloatingInputContainer"],
    div[data-testid="stChatInput"],
    section[data-testid="stChatInput"] {
        background: transparent !important;
        background-color: transparent !important;
        border-radius: 8px !important;
        border: 3px solid #025747 !important;
        padding: 0.75rem !important;
        box-shadow: none !important;
    }
    
    /* All nested divs in chat input - transparent but maintain parent border */
    .stChatInputContainer > div,
    .stChatInputContainer > div > div,
    .stChatInputContainer > div > div > div,
    [data-testid="stChatInput"] > div,
    [data-testid="stChatInput"] > div > div {
        background: transparent !important;
        background-color: transparent !important;
        border: none !important;
    }
    
    /* Input fields - transparent with dark green text #025747 */
    .stChatInputContainer input,
    .stChatInputContainer textarea,
    [data-testid="stChatInput"] input,
    [data-testid="stChatInput"] textarea,
    input[type="text"],
    textarea {
        background: transparent !important;
        background-color: transparent !important;
        color: #025747 !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        border: none !important;
    }
    
    /* Placeholder text - dark green #025747 */
    .stChatInputContainer input::placeholder,
    .stChatInputContainer textarea::placeholder,
    [data-testid="stChatInput"] input::placeholder,
    [data-testid="stChatInput"] textarea::placeholder {
        color: #025747 !important;
        opacity: 0.6 !important;
        font-weight: 600 !important;
    }
    
    /* Focus state - keep the same visible border */
    .stChatInputContainer:focus-within,
    [data-testid="stChatInputContainer"]:focus-within,
    [data-testid="stChatInput"]:focus-within {
        border: 3px solid #025747 !important;
        box-shadow: 0 0 0 2px rgba(2, 87, 71, 0.2) !important;
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Hover state - maintain border */
    .stChatInputContainer:hover,
    [data-testid="stChatInputContainer"]:hover,
    [data-testid="stChatInput"]:hover {
        border: 3px solid #025747 !important;
    }
    
    /* Bottom area where chat sits - white background */
    .stBottom,
    [data-testid="stBottom"],
    section.main > div:last-child,
    div[class*="stBottom"] {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
    }
    
    /* Chat input wrapper - ensure no dark background */
    div[data-baseweb="base-input"],
    div[data-baseweb="input"] {
        background: transparent !important;
        background-color: transparent !important;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 0.75rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    
    /* Assistant message - white background */
    .stChatMessage:has([alt="🎧"]),
    .stChatMessage:has(img[src*="child_agent"]) {
        background: #FFFFFF !important;
        border: 1px solid var(--care-border);
    }
    
    /* User message - highlighted background */
    .stChatMessage:has([alt="👤"]) {
        background: #f7f5f0 !important;
        border: 1px solid #e8e4db !important;
    }
    
    /* Additional fallback selectors for user messages */
    [data-testid="stChatMessageContent"]:has([alt="👤"]) {
        background: #f7f5f0 !important;
    }
    
    div[data-testid="stChatMessage"]:nth-of-type(odd) {
        background: #f7f5f0 !important;
    }
    
    /* Chat message text - larger font */
    .stChatMessage p, .stChatMessage div, .stChatMessage span, .stChatMessage li {
        font-size: 1.1rem !important;
        line-height: 1.6 !important;
        color: var(--care-text) !important;
    }
    
    .stChatMessage[data-testid="assistant"] {
        border-left: 3px solid var(--care-green);
    }
    
    .stChatMessage[data-testid="user"] {
        border-left: 3px solid var(--care-border);
    }
    
    /* Chat message text - readable */
    .stChatMessage p, .stChatMessage div, .stChatMessage span {
        color: var(--care-text) !important;
    }
    
    /* Button styling - transparent with green border for example questions */
    .stButton > button {
        background-color: transparent !important;
        color: var(--care-text-dark-green) !important;
        border-radius: 6px !important;
        border: 2px solid var(--care-green) !important;
        padding: 0.5rem 0.75rem !important;
        font-weight: 700 !important;
        font-family: 'Commissioner', sans-serif !important;
        transition: all 0.2s !important;
        width: 100% !important;
        text-align: left !important;
        margin-bottom: 0.25rem !important;
        font-size: 0.95rem !important;
    }
    
    .stButton > button:hover {
        background-color: var(--care-green-light) !important;
        border-color: var(--care-green-dark) !important;
        color: var(--care-green-dark) !important;
        box-shadow: 0 2px 4px rgba(0, 168, 107, 0.1) !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #F9FAFB;
    }
    
    .sidebar .sidebar-content {
        padding: 0.75rem;
    }
    
    /* Hide sidebar collapse button completely - no collapsing allowed */
    [data-testid="collapsedControl"] {
        display: none !important;
        visibility: hidden !important;
    }
    
    /* Sidebar text - readable */
    .sidebar p, .sidebar div, .sidebar span, .sidebar li {
        color: var(--care-text) !important;
    }
    
    .sidebar h2, .sidebar h3 {
        color: var(--care-text-dark-green) !important;
    }
    
    /* Expander styling - NOT USED ANYMORE but keeping for compatibility */
    .streamlit-expanderHeader {
        display: none !important;
    }
    
    .streamlit-expanderContent {
        padding: 0.75rem !important;
        background-color: #FFFFFF !important;
    }
    
    .streamlit-expanderContent p, 
    .streamlit-expanderContent div,
    .streamlit-expanderContent ul,
    .streamlit-expanderContent li {
        color: var(--care-text) !important;
        margin-bottom: 0.5rem !important;
        text-align: left !important;
    }
    
    .streamlit-expanderContent strong {
        color: var(--care-text-dark-green) !important;
        font-weight: 700 !important;
    }
    
    .streamlit-expanderContent a {
        color: var(--care-green) !important;
        text-decoration: none !important;
        display: inline-block !important;
        margin-top: 0.25rem !important;
    }
    
    .streamlit-expanderContent a:hover {
        text-decoration: underline !important;
    }
    
    /* Fix divider in expander */
    .streamlit-expanderContent hr {
        margin: 0.75rem 0 !important;
        border-color: var(--care-border) !important;
    }
    
    /* Remove button hover text/tooltips */
    button[title]:hover::after,
    button[aria-label]:hover::after {
        display: none !important;
        content: "" !important;
    }
    
    /* Remove all tooltips */
    [title]:hover::after,
    [aria-label]:hover::after {
        display: none !important;
    }
    
    /* Link styling */
    a {
        color: var(--care-green) !important;
        text-decoration: none;
        font-weight: 500;
    }
    
    a:hover {
        color: var(--care-green-dark) !important;
        text-decoration: underline;
    }
    
    /* Feature cards - compact */
    .feature-card {
        background: #FFFFFF;
        padding: 0.75rem;
        border-radius: 8px;
        border: 1px solid var(--care-border);
        margin-bottom: 0.5rem;
    }
    
    .feature-card h3 {
        color: var(--care-text-dark-green) !important;
        margin-bottom: 0.5rem;
        font-weight: 700;
        font-size: 1rem;
    }
    
    .feature-card ul {
        color: var(--care-text) !important;
        margin: 0;
        padding-left: 1.25rem;
    }
    
    .feature-card li {
        color: var(--care-text) !important;
        margin-bottom: 0.25rem;
    }
    
    .feature-card strong {
        color: var(--care-text-dark-green) !important;
        font-weight: 600;
    }
    
    /* Footer - compact */
    .care-footer {
        text-align: center;
        padding: 1rem;
        color: var(--care-text) !important;
        margin-top: 1.5rem;
        border-top: 1px solid var(--care-border);
        background: #FFFFFF;
    }
    
    .care-footer p {
        color: var(--care-text) !important;
        margin-bottom: 0.25rem;
    }
    
    .care-footer strong {
        color: var(--care-text-dark-green) !important;
        font-weight: 700;
    }
    
    /* Markdown text - readable */
    .stMarkdown p, .stMarkdown div, .stMarkdown span, .stMarkdown li {
        color: var(--care-text) !important;
    }
    
    .stMarkdown strong {
        color: var(--care-text-dark-green) !important;
        font-weight: 700;
    }
    
    /* Remove excessive spacing */
    hr {
        margin: 0.25rem 0 !important;
        border-color: var(--care-border);
    }
    
    /* Sidebar spacing - remove padding between sections */
    .sidebar .element-container {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Remove spacing in sidebar sections */
    .sidebar h2, .sidebar h3 {
        margin-bottom: 0.15rem !important;
        margin-top: 0.15rem !important;
    }
    
    /* Remove spacing between sidebar elements */
    [data-testid="stSidebar"] [class*="element-container"] {
        margin-bottom: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Remove spacing from markdown dividers in sidebar */
    [data-testid="stSidebar"] hr {
        margin: 0.25rem 0 !important;
    }
    
    /* Remove padding from sidebar content sections */
    [data-testid="stSidebar"] .element-container {
        margin-top: 0 !important;
        margin-bottom: 0 !important;
        padding-top: 0 !important;
        padding-bottom: 0 !important;
    }
    
    /* Specific spacing for sidebar sections */
    [data-testid="stSidebar"] > div > div {
        gap: 0 !important;
    }
    </style>
    """, unsafe_allow_html=True)


# Initialize chatbot (only once)
if 'chatbot' not in st.session_state:
    with st.spinner("Initializing assistant..."):
        try:
            st.session_state.chatbot = SafeRAGChatbot()
        except Exception as e:
            st.error(f"Error initializing chatbot: {e}")
            st.stop()

if 'messages' not in st.session_state:
    st.session_state.messages = []

# Header with Care.com logo (left aligned)
st.markdown("""
<div class="care-header">
    <div class="care-logo">
""", unsafe_allow_html=True)

if care_logo:
    st.image(care_logo, width=240)
else:
    st.markdown("""
        <div style="display: flex; align-items: center; justify-content: flex-start; gap: 0.5rem;">
            <div style="width: 40px; height: 40px; background: #00A86B; border-radius: 8px; display: flex; align-items: center; justify-content: center; color: white; font-size: 24px; font-weight: bold;">C</div>
            <span style="font-size: 1.5rem; font-weight: 700; color: #025747;">Care.com</span>
        </div>
    """, unsafe_allow_html=True)

st.markdown("""
    </div>
    <h1>We're here to help</h1>
    <p>Ask me anything about hiring a nanny. I can help you find the right caregiver for your family.</p>
</div>
""", unsafe_allow_html=True)

# Hero section with child agent illustration (left aligned)
st.markdown("""
<div class="hero-section">
    <div class="child-agent-img">
""", unsafe_allow_html=True)

if child_agent_img:
    st.image(child_agent_img, width=120)

st.markdown("""
    </div>
    <div class="hero-title">Nanny Hiring Assistant</div>
    <div class="hero-subtitle">
        Get expert guidance on finding, interviewing, and hiring the perfect nanny for your family
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar - no logo, just text
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 0;">
    """, unsafe_allow_html=True)
    
    if child_agent_img:
        st.image(child_agent_img, width=70)
    
    st.markdown("""
        <h2 style="color: #025747; margin-bottom: 0; margin-top: 0.25rem; font-weight: 700; font-size: 1.5rem;">Care.com</h2>
        <h3 style="color: #025747; margin-bottom: 0; margin-top: 0; font-weight: 600; font-size: 1.1rem;">Nanny Hiring Assistant</h3>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("""
    <div class="feature-card" style="margin-bottom: 0;">
        <h3 style="margin-bottom: 0.25rem; margin-top: 0;">💡 What I can help with:</h3>
        <ul style="text-align: left; margin-top: 0.25rem; margin-bottom: 0;">
            <li style="margin-bottom: 0.15rem;"><strong>Finding the right nanny</strong></li>
            <li style="margin-bottom: 0.15rem;"><strong>Understanding costs and payment</strong></li>
            <li style="margin-bottom: 0.15rem;"><strong>Interview questions and screening</strong></li>
            <li style="margin-bottom: 0.15rem;"><strong>Safety checks and background checks</strong></li>
            <li style="margin-bottom: 0;"><strong>Contracts and employment benefits</strong></li>
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
    
    st.markdown("---")
    
    # Footer in sidebar - text only, no logo
    st.markdown("""
    <div style="text-align: center; padding: 0.5rem; background: #E6F5F0; border-radius: 6px;">
        <p style="color: #025747; font-weight: 700; margin-bottom: 0.15rem; font-size: 0.9rem;">Need more help?</p>
        <a href="https://www.care.com/c/guides/hiring-a-nanny-guide/" target="_blank" style="color: #00A86B; font-weight: 600; font-size: 0.9rem;">
            View Full Guides →
        </a>
    </div>
    """, unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    if message["role"] == "assistant":
        avatar = child_agent_path if os.path.exists(child_agent_path) else "🎧"
    else:
        avatar = "👤"
    
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if "links" in message and message["links"]:
            with st.expander("📚 Related Articles"):
                for link in message["links"]:
                    st.markdown(f"- [{link}]({link})")

# User input
user_input = st.chat_input("Search our help articles...")

# Handle example question clicks
if 'user_input' in st.session_state:
    user_input = st.session_state.user_input
    del st.session_state.user_input

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_input)
    
    avatar_path = child_agent_path if os.path.exists(child_agent_path) else "🎧"
    with st.chat_message("assistant", avatar=avatar_path):
        with st.spinner("Thinking..."):
            try:
                response = None
                error_occurred = False
                
                try:
                    response = st.session_state.chatbot.chat(user_input)
                except KeyboardInterrupt:
                    raise
                except SystemExit:
                    raise
                except Exception as e:
                    error_occurred = True
                    error_msg = str(e)
                    st.error(f"Error: {error_msg}")
                    import traceback
                    traceback.print_exc()
                    
                    response = {
                        'answer': f"I encountered an error: {error_msg}. Please try rephrasing your question.",
                        'links': [],
                        'sources': []
                    }
                
                if response and 'answer' in response:
                    st.markdown(response['answer'])
                    
                    # Sources are already included in the answer, no need to display separately
                    
                    try:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": response['answer'],
                            "links": response.get('links', []),
                            "sources": response.get('sources', [])
                        })
                    except Exception as e:
                        print(f"Error saving to history: {e}")
                else:
                    st.error("Invalid response format. Please try again.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "I encountered an error. Please try again."
                    })
                    
            except KeyboardInterrupt:
                st.stop()
            except SystemExit:
                st.stop()
            except Exception as e:
                st.error(f"Unexpected error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"I encountered an unexpected error. Please try again."
                })

# Footer - text only, no logo
st.markdown("---")
st.markdown("""
<div class="care-footer">
    <p style="margin-bottom: 0.25rem;">
        <strong>Powered by Care.com's Nanny Hiring Guides</strong>
    </p>
    <p style="font-size: 0.85rem;">
        <a href="https://www.care.com/c/guides/hiring-a-nanny-guide/" target="_blank">
            View Full Guides
        </a> | 
        <a href="https://www.care.com" target="_blank">
            Visit Care.com
        </a>
    </p>
</div>
""", unsafe_allow_html=True)
