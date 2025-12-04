# Care.com Nanny Hiring Assistant

A modern RAG (Retrieval-Augmented Generation) chatbot that helps users with questions about hiring a nanny. Built with Streamlit and powered by Google Gemini API, featuring Care.com's official branding and design.

![Care.com Logo](Care_new_logo.png)

## Features

- **🎨 Official Care.com Design**: Matches Care.com's branding with Commissioner font and green color scheme
- **🤖 Smart AI Responses**: Powered by Google Gemini API for accurate, context-aware answers
- **📚 RAG Architecture**: Uses FAISS vector search to retrieve relevant articles before generating responses
- **✅ No Hallucinations**: Answers strictly based on Care.com's nanny hiring guides
- **🎯 Source Citations**: Every answer includes links to relevant articles
- **⚡ Fast & Stable**: Cloud-based LLM (no local model loading)
- **💰 Free Tier**: Uses Google Gemini's free tier

## Quick Start

### 1. Prerequisites

- Python 3.8 or higher
- pip (Python package manager)

### 2. Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd Hackathon

# Install dependencies
pip install -r requirements.txt
```

### 3. Setup Environment Variables

Create a `.env` file in the project root:

```bash
GEMINI_API_KEY=your_gemini_api_key_here
```

To get a free Gemini API key:
1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create or sign in to your Google account
3. Click "Create API Key"
4. Copy the key to your `.env` file

### 4. Build Vector Database

```bash
python build_vector_db.py
```

This will process the articles and create a FAISS vector database in the `vector_db/` directory.

### 5. Run the Application

```bash
streamlit run app.py
```

The app will open automatically at `http://localhost:8501`

## Project Structure

```
Hackathon/
├── app.py                          # Streamlit web interface with Care.com UI
├── rag_chatbot_safe.py            # RAG chatbot implementation
├── build_vector_db.py             # Script to build FAISS vector database
├── requirements.txt               # Python dependencies
├── .env                           # Environment variables (not in repo)
├── .gitignore                     # Git ignore rules
├── Child_Care_Articles.csv        # Source articles about nanny hiring
├── Care_new_logo.png             # Care.com logo
├── child_agent.png               # Chatbot avatar image
└── vector_db/                    # Generated FAISS database (ignored by git)
    ├── faiss_index.index
    ├── metadata.pkl
    └── model_name.txt
```

## How It Works

1. **User Query**: User asks a question about hiring a nanny
2. **Relevance Check**: System checks if the question is related to nanny hiring
3. **Document Retrieval**: FAISS searches the vector database for relevant articles
4. **Response Generation**: Google Gemini API generates a concise 3-5 sentence answer
5. **Source Attribution**: System returns the answer with links to source articles

## Key Technologies

- **Streamlit**: Web interface framework
- **Google Gemini API**: Large language model for response generation
- **FAISS**: Facebook AI Similarity Search for vector database
- **Sentence Transformers**: For creating document embeddings (`all-MiniLM-L6-v2`)
- **Python-dotenv**: Environment variable management
- **Pillow**: Image processing for UI assets

## Features in Detail

### Smart Retrieval
- Uses semantic search to find the most relevant articles
- Keyword-based fallback for reliability
- Top 5 most relevant sources returned

### Answer Quality
- Strict 3-5 sentence format
- Removes author names and metadata from context
- Includes primary link and related articles
- Fallback extraction method if API fails

### UI/UX Design
- Care.com's Commissioner font
- Official color scheme (dark green #025747, light green #00A86B)
- Responsive design with sidebar navigation
- Example questions for easy exploration
- Clean chat interface with user/assistant avatars

## Configuration

### Customizing the Data Source

To use your own articles:
1. Replace `Child_Care_Articles.csv` with your CSV file
2. Ensure it has columns: `question`, `category`, `article`, `link`
3. Rebuild the vector database: `python build_vector_db.py`

### Adjusting Response Length

Edit `rag_chatbot_safe.py`:
```python
# Change max_output_tokens for longer/shorter responses
generation_config=genai.types.GenerationConfig(
    temperature=0.2,
    max_output_tokens=300,  # Adjust this value
    top_p=0.8,
    top_k=40
)
```

### Changing the Model

The app uses `gemini-2.5-flash` by default. To use a different model, edit `rag_chatbot_safe.py`:
```python
self.model = genai.GenerativeModel('gemini-pro')  # or other Gemini models
```

## Troubleshooting

### Vector Database Issues
If you see errors about missing vector database:
```bash
python build_vector_db.py
```

### API Key Issues
If you see Gemini API errors:
1. Check that `.env` file exists and contains `GEMINI_API_KEY`
2. Verify your API key is valid at [Google AI Studio](https://makersuite.google.com/)
3. Check if you've hit rate limits (wait a few minutes)

### Import Errors
Make sure all dependencies are installed:
```bash
pip install -r requirements.txt --upgrade
```

## Development

### Adding New Features
1. Edit `app.py` for UI changes
2. Edit `rag_chatbot_safe.py` for chatbot logic changes
3. Test locally before committing

### Code Style
- Follow PEP 8 Python style guidelines
- Use meaningful variable names
- Add comments for complex logic

## License

This project is for demonstration purposes. Care.com branding and assets are property of Care.com, Inc.

## Credits

- Developed for Care.com Nanny Hiring Assistant
- UI design inspired by [Care.com Help Center](https://help.care.com/s/?language=en_US)
- Powered by Google Gemini API

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Review [Streamlit documentation](https://docs.streamlit.io/)
3. Check [Google Gemini API docs](https://ai.google.dev/docs)

---

Built with ❤️ using Streamlit, Google Gemini, and FAISS
