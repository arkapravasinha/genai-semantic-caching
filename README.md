# Semantic Caching with Google Generative AI

A Python implementation of semantic caching using Google's Generative AI and Redis Vector Search. This project demonstrates how to cache LLM responses based on semantic similarity of queries.

## Features

- Semantic similarity search using Google's text embeddings
- Redis Vector Search for efficient similarity lookups
- TTL-based cache expiration
- Batch processing support
- Custom vectorizer implementation

## Prerequisites

- Python 3.8+
- Redis Stack (with Vector Search capability)
- Google API Key for Generative AI

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/genai-semantic-caching.git
cd genai-semantic-caching
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
.\.venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
set GOOGLE_API_KEY=your_api_key_here
```

## Usage

Basic example of semantic caching:

```python
from semantic_caching import answer_question

# First query - will call LLM API
question = "What was the name of the first US President?"
answer = answer_question(question)

# Similar query - will use cache
question = "Who was the first US President?"
answer = answer_question(question)
```

## Architecture

The project consists of two main components:

1. **GoogleGenAIVectorizer**: Converts text to embeddings using Google's Generative AI
2. **SemanticCache**: Handles caching and similarity search using Redis Vector Search

## Configuration

Key configuration options in `semantic-caching.py`:

```python
llmcache = SemanticCache(
    name="llmcache",                # index name
    redis_url="redis://localhost:6379",
    distance_threshold=0.1,         # similarity threshold
    vectorizer=GoogleGenAIVectorizer(model="text-embedding-004"),
    ttl=30,                        # cache TTL in seconds
)
```

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
