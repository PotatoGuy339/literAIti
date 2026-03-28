# Research Guide

An AI-powered tool that helps researchers clarify their research direction by combining real-time web scraping (Tinyfish) with intelligent LLM processing (OpenAI).

## Architecture

```
research_guide/
├── models/
│   └── context.py        # Data models (FieldContext, UserContext, etc.)
├── services/
│   ├── openai_service.py  # OpenAI API integration
│   ├── tinyfish_service.py # Tinyfish API integration
│   └── orchestrator.py    # Main orchestration logic
├── ui/
│   └── (future UI components)
├── utils/
│   └── config.py          # Configuration management
├── templates/
│   └── index.html         # Web UI
├── config.json            # API keys and settings
├── app.py                 # Flask application entry point
└── requirements.txt       # Python dependencies
```

## Features

- **Layer 1: Prompt Breakdown** - Analyzes user queries into field context, user context, and request
- **Layer 2a: Field Context Generation** - Real-time research via Tinyfish + AI processing
- **Layer 2b: User Context Generation** - Builds user profile based on background and feedback
- **Layer 3: Context-Aware Responses** - Generates informed answers with continuous refinement

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API keys in `config.json`:
```json
{
  "openai": {
    "api_key": "OPENAI KEY"
  },
  "tinyfish": {
    "api_key": "TINYFISH KEY"
  }
}
```

3. Run the application:
```bash
python app.py
```

4. Open http://localhost:5000 in your browser

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/session` | POST | Create new research session |
| `/api/query` | POST | Process initial research query |
| `/api/chat` | POST | Send follow-up messages |
| `/api/feedback` | POST | Submit feedback on suggestions |
| `/api/context/<session_id>` | GET | Get current context state |

## How It Works

1. User enters their background and research interest
2. System breaks down the query into components
3. Tinyfish scrapes real-time research data from the web
4. OpenAI processes and summarizes the research landscape
5. User receives research directions with continuous refinement based on feedback
