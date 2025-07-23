# Case Study: Building a RAG and Tool-Using AI Agent with Semantic Kernel

This project serves as a practical, real-world example of how to build an intelligent AI agent using Python and Microsoft's Semantic Kernel. The agent, **"Leo,"** is a specialized football assistant on Telegram that demonstrates a complete Retrieval-Augmented Generation (RAG) and Tool Use architecture.

It showcases how to orchestrate AI services, manage a knowledge base with semantic memory, and extend the agent's capabilities with custom plugins.

## ✨ Core Concepts Demonstrated

This project is a hands-on demonstration of the following key concepts in modern AI agent development:

### 1. AI Orchestration with Semantic Kernel

The Kernel is the core of the application, acting as the central orchestrator or "brain." It manages the flow of information, connects to various services, and decides when to use tools.

### 2. Connecting to AI Services (Google Gemini)

The agent connects to Google's AI services for its core intelligence:

- **Chat Completion**: Uses the Gemini 1.5 Flash model for natural language understanding, reasoning, and generating conversational responses.
- **Text Embeddings**: Uses the `embedding-001` model to convert the text of football matches into numerical vectors for efficient semantic search.

### 3. Retrieval-Augmented Generation (RAG) with Semantic Memory

To provide accurate, up-to-date answers, the agent is grounded with external data using a RAG pipeline:

- **Fetch**: At startup, the agent fetches data for the Premier League's last and next seasons from the API-Football service.
- **Embed & Index**: Each match's text description is converted into a vector embedding and stored in an in-memory `SemanticTextMemory` store.
- **Retrieve**: When a user asks a question, the agent searches this memory store to find the most semantically relevant match data.
- **Generate**: The retrieved data is injected into a prompt along with the user's question, allowing the Gemini model to generate a factually grounded answer.

### 4. Custom Tool Use with Plugins

To extend the agent's capabilities beyond conversation, we've built a custom `GoogleCalendarPlugin`.

- The Kernel is made aware of this plugin and its functions (like `create_calendar_event`).
- When a user's intent is to set a reminder, the AI model can choose to call this function, passing the necessary parameters it extracts from the conversation.

This demonstrates how to give an AI agent the ability to perform actions in external systems.

---

## 🛠️ Setup and Installation

Follow these steps to get the bot running on your local machine.

### 1. Prerequisites

- Python 3.10+ installed
- A Telegram account
- A Google account

### 2. Clone the Repository

```bash
git clone <your-repository-url>
cd <your-repository-folder>
```


### 3. Install Dependencies

Install the required Python packages:

```bash
pip install -r requirements.txt
```

### 4. Configure API Keys and Credentials

#### a. Create the `.env` File

Create a file named `.env` in the project root and add your secret API keys:

```env
# .env file
TELEGRAM_BOT_TOKEN="your_telegram_bot_token_here"
GEMINI_API_KEY="your_gemini_api_key_here"
API_FOOTBALL_KEY="your_api_football_key_here"
```

- `TELEGRAM_BOT_TOKEN`: From Telegram's BotFather.
- `GEMINI_API_KEY`: From the Google AI Studio.
- `API_FOOTBALL_KEY`: From the API-Football dashboard.

#### b. Set Up Google Calendar Credentials

- Go to the **Google Cloud Console**.
- Enable the **Google Calendar API**.
- Create an **OAuth 2.0 Client ID** for a **Desktop App**.
- Download the credentials file.
- Rename it to `credentials.json` and place it in the project root.

### 5. First-Time Google Authentication

The first time you ask the bot to create a calendar event:

- A browser window will open for you to **grant permission**.
- A `token.json` file will be created automatically to remember your approval.

---

## 🚀 Usage

Run the main agent script from your terminal:

```bash
python agentBot.py
```

The bot will initialize, index the football data, and start listening for messages on Telegram.

### Available Commands

- `/start`: Initializes the conversation.
- `/reset`: Clears the conversation history.
