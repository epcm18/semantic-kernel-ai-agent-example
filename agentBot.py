import asyncio
import nest_asyncio
import requests
from datetime import datetime, timedelta

# Telegram Imports
from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes

# Semantic Kernel Google AI Imports
import semantic_kernel as sk
from semantic_kernel.contents import ChatHistory
from semantic_kernel.functions import KernelArguments
from semantic_kernel.connectors.ai.google.google_ai import (GoogleAIChatCompletion, GoogleAITextEmbedding)
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatPromptExecutionSettings
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.memory import VolatileMemoryStore, SemanticTextMemory
from semantic_kernel.memory.memory_record import MemoryRecord
from GoogleCalendarPlugin import GoogleCalendarPlugin

nest_asyncio.apply()

from dotenv import load_dotenv

load_dotenv()

# --- Configuration ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
API_FOOTBALL_KEY = os.getenv("API_FOOTBALL_KEY")
API_FOOTBALL_URL = "https://v3.football.api-sports.io/fixtures"
GEMINI_MODEL_ID = "gemini-1.5-flash"
EMBEDDING_MODEL_ID = "models/embedding-001"
MEMORY_COLLECTION_NAME = "footballMatches"

# --- Global objects for the AI Kernel and Memory ---
kernel: sk.Kernel = None
memory: SemanticTextMemory = None
execution_settings: GoogleAIChatPromptExecutionSettings = None
system_message: str = ""

def fetch_match_data(days_past=1, days_future=7):
    print("Fetching match data...")
    all_formatted_matches = []
    today = datetime.now()
    for i in range(-days_past, days_future + 1):
        target_date = today + timedelta(days=i)
        date_str = target_date.strftime("%Y-%m-%d")
        params = {"date": date_str}
        headers = {"x-apisports-key": API_FOOTBALL_KEY}
        try:
            response = requests.get(API_FOOTBALL_URL, headers=headers, params=params)
            response.raise_for_status()
            data = response.json().get('response', [])
            if not data: continue
            for match in data:
                fixture = match.get('fixture', {})
                teams = match.get('teams', {})
                goals = match.get('goals', {})
                league = match.get('league', {})
                match_info = (
                    f"On {fixture.get('date', date_str)[:16].replace('T', ' at ')}, in the {league.get('name', 'N/A')}, a match between "
                    f"{teams.get('home', {}).get('name', 'N/A')} and {teams.get('away', {}).get('name', 'N/A')} "
                    f"is scheduled. Status: {fixture.get('status', {}).get('long', 'Scheduled')}."
                )
                if fixture.get('status', {}).get('short') == 'FT':
                     match_info += f" Final score was {goals.get('home', '?')} - {goals.get('away', '?')}."
                all_formatted_matches.append(match_info)
        except Exception as e:
            print(f"Error fetching data for {date_str}: {e}")
    print(f"✅ Successfully fetched {len(all_formatted_matches)} matches.")
    return all_formatted_matches


# --- Telegram Handler Functions  ---
async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Sends a welcome message when the /start command is issued."""
    # We will let the handle_message function create the history to avoid duplicates.
    await update.message.reply_text('Hello! I am Leo, your football assistant. Ask me about matches or for reminders!')

async def reset_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handles the /reset command. Clears the user's chat history."""
    context.user_data['history'] = ChatHistory(system_message=system_message)
    await update.message.reply_text("History has been reset. How can I help you?")

async def handle_text_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    # If history doesn't exist for this user, create it once and only once.
    if 'history' not in context.user_data:
        context.user_data['history'] = ChatHistory(system_message=system_message)
    
    history = context.user_data['history']
    user_input = update.message.text
    
    try:
        search_results = await memory.search(collection=MEMORY_COLLECTION_NAME, query=user_input, limit=10)
        context_text = "\n".join([result.text for result in search_results])
        history.add_user_message(user_input)
        
        arguments = KernelArguments(settings=execution_settings, history=history, context=context_text)
        prompt_template = """
        {{$history}}

        Use the following context to answer the user's question.
        If the user asks for a reminder, use the information in the context to call the calendar tool.
        
        CONTEXT:
        {{$context}}
        """
        result = await kernel.invoke_prompt(prompt=prompt_template, arguments=arguments)
        
        response_str = str(result)
        history.add_assistant_message(response_str)
        await update.message.reply_text(response_str)
    except Exception as e:
        error_message = f"An error occurred: {e}"
        print(error_message)
        await update.message.reply_text(error_message)

async def setup_agent():
    """Initializes all the AI components and indexes the data."""
    global kernel, memory, execution_settings, system_message
    
    kernel = sk.Kernel()
    chat_service = GoogleAIChatCompletion(gemini_model_id=GEMINI_MODEL_ID, api_key=GEMINI_API_KEY)
    embedding_service = GoogleAITextEmbedding(embedding_model_id=EMBEDDING_MODEL_ID, api_key=GEMINI_API_KEY)
    kernel.add_service(chat_service)

    memory_store = VolatileMemoryStore()
    memory = SemanticTextMemory(storage=memory_store, embeddings_generator=embedding_service)
    kernel.add_plugin(GoogleCalendarPlugin(), plugin_name="Calendar")
    print("✅ AI Kernel, Memory, and Plugins initialized.")

    match_data = fetch_match_data(days_past=1, days_future=2)
    if match_data:
        print("Indexing match data into memory...")
        # This simple loop is the correct way to bypass the error.
        for i, match in enumerate(match_data):
            await memory.save_information(
                collection=MEMORY_COLLECTION_NAME, id=f"match_{i}", text=match
            )
        print("✅ Indexing complete.")
    
    system_message = (
        "You are a helpful football assistant named Leo. Your primary goal is to answer user questions with the help on a given CONTEXT of match data. "
        "User wants have a conversation with you about fooball matches or fixtures with you."
        "When the user asks to set a reminder, you MUST call the `Calendar` tool. "
        "When calling the tool, for the 'match_context' argument, you MUST provide the full, single sentence of context that contains the match details. "
        "Always check the conversation history to understand which match the user is referring to. If unable to find ask to provide information again."
    )
    execution_settings = GoogleAIChatPromptExecutionSettings(function_choice_behavior=FunctionChoiceBehavior.Auto())

async def main() -> None:
    """Starts the Telegram bot listener."""
    print("🤖 Starting Telegram bot...")

    await setup_agent()
    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("reset", reset_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text_message))


    await application.run_polling()


if __name__ == "__main__":
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    loop.run_until_complete(main())
    