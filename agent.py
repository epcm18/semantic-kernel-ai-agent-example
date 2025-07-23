# agent.py

import asyncio

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.google.google_ai import (GoogleAIChatCompletion, GoogleAITextEmbedding)
from semantic_kernel.contents import ChatHistory
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.google.google_ai import GoogleAIChatPromptExecutionSettings
from semantic_kernel.functions import kernel_function
from semantic_kernel.functions import kernel_function
from semantic_kernel.memory import VolatileMemoryStore, SemanticTextMemory
from semantic_kernel.memory.memory_record import MemoryRecord
from semantic_kernel.functions import KernelArguments
# Import custom plugin
from GoogleCalendarPlugin import GoogleCalendarPlugin

from datetime import datetime, timedelta
import requests

# --- Configuration ---
GEMINI_API_KEY = "GEMINI_API_KEY"
GEMINI_MODEL_ID = "gemini-1.5-flash"
API_FOOTBALL_KEY = "API_FOOTBALL_KEY" 
API_FOOTBALL_URL = "https://v3.football.api-sports.io/fixtures"
EMBEDDING_MODEL_ID = "models/embedding-001"
MEMORY_COLLECTION_NAME = "footballMatches"

def fetch_match_data(days_past=1, days_future=10):
    """Fetches match data for a given date range."""
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

async def main():
    # --- 1. Initialize the Semantic Kernel ---
    kernel = Kernel()

    # --- 2. Add the Google AI Service for Chat Completion ---
    chat_completion_service = GoogleAIChatCompletion(
        gemini_model_id=GEMINI_MODEL_ID,
        api_key=GEMINI_API_KEY,
    )

    kernel.add_service(chat_completion_service)
    
    # ---3. Create embeddings to have more football knowledge
    embedding_service = GoogleAITextEmbedding(embedding_model_id=EMBEDDING_MODEL_ID, api_key=GEMINI_API_KEY)
    memory_store = VolatileMemoryStore()
    memory = SemanticTextMemory(storage=memory_store, embeddings_generator=embedding_service)

    # ---4. Add the plugin
    kernel.add_plugin(GoogleCalendarPlugin(), plugin_name="Calendar")

    print("✅ AI Kernel, Memory, and Plugins initialized.")

    # --- 5. Fetch and INDEX the Data into Memory ---
    match_data = fetch_match_data(days_past=-1, days_future=7)
    if match_data:
        print("Indexing match data into memory...")
        for i, match in enumerate(match_data):
            await memory.save_information(
                collection=MEMORY_COLLECTION_NAME, id=f"match_{i}", text=match
            )
        print("✅ Indexing complete.")
    
    # --- 6. Start the Chat Loop ---
    system_message = (
        "You are a helpful football assistant named Leo. Your primary goal is to answer user questions based on a given CONTEXT of match data. "
        "When the user asks to set a reminder, you MUST call the `Calendar` tool. "
        "When calling the tool, for the 'match_context' argument, you MUST provide the full, single sentence of context that contains the match details. "
        "Always check the conversation history to understand which match the user is referring to."
    )
    history = ChatHistory(system_message=system_message)

    execution_settings = GoogleAIChatPromptExecutionSettings(function_choice_behavior=FunctionChoiceBehavior.Auto())
    # --- 7. Start the Chat Loop ---
    print("\n----------------------------------------------------")
    print("🤖 Chat with Leo, your Football Bot!")
    print("   Ask about scores or ask for a reminder.")
    print("   Type 'exit' to quit.")
    print("----------------------------------------------------")

    while True:
        try:
            user_input = input("You: ")

            # --- Command Handling ---
            if user_input.lower() == "exit":
                print("Bot: Goodbye!")
                break
            
            if user_input.lower() == '/reset':
                # Create a new history and re-add the original system prompt
                history = ChatHistory()
                history.add_system_message(system_message)
                print("Bot: History has been reset. How can I help you?")
                continue # Skip to the next loop iteration

            search_results = await memory.search(
                collection=MEMORY_COLLECTION_NAME, query=user_input, limit=10
            )
            context_text = "\n".join([result.text for result in search_results])

            # The RAG and tool-use logic now happens inside the kernel's invoker
            history.add_user_message(user_input)
            
            # We pass the RAG results in the arguments so the prompt can use it
            arguments = KernelArguments(
                settings=execution_settings,
                history=history,
                # This makes the context available to the prompt template
                context=context_text 
            )

            # A more robust prompt template
            prompt_template = """
            {{$history}}

            Use the following context to answer the user's question.
            If the user asks for a reminder, use the information in the context to call the calendar tool.
            
            CONTEXT:
            {{$context}}
            """
            
            result = await kernel.invoke_prompt(
                prompt=prompt_template,
                arguments=arguments
            )
            
            history.add_assistant_message(str(result))
            print(f"Bot: {result}")

        except Exception as e:
            print(f"An error occurred: {e}")

# --- Run the main async function ---
if __name__ == "__main__":
    asyncio.run(main())