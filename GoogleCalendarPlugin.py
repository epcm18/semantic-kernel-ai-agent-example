# GoogleCalendarPlugin.py

import os.path
import json
import re
from datetime import datetime, timedelta, timezone

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from semantic_kernel.functions import kernel_function

class GoogleCalendarPlugin:
    """A plugin to interact with Google Calendar."""
    
    SCOPES = ["https://www.googleapis.com/auth/calendar.events"]

    def _get_credentials(self):
        """Gets valid user credentials for the Google Calendar API."""
        creds = None
        if os.path.exists("token.json"):
            creds = Credentials.from_authorized_user_file("token.json", self.SCOPES)
            
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file("credentials.json", self.SCOPES)
                creds = flow.run_local_server(port=0)
            with open("token.json", "w") as token:
                token.write(creds.to_json())
        return creds

    @kernel_function(
        description="Creates an event in the user's Google Calendar.",
        name="create_calendar_event"
    )
    def create_calendar_event(
        self,
        summary: str,
        match_context: str,
    ) -> str:
        """
        Creates a Google Calendar event based on the text description of a match.

        Args:
            summary (str): The title of the event (e.g., 'Germany W vs Spain W').
            match_context (str): The full text description of the match, including date and time.
        """
        print("\n--- [Calendar Plugin] Received a request to create an event ---")
        print(f"  Summary: {summary}")
        print(f"  Match Context: {match_context}")
        
        try:
            # --- NEW: Reliably parse date and time from the context string ---
            # This regex finds a date and time pattern like '2025-07-23 at 19:00'
            match = re.search(r"(\d{4}-\d{2}-\d{2}) at (\d{2}:\d{2})", match_context)
            if not match:
                return "Failed to create event: could not find a valid date and time in the match details."

            date_str, time_str = match.groups()
            start_utc = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
            end_utc = start_utc + timedelta(hours=2) # Assume 2-hour duration

            creds = self._get_credentials()
            service = build("calendar", "v3", credentials=creds)

            event = {
                "summary": summary,
                "description": match_context,
                # The API is smart enough to handle ISO format with UTC timezone info
                "start": {"dateTime": start_utc.isoformat()},
                "end": {"dateTime": end_utc.isoformat()},
            }

            print("\n  Sending the following event object to Google:")
            print(json.dumps(event, indent=2))
            created_event = service.events().insert(calendarId="primary", body=event).execute()
            print("\n  Received a successful response from Google.")
            
            return f"Successfully created a calendar event titled '{summary}'."

        except errors.HttpError as error:
            message = json.loads(error.content).get('error', {}).get('message', 'No details.')
            print(f"\n--- [Calendar Plugin] HTTP Error from Google! ---")
            print(f"  Error details: {message}")
            return f"Failed to create event. Google API Error: {message}"
        except Exception as e:
            print(f"\n--- [Calendar Plugin] A general error occurred! ---")
            print(f"  Error details: {e}")
            return f"An error occurred: {e}"