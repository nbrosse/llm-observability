# Import necessary libraries and modules.
import traceback  # For formatting exception tracebacks for logging.
from typing import Literal  # For creating type hints for specific string values.
import uuid  # For generating unique session IDs.
from fasthtml.common import *  # Core components from the FastHTML library.
from fasthtml.components import Zero_md  # A specific component for rendering Markdown.
from dotenv import load_dotenv  # To load environment variables from a .env file.
from google import genai  # The Google Generative AI client library.
from google.genai.chats import Chat  # The specific Chat object from the Google library.
from langfuse import get_client  # For getting the Langfuse client instance.
from langfuse._client.span import LangfuseSpan  # The Langfuse Span object for tracing.
import os  # For accessing environment variables.

from langfuse.model import ModelUsage  # Langfuse model for tracking token usage.

# Load environment variables from a .env file into the environment.
load_dotenv()

# --- Configuration Constants ---

# The name of the environment variable that holds the Google API key.
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"
# The specific Google Generative AI model to be used for the chat.
GOOGLE_MODEL_NAME = "gemini-2.0-flash"  # Ensure this model name is correct for your Google AI setup

# --- Langfuse Constants ---
# The name for the parent span that encompasses an entire chat conversation.
LANGFUSE_CONVERSATION_SPAN_NAME = "chat_conversation"
# The name for a generation span, representing a single turn from the language model.
LANGFUSE_GENERATION_NAME = "llm_turn"
# The name used in Langfuse for scores that come from user feedback.
LANGFUSE_SCORE_NAME = "user_feedback_score"

# Define a type alias 'Role' which can only be the string "user" or "assistant".
type Role = Literal["user", "assistant"]

# Set up the HTML headers for the application.
# This includes CSS and JavaScript for styling and functionality.
hdrs = (
    picolink,  # A minimal CSS framework.
    Script(src="https://cdn.tailwindcss.com"),  # Tailwind CSS for utility-first styling.
    Link(
        rel="stylesheet", href="https://cdn.jsdelivr.net/npm/daisyui@4.11.1/dist/full.min.css"
    ),  # DaisyUI component library for Tailwind.
    Script(
        type="module", src="https://cdn.jsdelivr.net/npm/zero-md@3?register"
    ),  # Zero-md for rendering markdown content on the client side.
)


def render_local_md(md: str) -> Zero_md:
    """Renders a markdown string using the Zero-md component.

    It injects custom CSS to override the default white background and dark text,
    allowing the markdown to inherit the styling of its container (e.g., the chat bubble).
    """
    # CSS to unset the default background and color, making it transparent.
    css = ".markdown-body {background-color: unset !important; color: unset !important;}"
    # A template to hold the custom style.
    css_template = Template(Style(css), data_append=True)
    # The Zero_md component containing the style and the markdown content.
    return Zero_md(css_template, Script(md, type="text/markdown"))


class SessionsManager:
    """
    Manages all session-related states, including Google GenAI chat sessions
    and Langfuse tracing spans, mapping them by a unique session ID.
    """

    def __init__(self):
        """Initializes the SessionsManager, setting up Google and Langfuse clients."""
        # Retrieve the Google API key from environment variables.
        google_api_key = os.getenv(GOOGLE_API_KEY_ENV)
        if not google_api_key:
            raise ValueError(f"Environment variable '{GOOGLE_API_KEY_ENV}' is not set.")

        # Configure and create the Google GenAI client.
        self.google_client = genai.Client(api_key=google_api_key)

        # Check for and initialize the Langfuse client.
        required_langfuse_envs = ["LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST"]
        missing = [env for env in required_langfuse_envs if not os.getenv(env)]
        if missing:
            raise ValueError(f"Missing required Langfuse environment variables: {', '.join(missing)}")
        self.langfuse_client = get_client()
        # Verify that the Langfuse credentials are correct.
        if not self.langfuse_client.auth_check():
            raise RuntimeError("Failed to initialize Langfuse client. Check your environment variables.")

        # Dictionaries to store active sessions, keyed by session_id.
        self._chats: dict[str, Chat] = {}  # Stores Google GenAI chat objects.
        self._conversations_spans: dict[str, LangfuseSpan] = {}  # Stores Langfuse conversation spans.

    def get_google_chat_session(self, session_id: str) -> Chat:
        """
        Retrieves the Google GenAI chat session for a given session ID.
        If a session doesn't exist, it creates a new one.
        """
        if session_id not in self._chats:
            # Create a new chat session using the specified model.
            self._chats[session_id] = self.google_client.chats.create(model=GOOGLE_MODEL_NAME)
        return self._chats[session_id]

    def clear_google_chat_session(self, session_id: str) -> None:
        """Removes the Google GenAI chat session for a given session ID, effectively resetting it."""
        if session_id in self._chats:
            del self._chats[session_id]

    def get_conversation_span(self, session_id: str) -> LangfuseSpan:
        """
        Retrieves or creates the parent Langfuse span for the entire conversation.
        This span groups all related LLM turns (generations) into a single trace.
        """
        if session_id not in self._conversations_spans:
            # Start a new span (which also creates a new trace).
            self._conversations_spans[session_id] = self.langfuse_client.start_span(
                name=LANGFUSE_CONVERSATION_SPAN_NAME
            )
            # Associate the trace with the user's session ID for filtering in Langfuse.
            self._conversations_spans[session_id].update_trace(user_id=session_id)
        return self._conversations_spans[session_id]

    def end_conversation_span(self, session_id: str) -> None:
        """
        Ends the current Langfuse conversation span and ensures all buffered
        data is sent to the Langfuse server.
        """
        if session_id in self._conversations_spans:
            self._conversations_spans[session_id].end()
            del self._conversations_spans[session_id]
            self.langfuse_client.flush()  # Manually flush to ensure data is sent.


# Initialize the FastHTML application with the defined headers and a default CSS class.
app = FastHTML(hdrs=hdrs, cls="p-4 max-w-lg mx-auto", exts="ws")

# Create a single instance of the SessionsManager to be used by the entire application.
sessions_manager = SessionsManager()


def ChatMessage(msg: str, role: Role, trace_id: str | None = None, observation_id: str | None = None) -> Div:
    """
    A component function that renders a single chat message bubble.
    """
    rendered_msg = render_local_md(msg)  # Convert markdown text to a renderable component.
    # Determine bubble color based on the role (user or assistant).
    bubble_class = "chat-bubble-primary" if role == "user" else "chat-bubble-secondary"
    # Determine bubble alignment based on the role.
    chat_class = "chat-end" if role == "user" else "chat-start"

    feedback_buttons_html = ""

    # Only show feedback buttons for assistant messages that have tracing info.
    if role == "assistant" and trace_id and observation_id:
        feedback_container_id = f"feedback-{observation_id}"
        # Values to be sent when the user clicks the "like" button.
        vals_up = {"observation_id": observation_id, "trace_id": trace_id, "score": 1}
        # Values to be sent when the user clicks the "dislike" button.
        vals_down = {"observation_id": observation_id, "trace_id": trace_id, "score": 0}

        # Create the feedback buttons using htmx attributes for AJAX POST requests.
        feedback_buttons_html = Div(
            Button(
                "👍",
                hx_post="/score_message",
                hx_vals=vals_up,  # POST to /score_message with 'up' values.
                hx_target=f"#{feedback_container_id}",
                hx_swap="outerHTML",  # Replace the buttons with the response.
                cls="btn btn-xs btn-ghost",
            ),
            Button(
                "👎",
                hx_post="/score_message",
                hx_vals=vals_down,  # POST to /score_message with 'down' values.
                hx_target=f"#{feedback_container_id}",
                hx_swap="outerHTML",  # Replace the buttons with the response.
                cls="btn btn-xs btn-ghost",
            ),
            id=feedback_container_id,
            cls="flex space-x-1 mt-1",
        )

    # Construct the final chat message div.
    return Div(cls=f"chat {chat_class}")(
        Div(role, cls="chat-header"),
        Div(rendered_msg, cls=f"chat-bubble {bubble_class}"),
        feedback_buttons_html if feedback_buttons_html else "",
    )


def ChatInput() -> Input:
    """
    A component function that returns the chat input field.
    The `hx_swap_oob='true'` attribute allows this component to be targeted
    for an "Out of Band" swap, which is used to clear the input after a message is sent.
    """
    return Input(
        name="msg",
        id="msg-input",
        placeholder="Type a message",
        cls="input input-bordered w-full",
        hx_swap_oob="true",
        autocomplete="off",  # Disable browser's native autocomplete.
    )


# The main application route, handling GET requests to the root URL.
@app.get("/")
def index():
    """Defines the main chat page UI."""
    # The main form element that handles WebSocket communication.
    page = Form(
        ws_send=True,  # Automatically sends form data over the WebSocket on submit.
        hx_ext="ws",  # Enables the htmx WebSocket extension.
        ws_connect="/wscon",  # The WebSocket endpoint to connect to.
    )(
        # The container where chat messages will be appended.
        Div(id="chatlist", cls="chat-box h-[73vh] overflow-y-auto"),
        # The container for the input field and buttons.
        Div(cls="flex space-x-2 mt-2")(
            Group(
                ChatInput(),
                Button("Send", cls="btn btn-primary", hx_vals='{"action": "send"}'),
                Button(
                    "Clear Chat",
                    cls="btn btn-warning",
                    hx_post="/clear_chat",
                    hx_target="#chatlist",
                    hx_swap="innerHTML",
                    hx_include="[name='session_id']",
                ),
            ),
        ),
        # A hidden input to store the unique session ID for this client.
        Hidden(name="session_id", id="session-id", hx_swap_oob="true", value=str(uuid.uuid4())),
    )
    # Return the page wrapped in a title.
    return Titled("Chatbot Demo", page)


async def on_connect(ws, send):
    """Callback function executed when a new WebSocket connection is established."""
    # Generate a new unique ID for this session.
    session_id = str(uuid.uuid4())
    # Store the session ID in the WebSocket's scope for later access.
    ws.scope["session_id"] = session_id
    # Send a new hidden input field to the client via an OOB swap.
    # This updates the `session-id` input on the client with the server-generated ID.
    await send(Hidden(name="session_id", id="session-id", value=session_id, hx_swap_oob="true"))
    print(f"SERVER: WebSocket connected. Session ID: {session_id}.")


async def on_disconnect(ws):
    """
    Callback function executed when a WebSocket connection is closed.
    This is used for cleaning up server-side resources associated with the session.
    """
    # Retrieve the session ID from the connection's scope.
    session_id = ws.scope.get("session_id", None)
    if not session_id:
        print("ERROR: WebSocket disconnect called without a session ID. Cannot clean up.")
        return
    print(f"SERVER: WebSocket disconnected for Session ID: {session_id}. Cleaning up session.")
    try:
        # Get the session objects.
        current_chat_session = sessions_manager.get_google_chat_session(session_id=session_id)
        conv_span = sessions_manager.get_conversation_span(session_id=session_id)
        # Extract the full chat history from the Google chat object.
        messages = [
            {"role": message.role, "content": getattr(message.parts[0], "text", "")}
            for message in current_chat_session.get_history()
            if hasattr(message, "role") and hasattr(message, "parts") and message.parts
        ]
        # Update the Langfuse span with the full conversation history before ending it.
        if messages and conv_span:
            conv_span.update(
                input=messages[:-1],  # All messages except the last are considered input.
                output=messages[-1],  # The final message is the output of the whole conversation.
            )
        # Clean up the session data from the manager.
        sessions_manager.clear_google_chat_session(session_id=session_id)
        sessions_manager.end_conversation_span(session_id=session_id)
        print(f"SERVER: Cleanup complete for session: {session_id}.")
    except Exception as e:
        # Log any errors that occur during the cleanup process.
        print(f"ERROR during WebSocket disconnect cleanup for Session ID: {session_id}: {e}\n{traceback.format_exc()}")


@app.ws("/wscon", conn=on_connect, disconn=on_disconnect)
async def ws_chat_handler(msg: str, ws, send):
    """
    The main WebSocket message handler. This is called every time a message is
    received from a client over the WebSocket.
    """
    # Get the session ID associated with this WebSocket connection.
    session_id = ws.scope.get("session_id", None)
    if not session_id:
        print("ERROR: WebSocket handler called without a session ID. Cannot process message.")
        return

    # Ignore empty messages from the client.
    if not msg.strip():
        await send(ChatInput())  # Resend a clean input field.
        return

    # Ensure the conversation span and chat session are ready.
    conv_span = sessions_manager.get_conversation_span(session_id=session_id)
    current_chat_session = sessions_manager.get_google_chat_session(session_id=session_id)

    # Get the trace ID for this conversation to pass to the UI for feedback.
    trace_id = conv_span.trace_id

    # --- Optimistic UI Update ---
    # Immediately send the user's message back to them so it appears in the chat list.
    await send(Div(ChatMessage(msg=msg, role="user"), hx_swap_oob="beforeend", id="chatlist"))
    # Send a new, empty input field to clear the user's input.
    await send(ChatInput())

    try:
        # Start a Langfuse generation span to trace this specific LLM call.
        with conv_span.start_as_current_generation(
            name=LANGFUSE_GENERATION_NAME, input=msg, model=GOOGLE_MODEL_NAME
        ) as generation:
            # Send the user's message to the Google GenAI API.
            response = current_chat_session.send_message(msg)
            r = response.text.rstrip()  # Get the response text and clean it.

            # Create a ModelUsage object with token counts from the response metadata.
            usage = ModelUsage(
                input=response.usage_metadata.prompt_token_count,
                output=response.usage_metadata.candidates_token_count,
                total=response.usage_metadata.total_token_count,
            )
            # Update the Langfuse generation with the output and token usage.
            generation.update(output=r, usage_details=usage)
            # Get the unique ID of this generation for the feedback mechanism.
            observation_id = generation.id

        # Send the assistant's response to the client's chat list.
        await send(
            Div(
                ChatMessage(msg=r, role="assistant", trace_id=trace_id, observation_id=observation_id),
                hx_swap_oob="beforeend",
                id="chatlist",
            )
        )
    except Exception as e:
        # Log the full error on the server for debugging.
        print(f"ERROR in WebSocket handler during AI call: {e}\n{traceback.format_exc()}")

        # Log the error as an "event" within the Langfuse conversation trace for observability.
        if conv_span:
            conv_span.create_event(
                name="llm_turn_error",
                level="ERROR",
                status_message=str(e),
                metadata={"traceback": traceback.format_exc()},
            )

        # Send a user-friendly error message to the chat UI.
        error_ui_msg = "Sorry, I encountered an issue processing your message. Please try again."
        await send(Div(ChatMessage(msg=error_ui_msg, role="assistant"), hx_swap_oob="beforeend", id="chatlist"))


@app.post("/clear_chat")
def clear_chat(session_id: str):
    """
    HTTP endpoint to handle the 'Clear Chat' button press.
    It resets the chat session on the server.
    """
    # Retrieve the current session objects.
    current_chat_session = sessions_manager.get_google_chat_session(session_id=session_id)
    conv_span = sessions_manager.get_conversation_span(session_id=session_id)

    # Extract chat history before clearing, to update the Langfuse trace.
    messages = [
        {"role": message.role, "content": getattr(message.parts[0], "text", "")}
        for message in current_chat_session.get_history()
        if hasattr(message, "role") and hasattr(message, "parts") and message.parts
    ]
    # Update the Langfuse span with the full conversation history.
    if messages and conv_span:
        conv_span.update(
            input=messages[:-1],
            output=messages[-1],
        )

    # Clear the server-side session data and end the Langfuse span.
    sessions_manager.clear_google_chat_session(session_id=session_id)
    sessions_manager.end_conversation_span(session_id=session_id)

    # Return an empty chat list (to replace the old one) and a fresh input field.
    return Div(id="chatlist", cls="chat-box h-[73vh] overflow-y-auto"), ChatInput()


@app.post("/score_message")
def score_message(trace_id: str, observation_id: str, score: int):
    """
    HTTP endpoint to handle user feedback (👍/👎).
    It logs a score in Langfuse for the specific LLM generation.
    """
    try:
        # Use the Langfuse client to create a score.
        sessions_manager.langfuse_client.create_score(
            name=LANGFUSE_SCORE_NAME,
            trace_id=trace_id,  # Link the score to the correct conversation trace.
            observation_id=observation_id,  # Link the score to the specific message.
            value=score,  # The score value (1 for like, 0 for dislike).
            data_type="BOOLEAN",  # The type of the score value.
        )
        # Ensure the score is sent immediately.
        sessions_manager.langfuse_client.flush()
        # Return a "Thanks!" message to replace the feedback buttons in the UI.
        return P("Thanks!", cls="text-xs text-success mt-1 ml-2")
    except Exception as e:
        # Log any error that occurs while trying to record the score.
        print(f"Error scoring message: {observation_id}, score: {score}, error: {e}\n{traceback.format_exc()}")
        # Also, try to log this scoring failure as an event in Langfuse for better observability.
        try:
            sessions_manager.langfuse_client.event(
                trace_id=trace_id,
                parent_observation_id=observation_id,
                name="scoring_error",
                level="ERROR",
                input={"observation_id": observation_id, "score_attempted": score},
                output={"error_message": str(e)},
                metadata={"traceback": traceback.format_exc()},
            )
            sessions_manager.langfuse_client.flush()
        except Exception as langfuse_event_err:
            print(f"CRITICAL: Failed to log scoring error to Langfuse: {langfuse_event_err}")

        # Return an error message to the user.
        return P("Error.", cls="text-xs text-error mt-1 ml-2")


# Standard Python entry point to run the application server.
if __name__ == "__main__":
    serve()
