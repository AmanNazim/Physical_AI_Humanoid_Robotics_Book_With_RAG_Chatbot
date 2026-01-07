# OpenAI Agents SDK API Reference

## Core Classes and Functions

### Agent Class
```python
class Agent:
    def __init__(
        self,
        name: str,
        instructions: str,
        functions: Optional[List[Callable]] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        handoffs: Optional[List[Union[Agent, Handoff]]] = None,
        output_type: Optional[Type[BaseModel]] = None,
        handoff_description: Optional[str] = None,
        **provider_specific_kwargs
    ):
        """
        Create an agent with instructions, tools, and handoffs.

        Args:
            name: Name of the agent
            instructions: System prompt/instructions for the agent
            functions: List of callable functions available as tools
            model: LLM model to use (defaults to provider default)
            temperature: Temperature for generation (defaults to 0.7)
            handoffs: List of agents to hand off to
            output_type: Pydantic model for structured output
            handoff_description: Description of the agent for handoff purposes
            **provider_specific_kwargs: Additional arguments for specific LLM providers
                - For OpenAI: api_key, base_url, organization, etc.
                - For Anthropic: anthropic_api_key, etc.
                - For Google: google_api_key, etc.
                - For Azure: azure_api_key, azure_endpoint, etc.
                - For OpenRouter: openrouter_api_key, etc.
        """
```

### Global Configuration Functions
```python
def set_default_openai_key(key: str, use_for_tracing: bool = True):
    """
    Sets the default OpenAI API key for LLM requests and tracing.

    Args:
        key: The OpenAI API key to use
        use_for_tracing: Whether to use this key for tracing (default: True)
    """

def set_default_openai_client(client):
    """
    Sets the default OpenAI client for requests and tracing.

    Args:
        client: The OpenAI client instance to use
    """

def set_tracing_disabled(disabled: bool = True):
    """
    Globally enables/disables tracing.

    Args:
        disabled: Whether to disable tracing (default: True)
    """

def set_trace_processors(processors: List[Callable]):
    """
    Replaces current trace processors list.

    Args:
        processors: List of trace processor functions
    """

def enable_verbose_stdout_logging():
    """
    Enables debug logging to stdout.
    """

def set_tracing_export_api_key(api_key: str):
    """
    Sets API key for tracing backend exporter.

    Args:
        api_key: The API key for the tracing backend
    """
```

### Hosted Tools
```python
class WebSearchTool:
    """
    Built-in tool for searching the web.
    Available with OpenAIResponsesModel.
    """

class FileSearchTool:
    """
    Built-in tool for retrieving from OpenAI Vector Stores.
    Available with OpenAIResponsesModel.
    """

class ComputerTool:
    """
    Built-in tool for automating computer tasks.
    Available with OpenAIResponsesModel.
    """

class CodeInterpreterTool:
    """
    Built-in tool for executing code in sandbox.
    Available with OpenAIResponsesModel.
    """

class HostedMCPTool:
    """
    Built-in tool for exposing remote MCP server tools.
    Available with OpenAIResponsesModel.
    """

class ImageGenerationTool:
    """
    Built-in tool for generating images from prompts.
    Available with OpenAIResponsesModel.
    """

class LocalShellTool:
    """
    Built-in tool for running shell commands.
    Available with OpenAIResponsesModel.
    """
```

### Streaming Events
```python
class RawResponsesStreamEvent:
    """
    Provides raw LLM events in OpenAI Responses API format.
    Useful for token-by-token streaming to users.
    """

class RunItemStreamEvent:
    """
    Provides higher-level events when items are fully generated.
    Enables progress updates at message/tool level rather than per token.
    """

class AgentUpdatedStreamEvent:
    """
    Event for handoff notifications during streaming.
    """
```

### Session Classes
```python
class SQLiteSession:
    """
    Default session implementation using SQLite.

    Args:
        session_id: Unique identifier for the session
        db_path: Path to SQLite database (defaults to in-memory)
    """

    async def get_items(self, limit: int | None = None) -> List[TResponseInputItem]:
        """Retrieve conversation history for this session."""

    async def add_items(self, items: List[TResponseInputItem]) -> None:
        """Store new items for this session."""

    async def pop_item(self) -> TResponseInputItem | None:
        """Remove and return the most recent item from this session."""

    async def clear_session(self) -> None:
        """Clear all items for this session."""


class OpenAIConversationsSession:
    """
    Session implementation using OpenAI's Conversations API.

    Args:
        conversation_id: Optional existing conversation ID to resume
    """


class SQLAlchemySession:
    """
    Production-ready session using SQLAlchemy-supported databases.

    Args:
        session_id: Unique identifier for the session
        url: Database URL in SQLAlchemy format
        create_tables: Whether to create tables if they don't exist
    """

    @classmethod
    def from_url(cls, session_id: str, url: str, create_tables: bool = True):
        """Create a session from a database URL."""


class AdvancedSQLiteSession:
    """
    Enhanced SQLite session with conversation branching and analytics.

    Args:
        session_id: Unique identifier for the session
        db_path: Path to SQLite database
        create_tables: Whether to create tables if they don't exist
    """

    async def store_run_usage(self, result) -> None:
        """Track token usage for analytics."""

    async def create_branch_from_turn(self, turn_number: int) -> None:
        """Create a conversation branch from a specific turn."""


class EncryptedSession:
    """
    Transparent encryption wrapper for any session implementation.

    Args:
        session_id: Unique identifier for the session
        underlying_session: The session to wrap with encryption
        encryption_key: Key for encrypting/decrypting data
        ttl: Time-to-live in seconds for encrypted data
    """
```

### LiteLLM Model Support
```python
class LitellmModel:
    """
    LiteLLM model wrapper supporting 100+ models through a single interface.

    Args:
        model: Model identifier (e.g., "openai/gpt-4o", "anthropic/claude-3-5-sonnet")
        api_key: API key for the specific provider
    """
```

### Runner Class
```python
class Runner:
    @staticmethod
    def run_sync(
        agent: Agent,
        message: str,
        *,
        context_variables: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None
    ) -> RunResult:
        """
        Synchronously run an agent with a message.

        Args:
            agent: Agent to run
            message: Initial message/user input
            context_variables: Additional context for the agent
            session: Session for maintaining conversation history

        Returns:
            RunResult containing the final output and metadata
        """

    @staticmethod
    async def run(
        agent: Agent,
        message: str,
        *,
        context_variables: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None
    ) -> RunResult:
        """
        Asynchronously run an agent with a message.

        Args:
            agent: Agent to run
            message: Initial message/user input
            context_variables: Additional context for the agent
            session: Session for maintaining conversation history

        Returns:
            RunResult containing the final output and metadata
        """

    @staticmethod
    async def run_streamed(
        agent: Agent,
        message: str,
        *,
        context_variables: Optional[Dict[str, Any]] = None,
        session: Optional[Session] = None
    ) -> AsyncIterator[StreamEvent]:
        """
        Stream the agent's response as it's generated.

        Args:
            agent: Agent to run
            message: Initial message/user input
            context_variables: Additional context for the agent
            session: Session for maintaining conversation history

        Yields:
            StreamEvent objects as the response is generated
        """
```

### Handoff Class
```python
def handoff(
    agent: Agent,
    tool_name_override: Optional[str] = None,
    tool_description_override: Optional[str] = None,
    on_handoff: Optional[Callable[[Any], None]] = None,
    input_type: Optional[Type[BaseModel]] = None,
    input_filter: Optional[Callable[[List[Message]], List[Message]]] = None,
    is_enabled: Union[bool, Callable[[Dict[str, Any]], bool]] = True
) -> Handoff:
    """
    Create a handoff to another agent.

    Args:
        agent: Target agent for the handoff
        tool_name_override: Override the default tool name
        tool_description_override: Override the default tool description
        on_handoff: Callback function when handoff occurs
        input_type: Expected input data type for the handoff
        input_filter: Filter conversation history for the receiving agent
        is_enabled: Whether the handoff is available (bool or function)

    Returns:
        Handoff object that can be added to an agent
    """
```

### RunResult Class
```python
class RunResult:
    final_output: Any
    messages: List[Message]
    was_handoff: bool
    was_cancelled: bool
    context_wrapper: Any  # Contains usage data if enabled
```

### Message Class
```python
class Message(BaseModel):
    role: Literal["user", "assistant", "system", "tool"]
    content: str
    tool_calls: Optional[List[ToolCall]] = None
    tool_call_id: Optional[str] = None
```

### ToolCall Class
```python
class ToolCall(BaseModel):
    id: str
    function: FunctionCall
    type: Literal["function"] = "function"

class FunctionCall(BaseModel):
    name: str
    arguments: str  # JSON string of arguments
```

### Constants and Enums

#### Common constants for handoff filters
```python
from agents.extensions.handoff_filters import (
    collapse_messages,  # Collapse conversation history to summary
    keep_recent_messages,  # Keep only recent messages
    all_messages,  # Pass all messages to next agent
    remove_all_tools  # Remove tool messages from history
)
```

#### Handoff Prompt Constants
```python
from agents.extensions.handoff_prompt import RECOMMENDED_PROMPT_PREFIX
# Include this prefix in agent instructions to ensure proper handoff behavior
```

## Environment Variables

The SDK recognizes these environment variables:

- `OPENAI_API_KEY`: API key for OpenAI models
- `ANTHROPIC_API_KEY`: API key for Anthropic models
- `GOOGLE_API_KEY`: API key for Google models
- `OPENROUTER_API_KEY`: API key for OpenRouter models
- `AZURE_OPENAI_API_KEY`: API key for Azure OpenAI
- `AZURE_OPENAI_ENDPOINT`: Endpoint URL for Azure OpenAI
- `OPENAI_AGENTS_DISABLE_TRACING`: Set to "1" to disable tracing globally
- `LOGFIRE_TOKEN`: Token for Logfire tracing integration
- `AGENTOPS_API_KEY`: API key for AgentOps integration
- Custom provider API keys as needed

## Error Types

The SDK raises these specific exception types:

- `InputGuardrailTripwireTriggered`: When input guardrail blocks execution
- `OutputGuardrailTripwireTriggered`: When output guardrail blocks execution
- `HandoffInterrupt`: When a handoff occurs during execution
- `ToolExecutionError`: When a tool function fails
- `SessionError`: When session operations fail