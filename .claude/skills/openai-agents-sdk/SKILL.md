---
name: openai-agents-sdk
description: Complete OpenAI Agents SDK implementation covering all components: Agents, Tools, Handoffs, Guardrails, Sessions, and Tracing. Implements all documented patterns and best practices from official documentation. Use when building multi-agent systems with proper tool integration, safety checks, and session management.
---

# OpenAI Agents SDK Complete Skill

This skill implements the complete OpenAI Agents SDK framework with all documented components and best practices.

## Documentation References

Based on OpenAI Agents SDK documentation:
- Core Documentation: https://openai.github.io/openai-agents-python/
- Tools Documentation: https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/tools.md
- Guardrails Documentation: https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/guardrails.md
- Handoffs Documentation: https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/handoffs.md

## Overview

The OpenAI Agents SDK is a lightweight framework for building multi-agent workflows that supports OpenAI APIs and 100+ other LLMs.

## Installation

```bash
pip install openai-agents
```

For optional features:
- Voice support: `pip install 'openai-agents[voice]'`
- Redis sessions: `pip install 'openai-agents[redis]'`

## Quickstart

Create and run your first agent:

```python
from agents import Agent, Runner

# Create an agent
agent = Agent(
    name="Math Tutor",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples"
)

# Run the agent
result = Runner.run_sync(agent, "Solve 2x + 5 = 15")
print(result.final_output)
```

## Provider Configuration

The OpenAI Agents SDK supports 100+ LLM providers beyond OpenAI. To use alternative providers, configure your agent with the appropriate model and authentication:

### Global Configuration Functions

```python
from agents import set_default_openai_key, set_default_openai_client, set_tracing_disabled

# Set default OpenAI API key
set_default_openai_key("your-api-key", use_for_tracing=True)

# Set default OpenAI client for requests/tracing
# set_default_openai_client(your_openai_client)

# Disable tracing globally
set_tracing_disabled()
```

### Using Alternative Providers

```python
# For Anthropic models
assistant_agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant",
    model="claude-3-sonnet-20240229",  # Anthropic model
    # Additional provider-specific configuration
)

# For Google models
google_agent = Agent(
    name="Google Assistant",
    instructions="You are a helpful assistant",
    model="gemini-1.5-pro",  # Google model
    # Additional provider-specific configuration
)

# For OpenRouter models
openrouter_agent = Agent(
    name="OpenRouter Assistant",
    instructions="You are a helpful assistant",
    model="openai/gpt-4o",  # OpenRouter format: provider/model
    # Additional provider-specific configuration
)

# For Azure OpenAI
azure_agent = Agent(
    name="Azure Assistant",
    instructions="You are a helpful assistant",
    model="gpt-4",  # Your Azure deployment name
    # Additional Azure-specific configuration
)

# Using LiteLLM for provider flexibility
from agents.extensions.models.litellm_model import LitellmModel

litellm_agent = Agent(
    name="LiteLLM Agent",
    instructions="You are a flexible assistant using LiteLLM",
    model=LitellmModel(model="openai/gpt-4o", api_key="your-api-key"),  # Can be any supported provider/model combination
    # LiteLLM handles provider routing
)

# Tracking usage data with LiteLLM
from agents.model_settings import ModelSettings

agent_with_usage = Agent(
    model=LitellmModel(model="your/model", api_key="..."),
    model_settings=ModelSettings(include_usage=True),  # Enable usage tracking
)
# With include_usage=True, token and request counts are available through result.context_wrapper.usage
```

### LiteLLM Setup

Install the optional dependency for LiteLLM support:
```bash
pip install "openai-agents[litellm]"
```

Use `LitellmModel` in any agent after installation to access 100+ models through a single interface.

### Environment Variables for Different Providers

Different providers require different API keys and configuration:

```bash
# OpenAI
OPENAI_API_KEY=your-openai-api-key

# Anthropic
ANTHROPIC_API_KEY=your-anthropic-api-key

# Google
GOOGLE_API_KEY=your-google-api-key

# OpenRouter
OPENROUTER_API_KEY=your-openrouter-api-key

# Azure OpenAI
AZURE_OPENAI_API_KEY=your-azure-api-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/

# Other providers as required by your specific LLM provider
```

### Multi-Agent Configuration with Handoffs

Create multiple agents and configure handoffs between them:

```python
from agents import Agent, Runner

# Create specialized agents
history_tutor_agent = Agent(
    name="History Tutor",
    handoff_description="Specialist agent for historical questions",
    instructions="You provide assistance with historical queries. Explain important events and context clearly."
)

math_tutor_agent = Agent(
    name="Math Tutor",
    handoff_description="Specialist agent for math questions",
    instructions="You provide help with math problems. Explain your reasoning at each step and include examples",
)

# Create triage agent with handoffs
triage_agent = Agent(
    name="Triage Agent",
    instructions="You determine which agent to use based on the user's homework question",
    handoffs=[history_tutor_agent, math_tutor_agent]
)

# Run the agent orchestration
async def main():
    result = await Runner.run(triage_agent, "who was the first president of the united states?")
    print(result.final_output)
```

## Core Components

### 1. Agents

Agents are LLMs configured with instructions, tools, guardrails, and handoffs.

```python
from agents import Agent, Runner
from agents.decorators import function_tool
from agents.types import GuardrailFunctionOutput
from agents.extensions import SQLiteSession
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

# Basic agent example with provider-specific configuration
# See https://openai.github.io/openai-agents-python/#agents for agent definition
assistant_agent = Agent(
    name="Assistant",
    instructions="You are a helpful assistant",
    model="gpt-4o",  # Specify your preferred model
    temperature=0.7   # Adjust temperature as needed
)

# Run the agent
result = Runner.run_sync(assistant_agent, "Write a haiku about recursion in programming.")
print(result.final_output)
```

### 2. Agent Loop

When calling `Runner.run()`, the SDK runs a loop until final output:
1. Call LLM with agent model/settings and message history
2. Process LLM response (may include tool calls)
3. Return final output if present, else continue
4. Process handoffs or tool calls, then repeat

```python
# See https://openai.github.io/openai-agents-python/#agent-loop for loop specs
async def run_agent_with_loop(agent: Agent, message: str):
    """
    Execute agent with full loop processing
    """
    result = await Runner.run(agent, message)
    return result.final_output
```

### 3. Tools and Functions

The Agent SDK provides three classes of tools for agents to take actions:

#### Hosted Tools
Built-in tools available with `OpenAIResponsesModel`:
- `WebSearchTool` - search the web
- `FileSearchTool` - retrieve from OpenAI Vector Stores
- `ComputerTool` - automate computer tasks
- `CodeInterpreterTool` - execute code in sandbox
- `ImageGenerationTool` - generate images from prompts
- `LocalShellTool` - run shell commands

#### Function Tools
Decorate Python functions with `@function_tool` to automatically create tools:

```python
# See https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/tools.md#function-tools for tool specs
@function_tool
def get_weather(location: str) -> str:
    """Get the current weather for a location."""
    # Implementation here
    return "sunny"

@function_tool
async def fetch_user_data(user_id: str) -> Dict[str, Any]:
    """Fetch user data by ID."""
    # Implementation here
    return {"id": user_id, "name": "John Doe"}

# Function tools features:
# - Automatic schema generation from function signatures
# - Docstring parsing for descriptions
# - Support for sync/async functions
# - Context injection via RunContextWrapper
# - Custom name overrides
```

#### Agents as Tools
Orchestrate specialized agents by converting them to tools:

```python
# See https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/tools.md#agents-as-tools for specs
spanish_agent = Agent(name="Spanish Translator", instructions="Translate to Spanish")
spanish_tool = spanish_agent.as_tool(
    tool_name="translate_to_spanish",
    tool_description="Translate text to Spanish"
)
```

## Handoffs

Handoffs enable agent-to-agent delegation for specialized tasks.

```python
from agents import Agent, handoff
from agents.extensions.handoff_filters import collapse_messages

# Create specialized agents
billing_agent = Agent(
    name="Billing Agent",
    instructions="Handle billing-related inquiries"
)

refund_agent = Agent(
    name="Refund Agent",
    instructions="Handle refund requests and processes"
)

# Create triage agent with handoffs
# See https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/handoffs.md for handoff specs
triage_agent = Agent(
    name="Triage Agent",
    instructions="""
    You are a triage agent. Analyze incoming requests and hand off to the appropriate specialist:
    - Billing inquiries: transfer to Billing Agent
    - Refund requests: transfer to Refund Agent
    """,
    handoffs=[billing_agent, handoff(refund_agent)]
)

# Custom handoff with parameters
class HandoffData(BaseModel):
    reason: str
    priority: int

def on_refund_handoff(data: HandoffData):
    """Callback when handoff occurs"""
    print(f"Handing off with reason: {data.reason}")

refund_handoff = handoff(
    agent=refund_agent,
    on_handoff=on_refund_handoff,
    input_type=HandoffData,
    input_filter=collapse_messages,
    tool_name_override="escalate_to_refund_specialist",
    tool_description_override="Transfer to refund specialist with escalation data"
)
```

## Guardrails

Guardrails perform checks on user input and agent output to validate content and prevent unwanted usage.

```python
from agents.decorators import input_guardrail, output_guardrail

# Input guardrails validate user input before agent processing
# See https://raw.githubusercontent.com/openai/openai-agents-python/main/docs/guardrails.md for guardrail specs
@input_guardrail
async def math_homework_guardrail(ctx, agent, input) -> GuardrailFunctionOutput:
    """
    Input guardrail to detect math homework requests
    """
    # Check if input contains math homework
    is_math_homework = check_if_math_homework(input)

    return GuardrailFunctionOutput(
        output_info={"is_math_homework": is_math_homework},
        tripwire_triggered=is_math_homework,  # Trigger tripwire if math homework detected
    )

# Output guardrails check final agent output for compliance
@output_guardrail
async def content_safety_guardrail(ctx, agent, input, output) -> GuardrailFunctionOutput:
    """
    Output guardrail to ensure content safety
    """
    # Check if output contains unsafe content
    has_unsafe_content = check_unsafe_content(output)

    return GuardrailFunctionOutput(
        output_info={"is_safe": not has_unsafe_content},
        tripwire_triggered=has_unsafe_content,
    )

# Guardrails support two execution modes:
# - Parallel (default): Runs concurrently with agent for best latency
# - Blocking: Completes before agent starts to prevent token consumption
```

## Sessions

Sessions provide built-in memory to maintain conversation history across multiple agent runs, eliminating manual `.to_input_list()` handling.

### Quick Start

```python
from agents import Agent, Runner, SQLiteSession

agent = Agent(
    name="Assistant",
    instructions="Reply very concisely.",
)

session = SQLiteSession("conversation_123")

result = await Runner.run(
    agent,
    "What city is the Golden Gate Bridge in?",
    session=session
)
print(result.final_output)  # "San Francisco"

result = await Runner.run(
    agent,
    "What state is it in?",
    session=session
)
print(result.final_output)  # "California"
```

### How It Works

When session memory is enabled:
1. Before each run: The runner retrieves conversation history and prepends it to input items
2. After each run: New items generated during the run are automatically stored in the session
3. Context preservation: Each subsequent run includes full conversation history

### Session Types

#### SQLite sessions (default)
The default, lightweight implementation using SQLite:

```python
from agents import SQLiteSession

# In-memory database (lost when process ends)
session = SQLiteSession("user_123")

# Persistent file-based database
session = SQLiteSession("user_123", "conversations.db")
```

#### OpenAI Conversations API sessions
Use OpenAI's Conversations API through `OpenAIConversationsSession`:

```python
from agents import Agent, Runner, OpenAIConversationsSession

session = OpenAIConversationsSession()
# Optionally resume a previous conversation by passing a conversation ID
# session = OpenAIConversationsSession(conversation_id="conv_123")
```

#### SQLAlchemy sessions
Production-ready sessions using any SQLAlchemy-supported database:

```python
from agents.extensions.memory import SQLAlchemySession

session = SQLAlchemySession.from_url(
    "user_123",
    url="postgresql+asyncpg://user:pass@localhost/db",
    create_tables=True
)
```

#### Advanced SQLite sessions
Enhanced SQLite sessions with conversation branching and analytics:

```python
from agents.extensions.memory import AdvancedSQLiteSession

session = AdvancedSQLiteSession(
    session_id="user_123",
    db_path="conversations.db",
    create_tables=True
)

# Automatic usage tracking
result = await Runner.run(agent, "Hello", session=session)
await session.store_run_usage(result)  # Track token usage
```

#### Encrypted sessions
Transparent encryption wrapper for any session implementation:

```python
from agents.extensions.memory import EncryptedSession, SQLAlchemySession

underlying_session = SQLAlchemySession.from_url(
    "user_123",
    url="sqlite+aiosqlite:///conversations.db",
    create_tables=True
)

session = EncryptedSession(
    session_id="user_123",
    underlying_session=underlying_session,
    encryption_key="your-secret-key",
    ttl=600  # 10 minutes
)
```

### Session Management

Use meaningful session IDs that help organize conversations:
- User-based: `"user_12345"`
- Thread-based: `"thread_abc123"`
- Context-based: `"support_ticket_456"`

Memory persistence options:
- Use in-memory SQLite for temporary conversations
- Use file-based SQLite for persistent conversations
- Use SQLAlchemy-powered sessions for production systems
- Use OpenAI-hosted storage when preferring OpenAI Conversations API
- Use encrypted sessions to wrap any session with transparent encryption

### Memory Operations

```python
session = SQLiteSession("user_123", "conversations.db")

# Get all items in a session
items = await session.get_items()

# Add new items to a session
new_items = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"}
]
await session.add_items(new_items)

# Remove and return the most recent item
last_item = await session.pop_item()
print(last_item)  # {"role": "assistant", "content": "Hi there!"}

# Clear all items from a session
await session.clear_session()
```

Using pop_item for corrections:
```python
# User wants to correct their question
assistant_item = await session.pop_item()  # Remove agent's response
user_item = await session.pop_item()  # Remove user's question

# Ask a corrected question
result = await Runner.run(
    agent,
    "What's 2 + 3?",
    session=session
)
```

# See https://openai.github.io/openai-agents-python/sessions for session specs

## Complete Multi-Agent System Example

```python
from agents import Agent, Runner, handoff
from agents.decorators import function_tool, input_guardrail, output_guardrail
from agents.types import GuardrailFunctionOutput
from agents.extensions import SQLiteSession
from pydantic import BaseModel
import asyncio

# Define input model for handoffs
class EscalationData(BaseModel):
    reason: str
    priority: int = 1

# Define function tools
@function_tool
def search_knowledge_base(query: str, top_k: int = 3) -> List[Dict[str, str]]:
    """Search the knowledge base for relevant information."""
    # Implementation here
    return [{"id": "1", "content": "Sample result", "score": 0.9}]

@function_tool
def get_user_info(user_id: str) -> Dict[str, Any]:
    """Get information about a user."""
    # Implementation here
    return {"id": user_id, "name": "John Doe", "status": "active"}

# Define specialized agents
def create_support_agents():
    """
    Create specialized support agents with handoffs
    """
    # Tier 1 support agent
    tier1_agent = Agent(
        name="Tier 1 Support",
        instructions="""
        You are a first-level support agent. Handle basic inquiries and common questions.
        If the issue is complex or requires escalation, hand off to Tier 2 support.
        """,
        functions=[search_knowledge_base, get_user_info]
    )

    # Tier 2 support agent
    tier2_agent = Agent(
        name="Tier 2 Support",
        instructions="""
        You are a second-level support agent. Handle complex technical issues and escalations.
        """,
        functions=[search_knowledge_base, get_user_info]
    )

    # Create handoff to tier 2
    tier2_handoff = handoff(
        agent=tier2_agent,
        tool_name_override="escalate_to_tier2",
        tool_description_override="Escalate issue to Tier 2 support specialist",
        input_type=EscalationData
    )

    # Triage agent that can route to appropriate support level
    triage_agent = Agent(
        name="Triage Agent",
        instructions="""
        You are a triage agent. Analyze incoming support requests:
        - Simple questions: Answer directly using knowledge base
        - Complex issues: Hand off to Tier 2 support
        """,
        functions=[search_knowledge_base, get_user_info],
        handoffs=[tier2_handoff]
    )

    return triage_agent, tier1_agent, tier2_agent

# Define guardrails
@input_guardrail
async def support_request_guardrail(ctx, agent, input) -> GuardrailFunctionOutput:
    """
    Check if support request is appropriate
    """
    is_appropriate = check_support_request_appropriateness(input)
    return GuardrailFunctionOutput(
        output_info={"is_appropriate": is_appropriate},
        tripwire_triggered=not is_appropriate
    )

@output_guardrail
async def response_quality_guardrail(ctx, agent, input, output) -> GuardrailFunctionOutput:
    """
    Check response quality and compliance
    """
    quality_score = evaluate_response_quality(output, input)
    is_compliant = quality_score >= 0.8  # Threshold for quality
    return GuardrailFunctionOutput(
        output_info={"quality_score": quality_score},
        tripwire_triggered=not is_compliant
    )

# Complete system with all components
class SupportSystem:
    """
    Complete support system with agents, tools, handoffs, and guardrails
    """
    def __init__(self):
        self.triage_agent, self.tier1_agent, self.tier2_agent = create_support_agents()
        self.session = SQLiteSession("support_conversation")

    async def handle_request(self, user_request: str, user_id: str = None):
        """
        Handle a support request with full system
        """
        result = await Runner.run(
            self.triage_agent,
            user_request,
            context_variables={"user_id": user_id} if user_id else {},
            session=self.session
        )
        return result.final_output

# Example usage
async def main():
    support_system = SupportSystem()
    response = await support_system.handle_request("How do I reset my password?")
    print(response)

# Run the system
# asyncio.run(main())
```

## Tracing

The Agents SDK includes built-in tracing that collects comprehensive records of events during agent runs, including LLM generations, tool calls, handoffs, guardrails, and custom events. The Traces dashboard at platform.openai.com/traces enables debugging, visualization, and monitoring of workflows during development and production.

### Default Tracing Behavior

Tracing is enabled by default and captures:
- The entire Runner.run() operation wrapped in a trace()
- Each agent run wrapped in agent_span()
- LLM generations wrapped in generation_span()
- Function tool calls wrapped in function_span()
- Guardrails wrapped in guardrail_span()
- Handoffs wrapped in handoff_span()
- Audio inputs wrapped in transcription_span()
- Audio outputs wrapped in speech_span()

To disable tracing, set environment variable or configure RunConfig:
```python
import os
os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"  # Disable tracing globally

# Or disable for specific runs
from agents import set_tracing_disabled
set_tracing_disabled()  # Disable tracing globally
```

### Custom Tracing Options

Developers can create higher-level traces by wrapping multiple run() calls in a trace() context. Custom tracing processors can be added to send traces to alternative backends using add_trace_processor() or replace default processors with set_trace_processors().

For non-OpenAI models, tracing can be enabled using OpenAI API keys with the set_tracing_export_api_key() function.

### Sensitive Data Handling

The system can exclude sensitive data from traces through configuration settings to prevent capturing LLM inputs/outputs or audio data.

### External Integration Support

The SDK supports numerous external tracing processors including Weights & Biases, Arize-Phoenix, MLflow, Braintrust, LangSmith, Langfuse, and others that integrate with the OpenAI Agents SDK tracing system.

```python
# Enable tracing for debugging
# See https://openai.github.io/openai-agents-python/tracing for tracing specs
import os

# Set environment variable for tracing
os.environ["LOGFIRE_TOKEN"] = "your-logfire-token"  # For Logfire integration
# Or other tracing providers as supported
```

## Streaming

The OpenAI Agents SDK provides streaming capabilities through `Runner.run_streamed()` which returns a `RunResultStreaming` object with async stream events.

### Raw Response Events
`RawResponsesStreamEvent` provides raw LLM events in OpenAI Responses API format, useful for token-by-token streaming to users.

### Run Item Events
`RunItemStreamEvent` provides higher-level events when items are fully generated, enabling progress updates at message/tool level rather than per token. Includes `AgentUpdatedStreamEvent` for handoff notifications.

### Example Streaming Usage

```python
from agents import Runner

# Stream responses as they're generated
async def stream_agent_response(agent, message, session=None):
    async for event in Runner.run_streamed(agent, message, session=session):
        if hasattr(event, 'text'):
            print(f"Streaming: {event.text}")
        # Handle different event types as needed
```

The streaming API allows real-time progress updates and partial responses during agent execution, supporting both low-level token streaming and high-level item completion events.

## Final Output Rules

- With `output_type`: Loop runs until structured output matches type
- Without `output_type`: First response without tool calls/handoffs is final

## Development

Requires Python 3.9+ and uv for development:
```bash
make sync      # Install dependencies
make check     # Run tests, linter, typechecker
make tests     # Run tests only
```

## Best Practices

1. **Include handoff instructions in agent prompts** using `RECOMMENDED_PROMPT_PREFIX` to ensure LLMs understand transfer capabilities properly.

2. **Use appropriate session management** to maintain conversation context across interactions.

3. **Implement comprehensive guardrails** to ensure safety and quality of inputs and outputs.

4. **Structure tools properly** with clear function signatures and meaningful descriptions.

5. **Handle errors gracefully** with custom error functions via `failure_error_function` parameter.

## Error Handling

Function tools support custom error functions via `failure_error_function` parameter to provide user-friendly error responses to the LLM when tool calls fail.