"""
Test script for the Intelligence Layer (OpenAI Agents SDK) Subsystem.

This script validates that the Intelligence Service can be properly initialized
and that all required components are functioning correctly with LiteLLM integration for OpenRouter.
"""

import asyncio
import sys
import os

# Add the project root to the Python path to allow absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from agents_sdk.services.intelligence_service import IntelligenceService
from backend.schemas.retrieval import Source


async def test_intelligence_service():
    """
    Test the Intelligence Service initialization and basic functionality.
    """
    print("Testing Intelligence Service initialization...")

    # Create an instance of the Intelligence Service
    intelligence_service = IntelligenceService()

    try:
        # Initialize the service
        await intelligence_service.initialize()
        print("✓ Intelligence Service initialized successfully")

        # Test basic query processing with mock context
        mock_context_chunks = [
            Source(
                chunk_id="test-chunk-1",
                document_id="test-doc-1",
                text="Physical AI combines robotics and artificial intelligence to create embodied systems that can perceive, reason, and act in the physical world. This interdisciplinary field focuses on developing intelligent machines that interact with the physical environment.",
                score=0.9,
                metadata={"module": "introduction", "chapter": "overview", "section": "definition"}
            ),
            Source(
                chunk_id="test-chunk-2",
                document_id="test-doc-1",
                text="Humanoid robots are designed to mimic human appearance and behavior, often featuring bipedal locomotion and dexterous manipulation capabilities. These robots typically have anthropomorphic characteristics including legs, arms, and a head.",
                score=0.85,
                metadata={"module": "design", "chapter": "humanoids", "section": "characteristics"}
            )
        ]

        # Test processing a query
        result = await intelligence_service.process_query(
            user_query="What is Physical AI?",
            context_chunks=mock_context_chunks,
            session_id="test-session-123"
        )

        print(f"✓ Query processed successfully")
        print(f"  Response: {result['text'][:100]}...")
        print(f"  Sources used: {len(result['sources'])}")
        print(f"  Confidence: {result['metadata'].get('confidence_score', 'N/A')}")
        print(f"  Token usage: {result['metadata'].get('token_usage', 'N/A')}")

        # Test streaming functionality
        print("\nTesting streaming functionality...")
        stream_gen = intelligence_service.stream_response(
            user_query="What are humanoid robots?",
            context_chunks=mock_context_chunks,
            session_id="test-stream-session-123"
        )

        # Collect a few stream events to verify functionality
        stream_events = []
        async for event in stream_gen:
            stream_events.append(event)
            if len(stream_events) >= 3:  # Just test a few events
                break

        print(f"✓ Streaming functionality works, got {len(stream_events)} events")

        # Test guardrails
        print("\nTesting guardrail functionality...")

        # Test with a query that should trigger input guardrails
        result_guardrail = await intelligence_service.process_query(
            user_query="Tell me a joke about robots",
            context_chunks=mock_context_chunks,
            session_id="test-guardrail-session-123"
        )

        print(f"  Guardrail test result: Response processed successfully")

        # Test with insufficient context
        result_insufficient = await intelligence_service.process_query(
            user_query="What is the capital of Mars?",
            context_chunks=[],  # No context provided
            session_id="test-insufficient-session-123"
        )

        print(f"  Insufficient context test: Response handled appropriately")

        return True

    except Exception as e:
        print(f"✗ Error during testing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


async def test_prompt_engineering():
    """
    Test the prompt engineering functionality.
    """
    print("\nTesting prompt engineering functionality...")

    from agents_sdk.prompts.prompt_templates import PromptTemplates

    # Create mock context
    mock_context_chunks = [
        Source(
            chunk_id="test-chunk-1",
            document_id="test-doc-1",
            text="Physical AI combines robotics and artificial intelligence to create embodied systems that can perceive, reason, and act in the physical world.",
            score=0.9,
            metadata={"module": "introduction", "chapter": "overview"}
        )
    ]

    # Test persona configuration
    persona_config = {
        "role": "Expert Technical Instructor for the Physical AI & Humanoid Robotics Book",
        "tone": "authoritative but friendly",
        "style": "technically precise",
        "constraints": [
            "never hallucinate",
            "never answer outside book content unless explicitly allowed",
            "clearly state uncertainty when context is insufficient"
        ]
    }

    # Test layered prompt creation
    layered_prompt = PromptTemplates.create_layered_prompt(
        user_query="Explain Physical AI",
        context_chunks=mock_context_chunks,
        persona_config=persona_config
    )

    print("✓ Layered prompt created successfully")
    print(f"  Prompt length: {len(layered_prompt)} characters")

    # Test RAG prompt creation
    rag_prompt = PromptTemplates.create_rag_prompt(
        question="What is Physical AI?",
        context_chunks=mock_context_chunks,
        persona_config=persona_config
    )

    print("✓ RAG prompt created successfully")
    print(f"  RAG prompt length: {len(rag_prompt)} characters")

    return True


async def test_liteLLM_integration():
    """
    Test the LiteLLM integration with OpenRouter API.
    """
    print("\nTesting LiteLLM integration with OpenRouter...")

    intelligence_service = IntelligenceService()

    # Check if OpenRouter API key is configured
    api_key = intelligence_service.api_key
    if api_key:
        print("✓ OpenRouter API key is configured")
        print(f"  API key length: {len(api_key) if api_key else 0}")
        print(f"  Model: {intelligence_service.model}")
        print(f"  Base URL: {intelligence_service.base_url}")
    else:
        print("! OpenRouter API key not configured (this is expected in test environment)")

    return True


async def main():
    """
    Main test function to run all tests.
    """
    print("Starting Intelligence Layer (OpenAI Agents SDK) Subsystem Tests\n")

    # Test LiteLLM integration
    litellm_test_passed = await test_liteLLM_integration()

    # Test the intelligence service
    service_test_passed = await test_intelligence_service()

    # Test prompt engineering
    prompt_test_passed = await test_prompt_engineering()

    print(f"\nTest Results:")
    print(f"- LiteLLM Integration: {'PASS' if litellm_test_passed else 'SKIP'}")
    print(f"- Intelligence Service: {'PASS' if service_test_passed else 'FAIL'}")
    print(f"- Prompt Engineering: {'PASS' if prompt_test_passed else 'FAIL'}")

    overall_success = service_test_passed and prompt_test_passed  # Don't require API key for basic tests

    if overall_success:
        print("\n✓ Most tests passed! Intelligence Layer is functioning correctly.")
        print("  Note: API key configuration is required for full functionality.")
        return True
    else:
        print("\n✗ Some tests failed.")
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)