#!/usr/bin/env python3
"""
Minimal test for the Intelligence Service to check basic functionality
"""

import asyncio
import sys
import os

# Add the project root to the Python path to allow absolute imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

async def test_basic_imports():
    """Test basic imports work"""
    print("Testing basic imports...")
    try:
        from agents import Agent, Runner, SQLiteSession
        from agents import function_tool, input_guardrail, output_guardrail, GuardrailFunctionOutput
        from agents.extensions.models.litellm_model import LitellmModel
        from agents.model_settings import ModelSettings

        print("✓ All OpenAI Agents SDK imports successful")

        # Test that we can create a simple agent
        simple_agent = Agent(
            name="Test Agent",
            instructions="You are a test agent."
        )
        print("✓ Simple agent creation successful")

        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def test_intelligence_service_creation():
    """Test creating the IntelligenceService instance"""
    print("\nTesting IntelligenceService creation...")
    try:
        from agents_sdk.services.intelligence_service import IntelligenceService

        # Create instance without initializing to avoid API calls
        service = IntelligenceService()
        print("✓ IntelligenceService instance created")

        # Check if the service has the expected attributes
        assert hasattr(service, 'settings'), "Service should have settings"
        assert hasattr(service, 'logger'), "Service should have logger"
        assert hasattr(service, '_initialize_main_agent'), "Service should have _initialize_main_agent method"

        print("✓ IntelligenceService has expected attributes")

        return True
    except Exception as e:
        print(f"✗ IntelligenceService creation error: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test function"""
    print("Running minimal Intelligence Service tests...\n")

    imports_ok = await test_basic_imports()
    service_ok = await test_intelligence_service_creation()

    print(f"\nTest Results:")
    print(f"- Basic Imports: {'PASS' if imports_ok else 'FAIL'}")
    print(f"- Service Creation: {'PASS' if service_ok else 'FAIL'}")

    overall_success = imports_ok and service_ok

    if overall_success:
        print("\n✓ All basic tests passed!")
    else:
        print("\n✗ Some tests failed.")

    return overall_success

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)