"""
Prompt templates for the Intelligence Layer (OpenAI Agents SDK) Subsystem.

This module implements modular prompt structures with layered approach:
1. System Prompt - defines role, scope, refusal rules
2. Context Prompt - injects retrieved chunks with metadata
3. User Prompt - raw user question
4. Output Instruction Prompt - enforces formatting rules
"""

from typing import Dict, List, Any
from backend.schemas.retrieval import Source


class PromptTemplates:
    """Class containing all prompt templates for the agent system."""

    @staticmethod
    def create_system_prompt(persona_config: Dict[str, Any]) -> str:
        """
        Create system prompt with persona and role definition.

        Args:
            persona_config: Configuration dictionary with persona details

        Returns:
            Formatted system prompt string
        """
        return f"""SYSTEM INSTRUCTIONS:
You are an {persona_config['role']}. Your persona characteristics are:
- Tone: {persona_config['tone']}
- Style: {persona_config['style']}
- Constraints: {', '.join(persona_config['constraints'])}

You must:
- Answer only from retrieved context provided below
- Never speculate or fabricate information
- Clearly state when context is insufficient to answer
- Maintain technical precision and educational value
- Use bullet points and structured format when helpful
- Always cite sources from the retrieved context
"""

    @staticmethod
    def create_context_prompt(context_chunks: List[Source]) -> str:
        """
        Create context prompt with retrieved chunks and metadata.

        Args:
            context_chunks: List of Source objects containing retrieved context

        Returns:
            Formatted context prompt string
        """
        context_parts = ["SOURCE MATERIAL (RETRIEVED CONTEXT):"]

        for i, chunk in enumerate(context_chunks):
            chunk_info = f"""
CHUNK {i+1}:
Document ID: {chunk.document_id}
Chunk ID: {chunk.chunk_id}
Relevance Score: {chunk.score:.3f}
Metadata: {str(chunk.metadata)}
Content: {chunk.text}

"""
            context_parts.append(chunk_info)

        return "\n".join(context_parts)

    @staticmethod
    def create_user_prompt(user_query: str) -> str:
        """
        Create user prompt with the query.

        Args:
            user_query: The user's question

        Returns:
            Formatted user prompt string
        """
        return f"USER QUESTION:\n{user_query}"

    @staticmethod
    def create_output_instruction_prompt() -> str:
        """
        Create output instruction prompt for formatting.

        Returns:
            Formatted output instruction prompt string
        """
        return """OUTPUT INSTRUCTIONS:
- Respond based only on the SOURCE MATERIAL provided above
- Do not generate content outside the retrieved context
- If the context doesn't contain sufficient information, clearly state this
- Format your response in a clear, educational manner
- Use bullet points, numbered lists, or structured format when appropriate
- Include relevant citations to the source chunks if possible
- Maintain the specified tone and style
"""

    @staticmethod
    def create_layered_prompt(
        user_query: str,
        context_chunks: List[Source],
        persona_config: Dict[str, Any]
    ) -> str:
        """
        Create a complete layered prompt using the modular approach.

        Args:
            user_query: The user's question
            context_chunks: List of Source objects with retrieved context
            persona_config: Configuration dictionary with persona details

        Returns:
            Complete layered prompt string
        """
        system_prompt = PromptTemplates.create_system_prompt(persona_config)
        context_prompt = PromptTemplates.create_context_prompt(context_chunks)
        user_prompt = PromptTemplates.create_user_prompt(user_query)
        output_instruction = PromptTemplates.create_output_instruction_prompt()

        return f"{system_prompt}\n\n{context_prompt}\n\n{user_prompt}\n\n{output_instruction}"

    @staticmethod
    def create_rag_prompt(
        question: str,
        context_chunks: List[Source],
        persona_config: Dict[str, Any]
    ) -> str:
        """
        Create a RAG-style prompt that incorporates retrieved context.

        Args:
            question: The question to answer
            context_chunks: Retrieved context chunks
            persona_config: Configuration dictionary with persona details

        Returns:
            RAG-style prompt string
        """
        # Build context text from chunks
        context_text = "\n\n".join([chunk.text for chunk in context_chunks])

        return f"""CONTEXT:
{context_text}

QUESTION:
{question}

INSTRUCTIONS:
As a {persona_config['role']}, please answer the question based only on the provided context.
Do not use any external knowledge or information from your training data.
If the context doesn't contain sufficient information to answer the question, clearly state this.
Format your response in a clear, educational manner with appropriate citations to the context.
"""

    @staticmethod
    def create_chain_of_thought_prompt(question: str) -> str:
        """
        Create a chain-of-thought prompt that asks the model to think step by step.

        Args:
            question: The question to answer

        Returns:
            Chain-of-thought prompt string
        """
        return f"""QUESTION: {question}

Let's think step by step to arrive at the correct answer.

Step 1: [Analyze the key elements of the question]
Step 2: [Consider the relevant information from the provided context]
Step 3: [Apply the information to answer the question]
Step 4: [Verify the response is grounded in the provided context]

ANSWER:"""