---
name: prompt-engineering
description: Complete Prompt Engineering skill covering all techniques, patterns, and best practices for creating effective prompts. Implements all documented prompt engineering methodologies from official sources and research literature. Use when crafting prompts for any LLM to achieve optimal performance and reliability.
---

# Prompt Engineering Complete Skill

This skill implements comprehensive prompt engineering techniques and best practices for creating effective prompts that elicit optimal responses from Large Language Models.

## Documentation References

Based on official prompt engineering documentation and research:
- OpenAI Prompt Engineering Guide: https://platform.openai.com/docs/guides/prompt-engineering
- Anthropic Prompt Engineering Guide: https://docs.anthropic.com/claude/docs/prompt-engineering
- Google AI Prompt Engineering Guide: https://developers.google.com/machine-learning/resources/prompt-eng
- Microsoft Prompt Engineering Guide: https://learn.microsoft.com/en-us/azure/ai-services/openai/concepts/prompt-engineering
- Academic Research Papers on Prompt Engineering

## Overview

Prompt engineering is the practice of designing and structuring input prompts to effectively communicate with Large Language Models (LLMs) to produce desired outputs. This skill covers all major prompt engineering techniques, patterns, and best practices for achieving optimal performance across different LLM architectures and use cases.

## Core Principles

### 1. Clarity and Specificity
- Be explicit about what you want the model to do
- Use clear, unambiguous language
- Provide specific instructions rather than vague requests
- Define the expected output format explicitly

### 2. Context and Role Definition
- Provide relevant context for the task
- Define the role or persona the model should adopt
- Include background information when necessary
- Set the appropriate tone and style expectations

### 3. Structure and Formatting
- Use consistent formatting patterns
- Separate different parts of the prompt clearly
- Use delimiters, headings, or numbered lists for clarity
- Organize information logically

### 4. Constraints and Boundaries
- Set clear boundaries for the response
- Define limitations and constraints
- Specify what should be avoided
- Include safety and ethical considerations

## Major Prompt Engineering Techniques

### 1. Zero-Shot Prompting

Zero-shot prompting involves providing a task description without any examples.

```python
def create_zero_shot_prompt(task_description, input_text):
    """
    Create a zero-shot prompt that relies on the model's pre-trained knowledge.

    Args:
        task_description: Clear description of what the model should do
        input_text: The specific input to process

    Returns:
        Formatted prompt string
    """
    return f"""
    {task_description}

    Input: {input_text}

    Output:"""
```

**Example:**
```
Classify the following email as spam or not spam.

Email: "Congratulations! You've won $1000! Click here to claim your prize now!"

Classification:
```

### 2. Few-Shot Prompting

Few-shot prompting provides examples to guide the model's behavior.

```python
def create_few_shot_prompt(task_description, examples, input_text):
    """
    Create a few-shot prompt with examples to demonstrate the desired behavior.

    Args:
        task_description: Description of the task
        examples: List of (input, output) example pairs
        input_text: The input to process

    Returns:
        Formatted prompt with examples
    """
    example_strings = []
    for example_input, example_output in examples:
        example_strings.append(f"Input: {example_input}\nOutput: {example_output}")

    examples_text = "\n\n".join(example_strings)

    return f"""
    {task_description}

    Examples:
    {examples_text}

    Input: {input_text}

    Output:"""
```

**Example:**
```
Translate English to French. Provide only the translation.

English: "Hello, how are you?"
French: "Bonjour, comment allez-vous?"

English: "I love programming."
French: "J'adore la programmation."

English: "The weather is nice today."
French: "Il fait beau aujourd'hui."

English: "Can you help me with this?"
French:
```

### 3. Chain-of-Thought Prompting

Chain-of-thought prompting encourages the model to show its reasoning process.

```python
def create_chain_of_thought_prompt(question):
    """
    Create a chain-of-thought prompt that asks the model to think step by step.

    Args:
        question: The question to answer

    Returns:
        Prompt encouraging step-by-step reasoning
    """
    return f"""
    Question: {question}

    Let's think step by step to arrive at the correct answer.

    Step 1: [Identify the key elements of the question]
    Step 2: [Consider the relevant information or formulas]
    Step 3: [Apply the information to solve the problem]
    Step 4: [Verify the solution]

    Answer:"""
```

**Example:**
```
If Sarah has 5 apples and gives 2 to her friend, then buys 3 more apples, how many apples does she have?

Let's think step by step to arrive at the correct answer.

Step 1: Sarah starts with 5 apples
Step 2: She gives away 2 apples, so 5 - 2 = 3 apples remain
Step 3: She buys 3 more apples, so 3 + 3 = 6 apples total
Step 4: Verification: 5 - 2 + 3 = 6 ✓

Answer: Sarah has 6 apples.
```

### 4. Instruction Prompting

Instruction prompting provides clear, direct instructions for the task.

```python
def create_instruction_prompt(instruction, context, input_data):
    """
    Create an instruction-based prompt with clear directives.

    Args:
        instruction: The main instruction for the task
        context: Background context for the task
        input_data: The specific data to process

    Returns:
        Structured instruction prompt
    """
    return f"""
    Task: {instruction}

    Context: {context}

    Data: {input_data}

    Instructions:
    1. Follow the task requirements exactly as specified
    2. Use the provided context to inform your response
    3. Process the data according to the instructions
    4. Provide a clear and concise response

    Response:"""
```

### 5. Role-Based Prompting

Role-based prompting assigns a specific role or persona to the model.

```python
def create_role_based_prompt(role, expertise, task, input_data):
    """
    Create a role-based prompt that assigns a specific persona to the model.

    Args:
        role: The role the model should adopt
        expertise: The expertise level to apply
        task: The specific task to perform
        input_data: The data to process

    Returns:
        Role-based prompt with persona assignment
    """
    return f"""
    You are a {role} with {expertise} expertise.

    As a {role}, you should:
    - Apply your professional knowledge and experience
    - Maintain the appropriate tone and perspective
    - Provide insights based on your domain expertise

    Task: {task}

    Input: {input_data}

    Please respond from the perspective of a {role}:"""
```

**Example:**
```
You are a senior software engineer with 10 years of experience in Python development.

As a senior software engineer, you should:
- Apply your professional knowledge and experience
- Maintain the appropriate tone and perspective
- Provide insights based on your domain expertise

Task: Review the following code snippet and suggest improvements.

Input: def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        for j in range(0, n-i-1):
            if arr[j] > arr[j+1]:
                arr[j], arr[j+1] = arr[j+1], arr[j]
    return arr

Please respond from the perspective of a senior software engineer:
```

## Advanced Prompt Engineering Techniques

### 1. Self-Consistency

Prompt the model to generate multiple responses and select the most consistent one.

```python
def create_self_consistency_prompt(task, input_data, num_samples=3):
    """
    Create a prompt for self-consistency technique.

    Args:
        task: The task to perform
        input_data: The input data
        num_samples: Number of samples to generate

    Returns:
        Prompt for generating multiple responses
    """
    return f"""
    Task: {task}
    Input: {input_data}

    Please generate {num_samples} different responses to this task.
    Each response should be independent and follow the same reasoning process.

    Response 1:

    Response 2:

    Response 3:

    Most Consistent Answer:"""
```

### 2. Tree-of-Thoughts

Extends chain-of-thought by exploring multiple reasoning paths.

```python
def create_tree_of_thoughts_prompt(problem):
    """
    Create a tree-of-thoughts prompt that explores multiple reasoning paths.

    Args:
        problem: The problem to solve

    Returns:
        Prompt for multi-path reasoning
    """
    return f"""
    Problem: {problem}

    Let's explore multiple reasoning paths to solve this problem:

    Path 1: [First approach to solving the problem]
    - Step A: [Substep 1]
    - Step B: [Substep 2]
    - Step C: [Substep 3]

    Path 2: [Alternative approach to solving the problem]
    - Step A: [Substep 1]
    - Step B: [Substep 2]
    - Step C: [Substep 3]

    Path 3: [Another alternative approach]
    - Step A: [Substep 1]
    - Step B: [Substep 2]
    - Step C: [Substep 3]

    Evaluating all paths, the best solution is:

    Final Answer:"""
```

### 3. Program-Aided Language Models (PAL)

Prompt the model to generate executable code to solve the problem.

```python
def create_pal_prompt(problem):
    """
    Create a Program-Aided Language Model prompt.

    Args:
        problem: The problem that can be solved with code

    Returns:
        Prompt for code generation approach
    """
    return f"""
    Problem: {problem}

    Let's solve this step by step using Python code:

    # Step 1: Import necessary libraries
    import math

    # Step 2: Define the problem parameters
    # [Define variables based on the problem]

    # Step 3: Implement the solution
    def solve_problem():
        # [Implementation of the solution]
        pass

    # Step 4: Execute the solution
    result = solve_problem()

    # Step 5: Return the result
    print(f"Result: {{result}}")

    Solution Code:"""
```

### 4. Retrieval-Augmented Generation (RAG) Prompting

Combine external knowledge with model generation.

```python
def create_rag_prompt(context, question, retrieved_documents):
    """
    Create a RAG-style prompt that incorporates retrieved context.

    Args:
        context: Overall context for the task
        question: The question to answer
        retrieved_documents: Retrieved documents or context

    Returns:
        RAG-style prompt with context integration
    """
    documents_text = "\n\n".join(retrieved_documents)

    return f"""
    Context: {context}

    Retrieved Documents:
    {documents_text}

    Question: {question}

    Using the information from the retrieved documents, please answer the question.
    If the documents don't contain sufficient information, please state that clearly.

    Answer:"""
```

## Prompt Structure Patterns

### 1. CLEAR Framework
- **C**ontext: Provide background information
- **L**ength: Specify response length requirements
- **E**xample: Give examples of desired output
- **A**udience: Define the target audience
- **R**equirements: Specify constraints and requirements

### 2. CRISPE Framework
- **C**apability and Role: Define the model's role
- **I**nput: Specify the input data
- **S**tyle: Define the writing style
- **P**erspective: Define the viewpoint to take
- **E**xpectation: Define the output format and requirements

### 3. BROADS Framework
- **B**ackground: Provide context
- **R**ole: Define the model's role
- **O**bjective: State the goal
- **A**udience: Define the target audience
- **D**etails: Provide specific details
- **S**teps: Outline the required steps

## Best Practices and Guidelines

### 1. Formatting Techniques
- Use triple quotes for multi-line prompts
- Use clear delimiters like [INPUT], [OUTPUT], [CONTEXT]
- Use XML-style tags: <task>, <input>, <output>
- Use markdown-style headers and lists
- Use consistent indentation

### 2. Constraint Techniques
- Use "Do not" statements for explicit prohibitions
- Use "Only" statements for strict limitations
- Use "Except" clauses for exceptions to rules
- Use "Always/Never" for absolute requirements

### 3. Quality Enhancement Techniques
- Provide positive and negative examples
- Use temperature and sampling parameters appropriately
- Include validation criteria in the prompt
- Use iterative refinement approaches

### 4. Safety and Alignment Techniques
- Include safety instructions upfront
- Use moral and ethical framing
- Include harm reduction instructions
- Use value alignment statements

## Error Prevention and Troubleshooting

### Common Issues and Solutions

1. **Vague Instructions**: Use specific, concrete language
2. **Missing Context**: Provide adequate background information
3. **Ambiguous Requirements**: Define output format explicitly
4. **Overly Complex Prompts**: Break down into simpler components
5. **Insufficient Examples**: Add more diverse examples
6. **Wrong Tone/Style**: Specify role and audience clearly

### Validation Techniques

```python
def validate_prompt_effectiveness(prompt, test_cases):
    """
    Validate prompt effectiveness with test cases.

    Args:
        prompt: The prompt to validate
        test_cases: List of (input, expected_output) tuples

    Returns:
        Dictionary with validation results
    """
    results = {
        'success_rate': 0,
        'average_quality': 0,
        'common_issues': [],
        'suggestions': []
    }

    # Implementation would involve testing the prompt against test cases
    # and analyzing the results for effectiveness

    return results
```

## Specialized Applications

### 1. Creative Writing Prompts
- Character development instructions
- Setting and atmosphere creation
- Plot structure guidance
- Genre-specific requirements

### 2. Technical Documentation Prompts
- API specification generation
- Code documentation creation
- System architecture explanation
- Technical tutorial creation

### 3. Educational Prompts
- Learning objective alignment
- Difficulty level specification
- Assessment criteria inclusion
- Interactive element integration

### 4. Business Application Prompts
- Stakeholder requirement integration
- ROI and KPI consideration
- Compliance and regulatory requirements
- Market and competitive analysis

## Implementation Guidelines

### 1. Iterative Development
- Start with simple prompts and gradually add complexity
- Test with small datasets before scaling
- A/B test different prompt variations
- Collect and analyze response quality metrics

### 2. Version Control
- Maintain prompt version history
- Document changes and their impact
- Track performance metrics over time
- Create prompt libraries and templates

### 3. Performance Optimization
- Minimize token usage while maintaining effectiveness
- Balance detail with brevity
- Optimize for the specific model being used
- Consider cost-effectiveness of different approaches

This comprehensive skill provides all the necessary techniques and patterns for effective prompt engineering across various applications and use cases.