# Prompt Engineering Techniques Reference Guide

## Advanced Prompt Engineering Methods

### 1. Instruction Tuning Patterns

#### Direct Instruction
```
[INSTRUCTION]: [TASK DESCRIPTION]
[INPUT]: [SPECIFIC INPUT DATA]
[OUTPUT FORMAT]: [SPECIFIED FORMAT REQUIREMENTS]
[CONSTRAINTS]: [LIMITATIONS AND BOUNDARIES]
```

#### Template-Based Instructions
```
You are [ROLE] with [EXPERTISE LEVEL].
Your task is to [SPECIFIC TASK].
Context: [RELEVANT BACKGROUND INFORMATION]
Input: [DATA TO PROCESS]
Requirements: [SPECIFIC REQUIREMENTS]
Format: [OUTPUT STRUCTURE SPECIFICATION]
Constraints: [LIMITATIONS AND PROHIBITIONS]
Response:
```

### 2. Context Injection Techniques

#### Zero-Shot Contextualization
```
Given your knowledge of [DOMAIN], please [TASK] for [AUDIENCE].
```

#### In-Context Learning
```
Example 1:
Input: [INPUT_DATA_1]
Output: [EXPECTED_OUTPUT_1]

Example 2:
Input: [INPUT_DATA_2]
Output: [EXPECTED_OUTPUT_2]

Example 3:
Input: [INPUT_DATA_3]
Output: [EXPECTED_OUTPUT_3]

New Input: [ACTUAL_INPUT_DATA]
New Output:
```

### 3. Reasoning Path Techniques

#### Step-by-Step Reasoning
```
Problem: [STATE THE PROBLEM]
Let's think through this step by step:

Step 1: [FIRST STEP]
Step 2: [SECOND STEP]
Step 3: [THIRD STEP]
...
Final Answer: [CONCLUSION]
```

#### Chain-of-Thought with Verification
```
Question: [THE QUESTION]
Thought: [REASONING PROCESS]
Verification: [CHECKING THE REASONING]
Answer: [FINAL RESPONSE]
```

### 4. Role-Playing Prompts

#### Professional Roles
```
As a [PROFESSIONAL ROLE] with [YEARS] of experience in [FIELD],
please [TASK] considering [SPECIFIC FACTORS].

Your expertise includes:
- [SKILL 1]
- [SKILL 2]
- [SKILL 3]

When responding, ensure you:
- [REQUIREMENT 1]
- [REQUIREMENT 2]
- [REQUIREMENT 3]
```

#### Historical Personas
```
Act as [HISTORICAL FIGURE] during [TIME PERIOD].
Your worldview is shaped by:
- [BELIEF 1]
- [BELIEF 2]
- [BELIEF 3]

Regarding [TOPIC], your perspective would be:
```

### 5. Multi-Modal Prompting

#### Text-to-Text Transformation
```
Transform the following [SOURCE_FORMAT] into [TARGET_FORMAT]:
Source: [ORIGINAL TEXT]
Target:
```

#### Comparative Analysis
```
Compare [ITEM_1] and [ITEM_2] across the dimensions of:
- [DIMENSION_1]
- [DIMENSION_2]
- [DIMENSION_3]

Provide analysis in the following format:
[ITEM_1] vs [ITEM_2]:
- [DIMENSION_1]: [COMPARISON]
- [DIMENSION_2]: [COMPARISON]
- [DIMENSION_3]: [COMPARISON]

Conclusion: [SUMMARY]
```

### 6. Constraint-Based Prompts

#### Hard Constraints
```
You MUST follow these rules:
1. [MANDATORY RULE 1]
2. [MANDATORY RULE 2]
3. [MANDATORY RULE 3]

You MUST NOT do:
1. [FORBIDDEN ACTION 1]
2. [FORBIDDEN ACTION 2]
3. [FORBIDDEN ACTION 3]

Response:
```

#### Soft Constraints
```
Please consider the following preferences:
- [PREFERRED STYLE 1]
- [PREFERRED STYLE 2]
- [PREFERRED STYLE 3]

While not mandatory, adherence to these preferences is appreciated.

Task: [THE TASK]
```

### 7. Meta-Prompting Techniques

#### Self-Evaluation
```
Generate a response to: [QUESTION]

Then evaluate your own response by answering:
1. Is it accurate? [YES/NO]
2. Is it complete? [YES/NO]
3. Is it clear? [YES/NO]
4. What are its strengths? [ANSWER]
5. What are its weaknesses? [ANSWER]

If you identified any issues, please revise your response:
```

#### Prompt Refinement
```
Original Prompt: [ORIGINAL_PROMPT]
Issue: [WHAT_WENT_WRONG]

Please refine the original prompt to address the issue.
Refined Prompt:
```

### 8. Adversarial Prompting

#### Critique and Improve
```
[INITIAL_RESPONSE]

Critique this response by identifying:
1. Factual inaccuracies
2. Logical flaws
3. Missing information
4. Poor reasoning

Then provide an improved response that addresses these issues:
```

#### Devil's Advocate
```
Present a strong argument for [POSITION_A].
Now present a strong counterargument for [POSITION_B] as a devil's advocate.
Finally, synthesize both perspectives to reach a balanced conclusion:
```

### 9. Iterative Prompting

#### Progressive Disclosure
```
Round 1: Provide a brief overview of [TOPIC].
Round 2: Expand on the key components identified in Round 1.
Round 3: Provide detailed analysis of the most important component from Round 2.
Round 4: Synthesize all information and provide actionable recommendations.
```

#### Feedback Loop
```
Initial Response: [RESPONSE]

Feedback: [FEEDBACK_ON_RESPONSE]

Please revise your response considering the feedback:
```

### 10. Domain-Specific Templates

#### Technical Writing
```
Audience: [TECHNICAL_LEVEL]
Topic: [SUBJECT_MATTER]
Purpose: [EDUCATE/EXPLAIN/SOLVE]
Format: [DOCUMENT_TYPE]
Key Points: [IMPORTANT_ASPECTS]
Tone: [FORMALITY_LEVEL]
Response:
```

#### Creative Writing
```
Genre: [LITERARY_GENRE]
Setting: [LOCATION_TIME_PERIOD]
Characters: [CHARACTER_DESCRIPTIONS]
Conflict: [MAIN_CONFLICT]
Tone: [MOOD_ATMOSPHERE]
Word Count: [LENGTH_REQUIREMENT]
Opening Line: [STARTING_POINT]
Continue the story:
```

#### Analytical Writing
```
Subject: [ANALYSIS_TOPIC]
Perspective: [VIEWPOINT_TO_TAKE]
Evidence Required: [TYPE_OF_EVIDENCE]
Structure: [ORGANIZATIONAL_PATTERN]
Audience: [TARGET_READERSHIP]
Objective: [PURPOSE_OF_ANALYSIS]
Analysis:
```

## Quality Assurance Patterns

### 1. Validation Checks
```
Before providing your final answer, verify:
1. Accuracy: [FACT_CHECK]
2. Completeness: [COMPREHENSIVENESS_CHECK]
3. Relevance: [TASK_ALIGNEMENT_CHECK]
4. Clarity: [UNDERSTANDING_CHECK]

Only then provide your final response:
```

### 2. Confidence Indicators
```
Provide your answer to [QUESTION].

Then rate your confidence on a scale of 1-10, where:
1 = Completely guessing
5 = Some uncertainty
10 = Absolutely certain

Confidence Level: [NUMBER]
Justification: [REASON_FOR_CONFIDENCE_LEVEL]
```

### 3. Alternative Solutions
```
Provide the best solution to [PROBLEM].

Then provide 2-3 alternative solutions with their pros and cons:

Alternative 1:
Pros: [ADVANTAGES]
Cons: [DISADVANTAGES]

Alternative 2:
Pros: [ADVANTAGES]
Cons: [DISADVANTAGES]

Recommendation: [WHICH_IS_BEST_AND_WHY]
```

These patterns provide a comprehensive toolkit for creating effective prompts across various applications and contexts.