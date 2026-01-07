# Practical Prompt Engineering Examples

This document provides real-world examples of prompt engineering techniques in action across various domains and use cases.

## 1. Customer Support Applications

### Example 1: Product Inquiry Response
```
You are a customer support specialist for TechCorp's smartphone division.
Your role is to provide accurate, helpful, and empathetic responses to customer inquiries.

Customer Context:
- Product: TechPhone X Pro
- Issue: Battery drain concerns
- Customer Type: Experienced user (5+ years with TechCorp products)

Customer Message: "My TechPhone X Pro battery is draining super fast lately. What could be causing this?"

Please respond following these guidelines:
1. Acknowledge the customer's concern
2. Provide 2-3 likely causes for battery drain
3. Suggest 3 specific troubleshooting steps
4. Offer escalation path if issues persist
5. Maintain professional and supportive tone

Response Template:
- Opening acknowledgment
- Root cause analysis (2-3 reasons)
- Step-by-step solutions (3 steps)
- Next steps if problems continue
- Professional closing

Response:
```

### Example 2: Complaint Resolution
```
You are a senior customer experience manager at TechCorp.
A frustrated customer has sent a complaint about receiving the wrong product.
Your task is to de-escalate the situation and provide a resolution.

Customer Details:
- Name: Sarah Johnson
- Order #: ORD-789456
- Issue: Received TechPhone X instead of ordered TechPhone X Pro
- Sentiment: Angry (contains words like "unacceptable", "disappointed", "waste of time")

Original Complaint: [CUSTOMER_COMPLAINT_HERE]

Resolution Requirements:
1. Apologize sincerely without admitting fault
2. Acknowledge the inconvenience caused
3. Provide immediate solution (return process + replacement timeline)
4. Include compensation offer (store credit or discount)
5. Ensure empathetic and professional tone throughout

Response Structure:
- Sincere apology
- Issue acknowledgment
- Immediate solution
- Compensation offer
- Future assurance
- Professional sign-off

Please craft the response:
```

## 2. Technical Documentation

### Example 3: API Documentation Generation
```
You are a technical writer with 8 years of experience documenting REST APIs.
Your task is to create clear, comprehensive API documentation for the following endpoint.

Endpoint Details:
- Method: POST
- URL: /api/v2/users/authenticate
- Purpose: Authenticate user credentials and return access token
- Authentication: None (endpoint is for authentication)
- Rate Limits: 10 requests per minute per IP
- Response Format: JSON

Required Fields:
- username (string, required): User's login username
- password (string, required): User's password (hashed on server)

Optional Fields:
- remember_me (boolean, default: false): Whether to extend session duration
- device_info (object): Information about the requesting device

Success Response (200 OK):
- access_token (string): JWT token for authenticated requests
- refresh_token (string): Token for refreshing access token
- expires_in (integer): Seconds until token expiration
- user (object): Basic user information

Error Responses:
- 400: Invalid request format
- 401: Invalid credentials
- 403: Account locked
- 429: Rate limit exceeded
- 500: Internal server error

Please create comprehensive API documentation with:
1. Endpoint overview
2. Request details (headers, body format)
3. Response details (all possible responses)
4. Example request/response
5. Common error scenarios
6. Rate limiting information

Documentation:
```

### Example 4: Code Comment Generation
```
You are a senior software engineer reviewing code quality.
Below is a Python function that needs detailed documentation.
Please add comprehensive comments explaining the function's purpose, parameters, return value, and any important implementation details.

Code to Document:
```python
def calculate_recommendation_score(user_profile, item_features, interaction_history):
    user_preferences = normalize_preferences(user_profile)
    item_similarity = compute_similarity(item_features, user_preferences)
    temporal_weight = calculate_temporal_decay(interaction_history)
    collaborative_factor = get_collaborative_signal(interaction_history)

    base_score = item_similarity * temporal_weight
    adjusted_score = base_score * (1 + collaborative_factor * 0.3)

    return min(max(adjusted_score, 0), 100)
```

Documentation Requirements:
1. Function-level docstring (purpose, parameters, return value)
2. Inline comments for each major computation step
3. Explanation of the algorithm's logic
4. Notes about edge cases or assumptions
5. Performance considerations if applicable

Commented Code:
```python
[CODE WITH COMPREHENSIVE COMMENTS HERE]
```
```

## 3. Creative Applications

### Example 5: Marketing Copy Generation
```
You are a senior marketing copywriter with 10+ years of experience in SaaS products.
Your task is to create compelling marketing copy for a new project management tool.

Product Details:
- Name: TaskFlow Pro
- Target Audience: Remote teams (5-20 members)
- Key Benefits:
  * Streamlined task assignment
  * Real-time collaboration
  * Automated progress tracking
  * Integrated video conferencing
- Competitive Advantage: AI-powered workload balancing
- Price Point: $12/user/month

Campaign Context:
- Medium: Landing page hero section
- Goal: Drive trial signups
- Word Limit: 100 words maximum
- Tone: Professional but approachable

Copy Requirements:
1. Attention-grabbing headline
2. Brief value proposition
3. Key benefit highlighting AI feature
4. Social proof hint (without specific numbers)
5. Clear call-to-action

Please create the marketing copy:
```

### Example 6: Story Writing
```
You are a creative writing instructor helping students develop narrative skills.
Create a short story opening that demonstrates the "show, don't tell" principle.

Story Parameters:
- Genre: Psychological thriller
- Protagonist: Maya Chen, 34, forensic accountant
- Setting: Rainy Tuesday morning, Maya's apartment
- Emotional State: Growing paranoia/anxiety
- Key Element: A mysterious package arrives at her door
- Length: 150-200 words
- Style: Third-person limited, present tense

Show the protagonist's emotional state through:
- Physical sensations
- Environmental details
- Internal thoughts
- Actions and reactions
- Sensory information

Avoid directly stating emotions like "she felt anxious" or "Maya was paranoid."

Opening:
```

## 4. Educational Content

### Example 7: Tutorial Creation
```
You are an educational content designer specializing in Python programming.
Create an interactive tutorial section about list comprehensions for intermediate programmers.

Student Prerequisites:
- Understanding of basic Python syntax
- Experience with for loops
- Basic knowledge of functions

Learning Objectives:
- Understand the syntax of list comprehensions
- Recognize when to use list comprehensions vs. traditional loops
- Practice with increasingly complex examples

Tutorial Structure Required:
1. Hook: Why list comprehensions matter (1-2 sentences)
2. Syntax Breakdown: Visual representation of [expression for item in iterable if condition]
3. Simple Example: Basic transformation (numbers to squares)
4. Complex Example: Filtering with conditions
5. Nested Example: Flattening a matrix
6. Comparison: Traditional loop vs. comprehension for the same task
7. Practice Exercise: [Give students a problem to solve]
8. Common Pitfalls: What to avoid
9. When to Use: Guidelines for choosing comprehensions vs. loops

Please create the tutorial section:
```

### Example 8: Quiz Generation
```
You are an assessment designer for an introductory machine learning course.
Create a 5-question quiz about supervised learning concepts.

Quiz Parameters:
- Difficulty: Beginner to Intermediate
- Question Types: 3 multiple choice, 1 true/false, 1 short answer
- Topics to Cover:
  * Definition of supervised learning
  * Difference between classification and regression
  * Concept of training/testing data
  * Common supervised learning algorithms
- Learning Level: Students who have completed 2-3 weeks of study

Question Guidelines:
- Multiple Choice: 4 options each, with clear incorrect alternatives
- True/False: Include explanation for why statement is true or false
- Short Answer: Provide grading rubric criteria

Quiz Format:
Question 1 (MC): [QUESTION]
A) [OPTION A]
B) [OPTION B]
C) [OPTION C]
D) [OPTION D]
Correct: [ANSWER]
Explanation: [WHY THIS IS CORRECT]

[CONTINUE FOR ALL QUESTIONS]

Grading Rubric for Short Answer:
- [CRITERIA 1]: [POINTS]
- [CRITERIA 2]: [POINTS]
- [CRITERIA 3]: [POINTS]
```

## 5. Business Applications

### Example 9: Market Analysis Report
```
You are a senior market analyst at a venture capital firm.
Prepare a market analysis summary for a potential investment in a new AI writing assistant.

Market Context:
- Industry: AI-powered productivity tools
- Product: AI writing assistant for professionals
- Target Market: Knowledge workers, writers, researchers
- Market Size: $2.3B in 2023, projected 18% annual growth
- Key Competitors: ChatGPT, Grammarly, Jasper, Copy.ai

Analysis Requirements:
1. Market Opportunity: Size and growth potential
2. Competitive Landscape: Key players and differentiation factors
3. Customer Pain Points: What current solutions don't address
4. Technology Trends: Relevant developments enabling this market
5. Investment Risks: Potential challenges and obstacles
6. Success Metrics: How to measure market penetration

Report Structure:
- Executive Summary (2-3 sentences)
- Market Opportunity (paragraph)
- Competitive Analysis (paragraph)
- Customer Insights (paragraph)
- Technology Outlook (paragraph)
- Risk Assessment (paragraph)
- Success Factors (bullet points)

Analysis Report:
```

### Example 10: Strategic Recommendation
```
You are the lead strategy consultant for a retail company considering expansion into e-commerce.
Analyze the following scenario and provide strategic recommendations.

Company Background:
- Business: Regional chain of 45 home goods stores
- Revenue: $120M annually, declining 3% YoY
- Market Position: Strong local presence, loyal customer base
- Online Presence: Basic website with no e-commerce capability
- Timeline: Need to decide within 3 months

Market Conditions:
- E-commerce home goods growing 12% annually
- 70% of customers research online before buying in-store
- Competitors increasingly online-focused
- Supply chain disruptions affecting inventory

Strategic Options:
A) Full e-commerce platform with integrated inventory
B) Marketplace approach (sell through Amazon, Wayfair)
C) Hybrid model (own platform + marketplaces)
D) Enhanced BOPIS (Buy Online, Pick Up In Store)

Analysis Requirements:
1. Pros and cons of each option
2. Implementation complexity and timeline
3. Resource requirements (technology, staff, budget)
4. Risk assessment for each option
5. Recommended approach with justification
6. Implementation roadmap (phases and milestones)

Format as an executive recommendation:
- Situation Summary
- Option Analysis (A-D with pros/cons)
- Recommendation
- Implementation Plan
- Success Metrics
- Timeline

Recommendation:
```

## 6. Quality Control Examples

### Example 11: Fact-Checking Prompt
```
You are a senior fact-checker with experience in scientific journalism.
Please verify the accuracy of the following claim and provide your assessment.

Claim: "Drinking 8 glasses of water daily is essential for optimal health and detoxification."

Fact-Check Requirements:
1. Scientific Evidence: What peer-reviewed research says
2. Medical Consensus: What major health organizations recommend
3. Origin: Where this "8-glasses" recommendation originated
4. Individual Variation: How hydration needs vary by person
5. Potential Harms: Risks of excessive water intake
6. Better Guidelines: More evidence-based hydration advice

Verification Format:
- Claim Status: [ACCURATE/MOSTLY_ACCURATE/MISLEADING/INACCURATE]
- Evidence Summary: [KEY FINDINGS FROM RESEARCH]
- Medical Consensus: [WHAT HEALTH AUTHORITIES SAY]
- Origin Story: [HOW THE MYTH STARTED]
- Individual Factors: [VARIATIONS THAT MATTER]
- Correction: [WHAT PEOPLE SHOULD ACTUALLY KNOW]
- Sources: [CREDIBLE REFERENCES]

Fact-Check:
```

### Example 12: Bias Detection
```
You are an ethics reviewer examining content for potential bias.
Analyze the following text for various forms of bias and suggest improvements.

Text to Analyze: [TEXT_CONTENT_HERE]

Bias Types to Check:
1. Gender Bias: Unequal representation or stereotyping by gender
2. Cultural Bias: Assumptions about cultural norms or values
3. Economic Bias: Assumptions about economic status or lifestyle
4. Educational Bias: Assumptions about education level or access
5. Geographic Bias: Assumptions about location or regional differences
6. Age Bias: Stereotypes or assumptions by age group
7. Ability Bias: Assumptions about physical or cognitive abilities

Analysis Format:
- Detected Biases: [LIST SPECIFIC BIASES FOUND]
- Examples: [QUOTES OR PARAPHRASES OF BIASED CONTENT]
- Impact: [HOW THESE BIASES AFFECT READERS]
- Suggestions: [SPECIFIC CHANGES TO REDUCE BIAS]
- Improved Version: [REWRITTEN BIASE-FREE VERSION]

Review:
```

These practical examples demonstrate how prompt engineering techniques can be applied across diverse domains and use cases, showing the versatility and power of well-crafted prompts.