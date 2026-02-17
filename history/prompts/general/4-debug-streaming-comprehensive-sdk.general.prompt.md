---
id: 4
title: PHR for debugging streaming implementation to catch empty messages
stage: general
date: 2026-02-16
surface: "agent"
model: "opus"
feature: "none"
branch: "main"
user: "user"
command: "Debug streaming implementation"
labels: ["debug", "streaming", "agents-sdk", "token-handling"]
links:
  spec: "null"
  ticket: "null"
  adr: "null"
  pr: "null"
files:
  - "rag_chatbot/agents_sdk/services/intelligence_service.py"
tests:
  - "Manual testing to verify if tokens are now being caught from Agents SDK"
---

# PHR for debugging streaming implementation to catch empty messages

## Prompt

Debug the issue where chatbot messages are empty by adding comprehensive debugging to catch all possible token sources from the OpenAI Agents SDK. The console shows that tokens are not being sent during streaming, despite the stream completing successfully.

## Response

Added comprehensive debugging to catch all possible token sources from the OpenAI Agents SDK:

1. Added checks for direct text, delta, and content properties on event objects (not just in event.data)
2. Enhanced logging to show what event properties are available
3. Added token_found tracking to all token-yielding paths
4. Added fallback handling for different event data structures
5. Improved event string inspection to catch text/delta content in event representations

The root cause identified through debugging logs was that the Agents SDK events weren't matching our expected event type patterns, so no tokens were being extracted from the stream. The new comprehensive approach checks multiple possible locations for token content within Agent SDK events.

## Outcome

The debugging should now catch tokens regardless of how the OpenAI Agents SDK structures the event objects, resolving the empty messages issue.

## Evaluation

After deploying these changes, we should see tokens being received and displayed in chat messages instead of empty content.