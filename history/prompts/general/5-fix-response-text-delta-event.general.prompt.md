---
id: 5
title: Fix for ResponseTextDeltaEvent handling in streaming
stage: general
date: 2026-02-17
surface: "agent"
model: "opus"
feature: "none"
branch: "main"
user: "user"
command: "Fix streaming to handle ResponseTextDeltaEvent"
labels: ["bug-fix", "streaming", "agents-sdk", "event-handling"]
links:
  spec: "null"
  ticket: "null"
  adr: "null"
  pr: "null"
files:
  - "rag_chatbot/agents_sdk/services/intelligence_service.py"
tests:
  - "Manual testing of chat functionality to verify ResponseTextDeltaEvent content is captured"
---

# Fix for ResponseTextDeltaEvent handling in streaming

## Prompt

Fix the issue where chatbot messages are empty by adding specific handling for OpenAI ResponseTextDeltaEvent which contains the actual token content from the OpenAI Agents SDK.

## Response

Added specific handling for ResponseTextDeltaEvent which contains the actual token content:

1. Added specific check for ResponseTextDeltaEvent type name to extract delta content directly
2. Process ResponseTextDeltaEvent first before falling back to generic event type matching
3. Added comprehensive logging to see actual event type names and data structures
4. Maintained backward compatibility with other event types

The backend logs showed that ResponseTextDeltaEvent instances were being generated with content, but they were not being caught by the generic string matching approach. The new specific handling for this event type should resolve the empty messages issue.

## Outcome

The streaming should now properly capture and send token content from ResponseTextDeltaEvent, resulting in populated chat messages instead of empty ones.

## Evaluation

After deploying this fix, chat responses should appear in the UI instead of remaining empty.