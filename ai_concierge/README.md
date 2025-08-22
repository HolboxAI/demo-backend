# AI Concierge Module

## Overview
This module provides an AI-powered concierge tool that answers user questions about businesses using Strands Agent SDK with AWS Bedrock's LLM (Anthropic Claude 3 Sonnet). It uses AgentCore short-term memory via MemoryHookProvider to maintain conversation history across sessions, restricting responses to business-related topics only.

## Endpoints Added
- **/ai_concierge/ask** (POST): Accepts a JSON body with "session_id" and "question" fields and returns the AI's response. Use a unique session_id (e.g., UUID) for each conversation to maintain history.

## Setup/Installation Steps
1. Ensure AWS credentials are set up with access to Bedrock Runtime and AgentCore Memory in a supported region (e.g., us-west-2).
2. The memory resource is automatically created or retrieved on module import.
3. Install Strands SDK if not already: pip install strands (assuming it's available).
4. Test the endpoint with a tool like Postman or curl.

## Example Request/Response
**Request (POST /ai_concierge/ask):**
```json
{
    "session_id": "test",
    "question": "What are the opening hours of a typical Starbucks location?"
}