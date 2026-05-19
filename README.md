# Power Platform Licensing Assistant

A RAG (Retrieval-Augmented Generation) chatbot that answers questions about Microsoft Power Platform licensing using Azure AI Search and Azure OpenAI.

## Architecture

```
User question
     │
     ▼
┌─────────────────────────────────────────────────┐
│  Gradio Web UI  (http://localhost:7860)          │
│  - Model selector (gpt-4o-mini / DeepSeek-V3.2) │
│  - Chat history displayed in browser             │
└───────────────────┬─────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────┐
│  Agentic Loop  (bot_respond in rag_chatbot.py)  │
│                                                  │
│  1. Build messages: [system prompt] + history   │
│  2. Call LLM with search tool available          │
│  3. If LLM calls search tool → run search        │
│     → append results → repeat from step 2       │
│  4. When LLM returns final answer → stream it    │
└────────┬────────────────────────┬────────────────┘
         │                        │
         ▼                        ▼
┌─────────────────┐   ┌───────────────────────────┐
│  Azure OpenAI   │   │  Azure AI Search           │
│  (AsyncOpenAI)  │   │  (azure-search-documents)  │
│                 │   │                            │
│  gpt-4o-mini    │   │  Index: Power Platform     │
│  DeepSeek-V3.2  │   │  Licensing Guide chunks    │
└─────────────────┘   └───────────────────────────┘
```

## Logic Flow

1. **User submits a question** — Gradio appends it to the chat history as `{"role": "user", "content": "..."}`.

2. **Agentic loop starts** — The full conversation history plus a system prompt ("You are a Power Platform licensing specialist…") is sent to the LLM with the `search_power_platform_licensing` tool available.

3. **LLM decides to search** — The model almost always calls the search tool first. The tool takes a `query` string, runs a keyword search against the Azure AI Search index, and returns the top-5 matching document chunks concatenated as plain text.

4. **Tool result fed back** — The search result is appended to the messages as a `tool` role message. The LLM is called again with the updated context.

5. **LLM generates the answer** — Once the model has enough retrieved context it produces its final answer, which is streamed token-by-token back to the Gradio UI.

6. **Conversation history preserved** — The full `history` list (stored in Gradio's client-side state) is passed on every turn, so the model can reference prior exchanges within the same session.

## Key Files

| File | Purpose |
|------|---------|
| `rag_chatbot.py` | Main application — search client, agentic loop, Gradio UI |
| `test_components.py` | Smoke tests for LLM and Azure AI Search connectivity |
| `.env` | Runtime secrets (not committed) — see `.env.example` |
| `.env.example` | Template showing required environment variables |

## Environment Variables

| Variable | Description |
|----------|-------------|
| `AZURE_OPENAI_BASE_URL` | Azure OpenAI endpoint (`https://<resource>.openai.azure.com/openai/v1`) |
| `AZURE_OPENAI_API_KEY` | Azure OpenAI API key |
| `AZURE_SEARCH_ENDPOINT` | Azure AI Search service URL |
| `AZURE_SEARCH_API_KEY` | Azure AI Search admin/query key |
| `AZURE_SEARCH_INDEX` | Name of the search index containing the licensing guide |
| `AZURE_TENANT_ID` | Azure tenant ID (used for CLI credential fallback) |

## Running

```bash
# Verify components are working
python test_components.py

# Launch the chatbot
python rag_chatbot.py
# → open http://localhost:7860
```

## Dependencies

- `openai` — async OpenAI/Azure OpenAI client
- `azure-search-documents` — Azure AI Search SDK
- `gradio` — web UI framework
- `python-dotenv` — loads `.env` at startup
