# Copyright (c) Microsoft. All rights reserved.

from dotenv import load_dotenv
load_dotenv()

import json
import os

import gradio as gr
from azure.core.credentials import AzureKeyCredential
from azure.search.documents.aio import SearchClient
from ddgs import DDGS
from openai import AsyncOpenAI

# Ensure Azure CLI is on PATH
os.environ["PATH"] += r";C:\Program Files (x86)\Microsoft SDKs\Azure\CLI2\wbin"

# ── Config (loaded from .env) ─────────────────────────────────────────────────
BASE_URL        = os.environ["AZURE_OPENAI_BASE_URL"]
API_KEY         = os.environ["AZURE_OPENAI_API_KEY"]
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_API_KEY  = os.environ["AZURE_SEARCH_API_KEY"]
SEARCH_INDEX    = os.environ["AZURE_SEARCH_INDEX"]


# ── Search tool ───────────────────────────────────────────────────────────────
_search_client = SearchClient(
    endpoint=SEARCH_ENDPOINT,
    index_name=SEARCH_INDEX,
    credential=AzureKeyCredential(SEARCH_API_KEY),
)


async def _search(query: str, top: int = 5) -> str:
    print(f"[TOOL] Azure AI Search | query: \"{query}\"")
    results = await _search_client.search(query, top=top)
    chunks = [doc["chunk"] async for doc in results if doc.get("chunk")]
    output = "\n\n".join(chunks) if chunks else "No results found."
    preview = output[:120].replace("\n", " ")
    print(f"         ->{len(chunks)} chunk(s) returned | preview: {preview}...")
    return output


def _web_search(query: str, max_results: int = 5) -> str:
    print(f"[TOOL] DuckDuckGo Web Search | query: \"{query}\"")
    with DDGS() as ddgs:
        results = list(ddgs.text(query, max_results=max_results))
    if not results:
        print("         ->0 results returned")
        return "No web results found."
    parts = []
    for r in results:
        parts.append(f"**{r['title']}**\n{r['href']}\n{r['body']}")
    output = "\n\n".join(parts)
    preview = results[0]['title'] + " — " + results[0]['body'][:80].replace("\n", " ")
    print(f"         ->{len(results)} result(s) returned | top: {preview}...")
    return output


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_power_platform_licensing",
            "description": (
                "Search the Power Platform Licensing Guide for licensing information, "
                "pricing, plans, entitlements, and policies."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "The search query"},
                    "top":   {"type": "integer", "description": "Number of results to return", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the internet via DuckDuckGo for current information, news, "
                "general knowledge, or anything not covered by the Power Platform Licensing Guide."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query":       {"type": "string",  "description": "The search query"},
                    "max_results": {"type": "integer", "description": "Number of results to return", "default": 5},
                },
                "required": ["query"],
            },
        },
    },
]


_SYSTEM_PROMPT = (
    "You are a helpful assistant with two search tools:\n"
    "1. search_power_platform_licensing — searches the Power Platform Licensing Guide (use for licensing questions)\n"
    "2. web_search — searches the internet via DuckDuckGo (use for general or current information)\n"
    "Always use the most appropriate tool before answering. Cite sources where helpful."
)

_client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)


# ── Gradio handlers ───────────────────────────────────────────────────────────
async def user_submit(message: str, history: list):
    history = history + [{"role": "user", "content": message}]
    return "", history


async def bot_respond(history: list, model: str):
    """Agentic loop: handle tool calls then stream the final answer."""
    from openai import BadRequestError

    messages = [{"role": "system", "content": _SYSTEM_PROMPT}] + history
    history = history + [{"role": "assistant", "content": ""}]

    try:
        # Tool call loop (non-streaming until no more tool calls)
        while True:
            resp = await _client.chat.completions.create(
                model=model,
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
            )
            choice = resp.choices[0]

            if choice.finish_reason == "tool_calls" and choice.message.tool_calls:
                messages.append({
                    "role": "assistant",
                    "content": choice.message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in choice.message.tool_calls
                    ],
                })
                for tc in choice.message.tool_calls:
                    args = json.loads(tc.function.arguments)
                    if tc.function.name == "search_power_platform_licensing":
                        result = await _search(**args)
                    else:
                        result = _web_search(**args)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
            else:
                break

        # Stream the final answer
        stream = await _client.chat.completions.create(
            model=model,
            messages=messages,
            stream=True,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                history[-1]["content"] += chunk.choices[0].delta.content
                yield history

    except BadRequestError as e:
        err = e.body or {}
        inner = err.get("innererror", {})
        # Identify which content filter triggered
        if inner.get("code") == "ResponsibleAIPolicyViolation":
            filters = inner.get("content_filter_result", {})
            triggered = [k for k, v in filters.items() if v.get("filtered") or v.get("detected")]
            reason = ", ".join(triggered) if triggered else "policy violation"
            msg = (
                f"**Request blocked by Azure OpenAI content filter** ({reason}).\n\n"
                "Your message was flagged and could not be processed. "
                "Please rephrase your question and try again."
            )
        else:
            msg = f"**Request error ({e.status_code}):** {err.get('message', str(e))}"
        print(f"[ERROR] Azure OpenAI content filter blocked request: {inner.get('code', '')} | triggered: {triggered if 'triggered' in dir() else 'unknown'}")
        history[-1]["content"] = msg
        yield history
    except Exception as e:
        print(f"[ERROR] Unexpected error in bot_respond: {e}")
        history[-1]["content"] = f"**Unexpected error:** {e}"
        yield history


def clear_chat():
    return []


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(title="Power Platform Licensing Assistant") as demo:

    gr.Markdown(
        """
        # Power Platform Licensing Assistant
        Ask questions about Microsoft Power Platform licensing, plans, pricing, and entitlements.
        Powered by **Azure AI Search RAG** (licensing guide) + **DuckDuckGo** (internet search).
        """
    )

    with gr.Row():
        model_selector = gr.Dropdown(
            choices=["gpt-4o-mini", "DeepSeek-V3.2"],
            value="gpt-4o-mini",
            label="Model",
            scale=1,
        )
        gr.Markdown("")

    chatbot = gr.Chatbot(label="Conversation", height=520)

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="e.g. What licences are needed to use Power Automate premium connectors?",
            label="Your question",
            scale=5,
            autofocus=True,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    clear_btn = gr.Button("Clear conversation", variant="secondary")

    msg_box.submit(
        user_submit,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot],
    ).then(
        bot_respond,
        inputs=[chatbot, model_selector],
        outputs=[chatbot],
    )

    send_btn.click(
        user_submit,
        inputs=[msg_box, chatbot],
        outputs=[msg_box, chatbot],
    ).then(
        bot_respond,
        inputs=[chatbot, model_selector],
        outputs=[chatbot],
    )

    clear_btn.click(clear_chat, outputs=[chatbot])

    gr.Markdown(
        "_Switching models mid-conversation starts a fresh context for that model._"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Soft())
