# Copyright (c) Microsoft. All rights reserved.

from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
from dataclasses import dataclass
from typing import Annotated

import gradio as gr
from agent_framework import AgentResponseUpdate
from agent_framework_openai import OpenAIChatClient
from azure.identity import AzureCliCredential
from semantic_kernel.connectors.azure_ai_search import AzureAISearchCollection
from semantic_kernel.data import VectorStoreRecordKeyField, VectorStoreRecordDataField
from semantic_kernel.functions import KernelParameterMetadata

# Ensure Azure CLI is on PATH
os.environ["PATH"] += r";C:\Program Files (x86)\Microsoft SDKs\Azure\CLI2\wbin"

# ── Config (loaded from .env) ─────────────────────────────────────────────────
BASE_URL        = os.environ["AZURE_OPENAI_BASE_URL"]
API_KEY         = os.environ["AZURE_OPENAI_API_KEY"]
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_API_KEY  = os.environ["AZURE_SEARCH_API_KEY"]
SEARCH_INDEX    = os.environ["AZURE_SEARCH_INDEX"]
TENANT_ID       = os.environ["AZURE_TENANT_ID"]


# ── Data model for the Power Platform Licensing Guide index ───────────────────
@dataclass
class PowerPlatformDoc:
    id:      Annotated[str,      VectorStoreRecordKeyField()]
    content: Annotated[str | None, VectorStoreRecordDataField()] = None


# ── Build search tool ─────────────────────────────────────────────────────────
_collection = AzureAISearchCollection[str, PowerPlatformDoc](
    record_type=PowerPlatformDoc,
    collection_name=SEARCH_INDEX,
    search_endpoint=SEARCH_ENDPOINT,
    api_key=SEARCH_API_KEY,
)

_search_function = _collection.create_search_function(
    function_name="search_power_platform_licensing",
    description=(
        "Search the Power Platform Licensing Guide for licensing information, "
        "pricing, plans, entitlements, and policies."
    ),
    search_type="keyword",
    parameters=[
        KernelParameterMetadata(
            name="query",
            description="The search query to find relevant licensing information.",
            type="str",
            is_required=True,
            type_object=str,
        ),
        KernelParameterMetadata(
            name="top",
            description="Number of results to return.",
            type="int",
            default_value=5,
            type_object=int,
        ),
    ],
    string_mapper=lambda x: x.record.content or "",
)

_search_tool = _search_function.as_agent_framework_tool()

_INSTRUCTIONS = (
    "You are a Microsoft Power Platform licensing specialist. "
    "Always use the search tool to retrieve relevant content from the Power Platform Licensing Guide "
    "before answering. Base your answers strictly on the retrieved content. "
    "Cite relevant sections where helpful."
)

# ── Build one agent per model ─────────────────────────────────────────────────
credential = AzureCliCredential(tenant_id=TENANT_ID)

AGENTS = {
    "DeepSeek-V3.2": OpenAIChatClient(
        model="DeepSeek-V3.2",
        api_key=API_KEY,
        base_url=BASE_URL,
    ).as_agent(
        name="rag_deepseek",
        instructions=_INSTRUCTIONS,
        tools=[_search_tool],
    ),
    "gpt-4o-mini": OpenAIChatClient(
        model="gpt-4o-mini",
        api_key=API_KEY,
        base_url=BASE_URL,
    ).as_agent(
        name="rag_gpt4o",
        instructions=_INSTRUCTIONS,
        tools=[_search_tool],
    ),
}


# ── Gradio handlers ───────────────────────────────────────────────────────────
async def user_submit(message: str, history: list, sessions: dict):
    """Immediately append the user message and clear the input box."""
    history = history + [[message, None]]
    return "", history, sessions


async def bot_respond(history: list, model: str, sessions: dict):
    """Stream the agent response token by token."""
    agent = AGENTS[model]
    user_message = history[-1][0]

    # Maintain one session per model so conversation history is preserved
    if model not in sessions:
        sessions[model] = agent.create_session()
    session = sessions[model]

    history[-1][1] = ""
    async for event in agent.run(user_message, session=session, stream=True):
        if isinstance(event, AgentResponseUpdate) and event.text:
            history[-1][1] += event.text
            yield history, sessions


def clear_chat():
    return [], {}


# ── Gradio UI ─────────────────────────────────────────────────────────────────
with gr.Blocks(
    title="Power Platform Licensing Assistant",
    theme=gr.themes.Soft(),
) as demo:

    gr.Markdown(
        """
        # Power Platform Licensing Assistant
        Ask questions about Microsoft Power Platform licensing, plans, pricing, and entitlements.
        Powered by the **Power Platform Licensing Guide** via Azure AI Search RAG.
        """
    )

    with gr.Row():
        model_selector = gr.Dropdown(
            choices=["gpt-4o-mini", "DeepSeek-V3.2"],
            value="gpt-4o-mini",
            label="Model",
            scale=1,
        )
        gr.Markdown("", scale=3)  # spacer

    chatbot = gr.Chatbot(
        label="Conversation",
        height=520,
        bubble_full_width=False,
        show_copy_button=True,
    )

    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="e.g. What licences are needed to use Power Automate premium connectors?",
            label="Your question",
            scale=5,
            autofocus=True,
        )
        send_btn = gr.Button("Send", variant="primary", scale=1)

    clear_btn = gr.Button("Clear conversation", variant="secondary")

    sessions = gr.State({})  # persists {model_name: AgentSession} per browser tab

    # Wire up submit (Enter key or Send button)
    submit_event = (
        msg_box.submit(
            user_submit,
            inputs=[msg_box, chatbot, sessions],
            outputs=[msg_box, chatbot, sessions],
        ).then(
            bot_respond,
            inputs=[chatbot, model_selector, sessions],
            outputs=[chatbot, sessions],
        )
    )

    send_btn.click(
        user_submit,
        inputs=[msg_box, chatbot, sessions],
        outputs=[msg_box, chatbot, sessions],
    ).then(
        bot_respond,
        inputs=[chatbot, model_selector, sessions],
        outputs=[chatbot, sessions],
    )

    clear_btn.click(clear_chat, outputs=[chatbot, sessions])

    gr.Markdown(
        "_Switching models mid-conversation starts a fresh session for that model. "
        "Switching back resumes the previous session._"
    )


if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)