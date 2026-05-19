from dotenv import load_dotenv
load_dotenv()

import os
import sys

os.environ["PATH"] += r";C:\Program Files (x86)\Microsoft SDKs\Azure\CLI2\wbin"

BASE_URL        = os.environ["AZURE_OPENAI_BASE_URL"]
API_KEY         = os.environ["AZURE_OPENAI_API_KEY"]
SEARCH_ENDPOINT = os.environ["AZURE_SEARCH_ENDPOINT"]
SEARCH_API_KEY  = os.environ["AZURE_SEARCH_API_KEY"]
SEARCH_INDEX    = os.environ["AZURE_SEARCH_INDEX"]

PASS = "[PASS]"
FAIL = "[FAIL]"


def test_llm():
    from openai import OpenAI
    try:
        client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say hello."}],
            max_tokens=50,
        )
        text = response.choices[0].message.content or ""
        if text:
            print(f'{PASS} LLM (gpt-4o-mini): "{text.strip()[:80]}"')
            return True
        print(f"{FAIL} LLM (gpt-4o-mini): empty response")
        return False
    except Exception as e:
        print(f"{FAIL} LLM (gpt-4o-mini): {e}")
        return False


def test_search():
    from azure.search.documents import SearchClient
    from azure.core.credentials import AzureKeyCredential
    try:
        client = SearchClient(
            endpoint=SEARCH_ENDPOINT,
            index_name=SEARCH_INDEX,
            credential=AzureKeyCredential(SEARCH_API_KEY),
        )
        results = [r for r in client.search("Power Automate premium connectors", top=3) if r.get("chunk")]
        if results:
            print(f'{PASS} Azure AI Search: got {len(results)} result(s) for "Power Automate premium connectors"')
            return True
        print(f"{FAIL} Azure AI Search: 0 results returned")
        return False
    except Exception as e:
        print(f"{FAIL} Azure AI Search: {e}")
        return False


if __name__ == "__main__":
    results = [test_llm(), test_search()]
    if not all(results):
        sys.exit(1)
