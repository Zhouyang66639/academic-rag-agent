"""Find free OpenRouter models that support tool calling."""
import requests
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(r'd:\python study\ai_coding\academic-rag-agent\.env'))
key = os.getenv('OPENAI_API_KEY')

# Get all free models
headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
r = requests.get('https://openrouter.ai/api/v1/models', headers=headers, timeout=15)
models = r.json()['data']
free = [m for m in models if m.get('pricing', {}).get('prompt', '1') == '0']

# Check which ones support tool calling (usually indicated in capabilities)
print(f"Total free models: {len(free)}, checking tool support...\n")

# Models known to possibly support tool calling
candidates = [m['id'] for m in free if any(kw in m['id'].lower() for kw in [
    'llama', 'qwen', 'mistral', 'phi', 'gemma', 'nemotron', 'glm', 'gpt-oss'
])]

print(f"Candidate models to test: {len(candidates)}")
for cid in candidates[:15]:
    print(f"  {cid}")

@tool
def dummy_search(query: str) -> str:
    """Search for information."""
    return "result"

print("\nTesting tool calling support:")
working = []
for model_id in candidates[:15]:
    try:
        llm = ChatOpenAI(model=model_id, temperature=0, streaming=False)
        llm_with_tools = llm.bind_tools([dummy_search])
        r = llm_with_tools.invoke([HumanMessage(content='hello, just say hi back')])
        content = r.content.encode('utf-8', errors='replace').decode('utf-8')
        print(f"  [OK TOOLS] {model_id}")
        print(f"     reply: {content[:60]}")
        working.append(model_id)
        if len(working) >= 2:
            break
    except Exception as e:
        err = str(e).encode('utf-8', errors='replace').decode('utf-8')[:80]
        print(f"  [FAIL] {model_id}: {err}")

print(f"\nWorking models with tool support: {working}")
