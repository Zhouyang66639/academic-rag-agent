"""
Quick verification script for next AI.
Run this FIRST to check if the surrogate bug is fixed.

Usage:
    conda activate rag-agent
    cd d:\python study\ai_coding\academic-rag-agent
    python handoff\quick_test.py
"""
import os, sys
sys.path.insert(0, r'd:\python study\ai_coding\academic-rag-agent')
os.chdir(r'd:\python study\ai_coding\academic-rag-agent')

from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path('.env'))

print("=" * 60)
print("Academic RAG Agent — Quick Verification")
print("=" * 60)

# Step 1: API connectivity
print("\n[1/4] Testing API connectivity...")
import requests
key = os.getenv('OPENAI_API_KEY')
base = os.getenv('OPENAI_BASE_URL')
model = os.getenv('OPENAI_MODEL')

headers = {'Authorization': f'Bearer {key}', 'Content-Type': 'application/json'}
r = requests.post(f'{base}/chat/completions', headers=headers,
    json={'model': model, 'messages': [{'role':'user','content':'say: OK'}], 'max_tokens': 10},
    timeout=20)

if r.status_code == 200:
    reply = r.json()['choices'][0]['message']['content']
    print(f"  ✅ API OK — model: {model}, reply: {reply}")
else:
    print(f"  ❌ API FAIL [{r.status_code}]: {r.text[:100]}")
    sys.exit(1)

# Step 2: Tool calling
print("\n[2/4] Testing tool calling...")
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool

@tool
def dummy_search(query: str) -> str:
    """Search for papers."""
    return f"Papers about: {query}"

llm = ChatOpenAI(model=model, temperature=0, streaming=False)
llm_t = llm.bind_tools([dummy_search])
r2 = llm_t.invoke([HumanMessage(content="Search for RAG papers")])
if r2.tool_calls:
    print(f"  ✅ Tool calling works — called: {r2.tool_calls[0]['name']}")
else:
    print(f"  ⚠️ No tool call made — content: {(r2.content or '')[:80]}")

# Step 3: Full agent chat (THE KEY TEST — this is where the bug is)
print("\n[3/4] Testing FULL AGENT chat (checks surrogate bug)...")
from src.rag.vector_store import VectorStoreManager
from src.agent.agent import AcademicRAGAgent
from src.memory.persistent_memory import PersistentMemory

vs = VectorStoreManager('./vector_store', 'BAAI/bge-small-en-v1.5')
pm = PersistentMemory('./memory')
agent = AcademicRAGAgent(vs, model_name=model, persistent_memory=pm)

response = agent.chat("你好，请简单介绍你的功能，不要搜索任何论文")
if response.startswith("Error:"):
    print(f"  ❌ SURROGATE BUG STILL PRESENT: {response[:150]}")
    print("\n  → See handoff/README.md Section 3 for fix options")
else:
    has_surr = any(0xD800 <= ord(c) <= 0xDFFF for c in response)
    if has_surr:
        print(f"  ❌ Response has surrogates!")
    else:
        print(f"  ✅ Agent works! Response: {response[:100]}...")

# Step 4: Summary
print("\n[4/4] Environment check...")
import langchain_core, langgraph, langchain_openai
print(f"  langchain-core: {langchain_core.__version__}")
print(f"  langgraph: {langgraph.__version__}")
print(f"  langchain-openai: {langchain_openai.__version__}")

print("\n" + "=" * 60)
print("Done. Check results above.")
print("If [3/4] fails → follow README.md Section 3 fix options")
print("=" * 60)
