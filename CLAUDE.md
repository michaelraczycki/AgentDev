# AgentDev — Claude Code Context

## Environment

- **Python**: 3.12.10
- **Venv**: `./venv/` — always run scripts as `venv/Scripts/python <file>.py`
- **Platform**: Windows 11, bash shell via VSCode

## Key Package Versions

| Package | Version |
|---|---|
| langchain | 1.2.10 |
| langchain-core | 1.2.17 |
| langchain-ollama | 1.0.1 |
| langchain-chroma | 1.1.0 |
| langgraph | 1.0.10 |
| langgraph-prebuilt | 1.0.8 |
| chromadb | 1.5.2 |
| ollama | 0.6.1 |
| pydantic | 2.12.5 |

## LangChain 1.x Import Patterns

In this version, the API has shifted significantly from older tutorials:

```python
# Agents — use create_agent (NOT create_react_agent from langgraph)
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest

# Ollama — use ChatOllama (NOT OllamaLLM) for agents; LLM doesn't support tool-binding
from langchain_ollama import ChatOllama

# Tools
from langchain.tools import tool

# Vector store
from langchain_chroma import Chroma
```

`create_react_agent` from `langgraph.prebuilt` still works but is deprecated — prefer `create_agent` from `langchain.agents`.

## Agent Pattern

```python
from typing import TypedDict
from langchain.agents import create_agent
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain_ollama import ChatOllama
from langchain.tools import tool

class Context(TypedDict):
    user_role: str

model = ChatOllama(model='llama3.2')

@dynamic_prompt
def my_prompt(request: ModelRequest[Context]) -> str:
    user_role = request.runtime.context.get("user_role", "user")
    return f"You are a helpful assistant. Role: {user_role}"

agent = create_agent(
    model=model,
    tools=[...],
    middleware=[my_prompt],
    context_schema=Context,
)

result = agent.invoke(
    {"messages": [{"role": "user", "content": "..."}]},
    context={"user_role": "expert"},
)
print(result["messages"][-1].content)
```

## Project Files

- `main.py` — main agent entrypoint
- `vector.py` — ChromaDB vector store logic
- `langchain_docs.py` — sandbox for learning/experimenting with LangChain patterns
- `reviews.csv` — sample data for vector store
- `chrome_langchain_db/` — persisted Chroma vector DB (gitignored)

## Local Model

Ollama is running locally with `llama3.2`. Use `ChatOllama(model='llama3.2')`.
