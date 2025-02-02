# Langchain
LangChain is a framework for developing applications powered by large language models (LLMs).

## Langchain Packages
- langchain-core: It contains the base abstractions of different components. The interfaces for core components like LLMs, vector stores, retrievers and more are defined here. No third-party integrations in it.

- langchain: Contains chains, agents, and retrical strategies that make up and application's cognitive architecture. No third-partu integrations.

- langchain-community: Contains third-party integrations that are maintained by Langchain Community.

While there is long trail of integrations in langchain-community, there are some popular integrations into their own packages. e.g. langchain-openai, and langchain-anthropic, etc...

- langchain-openai: Contains classes to connect with OpenAI and Azure Open AI.

- langgraph: is an extension of langchain aimed at building robust and stateful multi-actor applications with LLMs by modeling steps as edges and nodes in a graph. It exposes high level interfaces for creating common type of agents as well as a low-level API for composing custom flows.

- langserve: A package to deploy LangChain chains as REST APIs. Makes it easy to get a production ready API up and running.

- langsmith: a developer platform that lets you debug, test, evaluate and monitor LLM applications.

## LLMChain vs ConversationalRetrievalChain vs LangChain Expression Language (LCEL) 

### LLMChain
LLMChain is used to automate the interaction between language models and tools or data sources by chaining different operations. It focuses primarily on managing the **flow of data between LLMs and other components in a sequence**.

Example:
```python
from langchain.chains import LLMChain
from langchain.llms import OpenAI

# Initialize the LLM
llm = OpenAI(model="text-davinci-003")

# Define a simple prompt template
prompt_template = "Answer the following question: {question}"

# Create the LLMChain
llm_chain = LLMChain(prompt=prompt_template, llm=llm)

# Run the chain
question = "What is the capital of France?"
response = llm_chain.run(question)
print(response)
```
Output:
```
The capital of France is Paris.

```

### ConversationalRetrievalChain 
ConversationalRetrievalChain is used to **chain LLM with document store (e.g. vector store, cache)**. It's used to build RAG application.
It helps in creating embedding the query, and sends the "retrieved documents + query" to llm to get the answer structured.

Example:
```python
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings

# Initialize LLM and document retrieval setup
llm = OpenAI(model="text-davinci-003")
vectorstore = FAISS.load_local("document_store", OpenAIEmbeddings())

# Create the ConversationalRetrievalChain
convo_chain = ConversationalRetrievalChain.from_llm(llm, vectorstore)

# Run the conversational chain with a query
query = "What is the process to reset my password?"
response = convo_chain.run(query)
print(response)
```
Output:
```
The capital of France is Paris.

```

### LangChain Expression Language (LCEL)
LCEL is a language designed for writing flexible, dynamic expressions and logic within the LangChain framework. It is used for creating complex, context-sensitive, and adaptable behaviors when interacting with models, data sources, or tools in LangChain.

LCEL is used to define expressions that determine which action to take based on dynamic conditions:

Example:
```python
lc_expression = """
if query.contains("reset password"):
    return "password_reset_tool"
else:
    return "general_support_tool"
"""
tool = evaluate_lcel(lc_expression, query="How do I reset my password?")

```
Output:
```
password_reset_tool

```
