## Langchain Demo App
## It demonstrates how chain is formed in langchain where input from one chain is passed to another chain.
## It demonstrates 2 chain interactions.
### Chain 1: connects a prompt, llm and parser. It answers a question based on the context.
### Chain 2: connects the output of chain 1 to a translator llm to translate the answer to another language.

"""
# Langchain Flow

![Langchain Flow](../_docs/images/langchain.png)
"""

from dotenv import load_dotenv
from utility.llm_factory import LLMFactory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


load_dotenv() # Load the environment variables into memory

## Setting up the LLM
#llm2 = LLMFactory.get_llm("azure")
llm2 = LLMFactory.get_llm("openai")
llm = LLMFactory.get_llm("openai")

## Setup the prompt template. This tells the llm what role it has to play and what context to use for answering the queries.
template = """
You are an assistant. 
You have to answer the questions based on the context below. If you can't answer the question, you can say "I don't know".

Context: {context}

Question: {question}

"""

## Create a prompt object from the template
prompt = ChatPromptTemplate.from_template(template)

## Setup the parser
parser = StrOutputParser()

## Form the langchain chain (build a sequence of operations/flow)
chain = prompt | llm | parser 
response = chain.invoke({
    "context": f"Mary's sister is suzane.",
    "question": "Who is Suzane?"
})
print(response)

## Setup the prompt template2 that translates the input into another language
translate_template = """
You are a translator.
Translate the {answer} into {language}
"""

## Create a prompt object2 from the template
translate_prompt = ChatPromptTemplate.from_template(translate_template)

## Form the langchain for translation. It sends the input of the previous chain to the translator chain.
from operator import itemgetter
translate_chain = (
    {"answer": chain, "language": itemgetter("language")} | translate_prompt | llm2 | parser
)
response = translate_chain.invoke({
    "context": f"Mary's sister is suzane.",
    "question": "Who is Suzane?",
    "language": "fr"
})
print(response)

