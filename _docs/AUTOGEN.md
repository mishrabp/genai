## Reference
https://microsoft.github.io/autogen/0.2/docs/Getting-Started/

https://www.youtube.com/watch?v=V2qZ_lgxTzg&list=PLp9pLaqAQbY2vUjGEVgz8yAOdJlyy3AQb

https://microsoft.github.io/autogen/0.2/docs/Examples

# Autogen
Autogen is a framework for simplifying the orchestration, optimization and automation of LLM workflows. It offers customizable and coversable agents that leverage the strongest capabilities of the most advanced LLMs, like GPT-4 while addressing their limitation by integrating with the humans and tools and having convesations between multiple agents via automated chat.

AutoGen is powered by collaborative research studies from Microsoft, Penn State University, and University of Washington. It works with all kind of LLMs, not just Open AI.

## Type of Agents in Autogen
- ConversableAgent
- AssistantExecutor
- GroupChat
- CodeGenerationAgent
- CodeReviewerAgent
- ValidatorAgent
- Custom Agent (created by extending ConversableAgent)

## Ways to End a Coverstation
**max-turns parameter in initiate_chat()** 
```python
result = joe.initiate_chat(
    cathy, message="Cathy, tell me a joke.", max_turns=3
)  # increase the number of max turns before termination
```

**max_consecutive_auto_reply parameter in agent**
when auto-reply from the agent reaches the threshold, it terminates the conversation.
```python
joe = ConversableAgent(
    "joe",
    system_message="Your name is Joe and you are a part of a duo of comedians.",
    llm_config={"config_list": [{"model": "gpt-4", "temperature": 0.7, "api_key": os.environ.get("OPENAI_API_KEY")}]},
    human_input_mode="NEVER",  # Never ask for human input.
    max_consecutive_auto_reply=1,  # Limit the number of consecutive auto-replies.
)

result = joe.initiate_chat(cathy, message="Cathy, tell me a joke.")
```

**is_termination_msg parameer in agent**
You can specify a message that would be treated as termination.
```python
joe = ConversableAgent(
    "joe",
    system_message="Your name is Joe and you are a part of a duo of comedians.",
    llm_config={"config_list": [{"model": "gpt-4", "temperature": 0.7, "api_key": os.environ.get("OPENAI_API_KEY")}]},
    human_input_mode="NEVER",  # Never ask for human input.
    is_termination_msg=lambda msg: "good bye" in msg["content"].lower(),
)

result = joe.initiate_chat(cathy, message="Cathy, tell me a joke and then say the words GOOD BYE.")
```

# What is an Agent?
Agents are systems that use LLMs as reasoning engines to determine which actions to take and the inputs necessary to perform the action. After executing actions, the results can be fed back into the LLM to determine whether more actions are needed, or whether it is okay to finish. This is often achieved via tool-calling.

