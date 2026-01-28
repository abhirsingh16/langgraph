from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from typing import TypedDict, Literal, Annotated
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver


# ================== Model

# Model Building

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key="REMOVED"
)


# ================= State

class ChatState(TypedDict):

    messages : Annotated[list[BaseMessage], add_messages]


# ==================== chat function

def chat_node(state : ChatState):

    messages = state["messages"]

    response = llm.invoke(messages)

    return {"messages" : response}



# =================== Graph, node, edge
checkpointer = MemorySaver()
graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)

graph.add_edge(START, "chat_node")
graph.add_edge("chat_node", END)


chatbot = graph.compile(checkpointer=checkpointer)

print(chatbot)


# initial_state = {"messages":[HumanMessage(content="What is the capital of India?")]}

# result = chatbot.invoke(initial_state)

# print("AI:", result["messages"][-1].content)


thread_id = "1"

while True:

    user_message = input("Type here:  ")

    print("user :", user_message)

    if user_message.strip().lower() in ['exit','quit', 'bye']:
        break
    
    config = {"configurable" : {"thread_id":thread_id}}
    response = chatbot.invoke({"messages":[HumanMessage(content=user_message)]}, config= config)

    print("AI", response["messages"][-1].content)



