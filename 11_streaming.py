from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from typing import TypedDict, Literal, Annotated
from dotenv import load_dotenv
import operator


load_dotenv()

llm = ChatGroq(
    model = "llama-3.1-8b-instant"
)



class ChatState(TypedDict):

    messages : Annotated[list[BaseMessage], add_messages]



def chat_node(state: ChatState):

    messages = state['messages']

    response = llm.invoke(messages)

    return {"messages":response}


graph = StateGraph(ChatState)


graph.add_node("chat_node", chat_node)

graph.add_edge(START,"chat_node")
graph.add_edge("chat_node", END)

checkpointer = InMemorySaver()
chatbot = graph.compile(checkpointer=checkpointer)


# =================== Streaming 



for message_chunk, metadata in chatbot.stream({"messages":[HumanMessage(content="Write an essay on politics in India")]}, config={'configurable':{"thread_id":"1"}}, stream_mode="messages"):

    if message_chunk.content:
        print(message_chunk.content, end=" ", flush=True)

