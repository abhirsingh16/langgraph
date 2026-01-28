from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from typing import TypedDict, Annotated, List
from dotenv import load_dotenv
import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient


load_dotenv()

model = ChatGroq(
    model = "llama-3.1-8b-instant"
)

# MCP client
client = MultiServerMCPClient(
    {
        "arith": {
            "transport" : "stdio",
            "command" : "C:/Users/abi/AppData/Local/anaconda3/envs/langgraphenv/python.exe",
            "args":[
                "D:/MCP/Local MCP Server/expense-tracker-mcpserver/main.py"
                ],
        },
        "expense": {
            "transport" : "streamable_http",
            "url" : "https://lively-blush-dog.fastmcp.app/mcp"
        }
    }
)



# @tool
# def calculator(first_num: float, second_num: float, operation:str) ->dict:
#     """
#     Performs basic arithmetic operations between the two numbers
#     Supported operations : add, subtract, multiply, divide"""

#     try:
#         if operation == "add":
#             result = first_num + second_num
#         elif operation == "subtract":
#             result = first_num - second_num
#         elif operation == "multiply":
#             result = first_num * second_num
#         elif operation == "divide":
#             result = first_num / second_num
#         else:
#             return {"error" : "Unsupported Operation"}

#         return {"first_num":first_num, "second_num":second_num, "operation": operation, "result": result}
    
#     except Exception as e:
#         return {"error":str(e)}
    



class ChatState(TypedDict):

    messages : Annotated[List[BaseMessage], add_messages]




async def build_graph():

    tools = await client.get_tools()

    print(tools)

    llm_with_tools = model.bind_tools(tools)

    async def chat_node(state: ChatState):

        messages = state["messages"]

        response = await llm_with_tools.ainvoke(messages)

        return {"messages": response}

    tool_node = ToolNode(tools)

    graph = StateGraph(ChatState)

    graph.add_node("chat_node", chat_node)
    graph.add_node("tools", tool_node)

    graph.add_edge(START, "chat_node")
    graph.add_conditional_edges("chat_node", tools_condition)
    graph.add_edge("tools", "chat_node")
    graph.add_edge("chat_node", END)

    chatbot = graph.compile()

    return chatbot

async def main():

    chatbot = await build_graph()

    response = await chatbot.ainvoke({"messages":[HumanMessage(content= "What is two plus two")]})


    print(response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())

