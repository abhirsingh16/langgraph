from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline, HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# Model Building

MODEL_ID =  "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map = "auto",
    dtype = "auto"
)

pipe = pipeline(
    model = model,
    task = "text-generation",
    tokenizer = tokenizer,
    max_new_tokens = 512,
    temperature = 0.5
)

llm = HuggingFacePipeline(pipeline = pipe)

chat_model = ChatHuggingFace(llm = llm)



# -------------------------Create a State

class BlogState(TypedDict):

    title : str
    outline : str
    content :str


# -------------------------Create Graph

graph = StateGraph(BlogState)



# ------------------------Create functions

def create_outline(State: BlogState) -> BlogState:

    # fetch title
    title = State['title']

    # call llm to generate outline
    prompt= f"Generate a detailed outline for a blog on the topic - {title}"

    outline = chat_model.invoke(prompt).content

    # update status
    State['outline'] = outline

    return State



def create_blog(State: BlogState) -> BlogState:

    title = State['title']
    outline = State['outline']

    # Call llm to generate a blog
    prompt = f"Write a detailed bog on the title - {title} using the following outline \n {outline}"

    content = chat_model.invoke(prompt).content

    State['content'] = content

    return State


# -----------------------------Create node

graph.add_node("create_outline", create_outline)
graph.add_node("create_blog", create_blog)



# ----------------------------- create edge

graph.add_edge(START,"create_outline")
graph.add_edge("create_outline", "create_blog")
graph.add_edge("create_blog", END)


# ---------------------------------Compile

workflow = graph.compile()

# ---------------------------------Execute

initial_state = {"title":"Impact of AI on Jobs in India"}

final_state = workflow.invoke(initial_state)

print(final_state)