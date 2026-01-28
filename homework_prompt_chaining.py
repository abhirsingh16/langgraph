from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict


# ------------ MOdel Building

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

Chat_model = ChatHuggingFace(llm = llm)


# ------------------Create State

class BlogState(TypedDict):

    title:str
    outline:str
    blog:str
    evaluate:int


# -----------------Create function for each State

def create_outline(state : BlogState) -> BlogState:

    title = state['title']

    # call llm to generate outline
    prompt = f"Generate a detailed outline for the given topic - {title}"

    outline = Chat_model.invoke(prompt).content

    state['outline'] = outline

    return state



def create_blog(state: BlogState) -> BlogState:

    title=state['title']
    outline=state['outline']

    # call llm to generate blog
    prompt = f"Write a detailed blog on topic - {title} using the following outline -\n {outline}"

    blog = Chat_model.invoke(prompt).content

    state['blog'] = blog

    return state


def create_evaluation(state:BlogState) ->BlogState:

    title=state['title']
    outline=state['outline']
    blog=state['blog']

    #call llm to generate scores
    prompt=f"Based on the topic -{title} and the outline -\n{outline}, evaluate and rate my blog out of 10. My blog -\n{blog}"

    evaluate = Chat_model.invoke(prompt).content

    state['evaluate'] = evaluate

    return state


# ----------------Define Graph

graph = StateGraph(BlogState)


# -----------------create note

graph.add_node("create_outline", create_outline)
graph.add_node("create_blog", create_blog)
graph.add_node("create_evaluation", create_evaluation)

# ---------------add edges

graph.add_edge(START, "create_outline")
graph.add_edge("create_outline", "create_blog")
graph.add_edge("create_blog", "create_evaluation")
graph.add_edge("create_evaluation", END)

# --------------------workflow compilec

workflow = graph.compile()


# ----------------execute
initial_state = {"title":"Unemployment in India"}
final_state = workflow.invoke(initial_state)

print(final_state)