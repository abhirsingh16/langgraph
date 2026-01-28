from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
from huggingface_hub import login
import os



# MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
# MODEL_ID ="Qwen/Qwen2.5-3B-Instruct"
MODEL_ID = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map ="auto",
    dtype="auto",
    use_auth_token = True
)

pipe = pipeline(
    model = model,
    task = "text-generation",
    tokenizer = tokenizer,
    max_new_tokens = 512,
    temperature = 0.5
)

llm = HuggingFacePipeline(pipeline = pipe)

chat_model = ChatHuggingFace(llm=llm)




#=================== Create a State

class BatsmanState(TypedDict):

    runs : int
    balls : int
    fours : int
    sixes : int

    sr : float
    bpb : float
    boundary_percent : float
    summary : str




# ====================== Create function

def calculate_sr(state : BatsmanState) -> BatsmanState:

    sr = (state['runs']/state['balls'])*100

    return {'sr':sr}


def calculate_bpb(state : BatsmanState) -> BatsmanState:

    bpb = (state['fours'] + state['sixes']) / state['balls']

    return {'bpb':bpb}


def calculate_boundary_percent(state : BatsmanState) ->BatsmanState:

    boundary_percent = (((state['fours']*4) + (state['sixes']*6)) / state['runs']) * 100

    return {'boundary_percent':boundary_percent}

def summary(state: BatsmanState)->BatsmanState:

    summary = f"""
Strike Rate : {state['sr']}\n
Boundaries Per Ball : {state['bpb']}\n
Boundaries Percentage : {state['boundary_percent']}
"""
    return {'summary':summary}



# ============Create Graph

graph = StateGraph(BatsmanState)



# ============ Create Nodes

graph.add_node('calculate_sr', calculate_sr)
graph.add_node('calculate_bpb', calculate_bpb)
graph.add_node('calculate_boundary_percent', calculate_boundary_percent)
graph.add_node('summary', summary)


#===============Define Edges

graph.add_edge(START,'calculate_sr')
graph.add_edge(START,'calculate_bpb')
graph.add_edge(START,'calculate_boundary_percent')

graph.add_edge('calculate_sr', 'summary')
graph.add_edge('calculate_bpb', 'summary')
graph.add_edge('calculate_boundary_percent', 'summary')

graph.add_edge('summary', END)


# =========== Compile

workflow = graph.compile()

# ============ Execute

initial_state = {
    "runs":75,
    "balls": 55,
    "fours":10,
    "sixes": 2}

final_state = workflow.invoke(initial_state)

print(final_state['summary'])