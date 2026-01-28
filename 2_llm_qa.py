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
    max_new_tokens = 256,
    temperature = 0.5
)

llm = HuggingFacePipeline(pipeline = pipe)

chat_model = ChatHuggingFace(llm = llm)



#---------------------- Create a state

class LLMState(TypedDict):

    question : str
    answer : str


# ----------------------- Create our graph

graph = StateGraph(LLMState)


# ------------------------ Create a function of LLMqa

def llm_qa(State: LLMState) -> LLMState:

    # extract the question from the state
    question = State['question']

    # form a prompt
    prompt = f"Answer the following question{question}"

    # Ask the qeustion to the llm
    answer = chat_model.invoke(prompt).content

    # update the answer in the state
    State['answer'] = answer

    return State


# ------------------------Add Nodes

graph.add_node('llm_qa', llm_qa)


# ------------------------Add Edges

graph.add_edge(START, "llm_qa")
graph.add_edge('llm_qa', END)


# -----------------------compile

workflow = graph.compile()

# ------------------------Execute

initial_state = {"question":"How far is moon from earth?"}

final_state = workflow.invoke(initial_state)

print(final_state)