from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Literal
from pydantic import Field, BaseModel


# ========================= model

model = ChatOpenAI(
    model ="phi",
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)

 
# ============================ Structured output

class SentimentSchema(BaseModel):

    sentiment:Literal["positive","negative"] = Field(description="Sentiment of the review")



class DiagnosisSchema(BaseModel):

    issue_type: Literal["UX", "Performance", "Bug", "Support", "Other"] = Field(description='The category of issue mentioned in the review')

    tone: Literal["angry", "frustrated", "disappointed", "calm"] = Field(description='The emotional tone expressed by the user')
    
    urgency: Literal["low", "medium", "high"] = Field(description='How urgent or critical the issue appears to be')


structured_model1 = model.with_structured_output(SentimentSchema)
structured_model2 = model.with_structured_output(DiagnosisSchema)

# ========================= Create State

class ReviewState(TypedDict):

    review:str
    sentiment:Literal["positive","negative"]
    diagnosis:dict
    response:str
# ========================= CREATE node functions

def find_sentiment(state: ReviewState):

    prompt = f"For the following review find out the sentiment \n {state["review"]}"

    sentiment = structured_model1.invoke(prompt)

    return {"sentiment" : sentiment}



def positive_response(state: ReviewState):

    prompt = f"""Write a warm thank you message in response to this review: \n\n
    "{state["review"]}" Also ask the user to leave a feedback on the websiite"""

    response = model.invoke(prompt).content

    return {"response": response}



def run_diagnosis(state: ReviewState):

    prompt = f" Diagnose the negitive review: \n\n {state["review"]} \n\n return issue_type, tone and urgency"

    response = structured_model2.invoke(prompt)

    return {"diagnosis" : response.model_dump()}


def negative_response(state: ReviewState):

    diagnosis = state["diagnosis"]

    prompt = f"""You are a support assistant.
The user had a '{diagnosis['issue_type']}' issue, sounded '{diagnosis['tone']}', and marked urgency as '{diagnosis['urgency']}'.
Write an empathetic, helpful resolution message.
"""
    response = model.invoke(prompt).content

    return {'response': response}

def check_sentiment(state: ReviewState) -> Literal["positive_response", "run_diagnosis"]:

    if state['sentiment'] == 'positive':
        return 'positive_response'
    else:
        return 'run_diagnosis'
    


# ========================== create nodes
graph = StateGraph(ReviewState)

graph.add_node("find_sentiment", find_sentiment)
graph.add_node("positive_response", positive_response)
graph.add_node("run_diagnosis", run_diagnosis)
graph.add_node("negative_response", negative_response)


# ============================ create Edges

graph.add_edge(START, "find_sentiment")
graph.add_conditional_edges("find_sentiment", check_sentiment)

graph.add_edge("positive_response",END)

graph.add_edge("run_diagnosis","negative_response")
graph.add_edge("negative_response",END)


# =============================compile

workflow = graph.compile()

# ============================= execute

initial_state = {
    "review": """I recently purchased the Jacket and I m very satisfied with it. The material feels soft yet durable making it comfortable to wear for long hours. It is truly lightweight so I can carry it easily while traveling, and it doesn’t feel heavy at alThe design is modern and stylish making it suitable for both casual and semi-formal outings. The fitting is perfect, and the stitching quality looks strong and neat the pockets are quite handy for carrying essentials like a wallet or phone. Very good product go for it."""
}

final_state = workflow.invoke(initial_state)

print(final_state)