from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
import operator

llm = ChatOpenAI(
    model ="phi",
    base_url="http://localhost:11434/v1",
    api_key="ollama"
)


# ========================= Create Class for structured output

class EvaluationSchema(BaseModel):

    feedback:str = Field(description="Detailed Feedback for the essay")
    score:int = Field(description="Score out of 10",ge=0, le=10)

structured_model = llm.with_structured_output(EvaluationSchema)



# ========================= Essay

essay = """
The Future of Artificial Intelligence in India

The future of artificial intelligence in India promises to be transformative, reshaping industries, governance, and everyday life. With a rapidly expanding digital ecosystem and one of the world’s largest pools of young technical talent, India is positioned to become a global hub for AI innovation. Sectors such as healthcare, agriculture, finance, and education stand to benefit enormously. For example, AI-powered diagnostic tools can make healthcare more accessible in rural areas, while smart farming solutions can help farmers increase productivity and adapt to climate challenges.

India’s government has also recognized AI’s strategic importance, launching initiatives like Digital India, IndiaAI, and programs supporting AI research, startups, and skilling. These investments aim to build robust AI infrastructure and ensure widespread adoption. At the same time, Indian enterprises are increasingly integrating AI to enhance decision-making, automate operations, and drive economic growth.

However, the future will also require addressing challenges such as data privacy, ethical use, and workforce upskilling. With balanced regulation, strong public-private collaboration, and continuous innovation, AI has the potential to accelerate India’s development and make technology more inclusive. Ultimately, AI’s future in India is not just about automation—it is about empowering people, enabling opportunity, and shaping a smarter nation.
"""



essay2  = """
Youth of India

The youth of india is very important but many times people not understand how much they doing in the country and all the places. Today the young pepole are facing so much problums like no jobs, pressure from parents and also the society keeps telling them what to do or not to doing. Many youth want to be success fastly but they dont have proper guidence so they get confussed and run in wrong derection. The country leaders always talking about future of nation but they not focus correct on youth skills and there talant is wasting.

Young pepole are also spending to much time on mobiles and internet and they forgeting the real life things. Many of them want to go out of india because they thinking life is easy outside, but actually if get support here they can be more good. Youth should be motivating and getting opertunities so they can grow nice and build strong india.

Teachers and parents also have to understand that youth need freedom and not every thing is about marks or exam. India youth is full of energy but it is not using proper. If right steps taking then the youth can make india very big and powerfull country.
"""

# ====================== prompt to check structured output


prompt = f'Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {essay}'

print(structured_model.invoke(prompt).score)



# ================================ Create State

class UPSCState(TypedDict):

    essay:str
    language_feedback :str
    analysis_feedback :str
    clarity_feedback :str
    individual_scores : Annotated[list[int], operator.add]
    avg_score : float


# ================================== Create function for each State


def evaluate_language(state : UPSCState) -> UPSCState:
    
    prompt = f"Evaluate the language quality of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}"

    output = structured_model.invoke(prompt)

    return {"language_feedback": output.feedback, "individual_scores":[output.score]}




def evaluate_analysis(state : UPSCState) -> UPSCState:
    
    prompt = f"Evaluate the depth of analysis of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}"

    output = structured_model.invoke(prompt)

    return {"analysis_feedback": output.feedback, "individual_scores":[output.score]}




def evaluate_thought(state : UPSCState) -> UPSCState:
    
    prompt = f"Evaluate the clarity of thought of the following essay and provide a feedback and assign a score out of 10 \n {state["essay"]}"

    output = structured_model.invoke(prompt)

    return {"clarity_feedback": output.feedback, "individual_scores":[output.score]}



def final_evaluation(state: UPSCState) -> UPSCState:

    #summary feedback
    prompt = f"Based on the following feedbacks, create a summerized feedback \n language feedback - {state["language_feedback"]}\n depth of analysis - {state['analysis_feedback']}  \n clarity of thought = {state["clarity_feedback"]}"

    overall_feedback = llm.invoke(prompt).content

    #avg calulator
    avg_score = sum(state["individual_scores"])/len(state["individual_scores"])

    return {"overall_feedback": overall_feedback, "avg_score":avg_score}




# ========================== create nodes

graph = StateGraph(UPSCState)

graph.add_node("evaluate_language", evaluate_language)
graph.add_node("evaluate_analysis", evaluate_analysis)
graph.add_node("evaluate_thought", evaluate_thought)
graph.add_node("final_evaluation", final_evaluation)


# ============================= Create edges

graph.add_edge(START, "evaluate_language")
graph.add_edge(START, "evaluate_analysis")
graph.add_edge(START, "evaluate_thought")

graph.add_edge("evaluate_language", "final_evaluation")
graph.add_edge("evaluate_analysis", "final_evaluation")
graph.add_edge("evaluate_thought", "final_evaluation")

graph.add_edge("final_evaluation", END)


# ============================= create workflow

workflow = graph.compile()

# ================================ Execute

initial_state= {"essay" : essay2}

final_state = workflow.invoke(initial_state)

print(final_state)