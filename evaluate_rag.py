import os
import json
import pandas as pd
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import re

load_dotenv()

# Initialize embeddings and vectorstore
print("Loading vector store...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"batch_size": 64}
)
vectorstore = FAISS.load_local("medquad_index", embeddings, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize LLM for the RAG chain
print("Initializing LLMs...")
llm = ChatGroq(model_name="llama-3.3-70b-versatile", temperature=0.3)
# Using a different model for evaluation but with the same temperature parameter
judge_llm = ChatGroq(model_name="qwen/qwen3-32b", temperature=0.3)

# RAG Chain setup
prompt = PromptTemplate.from_template("""You are Sahayak AI, a safe medical assistant.
Use ONLY the provided context to answer. Do NOT add outside knowledge.

Based on the symptoms described in the query and the provided context, identify the most likely common name of the disease or condition.

Answer strictly in this format:

**1. Potential Condition:**
<Name the most likely common disease/condition connecting these symptoms>

**2. Brief Explanation:**
<Explain how the symptoms link to this condition in 2-3 lines>

**3. Next Steps & When to See a Doctor:**
<List warning signs of when to see a doctor immediately. Also, explicitly advise the user to visit the 'Medicine Recommender' module for personalized medication, diet, and precaution details.>

If the answer is not in the context, say exactly:
"I don't have enough information. Please consult a doctor."

Context:
{context}

Question:
{question}

Answer:""")

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

qa_chain = (
    {"context": lambda x: format_docs(retriever.invoke(x)), "question": lambda x: x}
    | prompt
    | llm
    | StrOutputParser()
)

# Evaluator setup
eval_prompt = PromptTemplate.from_template("""You are an expert AI judge evaluating a Retrieval-Augmented Generation (RAG) system in the medical domain.
You will be given a question, the retrieved context, and the AI's generated answer.
Please evaluate the generated answer based on two criteria:
1. Faithfulness: Is the answer entirely grounded in the retrieved context? (Score 0-50)
2. Answer Relevance: Does the answer directly address the question? (Score 0-50)

Provide your evaluation as a JSON object with 'faithfulness_score', 'relevance_score', and 'total_score' (out of 100). Do not output anything other than the JSON object.

Question:
{question}

Retrieved Context:
{context}

Generated Answer:
{answer}

JSON Evaluation:""")

eval_chain = eval_prompt | judge_llm | StrOutputParser()

# Test queries
test_queries = [
    "I have a severe headache, fever, and nausea. What could it be?",
    "I am a 60-year-old male with Type 2 diabetes. I recently started experiencing chest pain, shortness of breath, and swollen ankles. What conditions could be responsible?",
    "I have been feeling incredibly thirsty lately and have to pee all the time. I also feel exhausted all day. What could be wrong with me?",
    "My skin is itchy, and I have small red bumps all over my arms. What is this?",
    "I have a sharp pain in my lower right abdomen, and I feel like vomiting. I also have a mild fever. Is this an emergency?"
]

print("\n--- Starting Evaluation ---")
total_percentage = 0
results_data = []

for i, query in enumerate(test_queries):
    print(f"\nEvaluating Query {i+1}/{len(test_queries)}: {query}")
    
    # Get retrieved context
    docs = retriever.invoke(query)
    context_text = format_docs(docs)
    
    # Get generated answer
    generated_answer = qa_chain.invoke(query)
    
    # Evaluate
    eval_input = {
        "question": query,
        "context": context_text,
        "answer": generated_answer
    }
    eval_result_str = eval_chain.invoke(eval_input)
    
    try:
        # Remove <think>...</think> blocks if present
        if "<think>" in eval_result_str:
            eval_result_str = re.sub(r'<think>.*?</think>', '', eval_result_str, flags=re.DOTALL)
            
        # Extract JSON if there are markdown blocks
        if "```json" in eval_result_str:
            eval_result_str = eval_result_str.split("```json")[1].split("```")[0]
        elif "```" in eval_result_str:
            eval_result_str = eval_result_str.split("```")[1].split("```")[0]
        
        eval_result = json.loads(eval_result_str.strip())
        score = eval_result.get("total_score", 0)
        total_percentage += score
        
        faithfulness = eval_result.get('faithfulness_score', 0)
        relevance = eval_result.get('relevance_score', 0)
        
        print(f"Faithfulness Score: {faithfulness}/50")
        print(f"Relevance Score: {relevance}/50")
        print(f"Total Score: {score}/100")
        
        # Save to our results list
        results_data.append({
            "Question": query,
            "Faithfulness_Score": faithfulness,
            "Relevance_Score": relevance,
            "Total_Score": score,
            "Generated_Answer": generated_answer
        })
        
    except Exception as e:
        print(f"Failed to parse evaluation result: {e}")
        print(f"Raw Output: {eval_result_str}")
        
        results_data.append({
            "Question": query,
            "Faithfulness_Score": 0,
            "Relevance_Score": 0,
            "Total_Score": 0,
            "Generated_Answer": "ERROR_PARSING"
        })

average_percentage = total_percentage / len(test_queries)
print("\n" + "="*40)
print(f"Final RAG Model Evaluation Score: {average_percentage:.2f}%")
print("="*40)

# Save the final output to a CSV file
results_df = pd.DataFrame(results_data)
# Add an average row
avg_row = pd.DataFrame([{
    "Question": "AVERAGE SCORE", 
    "Faithfulness_Score": results_df["Faithfulness_Score"].mean() if not results_df.empty else 0,
    "Relevance_Score": results_df["Relevance_Score"].mean() if not results_df.empty else 0,
    "Total_Score": average_percentage,
    "Generated_Answer": ""
}])
results_df = pd.concat([results_df, avg_row], ignore_index=True)
results_df.to_csv("rag_evaluation_results.csv", index=False)
print("\n✅ Results successfully saved to 'rag_evaluation_results.csv'")
