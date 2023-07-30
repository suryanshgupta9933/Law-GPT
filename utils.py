# Importing Dependencies
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from langchain import PromptTemplate, HuggingFacePipeline
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Faiss Index Path
FAISS_INDEX = "vectorstore/"

# Custom prompt template
custom_prompt_template = """[INST] <<SYS>>
You are a trained bot to guide people about Indian Law. You will answer user's query with your knowledge and the context provided. 
If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.
Do not say thank you and tell you are an AI Assistant and be open about everything.
<</SYS>>
Use the following pieces of context to answer the users question.
Context : {context}
Question : {question}
Answer : [/INST]
"""

# Return the custom prompt template
def set_custom_prompt_template():
    """
    Set the custom prompt template for the LLMChain
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

    return prompt

# Return the LLM
def load_llm():
    """
    Load the LLM
    """
    # Model ID
    repo_id = 'meta-llama/Llama-2-7b-chat-hf'

    # Load the model
    model = AutoModelForCausalLM.from_pretrained(
        repo_id,
        device_map='auto',
        load_in_4bit=True
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id,
        use_fast=True
    )

    # Create pipeline
    pipe = pipeline(
        'text-generation',
        model=model,
        tokenizer=tokenizer,
        max_length=512
    )

    # Load the LLM
    llm = HuggingFacePipeline(pipeline=pipe)

    return llm

# Return the chain
def retrieval_qa_chain(llm, prompt, db):
    """
    Create the Retrieval QA chain
    """
    # Create the chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type='stuff',
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )

    return qa_chain

# Return the chain
def qa_pipeline():
    """
    Create the QA pipeline
    """
    # Load the HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings()

    # Load the index
    db = FAISS.load_local("vectorstore/", embeddings)

    # Load the LLM
    llm = load_llm()

    # Set the custom prompt template
    qa_prompt = set_custom_prompt_template()

    # Create the retrieval QA chain
    chain = retrieval_qa_chain(llm, qa_prompt, db)

    return chain