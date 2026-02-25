"""
RAG agent for answering questions based on the context retrieved from a vector database.

Author: Liang Zhao

Usage: python rag_agent.py -m llm -d db_path
"""

import sys
import argparse
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline
from langchain_classic.chains import RetrievalQA

def rag(model_name, db_path):
    # embedding
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # vector db
    vector_db = Chroma.from_documents(
        persist_directory=db_path,
        embedding_function=embeddings
    )

    # retriever
    retriever = vector_db.as_retriever(
	search_type="similarity",
	search_kwargs={"k": 3} 
    )

    # llm
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(
	model_name,
	device_map="auto"
    )

    pipe = pipeline(
	"text2text-generation",
	model=model,
	tokenizer=tokenizer,
	max_new_tokens=512
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    # agent
    agent = RetrievalQA.from_chain_type(
	llm=llm,
	retriever=retriever,
	chain_type="stuff",
	return_source_documents=True
    )
    return agent

def main(argv):
    cmdparser = argparse.ArgumentParser(
        "RAG agent")
    cmdparser.add_argument(
        '-m',
        '--model',
        help='LLM model, ')
    cmdparser.add_argument(
        '-d',
        '--db',
        help='database path, ')

    args = cmdparser.parse_args(argv)
    if not args.model or not args.db:
        print("Usage: python rag_agent.py -m llm -d db_path")
        return
    else:
        agent = rag(args.model, args.db)
	while True:
	    query = input("\nAsk a question (or type 'exit'): ")
	    if query.lower() == 'exit': break

	    print("Thinking...")
	    response = agent.invoke(query)
	    print(f"\nAnswer: {response['result']}")

if __name__ == "__main__":
    main(sys.argv[1:])
