"""
Convert a document to embeddings and save to a vector database.

Author: Liang Zhao

Usage: python doc2db.py -i doc_file.txt -o db_path
"""

import sys
import argparse
from langchain_community.document_loaders import TextLoader 
from langchain_text_splitters import RecursiveCharacterTextSplitter

def doc2db(doc_file, db_path):
    # load data
    loader = TextLoader(doc_file)
    docs = loader.load()
    
    # chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100
    )
    chunks = splitter.split_documents(docs)
    
    # embedding
    from langchain_huggingface import HuggingFaceEmbeddings
    
    embeddings = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )
    
    # vector db
    from langchain_chroma import Chroma
    
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_path
    )

def main(argv):
    cmdparser = argparse.ArgumentParser(
        "Convert a doc file to vector db.")
    cmdparser.add_argument(
        '-i',
        '--input',
        help='input filename of document, ')
    cmdparser.add_argument(
        '-o',
        '--output',
        help='output path of database, ')

    args = cmdparser.parse_args(argv)
    if not args.input or not args.output:
        print("Usage: python doc2db.py -i doc_file.txt -o db_path")
        return
    else:
        doc2db(args.input, args.output)

if __name__ == "__main__":
    main(sys.argv[1:])
