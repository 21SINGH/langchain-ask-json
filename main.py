from dotenv import load_dotenv
import streamlit as st
import json
import random
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.callbacks import get_openai_callback
import os

def main():
    load_dotenv()
    st.set_page_config(page_title="Ask your JSON")
    st.header("Ask your JSON 💬")
    
    # Upload JSON file
    json_file = st.file_uploader("Upload your JSON", type="json")
    
    # Process the JSON
    if json_file is not None:
        try:
            # Read and parse JSON
            json_data = json.load(json_file)
            text = json.dumps(json_data, indent=2)
            
            # Split into chunks
            text_splitter = CharacterTextSplitter(
                separator="\n",
                chunk_size=2000,
                chunk_overlap=100,
                length_function=len
            )
            chunks = text_splitter.split_text(text)
            # Log number of chunks created
            st.write(f"Total chunks created: {len(chunks)}")
            
            # Create embeddings
            embeddings = AzureOpenAIEmbeddings(
                azure_deployment="text-embedding-3-large",
                api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                api_version="2024-06-01"
            )
            knowledge_base = FAISS.from_texts(chunks, embeddings)
            
            # Show user input
            user_question = st.text_input("Ask a question about your JSON:", value="Give me 1 product that will work well with moisturiser")
            if user_question:
                # Retrieve multiple documents for variety (e.g., top 3)
                docs = knowledge_base.similarity_search(user_question, k=3)
                # Randomly select one document to vary the context
                selected_doc = random.choice(docs)
                # Log number of chunks sent to AI
                st.write(f"Chunks sent to AI: 1")
                
                # Use AzureChatOpenAI with higher temperature for diversity
                llm = AzureChatOpenAI(
                    azure_deployment="gpt-4o",
                    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
                    api_version="2024-06-01",
                    temperature=0.7
                )
                
                # Custom prompt to minimize tokens and ensure single product name with use case
                prompt_template = """Based on the context, return only the product name with a small use case in less than 80 characters, nothing else.
                Context: {context}
                Question: {question}"""
                prompt = PromptTemplate(
                    template=prompt_template,
                    input_variables=["context", "question"]
                )
                
                # Use RetrievalQA with invoke, passing single selected document
                qa_chain = RetrievalQA.from_chain_type(
                    llm=llm,
                    chain_type="stuff",
                    retriever=knowledge_base.as_retriever(search_kwargs={"k": 1}),
                    chain_type_kwargs={"prompt": prompt}
                )
                
                with get_openai_callback() as cb:
                    response = qa_chain.invoke({"query": user_question, "context": selected_doc.page_content})
                    print(cb)
                
                # Extract only the product name with use case
                st.write(response["result"])
                
        except Exception as e:
            st.error(f"Error: {str(e)}")

if __name__ == '__main__':
    main()