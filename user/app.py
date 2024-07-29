import boto3
import streamlit as st
import os
import uuid
from streamlit_option_menu import option_menu

## s3_client
s3_client = boto3.client("s3")
#BUCKET_NAME = os.getenv("BUCKET_NAME")
BUCKET_NAME = "rag-streamlit"

## Bedrock
from langchain_community.embeddings import BedrockEmbeddings
from langchain.llms.bedrock import Bedrock
from langchain_community.chat_models import BedrockChat

## prompt and chain
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

## Text Splitter
from langchain.text_splitter import RecursiveCharacterTextSplitter

## Pdf Loader
from langchain_community.document_loaders import PyPDFLoader

## import FAISS
from langchain_community.vectorstores import FAISS

bedrock_client = boto3.client(service_name="bedrock-runtime", region_name = "ap-south-1")
bedrock_embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1", client=bedrock_client)

folder_path="/tmp/"

def get_unique_id():
    return str(uuid.uuid4())

## load index
def load_index():
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")

def get_llm():
    llm = BedrockChat(model_id="anthropic.claude-3-sonnet-20240229-v1:0", client=bedrock_client)
    return llm

# get_response()
def get_response(llm,vectorstore, question ):
    ## create prompt / template
    prompt_template = """

    Human: Please use the given context to provide concise answer to the question
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": 5}
    ),
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)
    answer=qa({"query":question})
    return answer['result']



def get_unique_id():
    return str(uuid.uuid4())


## Split the pages / text into chunks
def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

## create vector store
def create_vector_store(request_id, documents):
    vectorstore_faiss=FAISS.from_documents(documents, bedrock_embeddings)
    file_name=f"{request_id}.bin"
    folder_path="/tmp/"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    ## upload to S3
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".faiss", Bucket=BUCKET_NAME, Key="my_faiss.faiss")
    s3_client.upload_file(Filename=folder_path + "/" + file_name + ".pkl", Bucket=BUCKET_NAME, Key="my_faiss.pkl")

    return True


def upload_document():
    uploaded_file = st.file_uploader("Choose a file", "pdf")
    if uploaded_file is not None:
        request_id = get_unique_id()
        st.write(f"Request Id: {request_id}")
        saved_file_name = f"{request_id}.pdf"
        with open(saved_file_name, mode="wb") as w:
            w.write(uploaded_file.getvalue())

        loader = PyPDFLoader(saved_file_name)
        pages = loader.load_and_split()

        st.write(f"Total Pages: {len(pages)}")

        ## Split Text
        splitted_docs = split_text(pages, 1000, 200)
        st.write(f"Splitted Docs length: {len(splitted_docs)}")
        st.write("===================")
        st.write(splitted_docs[0])
        st.write("===================")
        st.write(splitted_docs[1])

        st.write("Creating the Vector Store")
        result = create_vector_store(request_id, splitted_docs)

        if result:
            st.write("Hurray!! PDF processed successfully")
            load_index()
        else:
            st.write("Error!! Please check logs.")


## main method
def main():
    
    st.image("user/PwCLogo.png", width=200)
    st.title("Virtual Assitant for Managed Services")
    st.markdown("The assitant helps our team respond to managed services incidents by using previously recorded support incidents and performing RAG.") 
    st.write("Bedrock LLM used : Anthropic Claude 3 Sonnet v1")
    st.write("Bedrock Embedding model used : Amazon Titan Embed Image v1")
    st.write("Vector Store : FAISS")
    selected = option_menu(menu_title=None, options = ["Query", 'Document Upload'], icons=[':robot_face:', ':page_facing_up:'], menu_icon="cast", default_index=0, orientation = "horizontal")

    

    dir_list = os.listdir(folder_path)
    #st.write(f"Files and Directories in {folder_path}")
    #st.write(dir_list)

    ## create index
    faiss_index = FAISS.load_local(
        index_name="my_faiss",
        folder_path = folder_path,
        embeddings=bedrock_embeddings,
        allow_dangerous_deserialization=True
    )

    #st.write("INDEX IS READY")
    if selected == "Query":
        question = st.text_input("Please ask your question")
        if st.button("Ask Question"):
            with st.spinner("Querying..."):

                llm = get_llm()

                # get_response
                st.write(get_response(llm, faiss_index, question))
                st.success("Done")
    
    if selected == "Document Upload":
        upload_document()

if __name__ == "__main__":
    main()




