import streamlit as st
from langchain import OpenAI
from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

def generate_response(txt):
    #instantiate the llm model

    llm=OpenAI(temperature=0.7 ,openai_api_key=openai_api_key)


    #split text using text splitter
    text_splitter=CharacterTextSplitter()
    texts=text_splitter.split_text(txt)

    #create multiple documents for the text above 
    docs=[Document(page_content=t)for t in texts]

    #text summarization
    chain=load_summarize_chain(llm,chain_type='map_reduce')
    return chain.run(docs)

#streamlit web
st.set_page_config(page_title="Text Summarization app")
st.title("Text summarization app")

#text input
txt_input=st.text_area("Enter the text ", '' , height=300)

#form to accpet user text
result=[]
with st.form("summarize form",clear_on_submit=True):
    openai_api_key=st.text_input("openai api key",type='password')
    submitted=st.form_submit_button("Submit")

    if submitted and openai_api_key.startswith('sk-'):
        with st.spinner('Calculating...'):
            response=generate_response(txt_input)
            result.append(response)
            del openai_api_key

    if len(result):
        st.info(response)      