import streamlit as st
from tempfile import NamedTemporaryFile
from langchain.agents import initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from tools import ImageCaptionTool, ObjectDetectionTool

tools = [ImageCaptionTool(), ObjectDetectionTool()]

Conversational_memory = ConversationBufferWindowMemory(
    memory_key = 'chat_hostory',
    k=5,
    return_messages=True
)
llm = ChatOpenAI(
    openai_api_key='None',
    temperature=0,
    model_name="gpt-3.5-turbo"
)

agent= initialize_agent(
    agent="chat-conversational-react-description",
    tools = tools,
    llm=llm,
    max_iterations=5,
    verbose=True,
    memory=Conversational_memory,
    early_stoppy_method='generate'
)


st.title('Ask a queston to an image')

st.header('Please upload an image')

file=st.file_uploader("",type=["jpeg","jpg","png"])

if file:
    st.image(file, use_column_width =True)

    user_question = st.text_input('Ask a question about your image:')

    
    with NamedTemporaryFile(dir='.') as f :
        f.write(file.getbuffer())
        image_path = f.name

        if user_question and user_question !="":
            with st.spinner(text="In progress...."):
                response = agent.run({
                    'input': '{}, this is the image path: {}'.format(user_question, image_path),
                    'chat_history': []  # or whatever your chat history is
                })
                st.write(response)
