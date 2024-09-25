from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from streamlit_mic_recorder import speech_to_text
from gtts.lang import tts_langs
import streamlit as st
from gtts import gTTS
import os

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from PyPDF2 import PdfReader
import docx
from langchain.vectorstores import FAISS





# Initialize conversation history


# Initialize conversation history
if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

# Ensure chat_history is initialized before making the request
if "chat_history" not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = []    

# Initialize session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []  # Store AI generated responses

if 'past' not in st.session_state:
    st.session_state['past'] = []  # Store past user inputs

if 'entered_prompt' not in st.session_state:
    st.session_state['entered_prompt'] = ""  # Store the latest user input







if "conversation" not in st.session_state:
        st.session_state.conversation = None

if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

if "processComplete" not in st.session_state:
        st.session_state.processComplete = None    


if 'audio_files' not in st.session_state:
        st.session_state.audio_files = []  # For storing separate audio files


# Function to format chat history as text for download
def format_chat_for_download(chat_history):
    formatted_text = ""
    for i, message in enumerate(chat_history):
        if message["role"] == "user":
            formatted_text += f"User {i+1}: {message['content']}\n"
        elif message["role"] == "bot":
            formatted_text += f"Bot {i+1}: {message['content']}\n"
    return formatted_text

# Function to display conversation and play audio separately for each question
def display_conversation_and_audio():
    if 'conversation_history' in st.session_state:
        for i, message in enumerate(st.session_state.conversation_history):
            # Display user question and bot response in the correct format
            if isinstance(message, dict):
                if message["role"] == "user":
                    st.markdown(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
                elif message["role"] == "bot":
                    st.markdown(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)
                    
                    # Play the corresponding audio file for the bot response
                    response_audio_file = f"response_audio_{(i//2)+1}.mp3"  # Create unique audio file for each response
                    st.audio(response_audio_file) 

# Apply custom CSS for chat style, background, and sidebar
css = '''
<style>
    .chat-message {
        padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex;
    }
    .chat-message.user {
        background-color: #2b313e;
    }
    .chat-message.bot {
        background-color: #475063;
    }
    .chat-message .avatar {
        width: 20%;
    }
    .chat-message .avatar img {
        max-width: 78px;
        max-height: 78px;
        border-radius: 50%;
        object-fit: cover;
    }
    .chat-message .message {
        width: 80%;
        padding: 0 1.5rem;
        color: #fff;
    }
    /* Background and Sidebar */
    body {
        background-color: skyblue !important;
    }
    [data-testid="stSidebar"] {
        background-color: lightgray !important;
        }
</style>
'''

st.markdown(css, unsafe_allow_html=True)

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.pinimg.com/originals/0c/67/5a/0c675a8e1061478d2b7b21b330093444.gif" style="max-height: 70px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://th.bing.com/th/id/OIP.uDqZFTOXkEWF9PPDHLCntAHaHa?pid=ImgDet&rs=1" style="max-height: 80px; max-width: 50px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''




# File upload and processing
def get_files_text(uploaded_files):
    text = ""
    for uploaded_file in uploaded_files:
        _, file_extension = os.path.splitext(uploaded_file.name)
        if file_extension == ".pdf":
            text += get_pdf_text(uploaded_file)
        elif file_extension == ".docx":
            text += get_docx_text(uploaded_file)
        else:
            text += "Unsupported file type."
    return text

def get_pdf_text(pdf):
    reader = PdfReader(pdf)
    return "".join([page.extract_text() for page in reader.pages])

def get_docx_text(doc_file):
    doc = docx.Document(doc_file)
    return ' '.join([para.text for para in doc.paragraphs])

def get_text_chunks(text):
    splitter = CharacterTextSplitter(separator="\n", chunk_size=900, chunk_overlap=100)
    return splitter.split_text(text)

def get_vectorstore(text_chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embeddings)
    return vector_store

# Setup conversation chain using Google Generative AI
def get_conversation_chain(vectorstore, api_key):
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=api_key)
    return ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=vectorstore.as_retriever(),
    )

def handle_user_input(user_question):
    response_container = st.container()
    
    # Ensure chat_history is initialized before making the request
    if "chat_history" not in st.session_state or st.session_state.chat_history is None:
        st.session_state.chat_history = []
    
    # Pass chat_history when calling the conversation chain
    response = st.session_state.conversation({
        'question': user_question,
        'chat_history': st.session_state.chat_history
    })

    # Update the chat history with the response
    st.session_state.chat_history = response['chat_history']

    # Display conversation in the response container
    with response_container:
        for i, messages in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.markdown(user_template.replace("{{MSG}}", messages.content), unsafe_allow_html=True)
            else:
                st.markdown(bot_template.replace("{{MSG}}", messages.content), unsafe_allow_html=True)
 





#api_key =   # Replace with your actual API key

st.title("🎙️RAG Voice Conversation Chatbot 🤖")

# Sidebar options for language selection and Q/A type
language = st.sidebar.selectbox("Select Language", ["Urdu", "English"])
option = st.sidebar.selectbox("Choose an option", ["General Q/A", "Document Q/A"])
 

# Clear chat button
if st.sidebar.button("Clear Chat"):
    st.session_state.conversation_history = []
    st.success("Chat history cleared.")




















# Set response language and prompt template based on user's language selection
if language == "Urdu":
    chat_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant. Please always respond to user queries in Urdu.",
            ),
            ("human", "{human_input}"),
        ]
    )
    response_lang = "ur"
else:
    chat_template = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful AI assistant. Please always respond to user queries in English.",
            ),
            ("human", "{human_input}"),
        ]
    )
    response_lang = "en"

# Initialize the language model
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", google_api_key=api_key
)

chain = chat_template | model | StrOutputParser()


# General Q/A functionality
if option == "General Q/A":
    st.subheader("General Question and Answer")

    # Initialize conversation history and audio files list if not already present
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []


    # Speech-to-text for input based on selected language
    if language == "Urdu":
        text = speech_to_text(language="ur", use_container_width=True, just_once=True, key="STT_Urdu")
    else:
        text = speech_to_text(language="en", use_container_width=True, just_once=True, key="STT_English")

    # Check if text was successfully recognized
    if text:
        st.subheader(f"Recognized {language} Text:")
        st.write(f"**User:** {text}")  # Display user's question

        with st.spinner("Fetching Response and Converting Text To Speech..."):
            # Get AI response
            res = chain.invoke({"human_input": text})

            # Save user message and AI response in conversation history
            st.session_state.conversation_history.append({"role": "user", "content": text})
            st.session_state.conversation_history.append({"role": "bot", "content": res})

            # Generate separate audio file for the latest bot response
            response_audio_file = f"response_audio_{len(st.session_state.audio_files) + 1}.mp3"
            tts = gTTS(text=res, lang="ur" if language == "Urdu" else "en")
            tts.save(response_audio_file)

            # Store the audio file in session state
            st.session_state.audio_files.append(response_audio_file)

    # Display the conversation history in the correct order
    bot_audio_index = 0  # Track audio files index
    for i, message in enumerate(st.session_state.conversation_history):
        if message["role"] == "user":
            st.markdown(user_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)  # User query in style
        elif message["role"] == "bot":
            st.markdown(bot_template.replace("{{MSG}}", message["content"]), unsafe_allow_html=True)  # Bot response in style
            
            # Display corresponding bot audio file if it exists
            if bot_audio_index < len(st.session_state.audio_files):
                st.audio(st.session_state.audio_files[bot_audio_index])  # Play the audio for the bot response
                bot_audio_index += 1  # Move to the next audio file

    # Display the option to download the chat history
    if st.session_state.conversation_history:
        # Format the chat history for download
        formatted_chat = ""
        for i, message in enumerate(st.session_state.conversation_history):
            if message["role"] == "user":
                formatted_chat += f"User {i//2 + 1}: {message['content']}\n"
            elif message["role"] == "bot":
                formatted_chat += f"Bot {i//2 + 1}: {message['content']}\n"

        st.download_button(
            label="Download Chat History",
            data=formatted_chat,
            file_name="chat_history.txt",
            mime="text/plain"
        )
 
 

 

# Document Q/A functionality
elif option == "Document Q/A":
    st.subheader("Document Question and Answer")

    # File uploader for PDFs and DOCX files
    uploaded_files = st.file_uploader("Upload your PDF or DOC files", type=["pdf", "docx"], accept_multiple_files=True)

    if uploaded_files:
        files_text = get_files_text(uploaded_files)  # Extract text from the uploaded files
        st.write("File uploaded and loaded...")

        # Process the file into chunks and store in vectorstore
        text_chunks = get_text_chunks(files_text)  # Split the file text into smaller chunks
        vectorstore = get_vectorstore(text_chunks)  # Create a vector store for document search
        st.write("Vector store created with file chunks...")

        # Setup conversation chain with the vectorstore for question answering
        st.session_state.conversation = get_conversation_chain(vectorstore, api_key)
        st.session_state.processComplete = True

    # Once document is processed, start Q/A session
    if st.session_state.processComplete:
        # Language selection for Speech-to-Text (STT) 
        if language == "Urdu":
            user_question = speech_to_text(language="ur", key="STT_Urdu_Doc")
        else:
            user_question = speech_to_text(language="en", key="STT_English_Doc")

        # Once a question is detected, handle conversation
        if user_question:
            st.write(f"Recognized {language} Question: {user_question}")

            # Handle user's input and fetch the response
            response = st.session_state.conversation({
                'question': user_question,
                'chat_history': st.session_state.get('chat_history', [])
            })

            # Append user's question and chatbot's response to the conversation history
            st.session_state.conversation_history.append({"role": "user", "content": user_question})
            st.session_state.conversation_history.append({"role": "bot", "content": response['answer']})

            # Update chat history in session state
            st.session_state.chat_history = response['chat_history']

            # Convert chatbot response to audio (TTS) and create unique audio files for each response
            response_audio_file = f"response_audio_{len(st.session_state.conversation_history)//2}.mp3"
            if language == "Urdu":
                tts = gTTS(text=response['answer'], lang='ur')  # Text-to-Speech in Urdu
            else:
                tts = gTTS(text=response['answer'], lang='en')  # Text-to-Speech in English

            # Save the audio file with a unique name
            tts.save(response_audio_file)
            st.write(f"Audio for question {len(st.session_state.conversation_history)//2} generated.")

    # Display conversation and play audio files separately for each question/response
    display_conversation_and_audio()

    # Add download button for chat history
    if st.session_state.conversation_history:
        formatted_chat = format_chat_for_download(st.session_state.conversation_history)
        st.download_button(
            label="Download Chat History",
            data=formatted_chat,
            file_name="chat_history.txt",
            mime="text/plain"
        ) 
