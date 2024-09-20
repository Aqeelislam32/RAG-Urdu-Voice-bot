# RAG-Urdu-Voice-bot

 Here’s the description of the Streamlit chatbot app with **emojis**:

This Streamlit app is a **RAG Urdu voice-chatbot** powered by **Google Generative AI** (Gemini model) 🌟, perfect for conversational question answering in **Urdu** and **English** 🌐.

### Key Features:

1. **Speech-to-Text Input** 🎤:
   - Users can give voice input in **Urdu** or **English**, which is converted to text using the `speech_to_text()` function 📝.
   - The recognized text is shown for confirmation ✔️.

2. **Language Selection** 🗣️🌍:
   - A **sidebar** lets users choose between **Urdu** 🇵🇰 and **English** 🇬🇧. The chatbot responds in the selected language.
   
3. **AI-Powered Responses** 🤖💬:
   - The app uses **Google’s Gemini AI model** to generate **intelligent responses** based on the user’s input 💡.
   - For **General Q/A**, user voice inputs are processed, and the bot generates appropriate answers 📚.

4. **Text-to-Speech Output** 🔊:
   - The chatbot’s replies are converted to speech using `gTTS` 🗣️, and the audio is played automatically 🎧. Responses are saved and played back as audio 🔁.

5. **Conversation History** 🕰️:
   - Both user inputs and AI responses are displayed in a chat-like interface 💬, styled for better readability 🎨.
   - Users can download their conversation as a **text file** 📂 for later use.

6. **Document Q/A** 📄❓:
   - In this mode, users can upload **PDF** or **DOCX** files 📥. The app extracts the text 📝 and processes it for document-based queries.
   - The document’s content is split into smaller chunks 📚, making it easier for the AI to find relevant information when answering questions 🔍.

7. **Vector Store for Document Q/A** 🧠:
   - The app uses **FAISS** to create a **vector store** of the document’s text, allowing for efficient retrieval based on user questions ⚡.

8. **Chat Interface** 💻:
   - A custom-designed chat interface shows user questions and AI responses with **stylish formatting** ✨.
   - Users can also **clear the chat** 🧹 and **download** the conversation for future reference 📄.

### Additional Features:
- **Clear Chat Button** 🧽: A button to reset the chat history 🔄.
- **File Uploader** 🗂️: Upload **PDF** or **Word** files for document-based Q/A 📤.
- **API Integration** 🔑: Requires a **Google API key** to use the **Gemini model** for response generation 🌐.

This app merges **speech recognition**, **AI conversations**, and **document analysis** into a fun and interactive **multilingual chatbot experience** 🤩!
