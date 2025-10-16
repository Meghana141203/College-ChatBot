import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import speech_recognition as sr
from gtts import gTTS
import tempfile
from audio_recorder_streamlit import audio_recorder
import io
from pydub import AudioSegment

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


#####################################################
# FUNCTION TO READ THE TEXT FROM PDFs
#####################################################
def get_pdf_text(docs):
    text = ""
    for pdf in docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


#####################################################
# FUNCTION TO DIVIDE THE TEXT FROM PDFs INTO SMALLER CHUNKS
#####################################################
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


#####################################################
# GET TEXT EMBEDDINGS USING GEMINI MODEL & PREPARING THE VECTOR DATABASE
#####################################################
def get_embeddings(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectors = FAISS.from_texts(chunks, embedding=embeddings)
    vectors.save_local("faiss_index")


#####################################################
# PREPARING THE CONVERSATIONAL CHAIN
#####################################################
def get_conversational_chain():
    prompt_temp = """
    You are a helpful and friendly assistant. Respond in a natural, conversational way as if you're having a casual conversation. 
    Use the following information to help answer the question, but make your response sound natural and human-like.
    Don't mention that you're reading from documents or reference the context directly.
    
    If you don't know something, just say something like "I'm not sure about that" or "I don't have that information, but is there something else I can help you with?"
    
    Context:\n{context}\n
    Question:\n{question}\n
    """
    prompt = PromptTemplate(
        template=prompt_temp, input_variables=["context", "question"]
    )

    # Increased temperature for more natural responses
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.8)
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain


#####################################################
# PREPARING THE MODEL'S RESPONSE
#####################################################
def speech_to_text(audio_bytes):
    try:
        # Convert audio bytes to AudioSegment
        audio = AudioSegment.from_wav(io.BytesIO(audio_bytes))

        # Export as WAV file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            audio.export(temp_audio.name, format="wav")

            # Initialize recognizer
            recognizer = sr.Recognizer()
            with sr.AudioFile(temp_audio.name) as source:
                audio_data = recognizer.record(source)
                text = recognizer.recognize_google(audio_data)
                return text
    except Exception as e:
        st.error(f"Error in speech recognition: {str(e)}")
        return None
    finally:
        if "temp_audio" in locals():
            os.unlink(temp_audio.name)


def text_to_speech(text):
    try:
        # Clean the text for better speech output
        text = text.replace("*", "").replace("#", "").strip()
        # Remove any markdown or special characters that might affect speech

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_file:
            temp_path = temp_file.name

        tts = gTTS(text=text, lang="en")
        tts.save(temp_path)

        with open(temp_path, "rb") as audio_file:
            audio_bytes = audio_file.read()

        return audio_bytes

    except Exception as e:
        st.error(f"Error in text-to-speech conversion: {str(e)}")
        return None
    finally:
        try:
            if "temp_path" in locals():
                os.unlink(temp_path)
        except Exception:
            pass


def get_response(user_question):
    try:
        embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        new_db = FAISS.load_local(
            "faiss_index", embedding, allow_dangerous_deserialization=True
        )
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain(
            {"input_documents": docs, "question": user_question},
            return_only_outputs=True,
        )
        response_text = response["output_text"]

        # Convert to speech and play
        audio_bytes = text_to_speech(response_text)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")

    except RuntimeError as e:
        audio_bytes = text_to_speech(
            "I'm having trouble accessing the document. Please make sure it's properly processed."
        )
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mp3")


def process_default_pdf():
    try:
        with open("data.pdf", "rb") as file:
            text = get_pdf_text([file])
            chunks = get_text_chunks(text)
            get_embeddings(chunks)
            return True
    except Exception as e:
        st.error(
            "Error processing default PDF. Please ensure 'data.pdf' exists in the application directory."
        )
        return False


#####################################################
# CREATING THE FRONT END APPLICATION
#####################################################
def main():
    st.title("College Assistant 🎙️")

    # Process default PDF on startup
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = process_default_pdf()

    # Simple voice interface
    st.write("Click the microphone button and ask your question:")
    audio_bytes = audio_recorder()

    if audio_bytes:
        st.audio(audio_bytes, format="audio/wav")
        user_question = speech_to_text(audio_bytes)
        if user_question:
            get_response(user_question)


if __name__ == "__main__":
    main()
