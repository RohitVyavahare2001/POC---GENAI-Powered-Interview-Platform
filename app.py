import streamlit as st
import requests
import time
import base64
from audio_recorder_streamlit import audio_recorder
import io

# Constants
BACKEND_URL = "http://localhost:8000"

st.set_page_config(page_title="AI Interviewer")

def play_audio(audio_base64):
    try:
        audio_bytes = base64.b64decode(audio_base64)
        st.audio(audio_bytes, format='audio/mp3')
    except Exception as e:
        st.error(f"Error playing audio: {str(e)}")

def check_server_connection():
    try:
        requests.get(BACKEND_URL)
        return True
    except requests.exceptions.ConnectionError:
        st.error(" Cannot connect to the backend server. Please make sure the backend server is running (python backend.py)")
        return False

def main():
    st.title(" AI Interview System")
    
    # Initialize session state
    if "interview_started" not in st.session_state:
        st.session_state.interview_started = False
    if "current_question" not in st.session_state:
        st.session_state.current_question = 0
    if "interview_completed" not in st.session_state:
        st.session_state.interview_completed = False

    # Check server connection before proceeding
    if not check_server_connection():
        st.stop()

    if not st.session_state.interview_started:
        st.write("Welcome to your AI interview! Please upload your resume to begin.")
        
        uploaded_file = st.file_uploader("Upload your resume (PDF)", type=['pdf'])
        
        if uploaded_file:
            if not uploaded_file.name.endswith('.pdf'):
                st.error("Please upload a PDF file")
                return
                
            if st.button("Start Interview"):
                with st.spinner("Processing your resume..."):
                    try:
                        files = {"file": uploaded_file}
                        response = requests.post(f"{BACKEND_URL}/upload-resume/", files=files)
                        
                        if response.status_code == 200:
                            st.success("Resume processed successfully!")
                            st.session_state.interview_started = True
                            time.sleep(2)
                            st.experimental_rerun()
                        else:
                            error_msg = response.json().get('detail', 'Unknown error occurred')
                            st.error(f"Error processing resume: {error_msg}")
                    except requests.exceptions.RequestException as e:
                        st.error(f"Error connecting to server: {str(e)}")
    
    elif not st.session_state.interview_completed:
        st.write("### Interview in Progress")
        
        try:
            # Get next question
            response = requests.get(f"{BACKEND_URL}/next-question/")
            
            if response.status_code != 200:
                error_msg = response.json().get('detail', 'Unknown error occurred')
                st.error(f"Error getting next question: {error_msg}")
                return
                
            data = response.json()
            
            if "completed" in data and data["completed"]:
                st.session_state.interview_completed = True
                st.experimental_rerun()
            else:
                st.write(f"Question {data['question_number']}/{data['total_questions']}")
                st.write(f"**{data['question']}**")
                
                # Play question audio
                play_audio(data['audio'])
                
                # Record answer
                st.write("Record your answer:")
                audio_bytes = audio_recorder()
                
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/wav")
                    
                    if st.button("Submit Answer"):
                        with st.spinner("Processing your answer..."):
                            try:
                                files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
                                response = requests.post(f"{BACKEND_URL}/transcribe/", files=files)
                                
                                if response.status_code == 200:
                                    transcript = response.json()["transcript"]
                                    st.success("Answer submitted successfully!")
                                    st.write("Your answer (transcribed):", transcript)
                                    time.sleep(2)
                                    st.experimental_rerun()
                                else:
                                    error_msg = response.json().get('detail', 'Unknown error occurred')
                                    st.error(f"Error processing audio: {error_msg}")
                            except requests.exceptions.RequestException as e:
                                st.error(f"Error connecting to server: {str(e)}")
        except requests.exceptions.RequestException as e:
            st.error(f"Error connecting to server: {str(e)}")
    
    else:
        st.write("### Interview Completed!")
        
        if st.button("Generate Feedback"):
            with st.spinner("Analyzing your interview..."):
                try:
                    response = requests.get(f"{BACKEND_URL}/generate-feedback/")
                    
                    if response.status_code == 200:
                        feedback = response.json()
                        
                        st.write("## Interview Results")
                        st.write(f"### Score: {feedback['score']}/100")
                        
                        st.write("### Technical Assessment")
                        st.write(feedback['technical_feedback'])
                        
                        st.write("### Communication Skills")
                        st.write(feedback['communication_feedback'])
                        
                        st.write("### Areas for Improvement")
                        st.write(feedback['improvements'])
                    else:
                        error_msg = response.json().get('detail', 'Unknown error occurred')
                        st.error(f"Error generating feedback: {error_msg}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error connecting to server: {str(e)}")
        
        if st.button("Start New Interview"):
            st.session_state.interview_started = False
            st.session_state.interview_completed = False
            st.session_state.current_question = 0
            st.experimental_rerun()

if __name__ == "__main__":
    main()
