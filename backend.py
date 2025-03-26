from fastapi import FastAPI, UploadFile, File, HTTPException
from deepgram import Deepgram
import uvicorn
import asyncio
import json
import os
import io
from dotenv import load_dotenv
from groq import Groq
from gtts import gTTS
import PyPDF2
from fastapi.responses import FileResponse
import base64
from fastapi.middleware.cors import CORSMiddleware
import logging
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize FastAPI
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Keys
DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not DEEPGRAM_API_KEY:
    raise ValueError(" Deepgram API Key is missing!")
if not GROQ_API_KEY:
    raise ValueError(" Groq API Key is missing!")

# Initialize clients
dg_client = Deepgram(DEEPGRAM_API_KEY)
groq_client = Groq(api_key=GROQ_API_KEY)

# Constants
GROQ_MODEL = "deepseek-r1-distill-llama-70b"

class InterviewState:
    def __init__(self):
        self.questions = []
        self.answers = []
        self.current_question = 0
        self.resume_text = ""
        self.conversation_history = []

interview_state = InterviewState()

def extract_question_from_response(response_text):
    """Extract a question from the model's response text."""
    # Try to parse as JSON first
    try:
        response_json = json.loads(response_text)
        if isinstance(response_json, dict) and "question" in response_json:
            return response_json["question"]
    except json.JSONDecodeError:
        pass
    
    # Try to find a question in quotes
    match = re.search(r'"([^"]+\?)"', response_text)
    if match:
        return match.group(1)
    
    # Try to find any sentence ending with a question mark
    match = re.search(r'([^.!?]+\?)', response_text)
    if match:
        return match.group(1).strip()
    
    # If all else fails, return the whole response
    return response_text.strip()

@app.post("/upload-resume/")
async def upload_resume(file: UploadFile = File(...)):
    try:
        logger.info(f"Received resume upload request: {file.filename}")
        
        if not file.filename.endswith('.pdf'):
            raise HTTPException(status_code=400, detail="Only PDF files are supported")
        
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Empty file received")
            
        pdf_file = io.BytesIO(content)
        
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
        except Exception as e:
            logger.error(f"Error reading PDF: {str(e)}")
            raise HTTPException(status_code=400, detail="Invalid PDF file")
        
        text = ""
        try:
            for page in pdf_reader.pages:
                text += page.extract_text()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text content found in PDF")
        
        interview_state.resume_text = text
        interview_state.conversation_history = []
        interview_state.questions = []
        interview_state.answers = []
        interview_state.current_question = 0
        
        logger.info("Generating first question using Groq model")
        try:
            # Generate first question based on resume
            prompt = f"""Based on the following resume, generate a technical interview question:
            Resume: {text}
            
            Generate a single technical question that assesses the candidate's most important skill from their resume.
            The question should be specific and directly related to their experience.
            Respond with just the question, ending with a question mark."""
            
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert technical interviewer. Generate relevant technical questions based on the candidate's resume. Keep responses concise and focused."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            try:
                response_content = completion.choices[0].message.content.strip()
                logger.info(f"Raw model response: {response_content}")
                
                # Extract the question from the response
                first_question = extract_question_from_response(response_content)
                
                if not first_question.endswith('?'):
                    first_question += '?'
                
                interview_state.questions.append(first_question)
                interview_state.conversation_history.append({"role": "assistant", "content": first_question})
                
                logger.info(f"Successfully generated first question: {first_question}")
                return {"message": "Resume processed successfully", "question": first_question}
                
            except Exception as e:
                logger.error(f"Error processing model response: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")
            
        except Exception as e:
            logger.error(f"Error with Groq API: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error calling Groq API: {str(e)}")
            
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

@app.get("/next-question/")
async def get_next_question():
    try:
        if not interview_state.questions:
            raise HTTPException(status_code=400, detail="No questions available. Please upload a resume first.")
        
        if interview_state.current_question >= 3:
            return {"message": "Interview completed", "completed": True}
        
        question = interview_state.questions[interview_state.current_question]
        
        # Generate TTS for the question
        try:
            tts = gTTS(text=question, lang='en')
            audio_file = io.BytesIO()
            tts.write_to_fp(audio_file)
            audio_file.seek(0)
            
            # Convert to base64
            audio_base64 = base64.b64encode(audio_file.read()).decode()
            
            return {
                "question": question,
                "audio": audio_base64,
                "question_number": interview_state.current_question + 1,
                "total_questions": 3
            }
        except Exception as e:
            logger.error(f"Error generating audio: {str(e)}")
            raise HTTPException(status_code=500, detail="Error generating audio for question")
            
    except Exception as e:
        logger.error(f"Error in next_question: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No audio file provided")
            
        audio_data = await file.read()
        if not audio_data:
            raise HTTPException(status_code=400, detail="Empty audio file received")
            
        try:
            # Transcribe audio using Deepgram
            response = await dg_client.transcription.prerecorded(
                {"buffer": audio_data, "mimetype": file.content_type},
                {"punctuate": True}
            )
            
            transcript = response["results"]["channels"][0]["alternatives"][0]["transcript"]
            
            # Add answer to conversation history
            interview_state.answers.append(transcript)
            interview_state.conversation_history.append({"role": "user", "content": transcript})
            
            # Generate next question if not the last answer
            if interview_state.current_question < 2:  # We want 3 questions total
                try:
                    # Prepare conversation context
                    conversation_text = "\n".join([
                        f"Q: {interview_state.questions[i]}\nA: {interview_state.answers[i]}"
                        for i in range(len(interview_state.answers))
                    ])
                    
                    prompt = f"""Based on the following resume and previous conversation, generate the next technical interview question.
                    
                    Resume: {interview_state.resume_text}
                    
                    Previous conversation:
                    {conversation_text}
                    
                    Generate a single follow-up technical question that builds upon the candidate's previous answers.
                    The question should be specific and directly related to their responses.
                    Respond with just the question, ending with a question mark."""
                    
                    completion = groq_client.chat.completions.create(
                        model=GROQ_MODEL,
                        messages=[
                            {"role": "system", "content": "You are an expert technical interviewer. Generate relevant follow-up questions based on the conversation context. Keep responses concise and focused."},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=500
                    )
                    
                    response_content = completion.choices[0].message.content.strip()
                    logger.info(f"Raw model response for next question: {response_content}")
                    
                    next_question = extract_question_from_response(response_content)
                    
                    if not next_question.endswith('?'):
                        next_question += '?'
                    
                    interview_state.questions.append(next_question)
                    interview_state.conversation_history.append({"role": "assistant", "content": next_question})
                    logger.info(f"Generated next question: {next_question}")
                    
                except Exception as e:
                    logger.error(f"Error generating next question: {str(e)}")
                    raise HTTPException(status_code=500, detail="Error generating next question")
            
            interview_state.current_question += 1
            return {"transcript": transcript}
            
        except Exception as e:
            logger.error(f"Deepgram transcription error: {str(e)}")
            raise HTTPException(status_code=500, detail="Error transcribing audio")
            
    except Exception as e:
        logger.error(f"Error in transcribe: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate-feedback/")
async def generate_feedback():
    try:
        if len(interview_state.answers) == 0:
            raise HTTPException(status_code=400, detail="No interview answers to evaluate")
        
        # Prepare the conversation history
        conversation = []
        for i in range(len(interview_state.answers)):
            conversation.append(f"Q: {interview_state.questions[i]}")
            conversation.append(f"A: {interview_state.answers[i]}")
        
        prompt = f"""Based on the following interview conversation, provide a detailed evaluation:
        Resume: {interview_state.resume_text}
        
        Interview:
        {chr(10).join(conversation)}
        
        Provide an evaluation in JSON format with the following structure:
        {{
            "score": (number between 0-100),
            "technical_feedback": "detailed assessment of technical knowledge",
            "communication_feedback": "assessment of communication skills",
            "improvements": "specific areas for improvement"
        }}"""
        
        try:
            completion = groq_client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": "You are an expert technical interviewer providing detailed feedback on candidate interviews. Always respond with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            try:
                response_content = completion.choices[0].message.content.strip()
                logger.info(f"Raw feedback response: {response_content}")
                
                # Try to find a JSON object in the response
                start_idx = response_content.find('{')
                end_idx = response_content.rfind('}')
                
                if start_idx != -1 and end_idx != -1:
                    json_str = response_content[start_idx:end_idx + 1]
                    feedback = json.loads(json_str)
                else:
                    raise ValueError("Could not find JSON object in response")
                
                required_keys = ["score", "technical_feedback", "communication_feedback", "improvements"]
                if not all(key in feedback for key in required_keys):
                    raise ValueError("Missing required keys in feedback response")
                    
                return feedback
                
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing feedback response: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Error parsing feedback: {str(e)}")
            except ValueError as e:
                logger.error(f"Invalid feedback format: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Invalid feedback format: {str(e)}")
                
        except Exception as e:
            logger.error(f"Error with Groq API: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error generating feedback: {str(e)}")
            
    except Exception as e:
        logger.error(f"Error in generate_feedback: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Run server
if __name__ == "__main__":
    logger.info("Starting FastAPI server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
