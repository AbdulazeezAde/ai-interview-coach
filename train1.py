import streamlit as st
import google.generativeai as genai
import torch
import torchaudio
import tempfile
import os
import random
import json
from datetime import datetime
import numpy as np
from transformers import (
    AutoProcessor,
    AutoModelForCTC,
    pipeline
)
import scipy.io.wavfile
import soundfile as sf
from datasets import load_dataset
import librosa
from dotenv import load_dotenv

# Configure page
st.set_page_config(
    page_title="AI Interview Coach",
    page_icon="🎤",
    layout="wide"
)

# Initialize session state
if 'interview_history' not in st.session_state:
    st.session_state.interview_history = []
if 'current_question' not in st.session_state:
    st.session_state.current_question = None
if 'feedback_data' not in st.session_state:
    st.session_state.feedback_data = None
if 'stt_model' not in st.session_state:
    st.session_state.stt_model = None
if 'tts_model' not in st.session_state:
    st.session_state.tts_model = None

# Common interview questions
INTERVIEW_QUESTIONS = [
    "Tell me about yourself.",
    "What are your greatest strengths?",
    "What is your biggest weakness?",
    "Why do you want to work here?",
    "Where do you see yourself in 5 years?",
    "Why are you leaving your current job?",
    "Describe a challenging situation you faced and how you handled it.",
    "What motivates you?",
    "How do you handle stress and pressure?",
    "What are your salary expectations?",
    "Do you have any questions for us?",
    "Tell me about a time you failed and what you learned.",
    "How do you work in a team?",
    "What makes you unique?",
    "Describe your ideal work environment.",
    "How do you prioritize your work?",
    "Tell me about a time you disagreed with your boss.",
    "What's your leadership style?",
    "How do you handle criticism?",
    "What's your biggest professional achievement?"
]

@st.cache_resource
def load_stt_model():
    """Load Speech-to-Text model from Hugging Face"""
    try:
        processor = AutoProcessor.from_pretrained("facebook/wav2vec2-base-960h")
        model = AutoModelForCTC.from_pretrained("facebook/wav2vec2-base-960h")
        return processor, model
    except Exception as e:
        st.error(f"Error loading STT model: {str(e)}")
        return None, None

@st.cache_resource
def load_tts_model():
    """Load a faster TTS model - using Bark which is reasonably fast"""
    try:
        # Using Bark for TTS as it's faster than SpeechT5
        synthesiser = pipeline(
            "text-to-speech", 
            model="suno/bark-small",
            torch_dtype="auto",
            device="cpu"  # Change to "cuda" if you have GPU
        )
        return synthesiser
    except Exception as e:
        st.error(f"Error loading TTS model: {str(e)}")
        return None

def setup_apis():
    """Setup API configurations"""
    load_dotenv()
    gemini_key = os.getenv("GEMINI_API_KEY")
    # Gemini API
    #gemini_key = st.sidebar.text_input("Gemini API Key", type="password")
    if gemini_key:
        genai.configure(api_key=gemini_key)
        st.session_state.gemini_client = genai.GenerativeModel('gemini-2.0-flash')
        
    
    # # Model loading section
    # st.sidebar.header("🤖 Model Loading")
    
    # Load models on first run
    if st.session_state.stt_model is None:
        processor, model = load_stt_model()
        if processor and model:
            st.session_state.stt_model = (processor, model)
            st.sidebar.success("STT Model loaded!")
        else:
            st.sidebar.error("Failed to load STT model")
    
    if st.session_state.tts_model is None:
        tts_model = load_tts_model()
        if tts_model:
            st.session_state.tts_model = tts_model
            st.sidebar.success("TTS Model loaded!")
        else:
            st.sidebar.error("Failed to load TTS model")
    
    # Model status
    # st.sidebar.subheader("Model Status")
    # st.sidebar.write(f"STT Model: {'✅ Loaded' if st.session_state.stt_model else '❌ Not loaded'}")
    # st.sidebar.write(f"TTS Model: {'✅ Loaded' if st.session_state.tts_model else '❌ Not loaded'}")
    
    return gemini_key

def transcribe_audio_data(audio_data):
    """Transcribe recorded audio data using Hugging Face Wav2Vec2 model"""
    try:
        if not st.session_state.stt_model:
            return "Please load the STT model first using the sidebar."
        
        processor, model = st.session_state.stt_model
        
        # Save audio data to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            tmp_file.write(audio_data.getvalue())
            tmp_file_path = tmp_file.name
        
        # Load and preprocess audio
        try:
            # Load audio with librosa, ensuring 16kHz sample rate
            audio, sr = librosa.load(tmp_file_path, sr=16000)
            
            # Ensure audio is not empty
            if len(audio) == 0:
                return "Error: Audio recording appears to be empty."
            
            # Process audio through the model
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            
            # Generate transcription
            with torch.no_grad():
                logits = model(inputs.input_values).logits
            
            # Decode the prediction
            predicted_ids = torch.argmax(logits, dim=-1)
            transcription = processor.batch_decode(predicted_ids)[0]
            
            # Clean up temp file
            os.unlink(tmp_file_path)
            
            # Return cleaned transcription
            return transcription.strip() if transcription.strip() else "No speech detected in recording."
        
        except Exception as audio_error:
            # Clean up temp file
            if os.path.exists(tmp_file_path):
                os.unlink(tmp_file_path)
            return f"Audio processing error: {str(audio_error)}"
    
    except Exception as e:
        return f"Transcription error: {str(e)}"

def generate_speech_hf(text):
    """Generate speech using Hugging Face TTS model"""
    try:
        if not st.session_state.tts_model:
            return None, "Please load the TTS model first using the sidebar."
        
        tts_pipeline = st.session_state.tts_model
        
        # Limit text length to prevent errors
        if len(text) > 300:
            text = text[:300] + "..."
        
        # Generate speech
        speech = tts_pipeline(text)
        
        # Extract audio data and sample rate
        audio_data = speech["audio"]
        sample_rate = speech["sampling_rate"]
        
        # Convert to numpy array if needed
        if isinstance(audio_data, torch.Tensor):
            audio_data = audio_data.cpu().numpy()
        
        # Ensure audio data is in the right format
        if audio_data.ndim > 1:
            audio_data = audio_data.squeeze()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
            sf.write(tmp_file.name, audio_data, sample_rate)
            return tmp_file.name, None
    
    except Exception as e:
        return None, f"TTS error: {str(e)}"

def analyze_response(response_text, question, gemini_key):
    """Analyze response using Gemini"""
    try:
        if not gemini_key:
            return {"error": "Please provide Gemini API key for response analysis."}
        
        if not st.session_state.gemini_client:
            return {"error": "Gemini client not initialized."}
        
        prompt = f"""
        Analyze this interview response and provide detailed feedback:
        
        Question: {question}
        Response: {response_text}
        
        Please analyze and score the response on the following criteria (1-10 scale):
        
        1. CLARITY: How clear and well-structured is the response?
        2. CONFIDENCE: Does the response sound confident and assertive?
        3. RELEVANCE: How well does it answer the question?
        4. GRAMMAR: Are there grammatical errors or awkward phrasing?
        5. FILLER_WORDS: Count and identify filler words (um, uh, like, you know, etc.)
        6. EMOTIONAL_TONE: What emotions are conveyed? (positive/negative/neutral)
        7. COMPLETENESS: Is the response complete and comprehensive?
        8. PROFESSIONALISM: How professional is the tone and content?
        9. SPECIFICITY: Are there concrete examples and details?
        10. COMMUNICATION_SKILLS: Overall communication effectiveness?
        
        Please return your analysis in this JSON format:
        {{
            "scores": {{
                "clarity": score,
                "confidence": score,
                "relevance": score,
                "grammar": score,
                "filler_words": score,
                "emotional_tone": score,
                "completeness": score,
                "professionalism": score,
                "specificity": score,
                "communication_skills": score
            }},
            "overall_score": average_score,
            "filler_word_count": number,
            "identified_fillers": ["list", "of", "fillers"],
            "strengths": ["strength1", "strength2", "strength3"],
            "improvements": ["improvement1", "improvement2", "improvement3"],
            "detailed_feedback": "comprehensive feedback paragraph",
            "star_method_used": true/false,
            "recommended_improvements": ["specific suggestion1", "specific suggestion2"]
        }}
        
        Be constructive and specific in your feedback. Look for use of the STAR method (Situation, Task, Action, Result).
        """
        
        response = st.session_state.gemini_client.generate_content(prompt)
        
        # Try to parse JSON from response
        response_text = response.text
        
        # Extract JSON from response (in case there's extra text)
        json_start = response_text.find('{')
        json_end = response_text.rfind('}') + 1
        
        if json_start != -1 and json_end != -1:
            json_str = response_text[json_start:json_end]
            return json.loads(json_str)
        else:
            # Fallback if JSON parsing fails
            return {
                "error": "Could not parse AI response",
                "raw_response": response_text
            }
    
    except Exception as e:
        return {"error": f"Analysis error: {str(e)}"}

def display_feedback(feedback_data):
    """Display feedback in an organized way"""
    if "error" in feedback_data:
        st.error(f"Error: {feedback_data['error']}")
        return
    
    # Overall Score with color coding
    overall_score = feedback_data.get("overall_score", 0)
    score_color = "green" if overall_score >= 7 else "orange" if overall_score >= 5 else "red"
    st.markdown(f"<h2 style='color: {score_color}'>Overall Score: {overall_score}/10</h2>", unsafe_allow_html=True)
    
    # Score breakdown
    st.subheader("📊 Detailed Score Breakdown")
    scores = feedback_data.get("scores", {})
    
    # Create three columns for better layout
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("🎯 Clarity", f"{scores.get('clarity', 0)}/10")
        st.metric("💪 Confidence", f"{scores.get('confidence', 0)}/10")
        st.metric("📝 Relevance", f"{scores.get('relevance', 0)}/10")
        st.metric("✍️ Grammar", f"{scores.get('grammar', 0)}/10")
    
    with col2:
        st.metric("🚫 No use of Filler Words", f"{scores.get('filler_words', 0)}/10")
        st.metric("😊 Emotional Tone", f"{scores.get('emotional_tone', 0)}/10")
        st.metric("✅ Completeness", f"{scores.get('completeness', 0)}/10")
        st.metric("🎩 Professionalism", f"{scores.get('professionalism', 0)}/10")
    
    with col3:
        st.metric("🔍 Specificity", f"{scores.get('specificity', 0)}/10")
        st.metric("💬 Communication", f"{scores.get('communication_skills', 0)}/10")
    
    # STAR method indicator
    star_used = feedback_data.get("star_method_used", False)
    if star_used:
        st.success("⭐ Great! You used the STAR method (Situation, Task, Action, Result)")
    else:
        st.info("💡 Consider using the STAR method for better structured responses")
    
    # Filler words analysis
    filler_count = feedback_data.get("filler_word_count", 0)
    identified_fillers = feedback_data.get("identified_fillers", [])
    
    if filler_count > 0:
        st.warning(f"🚨 Found {filler_count} filler words: {', '.join(identified_fillers)}")
    else:
        st.success("✅ No filler words detected! Great job!")
    
    # Strengths, improvements, and recommendations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💪 Strengths")
        strengths = feedback_data.get("strengths", [])
        for strength in strengths:
            st.write(f"• {strength}")
    
    with col2:
        st.subheader("🎯 Areas for Improvement")
        improvements = feedback_data.get("improvements", [])
        for improvement in improvements:
            st.write(f"• {improvement}")
    
    with col3:
        st.subheader("📋 Recommended Actions")
        recommendations = feedback_data.get("recommended_improvements", [])
        for rec in recommendations:
            st.write(f"• {rec}")
    
    # Detailed feedback
    st.subheader("📝 Detailed Feedback")
    detailed_feedback = feedback_data.get("detailed_feedback", "No detailed feedback available.")
    st.markdown(f"*{detailed_feedback}*")

def main():
    st.title("AI-Based Interview Coach")
    st.markdown("**Practice interviews with real-time feedback using AI!**")
    
    # API Setup
    gemini_key = setup_apis()
    
    # Main interface
    st.header("🎯 Mock Interview Session")
    
    # Question selection
    col1, col2 = st.columns([3, 1])
    
    with col1:
        question_type = st.selectbox("Choose question type:", ["Random Question", "Select Specific Question"])
    
    with col2:
        if st.button("🎲 New Question"):
            if question_type == "Random Question":
                st.session_state.current_question = random.choice(INTERVIEW_QUESTIONS)
            else:
                st.session_state.current_question = None
    
    # Question display and TTS
    if question_type == "Select Specific Question":
        selected_question = st.selectbox("Select a question:", INTERVIEW_QUESTIONS)
        st.session_state.current_question = selected_question
    
    if st.session_state.current_question:
        st.info(f"**Question:** {st.session_state.current_question}")
        
        # Text-to-Speech for question
        if st.button("🔊 Listen to Question"):
            if st.session_state.tts_model:
                with st.spinner("Generating speech..."):
                    audio_file, error = generate_speech_hf(st.session_state.current_question)
                    if audio_file:
                        st.audio(audio_file)
                        st.success("Audio generated successfully!")
                        os.unlink(audio_file)
                    else:
                        st.error(f"TTS Error: {error}")
            else:
                st.error("Please load the TTS model first using the sidebar.")
        
        # Response input methods
        st.subheader("🎙️ Your Response")
        
        tab1, tab2 = st.tabs(["🎤 Voice Response", "⌨️ Text Response"])
        
        with tab1:
            st.markdown("**Record your response directly from your microphone:**")
            
            # Audio recording using st.audio_input
            audio_data = st.audio_input("🎙️ Record your answer")
            
            if audio_data is not None:
                st.audio(audio_data, format='audio/wav')
                
                if st.button("🎤 Analyze"):
                    if not st.session_state.stt_model:
                        st.error("Please load the STT model first using the sidebar.")
                    elif not gemini_key:
                        st.error("Please provide Gemini API key for response analysis.")
                    else:
                        with st.spinner("Transcribing audio..."):
                            transcribed_text = transcribe_audio_data(audio_data)
                        
                        st.subheader("📝 Transcribed Response")
                        st.write(transcribed_text)
                        
                        if not transcribed_text.startswith("Error:") and not transcribed_text.startswith("Please load") and not transcribed_text.startswith("Transcription error"):
                            with st.spinner("Analyzing response..."):
                                feedback_data = analyze_response(
                                    transcribed_text, 
                                    st.session_state.current_question, 
                                    gemini_key
                                )
                            
                            st.session_state.feedback_data = feedback_data
                            
                            # Store in history
                            st.session_state.interview_history.append({
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "question": st.session_state.current_question,
                                "response": transcribed_text,
                                "feedback": feedback_data,
                                "method": "voice"
                            })
                        else:
                            st.error("Transcription failed. Please try again with a clearer recording.")
        
        with tab2:
            text_response = st.text_area("Type your response here:", height=150, placeholder="Start typing your answer...")
            
            if st.button("📝 Analyze Text Response") and text_response:
                if not gemini_key:
                    st.error("Please provide Gemini API key for response analysis.")
                else:
                    with st.spinner("Analyzing response..."):
                        feedback_data = analyze_response(
                            text_response, 
                            st.session_state.current_question, 
                            gemini_key
                        )
                    
                    st.session_state.feedback_data = feedback_data
                    
                    # Store in history
                    st.session_state.interview_history.append({
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "question": st.session_state.current_question,
                        "response": text_response,
                        "feedback": feedback_data,
                        "method": "text"
                    })
    
    # Display feedback
    if st.session_state.feedback_data:
        st.header("📊 AI Feedback Analysis")
        display_feedback(st.session_state.feedback_data)
    
    # Interview history
    if st.session_state.interview_history:
        st.header("📚 Interview History")
        
        # Show progress chart
        if len(st.session_state.interview_history) > 1:
            scores = [session['feedback'].get('overall_score', 0) for session in st.session_state.interview_history if 'overall_score' in session.get('feedback', {})]
            if scores:
                st.line_chart(scores)
        
        # Show recent sessions
        for i, session in enumerate(reversed(st.session_state.interview_history[-5:])):  # Show last 5
            with st.expander(f"Session {len(st.session_state.interview_history) - i} - {session['timestamp']} ({session['method']})"):
                st.write(f"**Question:** {session['question']}")
                st.write(f"**Response:** {session['response'][:300]}...")
                
                if 'overall_score' in session['feedback']:
                    st.write(f"**Score:** {session['feedback']['overall_score']}/10")
                    
                    # Quick stats
                    scores = session['feedback'].get('scores', {})
                    if scores:
                        st.write(f"**Top Strength:** {max(scores.items(), key=lambda x: x[1])[0].replace('_', ' ').title()}")
                        st.write(f"**Needs Work:** {min(scores.items(), key=lambda x: x[1])[0].replace('_', ' ').title()}")
    
    # Tips section
    st.sidebar.header("💡 Interview Tips")
    st.sidebar.markdown("""
    **STAR Method:**
    - **Situation:** Set the context
    - **Task:** Describe what needed to be done
    - **Action:** Explain what you did
    - **Result:** Share the outcome
    
    **Best Practices:**
    - Speak clearly and confidently
    - Avoid filler words (um, uh, like)
    - Be specific with examples
    - Show enthusiasm
    - Ask thoughtful questions
    - Practice active listening
    - Research the company
    
    **Audio Recording Tips:**
    - Click the microphone icon to start recording
    - Speak clearly and at normal pace
    - Keep responses between 30-90 seconds
    - Use a quiet environment for best results
    - Wait for the recording to complete before transcribing
    """)

if __name__ == "__main__":
    main()