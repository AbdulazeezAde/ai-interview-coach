# AI Interview Coach

Practice your interview skills with real-time feedback powered by AI!

## Features
- Real-time voice and text interview practice
- Speech-to-text transcription using Gemini API
- Text-to-speech for questions using Hugging Face Bark TTS
- AI-powered feedback and scoring (STAR method, strengths, improvements)
- Interview history and progress tracking
- Secure Gemini API key management via `.env` file

## Setup
1. Clone this repository.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file in the project root and add your Gemini API key:
   ```env
   GEMINI_API_KEY=your_api_key_here
   ```
4. Run the app:
   ```bash
   streamlit run app.py
   ```

## Usage
- Enter your Gemini API key in the sidebar or use the `.env` file.
- Select or record your interview response.
- Get instant transcription and AI feedback.
- Review your interview history and track your progress.

## Requirements
See `requirements.txt` for all dependencies.

## Notes
- Audio transcription uses Gemini API and expects WAV format.
- Text-to-speech uses Hugging Face Bark ("suno/bark-small").
- For best results, use a clear microphone and a quiet environment.

## License
MIT
