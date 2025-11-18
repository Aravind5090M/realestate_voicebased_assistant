import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import numpy as np
import av
import requests
import time
import os
import queue
from dotenv import load_dotenv
import io
import wave
import uuid
import json
from datetime import datetime
import re
from typing import List

# Import CrewAI components
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI

# =====================================================================================
# 1. INITIAL CONFIGURATION AND STATE MANAGEMENT
# =====================================================================================

st.set_page_config(layout="wide", page_title="AI Voice Assistant")

# Initialize session state for screen navigation FIRST
if 'current_screen' not in st.session_state:
    st.session_state.current_screen = 'form_filler'

# Navigation header
st.markdown("# 🏠 Real Estate Inspection Assistant")
col1, col2 = st.columns(2)
with col1:
    current_screen = st.session_state.get('current_screen', 'form_filler')
    if st.button("🏠 Real Estate Inspection Form", type="primary" if current_screen == 'form_filler' else "secondary"):
        st.session_state.current_screen = 'form_filler'
        st.rerun()
with col2:
    current_screen = st.session_state.get('current_screen', 'form_filler')
    if st.button("🎙️ Property Q&A Analysis", type="primary" if current_screen == 'qna_analysis' else "secondary"):
        st.session_state.current_screen = 'qna_analysis'
        st.rerun()

st.markdown("---")

# --- Initialize session state for Form Filler ---
# Predefined fields for Real Estate House Inspections
PREDEFINED_INSPECTION_FIELDS = [
    {"id": "inspector_name", "name": "Inspector Name"},
    {"id": "inspection_company", "name": "Inspection Company"},
    {"id": "property_address", "name": "Property Address"},
    {"id": "inspection_date", "name": "Inspection Date"},
]

if 'form_fields' not in st.session_state:
    st.session_state.form_fields = PREDEFINED_INSPECTION_FIELDS.copy()
if 'custom_fields_added' not in st.session_state:
    st.session_state.custom_fields_added = []
if 'field_values' not in st.session_state:
    st.session_state.field_values = {}
if 'recording_field' not in st.session_state:
    st.session_state.recording_field = None
if 'audio_processor' not in st.session_state:
    st.session_state.audio_processor = None
if 'confirmation_field' not in st.session_state:
    st.session_state.confirmation_field = None
if 'temp_audio_bytes' not in st.session_state:
    st.session_state.temp_audio_bytes = None
if 'recording_started' not in st.session_state:
    st.session_state.recording_started = False

# --- Initialize session state for Q&A Analysis ---
PREDEFINED_QUESTIONS = [
    "Can you describe the key features of the property, including size, layout, and notable upgrades?",
    "What is the current asking price and how was that price determined or justified?",
    "Are there any outstanding liens, easements, or legal issues affecting the property?",
]
if 'answers' not in st.session_state:
    st.session_state.answers = {}
if 'selected_question_index' not in st.session_state:
    st.session_state.selected_question_index = 0
if 'custom_questions' not in st.session_state:
    st.session_state.custom_questions = []
if 'all_questions' not in st.session_state:
    st.session_state.all_questions = PREDEFINED_QUESTIONS.copy()
if 'qna_audio_bytes' not in st.session_state:
    st.session_state.qna_audio_bytes = None
if 'qna_audio_file_path' not in st.session_state:
    st.session_state.qna_audio_file_path = None
if 'qna_transcription' not in st.session_state:
    st.session_state.qna_transcription = None
if 'qna_language_code' not in st.session_state:
    st.session_state.qna_language_code = None
if 'qna_transcription_confidence' not in st.session_state:
    st.session_state.qna_transcription_confidence = None
if 'qna_converted_text' not in st.session_state:
    st.session_state.qna_converted_text = None
if 'qna_crew_result' not in st.session_state:
    st.session_state.qna_crew_result = None
if 'qna_relevancy_score' not in st.session_state:
    st.session_state.qna_relevancy_score = None
if 'qna_recording_started' not in st.session_state:
    st.session_state.qna_recording_started = False
if 'qna_audio_processor' not in st.session_state:
    st.session_state.qna_audio_processor = None

if 'audio_processor' not in st.session_state:
    st.session_state.audio_processor = None


# =====================================================================================
# 2. HELPER CLASSES & FUNCTIONS (COMBINED FROM BOTH APPS)
# =====================================================================================

class AudioRecorder(AudioProcessorBase):
    def __init__(self):
        self.audio_buffer = queue.Queue()
        self.resampler = av.AudioResampler(format='s16', layout='mono', rate=16000)

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        resampled_frames = self.resampler.resample(frame)
        for resampled_frame in resampled_frames:
            audio_bytes = resampled_frame.to_ndarray().tobytes()
            self.audio_buffer.put(audio_bytes)
        return frame

    def get_audio_bytes(self):
        audio_segments = []
        while not self.audio_buffer.empty():
            audio_segments.append(self.audio_buffer.get())
        return b"".join(audio_segments) if audio_segments else None

def pcm_to_wav_in_memory(audio_bytes, sample_rate=16000):
    if not audio_bytes: return None
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_bytes)
    return wav_buffer.getvalue()

def get_audio_processor():
    if 'audio_processor' not in st.session_state or st.session_state.audio_processor is None:
        st.session_state.audio_processor = AudioRecorder()
    return st.session_state.audio_processor

def get_qna_audio_processor():
    if 'qna_audio_processor' not in st.session_state or st.session_state.qna_audio_processor is None:
        st.session_state.qna_audio_processor = AudioRecorder()
    return st.session_state.qna_audio_processor

def poll_assemblyai_for_result(transcript_id, headers):
    polling_endpoint = f'https://api.assemblyai.com/v2/transcript/{transcript_id}'
    while True:
        try:
            polling_response = requests.get(polling_endpoint, headers=headers)
            polling_response.raise_for_status()
            polling_result = polling_response.json()
            if polling_result['status'] in ['completed', 'error']:
                return polling_result
            time.sleep(3)
        except requests.exceptions.RequestException as e:
            st.error(f"Error polling for transcription result: {e}")
            return None

def transcribe_for_form(api_key, audio_file_path, language_preference):
    headers = {'authorization': api_key}
    try:
        with open(audio_file_path, 'rb') as f:
            upload_response = requests.post('https://api.assemblyai.com/v2/upload', headers=headers, data=f)
        upload_response.raise_for_status()
        upload_url = upload_response.json()['upload_url']
    except Exception as e:
        st.error(f"Error uploading file to AssemblyAI: {e}")
        return None

    json_data = {
        'audio_url': upload_url,
        'punctuate': True,
        'format_text': True,
        'speech_model': 'best'
    }
    if language_preference == 'auto':
        json_data['language_detection'] = True
    else:
        json_data['language_code'] = language_preference

    try:
        transcript_response = requests.post('https://api.assemblyai.com/v2/transcript', json=json_data, headers=headers)
        transcript_response.raise_for_status()
        transcript_id = transcript_response.json().get('id')
        if not transcript_id:
            st.error("AssemblyAI did not return a transcript ID.")
            return None
    except Exception as e:
        st.error(f"Error requesting transcription: {e}")
        return None

    polling_result = poll_assemblyai_for_result(transcript_id, headers)
    if polling_result and polling_result['status'] == 'completed':
        text = polling_result.get('text', '').strip()
        if not text:
            st.warning("⚠️ No speech detected in the audio. Please ensure you speak clearly and try again.")
            return None
        return {
            "text": text,
            "language_code": polling_result.get('language_code', 'en'),
            "confidence": polling_result.get('confidence', 0.0)
        }
    elif polling_result:
        error_msg = polling_result.get('error', 'Unknown transcription error')
        st.error(f"Transcription failed: {error_msg}")
    else:
        st.error("Failed to get transcription result from AssemblyAI")
    return None

def transcribe_for_qna(api_key, audio_file_path):
    headers = {'authorization': api_key}
    try:
        with open(audio_file_path, 'rb') as f:
            upload_response = requests.post('https://api.assemblyai.com/v2/upload', headers=headers, data=f)
        upload_response.raise_for_status()
        upload_url = upload_response.json()['upload_url']
    except Exception as e:
        st.error(f"Error uploading file to AssemblyAI: {e}")
        return None

    json_data = {
        'audio_url': upload_url, 'speaker_labels': True, 'punctuate': True,
        'format_text': True, 'language_detection': True, 'disfluencies': True,
        'speech_model': 'best'
    }
    try:
        transcript_response = requests.post('https://api.assemblyai.com/v2/transcript', json=json_data, headers=headers)
        transcript_response.raise_for_status()
        transcript_id = transcript_response.json()['id']
    except Exception as e:
        st.error(f"Error requesting transcription: {e}")
        return None

    polling_result = poll_assemblyai_for_result(transcript_id, headers)
    if polling_result and polling_result['status'] == 'completed':
        return {
            "text": polling_result.get('text', ''),
            "language_code": polling_result.get('language_code', 'en'),
            "confidence": polling_result.get('confidence', 0.0)
        }
    elif polling_result:
        st.error(f"Transcription failed: {polling_result.get('error')}")
    return None

def convert_to_english(openai_api_key, text, language_code):
    """Convert text to English using CrewAI translator agent"""
    try:
        llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o", temperature=0.1)
        translator_agent = Agent(
            role='Expert Language Translator',
            goal='Translate text to English, but first verify if it is already English.',
            backstory='An expert linguist who trusts text content over potentially incorrect language codes.',
            llm=llm,
            verbose=False,
            allow_delegation=False
        )
        translation_task = Task(
            description=f"Text: '{text}'. Detected language: '{language_code}'. If the text content is English, return it as is. Otherwise, translate to English.",
            agent=translator_agent,
            expected_output="The text in English."
        )
        translation_crew = Crew(
            agents=[translator_agent],
            tasks=[translation_task],
            process=Process.sequential
        )
        english_text = translation_crew.kickoff()
        return str(english_text).strip()
    except Exception as e:
        st.error(f"English conversion error: {e}")
        return text

def extract_field_info_with_crewai(openai_api_key, field_name, transcript, language_code):
    try:
        llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o", temperature=0.1)
        translator_agent = Agent(
            role='Expert Language Translator',
            goal=f'Translate the text to English. If it is already English ({language_code}), return it unchanged.',
            backstory='A skilled translator ensuring accuracy.',
            verbose=False, llm=llm, allow_delegation=False
        )
        translation_task = Task(description=f"Translate to English: '{transcript}'. Source language code: '{language_code}'.", expected_output="The English translation.", agent=translator_agent)
        translation_crew = Crew(agents=[translator_agent], tasks=[translation_task], process=Process.sequential)
        english_transcript = translation_crew.kickoff()
        st.info(f"Translated: \"{english_transcript}\"")

        extractor_agent = Agent(
            role='Information Extractor Agent',
            goal=f"Extract the specific information for the form field: '{field_name}'. Output ONLY the value.",
            backstory=f"An AI expert at parsing English text to fill forms accurately.",
            verbose=False, llm=llm, allow_delegation=False
        )
        extraction_task = Task(description=f"From the text: '{english_transcript}', extract the value for the field '{field_name}'.", expected_output=f"The precise value for '{field_name}'.", agent=extractor_agent)
        extraction_crew = Crew(agents=[extractor_agent], tasks=[extraction_task], process=Process.sequential)
        return extraction_crew.kickoff()
    except Exception as e:
        st.error(f"CrewAI Error for field '{field_name}': {e}")
        return ""

def process_answer_with_crewai(openai_api_key, question, answer, language_code):
    if not answer: return "No answer provided."
    try:
        llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4o", temperature=0.2)
        translator_agent = Agent(role='Expert Language Translator', goal='Translate text to English, but first verify if it is already English.', backstory='An expert linguist who trusts text content over potentially incorrect language codes.', llm=llm)
        analyzer_agent = Agent(role='Answer Relevance Analyzer', goal='Analyze ENGLISH text for relevance to the question.', backstory='Expert in linguistic analysis.', llm=llm)
        relevancy_agent = Agent(role='Answer Quality Scorer', goal='Provide numerical scores (1-10) for answer quality.', backstory='Professional evaluator.', llm=llm)
        summarizer_agent = Agent(role='Concise Summarizer', goal='Summarize the key points of the ENGLISH answer.', backstory='Professional editor.', llm=llm)
        translation_task = Task(description=f"Text: '{answer}'. Detected language: '{language_code}'. If the text content is English, return it as is. Otherwise, translate to English.", agent=translator_agent, expected_output="The text in English.")
        analysis_task = Task(description=f"Analyze this ENGLISH answer for the question: '{question}'.", agent=analyzer_agent, context=[translation_task], expected_output="Key points and a conclusion on relevance.")
        relevancy_task = Task(description="Based on the analysis, score the answer's relevance, content match, completeness, and specificity from 1-10. Provide brief reasons.", agent=relevancy_agent, context=[analysis_task], expected_output="A formatted list of scores with one-line explanations.")
        summary_task = Task(description="Create a concise, two-sentence summary of the user's response.", agent=summarizer_agent, context=[analysis_task], expected_output="A polished two-sentence summary.")
        qa_crew = Crew(agents=[translator_agent, analyzer_agent, relevancy_agent, summarizer_agent], tasks=[translation_task, analysis_task, relevancy_task, summary_task], process=Process.sequential)
        crew_results = qa_crew.kickoff()
        task_outputs = crew_results.tasks_output
        relevancy_result = task_outputs[2].raw if len(task_outputs) > 2 else "Scoring unavailable."
        summary_result = task_outputs[3].raw if len(task_outputs) > 3 else "Summary unavailable."
        return {"summary": summary_result, "relevancy_score": relevancy_result}
    except Exception as e:
        st.error(f"CrewAI processing error: {e}")
        return "Could not process the answer."

# =====================================================================================
# 3. SIDEBAR (COMMON ELEMENTS)
# =====================================================================================

load_dotenv()

# --- Initialize keys from secrets (this runs only once) ---
if 'keys_loaded' not in st.session_state:
    st.session_state.assemblyai_api_key = st.secrets.get("ASSEMBLYAI_API_KEY", "")
    st.session_state.openai_api_key = st.secrets.get("OPENAI_API_KEY", "")
    st.session_state.keys_loaded = True

# REPLACE WITH THIS ENTIRE BLOCK
with st.sidebar:
    st.header("🔑 API Configuration")
    st.markdown("API keys are loaded from secrets and are not displayed.")

    # --- Logic for OpenAI API Key ---
    if st.session_state.openai_api_key:
        st.success("✅ OpenAI API Key Loaded")
        if st.button("Change OpenAI Key"):
            st.session_state.openai_api_key = ""
            st.rerun()
    else:
        st.subheader("OpenAI API Key")
        new_openai_key = st.text_input("Enter your AI API Key", type="password", key="openai_input")
        if new_openai_key:
            st.session_state.openai_api_key = new_openai_key
            st.rerun()

    st.markdown("---")

    # --- Logic for AssemblyAI API Key ---
    if st.session_state.assemblyai_api_key:
        st.success("✅ Audio AI API Key Loaded")
        if st.button("Change Audio AI Key"):
            st.session_state.assemblyai_api_key = ""
            st.rerun()
    else:
        st.subheader("AssemblyAI API Key")
        new_assemblyai_key = st.text_input("Enter your Audio AI API Key", type="password", key="assemblyai_input")
        if new_assemblyai_key:
            st.session_state.assemblyai_api_key = new_assemblyai_key
            st.rerun()

    # --- Audio Settings (shared for both screens) ---
    st.markdown("---")
    st.header("🎙️ Audio Settings")
    language_options = {
        "Automatic Detection": "auto",
        "English": "en",
        "Spanish": "es",
        "French": "fr",
        "German": "de",
        "Telugu": "te",
        "Hindi": "hi"
    }
    selected_language_label = st.selectbox("Select Recording Language", options=list(language_options.keys()))
    selected_language_code = language_options[selected_language_label]


# =====================================================================================
# 4. SCREEN RENDER: VOICE FORM FILLER
# =====================================================================================

if st.session_state.current_screen == 'form_filler':
    st.title("🏠 Real Estate Inspection Form Filler")
    st.markdown("### Fill Out Property Inspection Details Using Voice")
    st.markdown("This form includes essential fields for real estate house inspections. Use your voice to fill them out efficiently, or add custom fields as needed.")

    with st.sidebar:
        st.header("📋 Form Setup")
        predefined_count = len(PREDEFINED_INSPECTION_FIELDS)
        custom_count = len(st.session_state.custom_fields_added)
        total_fields = len(st.session_state.form_fields)
        st.markdown(f"**📋 Predefined Fields:** {predefined_count} (Real Estate Inspection)")
        if custom_count > 0:
            st.markdown(f"**➕ Custom Fields:** {custom_count}")
        st.markdown(f"**📊 Total Fields:** {total_fields}")
        if custom_count > 0:
            if st.button("🔄 Reset to Predefined Fields Only", help="Remove all custom fields"):
                st.session_state.form_fields = PREDEFINED_INSPECTION_FIELDS.copy()
                st.session_state.custom_fields_added = []
                for field_id in list(st.session_state.field_values.keys()):
                    if not any(f["id"] == field_id for f in st.session_state.form_fields):
                        del st.session_state.field_values[field_id]
                st.rerun()
        st.markdown("**➕ Add Custom Field:**")
        with st.form("add_field_form", clear_on_submit=True):
            new_field_name = st.text_input("Custom Field Name", placeholder="e.g., Pool Type, Basement Details")
            if st.form_submit_button("Add Custom Field") and new_field_name:
                new_field = {"id": str(uuid.uuid4()), "name": new_field_name}
                st.session_state.form_fields.append(new_field)
                st.session_state.custom_fields_added.append(new_field)
                st.success(f"Added: {new_field_name}")
                st.rerun()

    if not st.session_state.assemblyai_api_key or not st.session_state.openai_api_key:
        st.warning("Please enter your API keys in the sidebar to begin.")
        st.stop()

    if not st.session_state.form_fields:
        st.info("Get started by adding some fields to your form in the sidebar.")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.header("Fill Out Your Form")
            for field in st.session_state.form_fields:
                field_id, field_name = field["id"], field["name"]
                with st.container():
                    st.subheader(field_name)
                    st.session_state.field_values[field_id] = st.text_input(
                        f"Value for {field_name}",
                        value=st.session_state.field_values.get(field_id, ""),
                        key=f"text_{field_id}")

                    if st.session_state.recording_field == field_id:
                        audio_processor = get_audio_processor()
                        webrtc_ctx = webrtc_streamer(
                            key=f"recorder_{field_id}",
                            mode=WebRtcMode.SENDONLY,
                            audio_processor_factory=lambda: audio_processor,
                            media_stream_constraints={"video": False, "audio": True},
                            rtc_configuration={
                                    "iceServers": [
                                        # First, we still try the simple STUN server
                                        {"urls": ["stun:stun.l.google.com:19302"]},
                                        
                                        # If STUN fails, we use this free TURN server as a fallback
                                        {
                                            "urls": ["turn:global.turn.twilio.com:3478?transport=udp"],
                                            "username": st.secrets.get("TWILIO_USERNAME"),
                                            "credential": st.secrets.get("TWILIO_CREDENTIAL"),
                                        },
                                    ]
                                })
                        if webrtc_ctx.state.playing:
                            if not st.session_state.recording_started:
                                st.session_state.recording_started = True
                            st.info("🔴 Recording...")
                        elif not webrtc_ctx.state.playing and st.session_state.recording_started:
                            st.session_state.recording_started = False
                            raw_audio_bytes = audio_processor.get_audio_bytes()
                            if raw_audio_bytes and len(raw_audio_bytes) > 0:
                                st.session_state.temp_audio_bytes = pcm_to_wav_in_memory(raw_audio_bytes)
                                st.session_state.confirmation_field = field_id
                            else:
                                st.warning("Recording was too short or silent.")
                            st.session_state.recording_field = None
                            st.rerun()
                    elif st.session_state.confirmation_field == field_id:
                        st.audio(st.session_state.temp_audio_bytes, format="audio/wav")
                        c1, c2 = st.columns(2)
                        if c1.button("✅ Confirm & Fill the Field", key=f"confirm_btn_{field_id}"):
                            file_path = f"temp_audio_{field_id}.wav"
                            with open(file_path, "wb") as f:
                                f.write(st.session_state.temp_audio_bytes)
                            with st.spinner(f"Transcribing audio for {field_name}..."):
                                result = transcribe_for_form(st.session_state.assemblyai_api_key, file_path, selected_language_code)
                            if result and result.get("text"):
                                transcript = result["text"]
                                language_code = result["language_code"]
                                st.info(f"Heard ({language_code}): \"{transcript}\"")
                                with st.spinner(f"Translating and extracting '{field_name}'..."):
                                    extracted_value = extract_field_info_with_crewai(st.session_state.openai_api_key, field_name, result["text"], result["language_code"])
                                if extracted_value:
                                    st.session_state.field_values[field_id] = extracted_value
                                    st.success(f"Field '{field_name}' auto-filled!")
                                else:
                                    st.error(f"Extraction failed for '{field_name}'. Please try re-recording.")
                            else:
                                st.error("Transcription failed or returned no text. Please try re-recording.")
                            if os.path.exists(file_path):
                                os.remove(file_path)
                            st.session_state.confirmation_field = None
                            st.session_state.temp_audio_bytes = None
                            st.rerun()
                        if c2.button("🔄 Re-record", key=f"rerecord_btn_{field_id}"):
                            st.session_state.confirmation_field = None
                            st.session_state.temp_audio_bytes = None
                            st.rerun()
                    else:
                        if st.button(f"🎙️ Record Answer for {field_name}", key=f"record_btn_{field_id}"):
                            audio_processor = get_audio_processor()
                            audio_processor.audio_buffer = queue.Queue()
                            st.session_state.recording_field = field_id
                            st.session_state.recording_started = False
                            st.rerun()
        with col2:
            st.header("Live Form Preview")
            if any(st.session_state.field_values.values()):
                for field in st.session_state.form_fields:
                    value = st.session_state.field_values.get(field["id"], "")
                    st.markdown(f"**{field['name']}:** {value}")
                st.markdown("---")
                st.markdown("### Export Options")
                if st.button("💾 Save Form as JSON", use_container_width=True):
                    form_data = {field["name"]: st.session_state.field_values.get(field["id"], "") for field in st.session_state.form_fields}
                    filename = f"form_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                    with open(filename, "w", encoding="utf-8") as f:
                        json.dump(form_data, f, indent=4)
                    st.success(f"Form saved as `{filename}`")
                if st.button("🗑️ Clear Form", use_container_width=True):
                    st.session_state.form_fields, st.session_state.field_values = [], {}
                    st.rerun()
            else:
                st.info("Your completed form fields will appear here.")

# =====================================================================================
# 5. SCREEN RENDER: Q&A ANALYSIS
# =====================================================================================

elif st.session_state.current_screen == 'qna_analysis':
    st.title("🎙️ Property Q&A with AI Analysis")
    st.markdown("Select a question from the sidebar to begin. Answer using your voice, confirm the recording, and see the AI-powered analysis.")

    with st.sidebar:
        st.markdown("---")
        st.header("📋 Questions")
        st.write("Select a question to answer:")
        for i, q in enumerate(st.session_state.all_questions):
            if i < len(PREDEFINED_QUESTIONS):
                question_type = "🔖"
            else:
                question_type = "✨"
            status = "✅" if i in st.session_state.answers else "⭕"
            label = f"{status} {question_type} {q}"
            if st.button(label, key=f"q_button_{i}"):
                st.session_state.selected_question_index = i
                st.session_state.qna_audio_bytes = None
                st.session_state.qna_audio_file_path = None
                st.session_state.qna_transcription = None
                st.session_state.qna_language_code = None
                st.session_state.qna_transcription_confidence = None
                st.session_state.qna_converted_text = None
                st.session_state.qna_crew_result = None
                st.session_state.qna_relevancy_score = None
                st.rerun()
        st.markdown("---")
        st.subheader("➕ Add Custom Question")
        new_question = st.text_area("Enter your custom question:",
                                      placeholder="Type your question here...",
                                      height=80,
                                      key="new_question_input")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Add Question", disabled=not new_question.strip()):
                if new_question.strip() and new_question.strip() not in st.session_state.all_questions:
                    st.session_state.custom_questions.append(new_question.strip())
                    st.session_state.all_questions = PREDEFINED_QUESTIONS + st.session_state.custom_questions
                    st.success("Question added!")
                    st.rerun()
                elif new_question.strip() in st.session_state.all_questions:
                    st.warning("Question already exists!")
        with col2:
            if st.button("Clear Custom", disabled=not st.session_state.custom_questions):
                st.session_state.custom_questions = []
                st.session_state.all_questions = PREDEFINED_QUESTIONS.copy()
                if st.session_state.selected_question_index >= len(PREDEFINED_QUESTIONS):
                    st.session_state.selected_question_index = 0
                st.success("Custom questions cleared!")
                st.rerun()

    if not st.session_state.assemblyai_api_key or not st.session_state.openai_api_key:
        st.warning("Please enter your API keys in the sidebar to proceed.")
        st.stop()

    if len(st.session_state.answers) == len(st.session_state.all_questions) and len(st.session_state.all_questions) > 0:
        st.header("✅ All Questions Answered!")
        st.balloons()
        col1, col2 = st.columns(2)
        with col1:
            if st.button("💾 Save Q&A Analysis as JSON", type="primary"):
                qna_data = {
                    "session_info": {
                        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "total_questions": len(st.session_state.all_questions),
                        "analysis_type": "Property Q&A Analysis"
                    },
                    "questions_and_answers": []
                }
                for i, data in sorted(st.session_state.answers.items()):
                    qna_entry = {
                        "question_number": i + 1,
                        "question": data['question'],
                        "transcribed_text": data['transcription'],
                        "english_converted_text": data.get('english_converted_text', data['transcription']),
                        "ai_analysis": data['ai_analysis'],
                        "relevancy_score": data['relevancy_score'],
                        "audio_file_path": data['audio_file'],
                        "language_detected": data.get('language_code', 'en'),
                        "transcription_confidence": data.get('transcription_confidence', 'N/A')
                    }
                    qna_data["questions_and_answers"].append(qna_entry)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"qna_analysis_{timestamp}.json"
                try:
                    with open(filename, 'w', encoding='utf-8') as f:
                        json.dump(qna_data, f, indent=2, ensure_ascii=False)
                    st.success(f"✅ Q&A Analysis saved successfully as: `{filename}`")
                    with open(filename, 'r', encoding='utf-8') as f:
                        json_content = f.read()
                    st.download_button(
                        label="📥 Download Q&A Analysis JSON",
                        data=json_content,
                        file_name=filename,
                        mime="application/json",
                        key="download_qna_json"
                    )
                except Exception as e:
                    st.error(f"❌ Error saving Q&A analysis: {str(e)}")
        with col2:
            if st.button("🔄 Start New Q&A Session"):
                st.session_state.answers = {}
                st.session_state.selected_question_index = 0
                st.session_state.qna_audio_bytes = None
                st.session_state.qna_audio_file_path = None
                st.session_state.qna_transcription = None
                st.session_state.qna_language_code = None
                st.session_state.qna_transcription_confidence = None
                st.session_state.qna_converted_text = None
                st.session_state.qna_crew_result = None
                st.session_state.qna_relevancy_score = None
                st.success("🔄 New Q&A session started!")
                st.rerun()
        st.markdown("---")
        st.subheader("📋 Complete Q&A Analysis Summary")
        for i, data in sorted(st.session_state.answers.items()):
            with st.expander(f"**Question {i+1}: {data['question']}**"):
                st.markdown(f"**Transcription:** {data['transcription']}")
                st.markdown(f"**AI Analysis:**")
                st.success(data['ai_analysis'])
                st.markdown(f"**Quality Score:**")
                st.info(data['relevancy_score'])
                if os.path.exists(data['audio_file']):
                    st.audio(data['audio_file'])
    else:
        idx = st.session_state.selected_question_index
        current_question = st.session_state.all_questions[idx]
        st.header(f"Question {idx + 1}/{len(st.session_state.all_questions)}")
        st.subheader(current_question)
        if idx in st.session_state.answers:
            saved_data = st.session_state.answers[idx]
            st.markdown("### 📋 Previously Saved Answer")
            st.success("✅ This question has been answered. You can review your response below:")
            col_review1, col_review2 = st.columns(2)
            with col_review1:
                st.markdown("#### 📝 Original Transcription")
                st.text_area("Your recorded answer:", value=saved_data['transcription'], height=100, disabled=True, key=f"review_trans_{idx}")
                st.markdown("#### 🎵 Audio Recording")
                if os.path.exists(saved_data['audio_file']):
                    with open(saved_data['audio_file'], 'rb') as audio_file:
                        audio_bytes = audio_file.read()
                    st.audio(audio_bytes, format="audio/wav")
                else:
                    st.warning("Audio file not found")
            with col_review2:
                st.markdown("#### 🤖 AI Analysis")
                st.info(saved_data['ai_analysis'])
                st.markdown("#### 📊 Quality Score")
                st.info(saved_data['relevancy_score'])
            st.markdown("---")
            col_action1, col_action2, col_action3 = st.columns(3)
            with col_action1:
                if st.button("🔄 Re-record This Answer", key=f"rerecord_question_{idx}", help="Start over with this question"):
                    del st.session_state.answers[idx]
                    qna_audio_processor = get_qna_audio_processor()
                    qna_audio_processor.audio_buffer = queue.Queue()
                    st.session_state.qna_audio_bytes = None
                    st.session_state.qna_audio_file_path = None
                    st.session_state.qna_transcription = None
                    st.session_state.qna_language_code = None
                    st.session_state.qna_transcription_confidence = None
                    st.session_state.qna_converted_text = None
                    st.session_state.qna_crew_result = None
                    st.session_state.qna_relevancy_score = None
                    st.success("🔄 Ready to re-record! Answer cleared.")
                    st.rerun()
            with col_action2:
                next_unanswered = None
                for i in range(len(st.session_state.all_questions)):
                    if i not in st.session_state.answers:
                        next_unanswered = i
                        break
                if next_unanswered is not None:
                    if st.button(f"➡️ Go to Next Unanswered", key=f"next_unanswered_{idx}"):
                        st.session_state.selected_question_index = next_unanswered
                        st.rerun()
                else:
                    st.info("🎉 All questions answered!")
            with col_action3:
                if st.button("📋 View All Answers", key=f"view_all_{idx}"):
                    st.session_state.selected_question_index = 0
                    st.rerun()
        else:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("#### Step 1: Record Your Answer")
                if not st.session_state.qna_audio_bytes:
                    qna_audio_processor = get_qna_audio_processor()
                    webrtc_ctx = webrtc_streamer(
                        key=f"qna-recorder-{idx}",
                        mode=WebRtcMode.SENDONLY,
                        audio_processor_factory=lambda: qna_audio_processor,
                        media_stream_constraints={"video": False, "audio": True},
                        rtc_configuration={
                            "iceServers": [
                                # First, we still try the simple STUN server
                                {"urls": ["stun:stun.l.google.com:19302"]},
                                
                                # If STUN fails, we use this free TURN server as a fallback
                                {
                                    "urls": ["turn:global.turn.twilio.com:3478?transport=udp"],
                                    "username": st.secrets.get("TWILIO_USERNAME"),
                                    "credential": st.secrets.get("TWILIO_CREDENTIAL"),
                                },
                            ]
                        })
                    if webrtc_ctx.state.playing:
                        if not st.session_state.qna_recording_started:
                            st.session_state.qna_recording_started = True
                            qna_audio_processor.audio_buffer = queue.Queue()
                        st.markdown("🔴 **Recording...**")
                    elif not webrtc_ctx.state.playing and st.session_state.qna_recording_started:
                        st.session_state.qna_recording_started = False
                        raw_audio_bytes = qna_audio_processor.get_audio_bytes()
                        if raw_audio_bytes and len(raw_audio_bytes) > 16000:
                            wav_bytes = pcm_to_wav_in_memory(raw_audio_bytes)
                            file_path = f"qna_temp_audio_{idx}.wav"
                            with open(file_path, "wb") as f:
                                f.write(wav_bytes)
                            st.session_state.qna_audio_file_path = file_path
                            st.session_state.qna_audio_bytes = wav_bytes
                            st.rerun()
                        else:
                            st.warning("Recording was too short. Please try again.")
            with col2:
                if st.session_state.qna_audio_bytes and not st.session_state.qna_transcription:
                    st.markdown("#### Step 2: Confirm & Transcribe")
                    st.audio(st.session_state.qna_audio_bytes, format="audio/wav")
                    confirm_col, rerecord_col = st.columns(2)
                    if confirm_col.button("✅ Confirm and Transcribe", key=f"confirm_{idx}"):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        with st.spinner("Transcribing audio..."):
                            result = transcribe_for_qna(st.session_state.assemblyai_api_key, st.session_state.qna_audio_file_path)
                        if result:
                            progress_bar.progress(75)
                            status_text.text("✅ Transcription completed!")
                            st.session_state.qna_transcription = result.get("text")
                            st.session_state.qna_language_code = result.get("language_code")
                            st.session_state.qna_transcription_confidence = result.get("confidence")
                            if st.session_state.qna_language_code and st.session_state.qna_language_code.lower() != 'en':
                                status_text.text("🔄 Converting to English...")
                                progress_bar.progress(90)
                                with st.spinner("Converting to English..."):
                                    english_text = convert_to_english(
                                        st.session_state.openai_api_key,
                                        st.session_state.qna_transcription,
                                        st.session_state.qna_language_code
                                    )
                                st.session_state.qna_converted_text = english_text
                            else:
                                st.session_state.qna_converted_text = st.session_state.qna_transcription
                            progress_bar.progress(100)
                            status_text.text("🎉 Processing complete!")
                            time.sleep(1)
                        else:
                            st.error("❌ Transcription failed. Please try again.")
                        st.rerun()
                    if rerecord_col.button("🔄 Re-record", key=f"rerecord_{idx}"):
                        st.session_state.qna_audio_bytes = None
                        st.session_state.qna_audio_file_path = None
                        st.session_state.qna_transcription = None
                        st.session_state.qna_language_code = None
                        st.session_state.qna_transcription_confidence = None
                        st.session_state.qna_converted_text = None
                        st.session_state.qna_crew_result = None
                        st.session_state.qna_relevancy_score = None
                        st.session_state.qna_recording_started = False
                        st.rerun()

            if st.session_state.qna_transcription and not st.session_state.qna_crew_result:
                st.markdown("#### Step 3: Review & Process")
                col_lang, col_conf = st.columns(2)
                with col_lang:
                    st.info(f"🌐 **Detected Language:** {st.session_state.qna_language_code.upper() if st.session_state.qna_language_code else 'Unknown'}")
                with col_conf:
                    confidence_pct = round(st.session_state.qna_transcription_confidence * 100, 1) if st.session_state.qna_transcription_confidence else 0
                    st.info(f"📊 **Confidence:** {confidence_pct}%")
                st.text_area("📝 Original Transcription:", value=st.session_state.qna_transcription, height=80, disabled=True)
                if (st.session_state.qna_converted_text and
                        st.session_state.qna_converted_text != st.session_state.qna_transcription):
                    st.text_area("🔤 English Converted Text:", value=st.session_state.qna_converted_text, height=80, disabled=True)
                if st.button("🤖 Process with AI Agents", key=f"process_{idx}"):
                    with st.spinner("AI agents are analyzing your answer..."):
                        analysis_text = st.session_state.qna_converted_text or st.session_state.qna_transcription
                        crew_result = process_answer_with_crewai(st.session_state.openai_api_key, current_question, analysis_text, st.session_state.qna_language_code)
                    if isinstance(crew_result, dict):
                        st.session_state.qna_crew_result = crew_result.get("summary")
                        st.session_state.qna_relevancy_score = crew_result.get("relevancy_score")
                    st.rerun()

            if st.session_state.qna_crew_result:
                st.markdown("#### Step 4: AI Analysis Results")
                st.success(st.session_state.qna_crew_result)
                st.info(st.session_state.qna_relevancy_score)
                if st.button("💾 Save Answer", key=f"save_{idx}"):
                    st.session_state.answers[idx] = {
                        "question": current_question,
                        "audio_file": st.session_state.qna_audio_file_path,
                        "transcription": st.session_state.qna_transcription,
                        "english_converted_text": st.session_state.qna_converted_text or st.session_state.qna_transcription,
                        "ai_analysis": st.session_state.qna_crew_result,
                        "relevancy_score": st.session_state.qna_relevancy_score,
                        "language_code": st.session_state.qna_language_code,
                        "transcription_confidence": st.session_state.qna_transcription_confidence
                    }
                    st.session_state.qna_audio_bytes, st.session_state.qna_audio_file_path, st.session_state.qna_transcription, st.session_state.qna_converted_text, st.session_state.qna_crew_result = None, None, None, None, None
                    st.session_state.qna_relevancy_score = None
                    if idx + 1 < len(st.session_state.all_questions):
                        st.session_state.selected_question_index = idx + 1
                        st.success(f"✅ Answer saved! Moving to Question {idx + 2}...")
                    else:
                        st.success("🎉 All questions completed!")
                    st.rerun()