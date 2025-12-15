import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import plotly.express as px
from datetime import datetime
import tempfile
import os
from transformers import pipeline
import mediapipe as mp
import speech_recognition as sr
import threading
import queue
import time

# 페이지 설정
st.set_page_config(
    page_title="감정 일기 - Emotion Diary",
    page_icon="📔",
    layout="wide"
)

# 세션 상태 초기화
if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []
if 'diary_entries' not in st.session_state:
    st.session_state.diary_entries = []
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'video_frames' not in st.session_state:
    st.session_state.video_frames = []
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = 'neutral'
if 'emotion_timeline' not in st.session_state:
    st.session_state.emotion_timeline = []
if 'recording_start_time' not in st.session_state:
    st.session_state.recording_start_time = None
if 'webcam_active' not in st.session_state:
    st.session_state.webcam_active = False
if 'voice_recording' not in st.session_state:
    st.session_state.voice_recording = False
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""
if 'audio_queue' not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if 'pending_save' not in st.session_state:
    st.session_state.pending_save = False
if 'save_data' not in st.session_state:
    st.session_state.save_data = None
if 'last_text_update' not in st.session_state:
    st.session_state.last_text_update = time.time()
if 'emotion_confirmed' not in st.session_state:
    st.session_state.emotion_confirmed = False
if 'confirmed_emotion' not in st.session_state:
    st.session_state.confirmed_emotion = None

# 사이드바 (메인 UI보다 먼저 실행되어야 anonymize_option 변수가 정의됨)
with st.sidebar:
    st.header("📸 녹화 설정")
    
    anonymize_option = st.selectbox(
        "전체 화면 익명화 방식",
        ["원본", "블러", "픽셀화", "카툰"],
        key="anonymize",
        disabled=st.session_state.recording
    )
    
    show_emotion_overlay = st.checkbox(
        "감정 정보 오버레이 표시", 
        value=True,
        disabled=st.session_state.recording
    )
    
    if st.session_state.recording:
        st.warning("⚠️ 녹화 중에는 설정을 변경할 수 없습니다.")
    
    st.markdown("---")
    
    st.header("ℹ️ 사용 방법")
    st.markdown("""
    ### 📹 음성 영상 일기 작성 순서
    
    1. 🎨 **익명화 방식 선택**
    2. 🔴 **녹화 시작** 클릭
    3. 🎤 **말하며 감정 표현**
    4. ⏹️ **녹화 중지 & 저장**
    5. ✨ **오늘의 기분 선택**
    6. 📥 **영상 & 텍스트 다운로드**
    
    ### 🎭 지원 감정
    - 😊 Happy (행복)
    - 😢 Sad (슬픔)
    - 😠 Angry (화남)
    - 😲 Surprise (놀람)
    - 😐 Neutral (중립)
    - 😨 Fear (두려움)
    - 🤢 Disgust (혐오)
    
    ### 🔒 익명화 방식
    - **원본**: 얼굴 그대로
    - **블러**: 전체 화면 흐리게
    - **픽셀화**: 전체 화면 모자이크
    - **카툰**: 전체 화면 만화 스타일
    
    ### 🎤 음성 입력
    - **자동 활성화**: 녹화 시작 시 자동 활성화
    - **한국어 인식**: 실시간 한국어 음성 인식
    - **실시간 변환**: 말한 내용을 즉시 텍스트로 변환
    - **자동 저장**: 일기로 자동 저장
    
    ### 💡 촬영 팁
    - 💡 **밝은 조명** 사용
    - 📷 **정면 얼굴** 유지
    - 😀 **자연스러운 표정**
    - 🔇 **조용한 환경** (음성 인식 최적화)
    - 🎤 **마이크 가까이**에서 또렷하게 말하기
    - 🗣️ **천천히 명확하게** 발음하기
    """)
    
    st.markdown("---")
    
    # 통계
    st.subheader("📈 전체 통계")
    total_entries = len(st.session_state.diary_entries)
    total_frames = sum([e['frame_count'] for e in st.session_state.diary_entries])
    voice_entries = len(st.session_state.diary_entries)
    st.metric("총 일기 수", total_entries)
    st.metric("총 프레임 수", total_frames)
    st.metric("음성 입력 사용", f"{voice_entries}회")
    
    st.markdown("---")
    
    if st.button("🗑️ 모든 기록 초기화", type="secondary"):
        st.session_state.emotion_history = []
        st.session_state.diary_entries = []
        st.session_state.video_frames = []
        st.session_state.emotion_timeline = []
        st.session_state.recording = False
        st.session_state.webcam_active = False
        st.session_state.recording_start_time = None
        st.session_state.voice_recording = False
        st.session_state.transcribed_text = ""
        st.session_state.pending_save = False
        st.session_state.save_data = None
        st.session_state.emotion_confirmed = False
        st.session_state.confirmed_emotion = None
        st.success("✅ 초기화 완료!")
        st.rerun()

# 한글 폰트 로드 함수
@st.cache_resource
def load_korean_font(size=40):
    """한글을 지원하는 폰트 로드"""
    font_paths = [
        "malgun.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "AppleGothic.ttf",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/truetype/nanum/NanumGothicBold.ttf",
        "NanumGothic.ttf",
    ]
    
    for font_path in font_paths:
        try:
            return ImageFont.truetype(font_path, size)
        except:
            continue
    
    return ImageFont.load_default()

# 한글 텍스트를 이미지에 추가하는 헬퍼 함수
def put_korean_text(image, text, position, font_size=40, color=(255, 255, 255)):
    """OpenCV 이미지에 한글 텍스트를 추가"""
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)
    font = load_korean_font(font_size)
    draw.text(position, text, fill=color, font=font)
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

# 감정 분석 모델 로드
@st.cache_resource
def load_emotion_model():
    try:
        return pipeline("image-classification", model="dima806/facial_emotions_image_detection")
    except Exception as e:
        st.error(f"모델 로드 오류: {e}")
        return None

# MediaPipe 얼굴 검출
@st.cache_resource
def load_face_detector():
    try:
        return mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.5)
    except Exception as e:
        st.error(f"얼굴 검출기 로드 오류: {e}")
        return None

# 전체 화면 익명화 함수들
def blur_frame(image: np.ndarray, strength: int = 51) -> np.ndarray:
    """전체 프레임 블러 처리"""
    if strength % 2 == 0:
        strength += 1
    return cv2.GaussianBlur(image, (strength, strength), 0)

def pixelate_frame(image: np.ndarray, blocks: int = 16) -> np.ndarray:
    """전체 프레임 픽셀화"""
    h, w = image.shape[:2]
    if h > 0 and w > 0 and h > blocks and w > blocks:
        temp = cv2.resize(image, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
        return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    return image

def cartoonize_frame(image: np.ndarray) -> np.ndarray:
    """전체 프레임 카툰 스타일 변환"""
    try:
        if image.size == 0 or image.shape[0] < 10 or image.shape[1] < 10:
            return image
        
        color = cv2.bilateralFilter(image, 9, 250, 250)
        for _ in range(2):
            color = cv2.bilateralFilter(color, 9, 250, 250)
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 
            9, 2
        )
        
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(color, edges_colored)
        cartoon = cv2.convertScaleAbs(cartoon, alpha=1.2, beta=10)
        
        return cartoon
    except Exception as e:
        print(f"카툰 변환 오류: {e}")
        return image

# 음성 인식 함수
def record_audio_continuous(audio_queue, stop_event):
    """연속적으로 음성을 인식하는 함수"""
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    
    with sr.Microphone() as source:
        print("마이크 조정 중...")
        recognizer.adjust_for_ambient_noise(source, duration=1)
        print("음성 인식 시작!")
        
        while not stop_event.is_set():
            try:
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=10)
                
                try:
                    text = recognizer.recognize_google(audio, language='ko-KR')
                    if text:
                        audio_queue.put(text)
                        print(f"인식된 텍스트: {text}")
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    print(f"음성 인식 서비스 오류: {e}")
                    time.sleep(1)
            except sr.WaitTimeoutError:
                continue
            except Exception as e:
                print(f"음성 인식 오류: {e}")
                time.sleep(1)

# 감정 분석 함수
def analyze_emotion_quick(image: np.ndarray, model, face_detector) -> tuple[str, float, tuple]:
    """빠른 감정 분석 (실시간용) - 얼굴 위치만 반환"""
    emotion = "neutral"
    confidence = 0.0
    face_bbox = None
    
    if model is None or face_detector is None:
        return emotion, confidence, face_bbox
    
    try:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detector.process(image_rgb)
        
        if results.detections:
            detection = results.detections[0]
            bboxC = detection.location_data.relative_bounding_box
            h, w, _ = image.shape
            x = int(bboxC.xmin * w)
            y = int(bboxC.ymin * h)
            width = int(bboxC.width * w)
            height = int(bboxC.height * h)
            
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)
            
            face_bbox = (x, y, width, height)
            
            face = image[y:y+height, x:x+width]
            
            if face.size > 0:
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                emotion_results = model(face_pil)
                
                if emotion_results:
                    emotion = emotion_results[0]['label']
                    confidence = emotion_results[0]['score']
    
    except Exception as e:
        print(f"감정 분석 오류: {e}")
    
    return emotion, confidence, face_bbox

# 비디오 저장 함수
def save_video(frames: list, filename: str, fps: int = 20):
    """프레임 리스트를 비디오 파일로 저장"""
    if not frames or len(frames) == 0:
        return None
    
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    return filename

# AI 기반 기분 추천 함수
def suggest_mood_from_data(dominant_emotion: str, diary_text: str, emotion_timeline: list) -> list:
    """감정 분석과 일기 내용을 바탕으로 기분을 추천"""
    suggestions = []
    
    emotion_to_mood = {
        'happy': [('행복한', '밝은 표정이 자주 보였어요'), ('즐거운', '긍정적인 에너지가 느껴져요'), ('기쁜', '웃는 모습이 많았어요')],
        'joy': [('기쁜', '환한 미소가 인상적이었어요'), ('신나는', '활기찬 모습이 보였어요'), ('즐거운', '긍정적인 분위기였어요')],
        'sad': [('슬픈', '우울한 표정이 보였어요'), ('우울한', '힘든 하루였나봐요'), ('침울한', '기운이 없어 보였어요')],
        'angry': [('화난', '불편한 감정이 느껴졌어요'), ('짜증난', '스트레스가 있었나봐요'), ('불쾌한', '기분이 좋지 않아 보였어요')],
        'surprise': [('놀란', '예상치 못한 일이 있었나봐요'), ('당황한', '갑작스러운 상황이 있었나요'), ('의외의', '새로운 일이 있었던 것 같아요')],
        'fear': [('불안한', '걱정이 많아 보였어요'), ('두려운', '긴장된 모습이었어요'), ('초조한', '마음이 편치 않아 보였어요')],
        'disgust': [('불편한', '거북한 상황이 있었나봐요'), ('싫은', '마음에 들지 않는 일이 있었나요'), ('거북한', '불쾌한 감정이 느껴졌어요')],
        'neutral': [('평온한', '차분한 하루였어요'), ('고요한', '안정적인 상태였어요'), ('담담한', '잔잔한 하루였네요')]
    }
    
    if dominant_emotion.lower() in emotion_to_mood:
        suggestions.extend(emotion_to_mood[dominant_emotion.lower()])
    else:
        suggestions.extend([('평온한', '차분한 하루였어요'), ('담담한', '특별한 감정 변화가 없었어요')])
    
    if emotion_timeline and len(emotion_timeline) > 5:
        emotions = [e['emotion'] for e in emotion_timeline]
        unique_emotions = len(set(emotions))
        
        if unique_emotions >= 4:
            suggestions.insert(0, ('복잡한', '다양한 감정을 느낀 하루였네요'))
        elif unique_emotions == 1:
            suggestions.insert(0, ('일관된', '하루 종일 비슷한 기분이었어요'))
    
    positive_keywords = ['좋', '행복', '기쁨', '즐거', '감사', '뿌듯', '성공', '완성', '달성', '사랑']
    negative_keywords = ['힘들', '피곤', '지치', '우울', '슬프', '화', '짜증', '스트레스', '불안', '걱정']
    
    text_lower = diary_text.lower()
    positive_count = sum(1 for kw in positive_keywords if kw in text_lower)
    negative_count = sum(1 for kw in negative_keywords if kw in text_lower)
    
    if positive_count > negative_count + 2:
        if ('행복한', '밝은 표정이 자주 보였어요') not in suggestions:
            suggestions.insert(0, ('감사한', '긍정적인 단어들이 많았어요'))
    elif negative_count > positive_count + 2:
        if ('슬픈', '우울한 표정이 보였어요') not in suggestions:
            suggestions.insert(0, ('지친', '힘든 표현들이 많았어요'))
    
    seen = set()
    unique_suggestions = []
    for mood, reason in suggestions:
        if mood not in seen:
            seen.add(mood)
            unique_suggestions.append((mood, reason))
    
    return unique_suggestions[:5]

# 메인 UI
st.title("📔 감정 영상 일기 - Emotion Video Diary")
st.markdown("*웹캠으로 실시간 감정을 분석하며 음성으로 영상 일기를 작성하세요*")

st.markdown("---")

# 레이아웃 구성 - 웹캠과 음성 텍스트 영역
col_webcam, col_text = st.columns([2, 1])

with col_webcam:
    st.subheader("📹 웹캠 화면")
    
    # 1. 녹화/감정 확정/완료 통합 버튼 (맨 위)
    if st.session_state.pending_save and st.session_state.save_data:
        # 감정 확정 전 - 녹화 시작 버튼이 "감정 확정하기"로 변경
        if not st.session_state.emotion_confirmed:
            # 감정 선택 UI는 아래에 표시되고, 버튼만 여기에
            confirm_emotion = st.button("✅ 감정 확정하기", type="primary", use_container_width=True, key="confirm_top_btn")
        else:
            # 감정 확정 후 - "완료" 버튼으로 변경
            complete_action = st.button("✅ 완료", type="primary", use_container_width=True, key="complete_top_btn")
    elif not st.session_state.recording:
        start_recording = st.button("🔴 녹화 시작", type="primary", use_container_width=True)
    else:
        start_recording = False
        stop_recording = st.button("⏹️ 녹화 중지 & 저장", type="secondary", use_container_width=True)
    
    # 2. 녹화 상태 표시 (버튼 바로 아래)
    status_placeholder = st.empty()
    
    # 녹화 전 상태 표시
    if not st.session_state.webcam_active and not st.session_state.pending_save:
        status_placeholder.info("녹화 시작 버튼을 눌러주세요")
    
    # 3. 웹캠 캡처 영역 (상태 표시 아래) - 고정 크기
    webcam_placeholder = st.empty()
    
    # 녹화 시작 전 대기 화면 표시 (고정 크기 640x480)
    if not st.session_state.webcam_active and not st.session_state.pending_save:
        waiting_image = np.zeros((480, 640, 3), dtype=np.uint8)
        waiting_image[:] = (50, 50, 50)
        webcam_placeholder.image(waiting_image, channels="BGR", width=640)
    
    # 4. 다운로드 영역
    download_placeholder = st.empty()

# 음성 텍스트 영역
with col_text:
    st.subheader("🎤 음성 입력")
    voice_text_placeholder = st.empty()
    
    if st.session_state.recording and st.session_state.voice_recording:
        word_count = len(st.session_state.transcribed_text.split()) if st.session_state.transcribed_text else 0
        current_text = st.session_state.transcribed_text if st.session_state.transcribed_text else "(음성 인식 중... 말씀해주세요)"
        voice_text_placeholder.text_area(
            f"입력된 내용 (단어: {word_count}개)",
            value=current_text,
            height=480,
            disabled=True,
            key=f"voice_display_{time.time()}"
        )
    elif st.session_state.transcribed_text:
        word_count = len(st.session_state.transcribed_text.split())
        voice_text_placeholder.text_area(
            f"입력된 내용 (단어: {word_count}개)",
            value=st.session_state.transcribed_text,
            height=480,
            disabled=True,
            key="voice_display_saved"
        )
    else:
        voice_text_placeholder.text_area(
            "입력된 내용",
            value="(음성 입력 대기 중...)",
            height=480,
            disabled=True,
            key="voice_display_empty"
        )

# 모델 로드
with st.spinner("AI 모델 로딩 중..."):
    emotion_model = load_emotion_model()
    face_detector = load_face_detector()

if emotion_model is None or face_detector is None:
    st.error("⚠️ AI 모델 로드에 실패했습니다. 페이지를 새로고침해주세요.")
    st.stop()

# 녹화 시작 처리
if 'start_recording' in locals() and start_recording:
    if not st.session_state.recording:
        st.session_state.recording = True
        st.session_state.webcam_active = True
        st.session_state.video_frames = []
        st.session_state.emotion_timeline = []
        st.session_state.recording_start_time = datetime.now()
        st.session_state.transcribed_text = ""
        
        # 음성 인식 시작
        st.session_state.voice_recording = True
        st.session_state.audio_queue = queue.Queue()
        st.session_state.stop_event = threading.Event()
        st.session_state.audio_thread = threading.Thread(
            target=record_audio_continuous,
            args=(st.session_state.audio_queue, st.session_state.stop_event)
        )
        st.session_state.audio_thread.daemon = True
        st.session_state.audio_thread.start()
        
        st.rerun()

# 녹화 중지 처리
if st.session_state.recording and 'stop_recording' in locals() and stop_recording:
    st.session_state.recording = False
    st.session_state.webcam_active = False
    
    # 음성 인식 중지
    if st.session_state.voice_recording:
        st.session_state.stop_event.set()
        st.session_state.voice_recording = False
        time.sleep(0.5)
    
    final_text = st.session_state.transcribed_text if st.session_state.transcribed_text else "(음성 입력 없음)"
    
    if st.session_state.video_frames and len(st.session_state.video_frames) > 0:
        status_placeholder.info("💾 영상 일기 저장 중...")
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_filename = f"emotion_diary_{timestamp}.mp4"
            text_filename = f"emotion_diary_{timestamp}.txt"
            
            temp_dir = tempfile.gettempdir()
            video_path = os.path.join(temp_dir, video_filename)
            text_path = os.path.join(temp_dir, text_filename)
            
            # 비디오 파일 즉시 저장
            save_video(st.session_state.video_frames, video_path, fps=20)
            
            # 감정 통계 계산
            if st.session_state.emotion_timeline:
                emotions_list = [e['emotion'] for e in st.session_state.emotion_timeline]
                emotion_counts = pd.Series(emotions_list).value_counts()
                dominant_emotion = emotion_counts.index[0] if len(emotion_counts) > 0 else "neutral"
                avg_confidence = np.mean([e['confidence'] for e in st.session_state.emotion_timeline])
            else:
                dominant_emotion = "neutral"
                avg_confidence = 0.0
            
            # AI 기반 기분 추천
            suggested_moods = suggest_mood_from_data(
                dominant_emotion, 
                final_text, 
                st.session_state.emotion_timeline
            )
            
            # 녹화 시간 계산
            if st.session_state.recording_start_time:
                elapsed = datetime.now() - st.session_state.recording_start_time
                elapsed_seconds = int(elapsed.total_seconds())
                recording_duration = f"{elapsed_seconds // 60:02d}:{elapsed_seconds % 60:02d}"
            else:
                recording_duration = "00:00"
            
            # 저장 데이터를 세션에 보관
            st.session_state.save_data = {
                'timestamp': timestamp,
                'video_filename': video_filename,
                'video_path': video_path,
                'text_filename': text_filename,
                'text_path': text_path,
                'final_text': final_text,
                'dominant_emotion': dominant_emotion,
                'avg_confidence': avg_confidence,
                'suggested_moods': suggested_moods,
                'frame_count': len(st.session_state.video_frames),
                'recording_duration': recording_duration,
                'emotion_timeline': st.session_state.emotion_timeline.copy(),
                'anonymize_method': anonymize_option
            }
            
            st.session_state.pending_save = True
            st.session_state.video_frames = []
            st.session_state.recording_start_time = None
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ 저장 중 오류가 발생했습니다: {e}")
            import traceback
            st.error(traceback.format_exc())
    else:
        st.warning("⚠️ 녹화된 프레임이 없습니다!")
        st.session_state.video_frames = []
        st.session_state.recording_start_time = None

# 기분 선택 UI (pending_save 상태일 때) - col_webcam 영역에서 표시
if st.session_state.pending_save and st.session_state.save_data:
    save_data = st.session_state.save_data
    
    with col_webcam:
        # 감정 확정 전 단계
        if not st.session_state.emotion_confirmed:
            status_placeholder.info("✨ 오늘의 감정을 선택해주세요 (분석 결과를 기반으로 추천합니다)")
            
            # 감정 선택 (라디오 버튼)
            emotion_options = [
                "😊 Happy (행복)",
                "😢 Sad (슬픔)",
                "😠 Angry (화남)",
                "😲 Surprise (놀람)",
                "😐 Neutral (중립)",
                "😨 Fear (두려움)",
                "🤢 Disgust (혐오)"
            ]
            
            # AI 추천 감정을 기본 선택으로 설정
            dominant_emotion = save_data['dominant_emotion'].lower()
            emotion_map = {
                'happy': 0,
                'sad': 1,
                'angry': 2,
                'surprise': 3,
                'neutral': 4,
                'fear': 5,
                'disgust': 6,
                'joy': 0  # joy도 happy로 매핑
            }
            
            default_index = emotion_map.get(dominant_emotion, 4)
            
            selected_emotion = st.radio(
                f"🎭 AI 추천: **{save_data['dominant_emotion']}**",
                emotion_options,
                index=default_index,
                key="emotion_radio"
            )
            
            # 상단의 "감정 확정하기" 버튼이 클릭되었을 때 처리
            if 'confirm_emotion' in locals() and confirm_emotion:
                # 선택된 감정 추출 (이모지와 영문명 제거)
                final_mood = selected_emotion.split('(')[1].replace(')', '').strip()
                
                # 텍스트 파일 저장
                with open(save_data['text_path'], 'w', encoding='utf-8') as f:
                    f.write(f"=== 감정 영상 일기 ===\n")
                    f.write(f"날짜: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}\n")
                    f.write(f"오늘의 감정: {final_mood}\n")
                    f.write(f"익명화 방식: {save_data['anonymize_method']}\n")
                    f.write(f"\n=== 일기 내용 (음성 입력) ===\n\n")
                    f.write(save_data['final_text'])
                    f.write(f"\n\n=== 감정 분석 결과 ===\n")
                    if save_data['emotion_timeline']:
                        emotions_list = [e['emotion'] for e in save_data['emotion_timeline']]
                        emotion_counts = pd.Series(emotions_list).value_counts()
                        f.write(f"주요 감정: {save_data['dominant_emotion']}\n")
                        f.write(f"평균 확신도: {save_data['avg_confidence']*100:.1f}%\n")
                        f.write(f"\n감정 분포:\n")
                        for emotion, count in emotion_counts.items():
                            percentage = (count / len(emotions_list)) * 100
                            f.write(f"  - {emotion}: {count}회 ({percentage:.1f}%)\n")
                    f.write(f"\n=== AI 감정 분석 ===\n")
                    f.write(f"분석된 주요 감정: {save_data['dominant_emotion']}\n")
                    f.write(f"선택한 감정: {final_mood}\n")
                
                # 일기 항목 저장
                entry = {
                    'timestamp': save_data['timestamp'],
                    'emotion': final_mood,
                    'diary_text': save_data['final_text'],
                    'video_filename': save_data['video_filename'],
                    'video_path': save_data['video_path'],
                    'text_filename': save_data['text_filename'],
                    'text_path': save_data['text_path'],
                    'dominant_emotion': save_data['dominant_emotion'],
                    'avg_confidence': save_data['avg_confidence'],
                    'frame_count': save_data['frame_count'],
                    'recording_duration': save_data['recording_duration'],
                    'emotion_timeline': save_data['emotion_timeline'],
                    'anonymize_method': save_data['anonymize_method'],
                    'voice_input_used': True
                }
                
                st.session_state.diary_entries.append(entry)
                st.session_state.confirmed_emotion = final_mood
                st.session_state.emotion_confirmed = True
                
                st.rerun()
        
        # 감정 확정 후 - 세션 감정 분석 및 다운로드
        else:
            status_placeholder.success(f"✅ 영상 일기가 저장되었습니다! (감정: {st.session_state.confirmed_emotion})")
            
            # 상단의 "완료" 버튼이 클릭되었을 때 처리
            if 'complete_action' in locals() and complete_action:
                # 상태 초기화
                st.session_state.pending_save = False
                st.session_state.save_data = None
                st.session_state.emotion_confirmed = False
                st.session_state.confirmed_emotion = None
                
                st.rerun()
            
            # 다운로드 버튼
            st.subheader(f"📥 파일 다운로드: **{st.session_state.confirmed_emotion}**")
            
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                if os.path.exists(save_data['video_path']):
                    with open(save_data['video_path'], 'rb') as f:
                        video_bytes = f.read()
                        st.download_button(
                            label="📥 영상 일기 (MP4)",
                            data=video_bytes,
                            file_name=save_data['video_filename'],
                            mime="video/mp4",
                            type="primary",
                            use_container_width=True,
                            key="download_video_saved"
                        )
                else:
                    st.warning("⚠️ 영상 파일 없음")
            
            with col_dl2:
                if os.path.exists(save_data['text_path']):
                    with open(save_data['text_path'], 'r', encoding='utf-8') as f:
                        text_content = f.read()
                        st.download_button(
                            label="📄 일기 텍스트 (TXT)",
                            data=text_content,
                            file_name=save_data['text_filename'],
                            mime="text/plain",
                            type="secondary",
                            use_container_width=True,
                            key="download_text_saved"
                        )
                else:
                    st.warning("⚠️ 텍스트 파일 없음")
            
            # 현재 세션 감정 분석 표시
            st.markdown("---")
            st.subheader("📊 현재 세션 감정 분석")
            
            if save_data['emotion_timeline'] and len(save_data['emotion_timeline']) > 0:
                timeline_df = pd.DataFrame(save_data['emotion_timeline'])
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    emotion_counts = timeline_df['emotion'].value_counts()
                    fig_pie = px.pie(
                        values=emotion_counts.values,
                        names=emotion_counts.index,
                        title="감정 분포",
                        color_discrete_sequence=px.colors.qualitative.Set3
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_chart2:
                    # 프레임을 시간(초)으로 변환 (20fps 기준)
                    timeline_df['time_seconds'] = timeline_df['frame'] / 20
                    timeline_df['confidence_percent'] = timeline_df['confidence'] * 100
                    
                    # 영상 총 길이 계산
                    max_time = timeline_df['time_seconds'].max()
                    
                    fig_line = px.line(
                        timeline_df,
                        x='time_seconds',
                        y='confidence_percent',
                        color='emotion',
                        title="프레임별 감정 변화 (시간축)",
                        markers=True,
                        labels={
                            'time_seconds': '영상 시간 (초)',
                            'confidence_percent': '확신도 (%)',
                            'emotion': '감정'
                        }
                    )
                    
                    # Y축을 0~100%로 고정, 10% 단위
                    fig_line.update_yaxes(
                        range=[0, 100],
                        dtick=10,
                        title="확신도 (%)"
                    )
                    
                    # X축을 영상 길이에 맞게 10초 단위로 설정
                    import math
                    x_max = math.ceil(max_time / 10) * 10  # 10초 단위로 올림
                    fig_line.update_xaxes(
                        range=[0, x_max],
                        dtick=10,  # 10초 단위
                        title="영상 시간 (초)"
                    )
                    
                    st.plotly_chart(fig_line, use_container_width=True)
                
                st.subheader("📋 감정 타임라인")
                display_timeline = timeline_df[['frame', 'timestamp', 'emotion', 'confidence']].copy()
                display_timeline['confidence'] = display_timeline['confidence'].apply(lambda x: f"{x*100:.1f}%")
                st.dataframe(display_timeline, use_container_width=True, height=200)

# 익명화 맵핑
anonymize_map = {
    "원본": None,
    "블러": "blur",
    "픽셀화": "pixelate",
    "카툰": "cartoon"
}

# 웹캠 실행 (녹화 중일 때만)
if st.session_state.webcam_active:
    loading_image = np.zeros((480, 640, 3), dtype=np.uint8)
    loading_image[:] = (50, 50, 50)
    
    webcam_placeholder.image(loading_image, channels="BGR", width=640)
    status_placeholder.info("📹 웹캠을 시작하는 중입니다...")
    
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        status_placeholder.error("❌ 웹캠을 열 수 없습니다. 웹캠이 연결되어 있는지 확인하세요.")
        st.session_state.webcam_active = False
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        init_image = np.zeros((480, 640, 3), dtype=np.uint8)
        init_image[:] = (50, 50, 50)
        
        webcam_placeholder.image(init_image, channels="BGR", width=640)
        status_placeholder.info("📹 웹캠 초기화 중...")
        
        for _ in range(5):
            ret, _ = cap.read()
            if not ret:
                break
            time.sleep(0.1)
        
        ready_image = np.zeros((480, 640, 3), dtype=np.uint8)
        ready_image[:] = (50, 50, 50)
        
        webcam_placeholder.image(ready_image, channels="BGR", width=640)
        status_placeholder.success("✅ 웹캠 준비 완료! 🎤 음성 입력 활성화됨!")
        time.sleep(0.5)
        
        frame_count = 0
        
        while st.session_state.webcam_active:
            ret, frame = cap.read()
            
            if not ret:
                st.error("❌ 프레임을 읽을 수 없습니다.")
                break
            
            # 음성 인식 텍스트 업데이트
            text_updated = False
            if st.session_state.voice_recording:
                try:
                    while not st.session_state.audio_queue.empty():
                        new_text = st.session_state.audio_queue.get_nowait()
                        if st.session_state.transcribed_text:
                            st.session_state.transcribed_text += " " + new_text
                        else:
                            st.session_state.transcribed_text = new_text
                        text_updated = True
                except queue.Empty:
                    pass
            
            # 전체 프레임 익명화 적용
            anonymized_frame = frame.copy()
            if anonymize_map[anonymize_option] == "blur":
                anonymized_frame = blur_frame(anonymized_frame)
            elif anonymize_map[anonymize_option] == "pixelate":
                anonymized_frame = pixelate_frame(anonymized_frame)
            elif anonymize_map[anonymize_option] == "cartoon":
                anonymized_frame = cartoonize_frame(anonymized_frame)
            
            # 감정 분석 (3프레임마다)
            face_bbox = None
            if frame_count % 3 == 0:
                emotion, confidence, face_bbox = analyze_emotion_quick(
                    frame.copy(), emotion_model, face_detector
                )
                st.session_state.current_emotion = emotion
                
                if st.session_state.recording:
                    st.session_state.emotion_timeline.append({
                        'frame': len(st.session_state.video_frames) + 1,
                        'emotion': emotion,
                        'confidence': confidence,
                        'timestamp': datetime.now().strftime('%H:%M:%S')
                    })
            
            display_frame = anonymized_frame.copy()
            
            # 감정 오버레이 추가
            if show_emotion_overlay and face_bbox:
                x, y, w, h = face_bbox
                emotion = st.session_state.current_emotion
                
                if frame_count % 3 == 0 and 'confidence' in locals():
                    text = f"{emotion} ({confidence*100:.1f}%)"
                else:
                    text = f"{emotion}"
                
                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                display_frame = put_korean_text(
                    display_frame,
                    text,
                    (x, y-35),
                    font_size=20,
                    color=(0, 255, 0)
                )
            
            if st.session_state.recording:
                st.session_state.video_frames.append(display_frame)
            
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            webcam_placeholder.image(frame_rgb, channels="RGB", width=640)
            
            # 감정 이모지
            emotion_emoji = {
                'happy': '😊', 'sad': '😢', 'angry': '😠', 
                'surprise': '😲', 'neutral': '😐', 'fear': '😨',
                'disgust': '🤢', 'joy': '😄'
            }
            emoji = emotion_emoji.get(st.session_state.current_emotion.lower(), '😐')
            
            # 녹화 상태 업데이트
            if st.session_state.recording:
                if st.session_state.recording_start_time:
                    elapsed = datetime.now() - st.session_state.recording_start_time
                    elapsed_seconds = int(elapsed.total_seconds())
                    minutes = elapsed_seconds // 60
                    seconds = elapsed_seconds % 60
                    
                    voice_status = ""
                    if st.session_state.voice_recording:
                        word_count = len(st.session_state.transcribed_text.split()) if st.session_state.transcribed_text else 0
                        voice_status = f" | 🎤 음성 인식 중 (단어: {word_count}개)"
                    
                    emotion_status = f" | {emoji} {st.session_state.current_emotion}"
                    
                    status_placeholder.success(
                        f"🔴 녹화 중: {minutes:02d}:{seconds:02d} | 프레임: {len(st.session_state.video_frames)}{emotion_status}{voice_status}"
                    )
            else:
                if len(st.session_state.video_frames) > 0:
                    status_placeholder.warning(f"⏹️ 녹화 중지됨 ({len(st.session_state.video_frames)} 프레임)")
                else:
                    status_placeholder.info("⚪ 대기 중")
            
            frame_count += 1
        
        cap.release()

# 감정 변화 시각화 (녹화 중이거나 pending_save가 아닐 때만 표시)
if st.session_state.emotion_timeline and len(st.session_state.emotion_timeline) > 0 and not st.session_state.pending_save:
    st.markdown("---")
    st.subheader("📊 현재 세션 감정 분석")
    
    timeline_df = pd.DataFrame(st.session_state.emotion_timeline)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        emotion_counts = timeline_df['emotion'].value_counts()
        fig_pie = px.pie(
            values=emotion_counts.values,
            names=emotion_counts.index,
            title="감정 분포",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_chart2:
        # 프레임을 시간(초)으로 변환 (20fps 기준)
        timeline_df['time_seconds'] = timeline_df['frame'] / 20
        timeline_df['confidence_percent'] = timeline_df['confidence'] * 100
        
        fig_line = px.line(
            timeline_df,
            x='time_seconds',
            y='confidence_percent',
            color='emotion',
            title="프레임별 감정 변화 (시간축)",
            markers=True,
            labels={
                'time_seconds': '시간 (초)',
                'confidence_percent': '확신도 (%)',
                'emotion': '감정'
            }
        )
        
        # Y축을 0~100%로 고정, 10% 단위
        fig_line.update_yaxes(
            range=[0, 100],
            dtick=10,
            title="확신도 (%)"
        )
        
        # X축을 영상 길이에 맞게 설정
        max_time = timeline_df['time_seconds'].max()
        fig_line.update_xaxes(
            range=[0, max_time + 0.5],
            title="영상 시간 (초)"
        )
        
        st.plotly_chart(fig_line, use_container_width=True)
    
    st.subheader("📋 감정 타임라인")
    display_timeline = timeline_df[['frame', 'timestamp', 'emotion', 'confidence']].copy()
    display_timeline['confidence'] = display_timeline['confidence'].apply(lambda x: f"{x*100:.1f}%")
    st.dataframe(display_timeline, use_container_width=True, height=200)

# 저장된 일기 목록
st.markdown("---")
st.subheader("📚 저장된 영상 일기")

if st.session_state.diary_entries:
    for i, entry in enumerate(reversed(st.session_state.diary_entries)):
        emotion_display = f" - 감정: {entry.get('emotion', '미기록')}" if 'emotion' in entry else ""
        with st.expander(f"📔 일기 #{len(st.session_state.diary_entries)-i} - {entry['timestamp']}{emotion_display}"):
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                if 'emotion' in entry:
                    st.write(f"**✨ 오늘의 감정:** {entry['emotion']}")
                st.write("**📝 일기 내용 (음성 입력):**")
                st.write(entry['diary_text'])
                st.write(f"**🎭 주요 감정:** {entry['dominant_emotion']}")
                st.write(f"**📊 평균 확신도:** {entry['avg_confidence']*100:.1f}%")
                st.write("**🎤 음성 입력:** 사용됨 ✓")
            
            with col_info2:
                st.write(f"**🎬 프레임 수:** {entry['frame_count']}")
                st.write(f"**🔒 익명화:** {entry['anonymize_method']}")
                st.write(f"**⏱️ 녹화 시간:** {entry.get('recording_duration', '00:00')}")
                st.write(f"**📏 영상 길이:** 약 {entry['frame_count'] / 20:.1f}초")
                
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    if os.path.exists(entry['video_path']):
                        with open(entry['video_path'], 'rb') as f:
                            video_bytes = f.read()
                            st.download_button(
                                label="📥 영상",
                                data=video_bytes,
                                file_name=entry['video_filename'],
                                mime="video/mp4",
                                key=f"download_video_{i}",
                                use_container_width=True
                            )
                    else:
                        st.warning("⚠️ 영상 파일 없음")
                
                with col_dl2:
                    if os.path.exists(entry['text_path']):
                        with open(entry['text_path'], 'r', encoding='utf-8') as f:
                            text_content = f.read()
                            st.download_button(
                                label="📄 텍스트",
                                data=text_content,
                                file_name=entry['text_filename'],
                                mime="text/plain",
                                key=f"download_text_{i}",
                                use_container_width=True
                            )
                    else:
                        st.warning("⚠️ 텍스트 파일 없음")
            
            if entry.get('emotion_timeline') and len(entry['emotion_timeline']) > 0:
                st.markdown("---")
                st.write("**📊 감정 분석**")
                
                timeline_df = pd.DataFrame(entry['emotion_timeline'])
                
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    emotion_counts = timeline_df['emotion'].value_counts()
                    fig_pie = px.pie(
                        values=emotion_counts.values,
                        names=emotion_counts.index,
                        title="감정 분포",
                        color_discrete_sequence=px.colors.qualitative.Set3,
                        height=300
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col_chart2:
                    # 프레임을 시간(초)으로 변환 (20fps 기준)
                    timeline_df['time_seconds'] = timeline_df['frame'] / 20
                    timeline_df['confidence_percent'] = timeline_df['confidence'] * 100
                    
                    # 영상 총 길이 계산
                    max_time = timeline_df['time_seconds'].max()
                    
                    fig_line = px.line(
                        timeline_df,
                        x='time_seconds',
                        y='confidence_percent',
                        color='emotion',
                        title="프레임별 감정 변화 (시간축)",
                        markers=True,
                        height=300,
                        labels={
                            'time_seconds': '영상 시간 (초)',
                            'confidence_percent': '확신도 (%)',
                            'emotion': '감정'
                        }
                    )
                    
                    # Y축을 0~100%로 고정, 10% 단위
                    fig_line.update_yaxes(
                        range=[0, 100],
                        dtick=10,
                        title="확신도 (%)"
                    )
                    
                    # X축을 영상 길이에 맞게 10초 단위로 설정
                    import math
                    x_max = math.ceil(max_time / 10) * 10  # 10초 단위로 올림
                    fig_line.update_xaxes(
                        range=[0, x_max],
                        dtick=10,  # 10초 단위
                        title="영상 시간 (초)"
                    )
                    
                    st.plotly_chart(fig_line, use_container_width=True)
else:
    st.info("📭 아직 저장된 영상 일기가 없습니다. 위에서 녹화를 시작해보세요!")
