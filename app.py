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
import json
import pickle
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from collections import Counter

# 로컬 데이터 저장 경로 설정 (현재 폴더에 저장)
DATA_DIR = Path("emotion_diary_data")
DATA_DIR.mkdir(exist_ok=True)

DIARY_DATA_FILE = DATA_DIR / "diary_entries.json"
USER_MODEL_FILE = DATA_DIR / "user_emotion_model.pkl"
USER_STATS_FILE = DATA_DIR / "user_stats.json"
VIDEOS_DIR = DATA_DIR / "videos"
VIDEOS_DIR.mkdir(exist_ok=True)

# 영어-한글 감정 매핑
EMOTION_TRANSLATION = {
    'happy': '행복',
    'sad': '슬픔',
    'angry': '화남',
    'surprise': '놀람',
    'neutral': '중립',
    'fear': '두려움',
    'disgust': '혐오',
    'joy': '행복'  # joy는 happy와 동일하게 처리
}

# 한글-영어 감정 역매핑
EMOTION_REVERSE_TRANSLATION = {v: k for k, v in EMOTION_TRANSLATION.items() if k != 'joy'}

# 페이지 설정
st.set_page_config(
    page_title="감정 일기 - Emotion Diary",
    page_icon="📔",
    layout="wide"
)

# 로컬 데이터 로드 함수
def load_local_data():
    """로컬에 저장된 일기 데이터 로드"""
    if DIARY_DATA_FILE.exists():
        try:
            with open(DIARY_DATA_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                # 경로가 존재하는 항목만 필터링
                valid_entries = []
                for entry in data:
                    if os.path.exists(entry.get('video_path', '')):
                        valid_entries.append(entry)
                return valid_entries
        except Exception as e:
            st.warning(f"데이터 로드 중 오류: {e}")
            return []
    return []

def save_local_data(entries):
    """일기 데이터를 로컬에 저장"""
    try:
        with open(DIARY_DATA_FILE, 'w', encoding='utf-8') as f:
            json.dump(entries, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        st.error(f"데이터 저장 중 오류: {e}")
        return False

def load_user_stats():
    """사용자 통계 로드"""
    if USER_STATS_FILE.exists():
        try:
            with open(USER_STATS_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            pass
    return {
        'total_entries': 0,
        'emotion_distribution': {},
        'ai_vs_user_agreement': 0,
        'last_updated': None
    }

def save_user_stats(stats):
    """사용자 통계 저장"""
    try:
        with open(USER_STATS_FILE, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"통계 저장 오류: {e}")

# 사용자 맞춤 모델 클래스
class PersonalizedEmotionModel:
    def __init__(self):
        self.model = None
        self.emotion_mapping = {
            '행복': 0, '슬픔': 1, '화남': 2, '놀람': 3,
            '중립': 4, '두려움': 5, '혐오': 6
        }
        self.reverse_mapping = {v: k for k, v in self.emotion_mapping.items()}
        self.training_data = []
        self.load_model()
    
    def load_model(self):
        """저장된 모델 로드"""
        if USER_MODEL_FILE.exists():
            try:
                with open(USER_MODEL_FILE, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data.get('model')
                    self.training_data = data.get('training_data', [])
                    print(f"✅ 모델 로드 완료: {len(self.training_data)}개 학습 데이터")
                    return True
            except Exception as e:
                print(f"❌ 모델 로드 오류: {e}")
        else:
            print(f"ℹ️ 저장된 모델 없음 (경로: {USER_MODEL_FILE})")
        return False
    
    def save_model(self):
        """모델 저장"""
        try:
            with open(USER_MODEL_FILE, 'wb') as f:
                pickle.dump({
                    'model': self.model,
                    'training_data': self.training_data
                }, f)
            print(f"✅ 모델 저장 완료: {len(self.training_data)}개 학습 데이터")
            return True
        except Exception as e:
            print(f"❌ 모델 저장 오류: {e}")
            return False
    
    def extract_features(self, emotion_timeline, text):
        """감정 타임라인과 텍스트에서 특징 추출"""
        if not emotion_timeline:
            return None
        
        # 감정 분포
        emotions = [e['emotion'] for e in emotion_timeline]
        emotion_counts = Counter(emotions)
        
        # 특징 벡터 생성
        features = []
        
        # 각 감정의 비율
        for emotion in ['행복', '슬픔', '화남', '놀람', '중립', '두려움', '혐오']:
            features.append(emotion_counts.get(emotion, 0) / len(emotions))
        
        # 평균 확신도
        avg_confidence = np.mean([e['confidence'] for e in emotion_timeline])
        features.append(avg_confidence)
        
        # 감정 변화 횟수 (감정이 바뀐 횟수)
        emotion_changes = sum(1 for i in range(1, len(emotions)) if emotions[i] != emotions[i-1])
        features.append(emotion_changes / len(emotions))
        
        # 텍스트 길이
        text_length = len(text.split()) if text else 0
        features.append(min(text_length / 100, 1.0))  # 정규화
        
        # 가장 빈번한 감정
        most_common = emotion_counts.most_common(1)[0][0]
        features.append(self.emotion_mapping.get(most_common, 4))
        
        return features
    
    def add_training_sample(self, emotion_timeline, text, ai_emotion, user_emotion):
        """학습 샘플 추가"""
        features = self.extract_features(emotion_timeline, text)
        if features is None:
            print("❌ 학습 데이터 추가 실패: 특징 추출 오류")
            return False
        
        user_emotion_code = self.emotion_mapping.get(user_emotion, 4)
        
        self.training_data.append({
            'features': features,
            'ai_emotion': ai_emotion,
            'user_emotion': user_emotion_code,
            'timestamp': datetime.now().isoformat()
        })
        
        print(f"✅ 학습 데이터 추가 완료: 총 {len(self.training_data)}개 (AI: {ai_emotion}, 사용자: {user_emotion})")
        
        # 학습 데이터 추가 후 항상 저장
        self.save_model()
        
        return True
    
    def train(self):
        """모델 학습"""
        if len(self.training_data) < 3:
            return False
        
        X = [d['features'] for d in self.training_data]
        y = [d['user_emotion'] for d in self.training_data]
        
        self.model = RandomForestClassifier(n_estimators=50, random_state=42)
        self.model.fit(X, y)
        
        self.save_model()
        return True
    
    def predict(self, emotion_timeline, text, ai_emotion):
        """맞춤형 감정 예측"""
        if self.model is None or len(self.training_data) < 3:
            # 학습 데이터가 부족하면 AI 예측 사용
            return ai_emotion, 0.0, False
        
        features = self.extract_features(emotion_timeline, text)
        if features is None:
            return ai_emotion, 0.0, False
        
        try:
            prediction = self.model.predict([features])[0]
            probabilities = self.model.predict_proba([features])[0]
            confidence = max(probabilities)
            
            predicted_emotion = self.reverse_mapping.get(prediction, '중립')
            return predicted_emotion, confidence, True
        except Exception as e:
            print(f"예측 오류: {e}")
            return ai_emotion, 0.0, False

# 세션 상태 초기화
if 'diary_entries' not in st.session_state:
    st.session_state.diary_entries = load_local_data()
if 'recording' not in st.session_state:
    st.session_state.recording = False
if 'video_frames' not in st.session_state:
    st.session_state.video_frames = []
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = '중립'
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
if 'emotion_confirmed' not in st.session_state:
    st.session_state.emotion_confirmed = False
if 'confirmed_emotion' not in st.session_state:
    st.session_state.confirmed_emotion = None
if 'personalized_model' not in st.session_state:
    st.session_state.personalized_model = PersonalizedEmotionModel()
if 'user_stats' not in st.session_state:
    st.session_state.user_stats = load_user_stats()

# 사이드바
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
    
    st.header("🤖 AI 개인화 상태")
    
    model_trained = st.session_state.personalized_model.model is not None
    training_count = len(st.session_state.personalized_model.training_data)
    
    if model_trained:
        st.success(f"✅ 맞춤 모델 활성화됨")
        st.metric("학습 데이터", f"{training_count}개")
        
        # 일치율 표시
        if st.session_state.user_stats['total_entries'] > 0:
            agreement = st.session_state.user_stats['ai_vs_user_agreement']
            st.metric("AI-사용자 일치율", f"{agreement:.1f}%")
    else:
        st.info(f"📊 학습 데이터 수집 중")
        st.metric("수집된 데이터", f"{training_count}/3개")
        if training_count < 3:
            st.caption(f"맞춤 모델 활성화까지 {3-training_count}개 필요")
    
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
    
    ### 🤖 AI 개인화 기능
    - **자동 학습**: 일기를 3개 이상 작성하면 자동으로 맞춤 모델이 활성화됩니다
    - **맞춤 추천**: 사용자의 과거 선택 패턴을 학습하여 더 정확한 감정을 추천합니다
    - **지속 개선**: 일기를 작성할수록 AI가 사용자를 더 잘 이해합니다
    
    ### 🎭 지원 감정
    - 😊 행복
    - 😢 슬픔
    - 😠 화남
    - 😲 놀람
    - 😐 중립
    - 😨 두려움
    - 🤢 혐오
    
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
    
    ### 💾 데이터 저장 위치
    - **저장 폴더**: `emotion_diary_data/`
    - **영상 파일**: `emotion_diary_data/videos/`
    - 앱 실행 폴더에 자동으로 생성됩니다
    - 모든 데이터는 로컬에 안전하게 보관됩니다
    """)
    
    st.markdown("---")
    
    # 통계
    st.subheader("📈 전체 통계")
    total_entries = len(st.session_state.diary_entries)
    total_frames = sum([e.get('frame_count', 0) for e in st.session_state.diary_entries])
    voice_entries = len(st.session_state.diary_entries)
    st.metric("총 일기 수", total_entries)
    st.metric("총 프레임 수", total_frames)
    st.metric("음성 입력 사용", f"{voice_entries}회")
    
    # 감정 분포 표시
    if st.session_state.user_stats['emotion_distribution']:
        st.markdown("**감정 분포**")
        for emotion, count in sorted(
            st.session_state.user_stats['emotion_distribution'].items(), 
            key=lambda x: x[1], 
            reverse=True
        ):
            st.caption(f"{emotion}: {count}회")
    
    st.markdown("---")
    
    if st.button("🗑️ 모든 기록 초기화", type="secondary"):
        # 로컬 파일 삭제
        try:
            if DIARY_DATA_FILE.exists():
                DIARY_DATA_FILE.unlink()
            if USER_MODEL_FILE.exists():
                USER_MODEL_FILE.unlink()
            if USER_STATS_FILE.exists():
                USER_STATS_FILE.unlink()
        except Exception as e:
            st.error(f"파일 삭제 오류: {e}")
        
        # 세션 상태 초기화
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
        st.session_state.personalized_model = PersonalizedEmotionModel()
        st.session_state.user_stats = {
            'total_entries': 0,
            'emotion_distribution': {},
            'ai_vs_user_agreement': 0,
            'last_updated': None
        }
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
    emotion = '중립'
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
                    # 영어 감정을 한글로 변환
                    emotion_eng = emotion_results[0]['label'].lower()
                    emotion = EMOTION_TRANSLATION.get(emotion_eng, '중립')
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

# 메인 UI
st.title("📔 감정 영상 일기 - Emotion Video Diary")
st.markdown("*웹캠으로 실시간 감정을 분석하며 음성으로 영상 일기를 작성하세요*")

st.markdown("---")

st.subheader("📹 웹캠 화면")

# 1. 녹화 상태 표시
status_placeholder = st.empty()

# 녹화 전 상태 표시
if not st.session_state.webcam_active and not st.session_state.pending_save:
    status_placeholder.info("아래 녹화 시작 버튼을 눌러주세요")

# 레이아웃 구성
if st.session_state.pending_save:
    webcam_placeholder = st.empty()
else:
    col_webcam, col_text = st.columns([2, 1])

    with col_webcam:
        webcam_placeholder = st.empty()
        
        if not st.session_state.webcam_active and not st.session_state.pending_save:
            waiting_image = np.zeros((480, 640, 3), dtype=np.uint8)
            waiting_image[:] = (50, 50, 50)
            webcam_placeholder.image(waiting_image, channels="BGR", width=640)

    with col_text:
        voice_text_placeholder = st.empty()
        
        if st.session_state.recording and st.session_state.voice_recording:
            current_text = st.session_state.transcribed_text if st.session_state.transcribed_text else "(음성 인식 중... 말씀해주세요)"
            voice_text_placeholder.text_area(
                f"음성 텍스트",
                value=current_text,
                height=480,
                disabled=True,
                key=f"voice_display_{time.time()}"
            )
        elif st.session_state.transcribed_text:
            voice_text_placeholder.text_area(
                f"음성 텍스트",
                value=st.session_state.transcribed_text,
                height=480,
                disabled=True,
                key="voice_display_saved"
            )
        else:
            voice_text_placeholder.text_area(
                "음성 텍스트",
                value="(음성 입력 대기 중...)",
                height=480,
                disabled=True,
                key="voice_display_empty"
            )

# 3. 녹화 버튼
if not st.session_state.pending_save:
    if not st.session_state.recording:
        start_recording = st.button("🔴 녹화 시작", type="primary", use_container_width=True)
    else:
        start_recording = False
        stop_recording = st.button("⏹️ 녹화 중지 & 저장", type="secondary", use_container_width=True)

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
            
            # 로컬 폴더에 저장
            video_path = str(VIDEOS_DIR / video_filename)
            text_path = str(VIDEOS_DIR / text_filename)
            
            save_video(st.session_state.video_frames, video_path, fps=20)
            
            if st.session_state.emotion_timeline:
                emotions_list = [e['emotion'] for e in st.session_state.emotion_timeline]
                emotion_counts = pd.Series(emotions_list).value_counts()
                dominant_emotion = emotion_counts.index[0] if len(emotion_counts) > 0 else "중립"
                avg_confidence = np.mean([e['confidence'] for e in st.session_state.emotion_timeline])
            else:
                dominant_emotion = "중립"
                avg_confidence = 0.0
            
            # 맞춤형 AI 예측
            personalized_emotion, personalized_confidence, is_personalized = \
                st.session_state.personalized_model.predict(
                    st.session_state.emotion_timeline,
                    final_text,
                    dominant_emotion
                )
            
            if st.session_state.recording_start_time:
                elapsed = datetime.now() - st.session_state.recording_start_time
                elapsed_seconds = int(elapsed.total_seconds())
                recording_duration = f"{elapsed_seconds // 60:02d}:{elapsed_seconds % 60:02d}"
            else:
                recording_duration = "00:00"
            
            st.session_state.save_data = {
                'timestamp': timestamp,
                'video_filename': video_filename,
                'video_path': video_path,
                'text_filename': text_filename,
                'text_path': text_path,
                'final_text': final_text,
                'dominant_emotion': dominant_emotion,
                'avg_confidence': avg_confidence,
                'frame_count': len(st.session_state.video_frames),
                'recording_duration': recording_duration,
                'emotion_timeline': st.session_state.emotion_timeline.copy(),
                'anonymize_method': anonymize_option,
                'personalized_emotion': personalized_emotion,
                'personalized_confidence': personalized_confidence,
                'is_personalized': is_personalized
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

# 기분 선택 UI
if st.session_state.pending_save and st.session_state.save_data:
    save_data = st.session_state.save_data
    
    if not st.session_state.emotion_confirmed:
        # AI 추천 표시
        if save_data['is_personalized']:
            status_placeholder.success(
                f"✨ 맞춤형 AI 추천: **{save_data['personalized_emotion']}** "
                f"(확신도: {save_data['personalized_confidence']*100:.1f}%)"
            )
            st.info("🤖 사용자님의 과거 감정 패턴을 분석한 맞춤 추천입니다!")
        else:
            status_placeholder.info(
                f"✨ AI 추천: **{save_data['dominant_emotion']}** "
                f"(기본 분석 결과)"
            )
        
        emotion_options = [
            "😊 행복",
            "😢 슬픔",
            "😠 화남",
            "😲 놀람",
            "😐 중립",
            "😨 두려움",
            "🤢 혐오"
        ]
        
        # 추천 감정을 기본 선택으로
        recommended_emotion = save_data['personalized_emotion'] if save_data['is_personalized'] else save_data['dominant_emotion']
        emotion_map = {
            '행복': 0, '슬픔': 1, '화남': 2, '놀람': 3,
            '중립': 4, '두려움': 5, '혐오': 6
        }
        
        default_index = emotion_map.get(recommended_emotion, 4)
        
        selected_emotion = st.radio(
            "🎭 오늘의 감정을 선택해주세요:",
            emotion_options,
            index=default_index,
            key="emotion_radio"
        )
        
        confirm_emotion = st.button("✅ 감정 확정하기", type="primary", use_container_width=True, key="confirm_bottom_btn")
        
        if confirm_emotion:
            # 선택된 감정에서 한글 부분만 추출
            final_mood = selected_emotion.split()[1]
            
            # 학습 데이터에 추가
            st.session_state.personalized_model.add_training_sample(
                save_data['emotion_timeline'],
                save_data['final_text'],
                save_data['dominant_emotion'],
                final_mood
            )
            
            # 모델 학습 (3개 이상일 때)
            if len(st.session_state.personalized_model.training_data) >= 3:
                with st.spinner("🤖 맞춤형 AI 학습 중..."):
                    if st.session_state.personalized_model.train():
                        st.success("✅ AI 학습 완료! 다음 일기부터 더 정확한 추천을 받을 수 있습니다.")
            
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
                f.write(f"\n=== AI 추천 ===\n")
                if save_data['is_personalized']:
                    f.write(f"맞춤형 AI 추천: {save_data['personalized_emotion']} (확신도: {save_data['personalized_confidence']*100:.1f}%)\n")
                else:
                    f.write(f"기본 AI 분석: {save_data['dominant_emotion']}\n")
                f.write(f"최종 선택: {final_mood}\n")
            
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
                'voice_input_used': True,
                'ai_recommended': save_data['personalized_emotion'] if save_data['is_personalized'] else save_data['dominant_emotion'],
                'is_personalized': save_data['is_personalized']
            }
            
            st.session_state.diary_entries.append(entry)
            
            # 로컬에 저장
            save_local_data(st.session_state.diary_entries)
            
            # 통계 업데이트
            stats = st.session_state.user_stats
            stats['total_entries'] += 1
            stats['emotion_distribution'][final_mood] = stats['emotion_distribution'].get(final_mood, 0) + 1
            
            # AI-사용자 일치율 계산
            recommended = save_data['personalized_emotion'] if save_data['is_personalized'] else save_data['dominant_emotion']
            matches = sum(1 for e in st.session_state.diary_entries 
                         if e['emotion'] == e['ai_recommended'])
            stats['ai_vs_user_agreement'] = (matches / stats['total_entries']) * 100
            stats['last_updated'] = datetime.now().isoformat()
            
            save_user_stats(stats)
            st.session_state.user_stats = stats
            
            st.session_state.confirmed_emotion = final_mood
            st.session_state.emotion_confirmed = True
            
            st.rerun()
    
    else:
        status_placeholder.success(f"✅ 영상 일기가 저장되었습니다! (감정: {st.session_state.confirmed_emotion})")
        
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
                timeline_df['time_seconds'] = timeline_df['frame'] / 20
                timeline_df['confidence_percent'] = timeline_df['confidence'] * 100
                
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
                
                fig_line.update_yaxes(range=[0, 100], dtick=10, title="확신도 (%)")
                
                import math
                x_max = math.ceil(max_time / 10) * 10
                fig_line.update_xaxes(range=[0, x_max], dtick=10, title="영상 시간 (초)")
                
                st.plotly_chart(fig_line, use_container_width=True)
            
            st.markdown("**📋 감정 타임라인**")
            display_timeline = timeline_df[['frame', 'timestamp', 'emotion', 'confidence']].copy()
            display_timeline['confidence'] = display_timeline['confidence'].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(display_timeline, use_container_width=True, height=200)
        
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            if os.path.exists(save_data['video_path']):
                with open(save_data['video_path'], 'rb') as f:
                    video_bytes = f.read()
                    st.download_button(
                        label="📥 영상 일기 (MP4) 파일 다운로드",
                        data=video_bytes,
                        file_name=save_data['video_filename'],
                        mime="video/mp4",
                        type="secondary",
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
                        label="📄 일기 텍스트 (TXT) 파일 다운로드",
                        data=text_content,
                        file_name=save_data['text_filename'],
                        mime="text/plain",
                        type="secondary",
                        use_container_width=True,
                        key="download_text_saved"
                    )
            else:
                st.warning("⚠️ 텍스트 파일 없음")
        
        complete_action = st.button("✅ 확인 완료", type="primary", use_container_width=True, key="complete_bottom_btn")
        
        if complete_action:
            st.session_state.pending_save = False
            st.session_state.save_data = None
            st.session_state.emotion_confirmed = False
            st.session_state.confirmed_emotion = None
            
            st.rerun()

# 익명화 맵핑
anonymize_map = {
    "원본": None,
    "블러": "blur",
    "픽셀화": "pixelate",
    "카툰": "cartoon"
}

# 웹캠 실행
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
            
            anonymized_frame = frame.copy()
            if anonymize_map[anonymize_option] == "blur":
                anonymized_frame = blur_frame(anonymized_frame)
            elif anonymize_map[anonymize_option] == "pixelate":
                anonymized_frame = pixelate_frame(anonymized_frame)
            elif anonymize_map[anonymize_option] == "cartoon":
                anonymized_frame = cartoonize_frame(anonymized_frame)
            
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
            
            emotion_emoji = {
                '행복': '😊', '슬픔': '😢', '화남': '😠', 
                '놀람': '😲', '중립': '😐', '두려움': '😨',
                '혐오': '🤢'
            }
            emoji = emotion_emoji.get(st.session_state.current_emotion, '😐')
            
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

# 저장된 일기 목록
st.markdown("---")
st.subheader("📚 저장된 영상 일기")
if st.session_state.diary_entries:
    # 개별 일기
    for i, entry in enumerate(reversed(st.session_state.diary_entries)):
        emotion_display = f" - 감정: {entry.get('emotion', '미기록')}" if 'emotion' in entry else ""
        ai_rec = entry.get('ai_recommended', '없음')
        is_personalized = "AI 맞춤 추천" if entry.get('is_personalized', False) else "AI 기본 추천"
        
        with st.expander(f"📔 일기 #{len(st.session_state.diary_entries)-i} - {entry['timestamp']}{emotion_display} ({is_personalized}: {ai_rec})"):
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                if 'emotion' in entry:
                    st.write(f"**✨ 오늘의 감정:** {entry['emotion']}")
                st.write(f"**🤖 {is_personalized}:** {ai_rec}")
                if entry['emotion'] == ai_rec:
                    st.success("✅ AI 추천과 일치")
                else:
                    st.info("ℹ️ 사용자가 다른 감정 선택")
                
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
                st.write("**📊 AI 감정 분석**")
                
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
                    timeline_df['time_seconds'] = timeline_df['frame'] / 20
                    timeline_df['confidence_percent'] = timeline_df['confidence'] * 100
                    
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
                    
                    fig_line.update_yaxes(range=[0, 100], dtick=10, title="확신도 (%)")
                    
                    import math
                    x_max = math.ceil(max_time / 10) * 10
                    fig_line.update_xaxes(range=[0, x_max], dtick=10, title="영상 시간 (초)")
                    
                    st.plotly_chart(fig_line, use_container_width=True)
    
    # 전체 통계
    st.markdown("---")
    st.markdown("### 📊 전체 감정 분석")
    
    all_emotions = [e['emotion'] for e in st.session_state.diary_entries]
    emotion_df = pd.DataFrame({'감정': all_emotions})
    
    col_stat1, col_stat2 = st.columns(2)
    
    with col_stat1:
        emotion_counts = emotion_df['감정'].value_counts()
        fig_overall = px.pie(
            values=emotion_counts.values,
            names=emotion_counts.index,
            title="전체 감정 분포",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_overall, use_container_width=True)
    
    with col_stat2:
        # AI 추천 vs 사용자 선택 비교
        ai_matches = sum(1 for e in st.session_state.diary_entries 
                        if e['emotion'] == e.get('ai_recommended', ''))
        match_rate = (ai_matches / len(st.session_state.diary_entries)) * 100
        
        st.metric("AI-사용자 일치율", f"{match_rate:.1f}%")
        
        personalized_count = sum(1 for e in st.session_state.diary_entries 
                                if e.get('is_personalized', False))
        st.metric("맞춤형 추천 사용", f"{personalized_count}회")
        st.metric("학습 데이터", f"{len(st.session_state.personalized_model.training_data)}개")
else:
    st.info("📭 아직 저장된 영상 일기가 없습니다. 위에서 녹화를 시작해보세요!")