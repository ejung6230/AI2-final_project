import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import pandas as pd
import plotly.express as px
from datetime import datetime
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

# 제미나이 SDK 임포트
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("⚠️ google-genai 패키지가 설치되지 않았습니다.")
    print("설치 방법: pip install google-genai")

# 제미나이 API 설정
GEMINI_API_KEY = "AIzaSyDdOJZsmnmTjuC0Uc--j1ZKhXsXtUxvR2I"

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
    'happy': '행복함',
    'sad': '슬픔',
    'angry': '화남',
    'surprise': '놀람',
    'neutral': '담담함',
    'fear': '두려움',
    'disgust': '혐오',
    'joy': '행복함'  # joy는 happy와 동일하게 처리
}

# 감정 이모지 매핑
emotion_emoji_map = {
    '행복함': '😊',
    '슬픔': '😢',
    '화남': '😠',
    '놀람': '😲',
    '담담함': '😐',
    '두려움': '😨',
    '혐오': '🤢'
}

# 한글-영어 감정 역매핑
EMOTION_REVERSE_TRANSLATION = {v: k for k, v in EMOTION_TRANSLATION.items() if k != 'joy'}

# 페이지 설정
st.set_page_config(
    page_title="감정 일기 - Emotion Diary",
    page_icon="📔",
    layout="wide"
)

# 제미나이 API 호출 함수
def get_gemini_advice(emotion: str, diary_text: str, emotion_timeline: list) -> str:
    """제미나이 API를 호출하여 감정 일기에 대한 조언을 받습니다"""
    
    # SDK가 설치되지 않은 경우
    if not GEMINI_AVAILABLE:
        return f"⚠️ 제미나이 SDK가 설치되지 않았습니다\n\n설치 방법: pip install google-genai\n\n하지만 걱정하지 마세요! 오늘 '{emotion}' 감정을 솔직하게 기록하신 것만으로도 훌륭합니다. 감정 일기는 자기 성찰의 소중한 도구입니다. 🌈"
    
    try:
        # 감정 분포 계산
        emotions_list = [e['emotion'] for e in emotion_timeline] if emotion_timeline else []
        emotion_counts = Counter(emotions_list)
        
        # 주요 감정만 추출 (횟수 제외, 상위 3개)
        top_emotions = [k for k, v in emotion_counts.most_common(3)]
        emotion_summary = ", ".join(top_emotions) if top_emotions else "분석 데이터 없음"
        
        # 주요 감정 설명
        if len(top_emotions) > 1:
            main_emotion_desc = f"주로 {top_emotions[0]} 감정이 많았고, {', '.join(top_emotions[1:])}도 느끼셨네요"
        elif len(top_emotions) == 1:
            main_emotion_desc = f"대부분 {top_emotions[0]} 감정이었네요"
        else:
            main_emotion_desc = "감정 분석 데이터가 없습니다"
        
        # 프롬프트 구성
        prompt = f"""당신은 공감 능력이 뛰어나고 따뜻한 심리 상담사입니다. 
사용자가 오늘 작성한 감정 일기를 보고 진심어린 조언과 위로를 해주세요.

**사용자의 오늘 감정**: {emotion}
**일기 내용**: {diary_text}
**감정 분석**: {main_emotion_desc}

다음 가이드라인을 따라주세요:
1. 친근하고 따뜻한 톤으로 작성해주세요 (존댓말 사용)
2. 사용자의 감정을 공감하고 인정해주세요
3. 감정 분석 결과의 구체적인 횟수나 숫자는 절대 언급하지 마세요
4. "대부분", "주로", "많이" 같은 표현만 사용하세요
5. 긍정적이고 건설적인 조언을 제공해주세요
6. 필요하다면 구체적인 실천 방법을 제안해주세요
7. 200-300자 내외로 간결하게 작성해주세요
8. 이모지를 적절히 사용하여 친근감을 더해주세요

응답 형식:
[공감과 인정] → [조언 또는 격려] → [마무리 응원]

주의사항: "28회", "10번" 같은 구체적인 숫자나 횟수는 절대 언급하지 마세요!
"""

        # 제미나이 클라이언트 생성
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        print("제미나이 API 호출 시작...")
        
        # API 호출 - gemini-2.5-flash 사용
        response = client.models.generate_content(
            model="gemini-2.5-flash-lite",
            contents=prompt
        )
        
        print("✅ 제미나이 응답 받음!")
        
        # 응답 텍스트 추출
        advice = response.text.strip()
        
        if advice:
            return advice
        else:
            return f"⚠️ AI 조언을 생성할 수 없습니다.\n\n오늘 '{emotion}' 감정을 느끼셨군요. 감정을 기록하고 표현하는 것만으로도 큰 의미가 있습니다. 💙"
            
    except Exception as e:
        error_msg = str(e)
        print(f"❌ 제미나이 API 오류: {error_msg}")
        import traceback
        traceback.print_exc()
        
        # 사용자 친화적인 오류 메시지
        if "API_KEY" in error_msg.upper() or "authentication" in error_msg.lower():
            return f"⚠️ API 키 인증 오류\n\n하지만 오늘 '{emotion}' 감정을 기록하신 것은 매우 의미있는 행동입니다. 💚"
        elif "quota" in error_msg.lower() or "limit" in error_msg.lower():
            return f"⚠️ API 사용 한도 초과\n\n그래도 오늘 '{emotion}' 감정을 표현하신 것만으로 충분합니다. 💪"
        else:
            return f"⚠️ AI 조언 서비스 오류\n\n하지만 오늘 '{emotion}' 감정을 기록하신 것은 매우 의미있는 행동입니다. 감정을 글로 표현하는 것만으로도 마음이 정리되고 치유될 수 있습니다. 💚\n\n💪 지금 이 순간 느끼는 감정을 있는 그대로 받아들여 주세요. 내일은 또 다른 하루가 시작됩니다!"

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
                    video_path = entry.get('video_path', '')
                    # 상대 경로와 절대 경로 모두 확인
                    if os.path.exists(video_path):
                        valid_entries.append(entry)
                    elif os.path.exists(os.path.join(os.getcwd(), video_path)):
                        # 상대 경로를 절대 경로로 업데이트
                        entry['video_path'] = os.path.join(os.getcwd(), video_path)
                        entry['text_path'] = os.path.join(os.getcwd(), entry.get('text_path', ''))
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
            '행복함': 0, '슬픔': 1, '화남': 2, '놀람': 3,
            '담담함': 4, '두려움': 5, '혐오': 6
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
        for emotion in ['행복함', '슬픔', '화남', '놀람', '담담함', '두려움', '혐오']:
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
            
            predicted_emotion = self.reverse_mapping.get(prediction, '담담함')
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
    st.session_state.current_emotion = '담담함'
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
if 'audio_frames_queue' not in st.session_state:
    st.session_state.audio_frames_queue = queue.Queue()
if 'audio_frames' not in st.session_state:
    st.session_state.audio_frames = []
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
if 'gemini_advice' not in st.session_state:
    st.session_state.gemini_advice = None
if 'advice_loading' not in st.session_state:
    st.session_state.advice_loading = False
if 'processing_emotion' not in st.session_state:
    st.session_state.processing_emotion = None

# 사이드바
with st.sidebar:
    st.header("📸 녹화 설정")
    
    anonymize_option = st.selectbox(
        "전체 화면 익명화 방식",
        ["원본", "블러", "곰 얼굴 🐻", "토끼 얼굴 🐰", "고양이 얼굴 🐱"],
        key="anonymize",
        disabled=st.session_state.recording
    )
    
    if st.session_state.recording:
        st.warning("⚠️ 녹화 중에는 설정을 변경할 수 없습니다.")
    
    st.markdown("---")
    
    st.header("AI 개인화 상태")
    
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
    6. 🎭 **제미나이 AI 조언 받기**
    7. 📥 **영상 & 텍스트 다운로드**
    
    ### 🌟 제미나이 AI 조언
    - **개인화된 조언**: 일기 내용과 감정을 분석하여 맞춤 조언 제공
    - **따뜻한 위로**: 공감과 격려의 메시지
    - **실천 가능한 팁**: 구체적인 개선 방법 제안
    
    ### ✨ AI 개인화 기능
    - **자동 학습**: 일기를 3개 이상 작성하면 자동으로 맞춤 모델이 활성화됩니다
    - **맞춤 추천**: 사용자의 과거 선택 패턴을 학습하여 더 정확한 감정을 추천합니다
    - **지속 개선**: 일기를 작성할수록 AI가 사용자를 더 잘 이해합니다
    
    ### 🎭 지원 감정
    - 😊 행복함
    - 😢 슬픔
    - 😠 화남
    - 😲 놀람
    - 😐 담담함
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
        st.session_state.gemini_advice = None
        st.session_state.advice_loading = False
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

def bear_face_mask(image: np.ndarray, face_detector) -> np.ndarray:
    """얼굴을 귀여운 곰 얼굴로 대체"""
    try:
        if image.size == 0 or image.shape[0] < 10 or image.shape[1] < 10:
            return image
        
        result = image.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detector.process(image_rgb)
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # 경계 확인
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                if width <= 0 or height <= 0:
                    continue
                
                # 얼굴 영역을 약간 확장 (귀 포함)
                margin = int(width * 0.3)
                bear_x = max(0, x - margin)
                bear_y = max(0, y - margin)
                bear_w = min(w - bear_x, width + margin * 2)
                bear_h = min(h - bear_y, height + margin * 2)
                
                # 곰 얼굴 그리기
                center_x = bear_x + bear_w // 2
                center_y = bear_y + bear_h // 2
                
                # 얼굴 (원) - 연한 갈색
                face_radius = min(bear_w, bear_h) // 2
                cv2.circle(result, (center_x, center_y), face_radius, (150, 120, 80), -1)  # 연한 갈색
                cv2.circle(result, (center_x, center_y), face_radius, (100, 70, 40), 3)  # 진한 갈색 테두리
                
                # 귀 (2개) - 갈색
                ear_radius = face_radius // 3
                left_ear_x = center_x - int(face_radius * 0.7)
                right_ear_x = center_x + int(face_radius * 0.7)
                ear_y = center_y - int(face_radius * 0.7)
                
                # 귀 본체
                cv2.circle(result, (left_ear_x, ear_y), ear_radius, (150, 120, 80), -1)
                cv2.circle(result, (left_ear_x, ear_y), ear_radius, (100, 70, 40), 2)
                cv2.circle(result, (right_ear_x, ear_y), ear_radius, (150, 120, 80), -1)
                cv2.circle(result, (right_ear_x, ear_y), ear_radius, (100, 70, 40), 2)
                
                # 귀 안쪽 - 밝은 노란색
                inner_ear_radius = ear_radius // 2
                cv2.circle(result, (left_ear_x, ear_y), inner_ear_radius, (100, 200, 255), -1)  # 노란색
                cv2.circle(result, (right_ear_x, ear_y), inner_ear_radius, (100, 200, 255), -1)
                
                # 🎀 리본 추가 (오른쪽 귀 옆)
                ribbon_center_x = right_ear_x + int(ear_radius * 1.2)
                ribbon_center_y = ear_y - int(ear_radius * 0.3)
                ribbon_size = ear_radius // 2
                
                # 리본 왼쪽 나비
                ribbon_left = (ribbon_center_x - ribbon_size, ribbon_center_y)
                cv2.circle(result, ribbon_left, ribbon_size, (100, 100, 255), -1)  # 분홍색
                
                # 리본 오른쪽 나비
                ribbon_right = (ribbon_center_x + ribbon_size, ribbon_center_y)
                cv2.circle(result, ribbon_right, ribbon_size, (100, 100, 255), -1)
                
                # 리본 중앙 매듭
                cv2.circle(result, (ribbon_center_x, ribbon_center_y), ribbon_size // 2, (80, 80, 200), -1)
                
                # 얼굴 중앙 부분 - 밝은 노란색
                snout_radius = face_radius // 2
                snout_y = center_y + face_radius // 4
                cv2.circle(result, (center_x, snout_y), snout_radius, (120, 220, 255), -1)  # 밝은 노란색
                cv2.circle(result, (center_x, snout_y), snout_radius, (100, 180, 230), 2)  # 테두리
                
                # 눈 (2개) - 크고 반짝이는 눈
                eye_radius = face_radius // 5
                left_eye_x = center_x - face_radius // 3
                right_eye_x = center_x + face_radius // 3
                eye_y = center_y - face_radius // 5
                
                # 눈 흰자
                cv2.circle(result, (left_eye_x, eye_y), eye_radius, (255, 255, 255), -1)
                cv2.circle(result, (right_eye_x, eye_y), eye_radius, (255, 255, 255), -1)
                
                # 눈동자
                pupil_radius = eye_radius * 2 // 3
                cv2.circle(result, (left_eye_x, eye_y), pupil_radius, (50, 30, 20), -1)
                cv2.circle(result, (right_eye_x, eye_y), pupil_radius, (50, 30, 20), -1)
                
                # 눈 하이라이트 (반짝임)
                highlight_radius = eye_radius // 3
                cv2.circle(result, (left_eye_x - 3, eye_y - 3), highlight_radius, (255, 255, 255), -1)
                cv2.circle(result, (right_eye_x - 3, eye_y - 3), highlight_radius, (255, 255, 255), -1)
                
                # 코 (하트 모양 시도 - 타원)
                nose_w = snout_radius // 2
                nose_h = snout_radius // 3
                nose_y = snout_y - snout_radius // 4
                cv2.ellipse(result, (center_x, nose_y), (nose_w, nose_h), 0, 0, 360, (50, 30, 20), -1)
                
                # 입 (귀여운 미소)
                mouth_y = snout_y + snout_radius // 3
                # 아래 곡선
                cv2.ellipse(result, (center_x, mouth_y), (snout_radius // 2, snout_radius // 4), 
                           0, 0, 180, (50, 30, 20), 2)
                # 코에서 입으로 선
                cv2.line(result, (center_x, nose_y + nose_h), (center_x, mouth_y - snout_radius // 4), 
                        (50, 30, 20), 2)
                
                # 볼 (분홍색 블러시)
                blush_radius = face_radius // 6
                left_blush_x = center_x - int(face_radius * 0.5)
                right_blush_x = center_x + int(face_radius * 0.5)
                blush_y = center_y + face_radius // 6
                
                # 반투명 블러시 효과
                overlay = result.copy()
                cv2.circle(overlay, (left_blush_x, blush_y), blush_radius, (128, 128, 255), -1)
                cv2.circle(overlay, (right_blush_x, blush_y), blush_radius, (128, 128, 255), -1)
                cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
        
        return result
    except Exception as e:
        print(f"곰 얼굴 마스크 오류: {e}")
        return image

def rabbit_face_mask(image: np.ndarray, face_detector) -> np.ndarray:
    """얼굴을 귀여운 토끼 얼굴로 대체"""
    try:
        if image.size == 0 or image.shape[0] < 10 or image.shape[1] < 10:
            return image
        
        result = image.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detector.process(image_rgb)
        
        if results.detections:
            for detection in results.detections:
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
                
                if width <= 0 or height <= 0:
                    continue
                
                margin = int(width * 0.4)  # 토끼 귀가 길어서 여유 공간 더 필요
                rabbit_x = max(0, x - margin)
                rabbit_y = max(0, y - margin)
                rabbit_w = min(w - rabbit_x, width + margin * 2)
                rabbit_h = min(h - rabbit_y, height + margin * 2)
                
                center_x = rabbit_x + rabbit_w // 2
                center_y = rabbit_y + rabbit_h // 2
                
                # 얼굴 (원) - 흰색
                face_radius = min(rabbit_w, rabbit_h) // 2
                cv2.circle(result, (center_x, center_y), face_radius, (240, 240, 250), -1)
                cv2.circle(result, (center_x, center_y), face_radius, (200, 200, 210), 3)
                
                # 긴 귀 (2개) - 타원
                ear_width = face_radius // 4
                ear_height = int(face_radius * 0.8)
                left_ear_x = center_x - int(face_radius * 0.5)
                right_ear_x = center_x + int(face_radius * 0.5)
                ear_y = center_y - int(face_radius * 1.1)
                
                # 왼쪽 귀
                cv2.ellipse(result, (left_ear_x, ear_y), (ear_width, ear_height), -15, 0, 360, (240, 240, 250), -1)
                cv2.ellipse(result, (left_ear_x, ear_y), (ear_width, ear_height), -15, 0, 360, (200, 200, 210), 2)
                # 귀 안쪽 (분홍)
                cv2.ellipse(result, (left_ear_x, ear_y), (ear_width//2, ear_height-10), -15, 0, 360, (200, 150, 255), -1)
                
                # 오른쪽 귀
                cv2.ellipse(result, (right_ear_x, ear_y), (ear_width, ear_height), 15, 0, 360, (240, 240, 250), -1)
                cv2.ellipse(result, (right_ear_x, ear_y), (ear_width, ear_height), 15, 0, 360, (200, 200, 210), 2)
                # 귀 안쪽 (분홍)
                cv2.ellipse(result, (right_ear_x, ear_y), (ear_width//2, ear_height-10), 15, 0, 360, (200, 150, 255), -1)
                
                # 눈 (2개) - 큰 눈
                eye_radius = face_radius // 5
                left_eye_x = center_x - face_radius // 3
                right_eye_x = center_x + face_radius // 3
                eye_y = center_y - face_radius // 5
                
                cv2.circle(result, (left_eye_x, eye_y), eye_radius, (255, 255, 255), -1)
                cv2.circle(result, (right_eye_x, eye_y), eye_radius, (255, 255, 255), -1)
                
                pupil_radius = eye_radius * 2 // 3
                cv2.circle(result, (left_eye_x, eye_y), pupil_radius, (80, 50, 50), -1)
                cv2.circle(result, (right_eye_x, eye_y), pupil_radius, (80, 50, 50), -1)
                
                highlight_radius = eye_radius // 3
                cv2.circle(result, (left_eye_x - 3, eye_y - 3), highlight_radius, (255, 255, 255), -1)
                cv2.circle(result, (right_eye_x - 3, eye_y - 3), highlight_radius, (255, 255, 255), -1)
                
                # 코 (작은 삼각형 - 분홍)
                nose_y = center_y + face_radius // 8
                nose_size = face_radius // 8
                nose_pts = np.array([
                    [center_x, nose_y - nose_size//2],
                    [center_x - nose_size//2, nose_y + nose_size//2],
                    [center_x + nose_size//2, nose_y + nose_size//2]
                ], np.int32)
                cv2.fillPoly(result, [nose_pts], (180, 120, 255))
                
                # 입 (토끼 특유의 Y자 모양)
                mouth_y = nose_y + nose_size
                # 중앙 세로선
                cv2.line(result, (center_x, nose_y + nose_size//2), (center_x, mouth_y), (100, 70, 70), 2)
                # 왼쪽 곡선
                cv2.ellipse(result, (center_x - face_radius//6, mouth_y + face_radius//8), 
                           (face_radius//6, face_radius//8), 0, 180, 270, (100, 70, 70), 2)
                # 오른쪽 곡선
                cv2.ellipse(result, (center_x + face_radius//6, mouth_y + face_radius//8), 
                           (face_radius//6, face_radius//8), 0, 270, 360, (100, 70, 70), 2)
                
                # 볼 (분홍 블러시)
                blush_radius = face_radius // 7
                left_blush_x = center_x - int(face_radius * 0.5)
                right_blush_x = center_x + int(face_radius * 0.5)
                blush_y = center_y + face_radius // 6
                
                overlay = result.copy()
                cv2.circle(overlay, (left_blush_x, blush_y), blush_radius, (180, 150, 255), -1)
                cv2.circle(overlay, (right_blush_x, blush_y), blush_radius, (180, 150, 255), -1)
                cv2.addWeighted(overlay, 0.4, result, 0.6, 0, result)
                
                # 앞니 (2개)
                tooth_width = face_radius // 8
                tooth_height = face_radius // 6
                left_tooth_x = center_x - tooth_width // 2
                right_tooth_x = center_x + tooth_width // 2
                tooth_y = mouth_y + face_radius // 6
                
                cv2.rectangle(result, (left_tooth_x - tooth_width, tooth_y), 
                            (left_tooth_x, tooth_y + tooth_height), (255, 255, 255), -1)
                cv2.rectangle(result, (right_tooth_x, tooth_y), 
                            (right_tooth_x + tooth_width, tooth_y + tooth_height), (255, 255, 255), -1)
        
        return result
    except Exception as e:
        print(f"토끼 얼굴 마스크 오류: {e}")
        return image

def cat_face_mask(image: np.ndarray, face_detector) -> np.ndarray:
    """얼굴을 귀여운 고양이 얼굴로 대체"""
    try:
        if image.size == 0 or image.shape[0] < 10 or image.shape[1] < 10:
            return image
        
        result = image.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detector.process(image_rgb)
        
        if results.detections:
            for detection in results.detections:
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
                
                if width <= 0 or height <= 0:
                    continue
                
                margin = int(width * 0.3)
                cat_x = max(0, x - margin)
                cat_y = max(0, y - margin)
                cat_w = min(w - cat_x, width + margin * 2)
                cat_h = min(h - cat_y, height + margin * 2)
                
                center_x = cat_x + cat_w // 2
                center_y = cat_y + cat_h // 2
                
                # 얼굴 (원) - 주황색 (고양이)
                face_radius = min(cat_w, cat_h) // 2
                cv2.circle(result, (center_x, center_y), face_radius, (100, 160, 255), -1)  # 주황색
                cv2.circle(result, (center_x, center_y), face_radius, (70, 130, 220), 3)
                
                # 삼각형 귀 (2개)
                ear_size = face_radius // 2
                left_ear_x = center_x - int(face_radius * 0.6)
                right_ear_x = center_x + int(face_radius * 0.6)
                ear_y = center_y - int(face_radius * 0.8)
                
                # 왼쪽 귀
                left_ear_pts = np.array([
                    [left_ear_x, ear_y],
                    [left_ear_x - ear_size//2, ear_y - ear_size],
                    [left_ear_x + ear_size//2, ear_y - ear_size//3]
                ], np.int32)
                cv2.fillPoly(result, [left_ear_pts], (100, 160, 255))
                cv2.polylines(result, [left_ear_pts], True, (70, 130, 220), 2)
                
                # 왼쪽 귀 안쪽 (분홍)
                left_inner_ear = np.array([
                    [left_ear_x, ear_y - ear_size//4],
                    [left_ear_x - ear_size//4, ear_y - ear_size//2],
                    [left_ear_x + ear_size//4, ear_y - ear_size//4]
                ], np.int32)
                cv2.fillPoly(result, [left_inner_ear], (150, 150, 255))
                
                # 오른쪽 귀
                right_ear_pts = np.array([
                    [right_ear_x, ear_y],
                    [right_ear_x + ear_size//2, ear_y - ear_size],
                    [right_ear_x - ear_size//2, ear_y - ear_size//3]
                ], np.int32)
                cv2.fillPoly(result, [right_ear_pts], (100, 160, 255))
                cv2.polylines(result, [right_ear_pts], True, (70, 130, 220), 2)
                
                # 오른쪽 귀 안쪽 (분홍)
                right_inner_ear = np.array([
                    [right_ear_x, ear_y - ear_size//4],
                    [right_ear_x + ear_size//4, ear_y - ear_size//2],
                    [right_ear_x - ear_size//4, ear_y - ear_size//4]
                ], np.int32)
                cv2.fillPoly(result, [right_inner_ear], (150, 150, 255))
                
                # 눈 (고양이 눈 - 타원)
                eye_width = face_radius // 5
                eye_height = face_radius // 3
                left_eye_x = center_x - face_radius // 3
                right_eye_x = center_x + face_radius // 3
                eye_y = center_y - face_radius // 5
                
                # 녹색 고양이 눈
                cv2.ellipse(result, (left_eye_x, eye_y), (eye_width, eye_height), 0, 0, 360, (100, 255, 100), -1)
                cv2.ellipse(result, (right_eye_x, eye_y), (eye_width, eye_height), 0, 0, 360, (100, 255, 100), -1)
                
                # 세로 동공
                pupil_width = eye_width // 3
                pupil_height = int(eye_height * 0.8)
                cv2.ellipse(result, (left_eye_x, eye_y), (pupil_width, pupil_height), 0, 0, 360, (20, 20, 20), -1)
                cv2.ellipse(result, (right_eye_x, eye_y), (pupil_width, pupil_height), 0, 0, 360, (20, 20, 20), -1)
                
                # 하이라이트
                highlight_radius = eye_width // 4
                cv2.circle(result, (left_eye_x - pupil_width//2, eye_y - pupil_height//3), highlight_radius, (255, 255, 255), -1)
                cv2.circle(result, (right_eye_x - pupil_width//2, eye_y - pupil_height//3), highlight_radius, (255, 255, 255), -1)
                
                # 코 (작은 삼각형 - 분홍)
                nose_y = center_y + face_radius // 8
                nose_size = face_radius // 7
                nose_pts = np.array([
                    [center_x, nose_y + nose_size//2],
                    [center_x - nose_size//2, nose_y - nose_size//2],
                    [center_x + nose_size//2, nose_y - nose_size//2]
                ], np.int32)
                cv2.fillPoly(result, [nose_pts], (150, 120, 255))
                
                # 입 (W 모양)
                mouth_y = nose_y + nose_size
                # 왼쪽 곡선
                cv2.ellipse(result, (center_x - face_radius//6, mouth_y), 
                           (face_radius//6, face_radius//8), 0, 0, 180, (80, 60, 60), 2)
                # 오른쪽 곡선
                cv2.ellipse(result, (center_x + face_radius//6, mouth_y), 
                           (face_radius//6, face_radius//8), 0, 0, 180, (80, 60, 60), 2)
                
                # 수염 (3개씩 양쪽)
                whisker_length = face_radius // 2
                whisker_y_offset = face_radius // 8
                
                # 왼쪽 수염
                for i in range(3):
                    y_offset = whisker_y_offset * (i - 1)
                    cv2.line(result, (center_x - face_radius//2, center_y + y_offset), 
                            (center_x - face_radius - whisker_length//2, center_y + y_offset - i*5), 
                            (80, 60, 60), 2)
                
                # 오른쪽 수염
                for i in range(3):
                    y_offset = whisker_y_offset * (i - 1)
                    cv2.line(result, (center_x + face_radius//2, center_y + y_offset), 
                            (center_x + face_radius + whisker_length//2, center_y + y_offset - i*5), 
                            (80, 60, 60), 2)
        
        return result
    except Exception as e:
        print(f"고양이 얼굴 마스크 오류: {e}")
        return image
    """얼굴을 귀여운 곰 얼굴로 대체"""
    try:
        if image.size == 0 or image.shape[0] < 10 or image.shape[1] < 10:
            return image
        
        result = image.copy()
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_detector.process(image_rgb)
        
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                x = int(bboxC.xmin * w)
                y = int(bboxC.ymin * h)
                width = int(bboxC.width * w)
                height = int(bboxC.height * h)
                
                # 경계 확인
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                if width <= 0 or height <= 0:
                    continue
                
                # 얼굴 영역을 약간 확장 (귀 포함)
                margin = int(width * 0.3)
                bear_x = max(0, x - margin)
                bear_y = max(0, y - margin)
                bear_w = min(w - bear_x, width + margin * 2)
                bear_h = min(h - bear_y, height + margin * 2)
                
                # 곰 얼굴 그리기
                center_x = bear_x + bear_w // 2
                center_y = bear_y + bear_h // 2
                
                # 얼굴 (원) - 연한 갈색
                face_radius = min(bear_w, bear_h) // 2
                cv2.circle(result, (center_x, center_y), face_radius, (150, 120, 80), -1)  # 연한 갈색
                cv2.circle(result, (center_x, center_y), face_radius, (100, 70, 40), 3)  # 진한 갈색 테두리
                
                # 귀 (2개) - 갈색
                ear_radius = face_radius // 3
                left_ear_x = center_x - int(face_radius * 0.7)
                right_ear_x = center_x + int(face_radius * 0.7)
                ear_y = center_y - int(face_radius * 0.7)
                
                # 귀 본체
                cv2.circle(result, (left_ear_x, ear_y), ear_radius, (150, 120, 80), -1)
                cv2.circle(result, (left_ear_x, ear_y), ear_radius, (100, 70, 40), 2)
                cv2.circle(result, (right_ear_x, ear_y), ear_radius, (150, 120, 80), -1)
                cv2.circle(result, (right_ear_x, ear_y), ear_radius, (100, 70, 40), 2)
                
                # 귀 안쪽 - 밝은 노란색
                inner_ear_radius = ear_radius // 2
                cv2.circle(result, (left_ear_x, ear_y), inner_ear_radius, (100, 200, 255), -1)  # 노란색
                cv2.circle(result, (right_ear_x, ear_y), inner_ear_radius, (100, 200, 255), -1)
                
                # 🎀 리본 추가 (오른쪽 귀 옆)
                ribbon_center_x = right_ear_x + int(ear_radius * 1.2)
                ribbon_center_y = ear_y - int(ear_radius * 0.3)
                ribbon_size = ear_radius // 2
                
                # 리본 왼쪽 나비
                ribbon_left = (ribbon_center_x - ribbon_size, ribbon_center_y)
                cv2.circle(result, ribbon_left, ribbon_size, (100, 100, 255), -1)  # 분홍색
                
                # 리본 오른쪽 나비
                ribbon_right = (ribbon_center_x + ribbon_size, ribbon_center_y)
                cv2.circle(result, ribbon_right, ribbon_size, (100, 100, 255), -1)
                
                # 리본 중앙 매듭
                cv2.circle(result, (ribbon_center_x, ribbon_center_y), ribbon_size // 2, (80, 80, 200), -1)
                
                # 얼굴 중앙 부분 - 밝은 노란색
                snout_radius = face_radius // 2
                snout_y = center_y + face_radius // 4
                cv2.circle(result, (center_x, snout_y), snout_radius, (120, 220, 255), -1)  # 밝은 노란색
                cv2.circle(result, (center_x, snout_y), snout_radius, (100, 180, 230), 2)  # 테두리
                
                # 눈 (2개) - 크고 반짝이는 눈
                eye_radius = face_radius // 5
                left_eye_x = center_x - face_radius // 3
                right_eye_x = center_x + face_radius // 3
                eye_y = center_y - face_radius // 5
                
                # 눈 흰자
                cv2.circle(result, (left_eye_x, eye_y), eye_radius, (255, 255, 255), -1)
                cv2.circle(result, (right_eye_x, eye_y), eye_radius, (255, 255, 255), -1)
                
                # 눈동자
                pupil_radius = eye_radius * 2 // 3
                cv2.circle(result, (left_eye_x, eye_y), pupil_radius, (50, 30, 20), -1)
                cv2.circle(result, (right_eye_x, eye_y), pupil_radius, (50, 30, 20), -1)
                
                # 눈 하이라이트 (반짝임)
                highlight_radius = eye_radius // 3
                cv2.circle(result, (left_eye_x - 3, eye_y - 3), highlight_radius, (255, 255, 255), -1)
                cv2.circle(result, (right_eye_x - 3, eye_y - 3), highlight_radius, (255, 255, 255), -1)
                
                # 코 (하트 모양 시도 - 타원)
                nose_w = snout_radius // 2
                nose_h = snout_radius // 3
                nose_y = snout_y - snout_radius // 4
                cv2.ellipse(result, (center_x, nose_y), (nose_w, nose_h), 0, 0, 360, (50, 30, 20), -1)
                
                # 입 (귀여운 미소)
                mouth_y = snout_y + snout_radius // 3
                # 아래 곡선
                cv2.ellipse(result, (center_x, mouth_y), (snout_radius // 2, snout_radius // 4), 
                           0, 0, 180, (50, 30, 20), 2)
                # 코에서 입으로 선
                cv2.line(result, (center_x, nose_y + nose_h), (center_x, mouth_y - snout_radius // 4), 
                        (50, 30, 20), 2)
                
                # 볼 (분홍색 블러시)
                blush_radius = face_radius // 6
                left_blush_x = center_x - int(face_radius * 0.5)
                right_blush_x = center_x + int(face_radius * 0.5)
                blush_y = center_y + face_radius // 6
                
                # 반투명 블러시 효과
                overlay = result.copy()
                cv2.circle(overlay, (left_blush_x, blush_y), blush_radius, (128, 128, 255), -1)
                cv2.circle(overlay, (right_blush_x, blush_y), blush_radius, (128, 128, 255), -1)
                cv2.addWeighted(overlay, 0.3, result, 0.7, 0, result)
        
        return result
    except Exception as e:
        print(f"곰 얼굴 마스크 오류: {e}")
        return image

# 음성 인식 함수
def record_audio_continuous(audio_queue, stop_event, audio_frames_queue=None, start_time=None):
    """연속적으로 음성을 인식하고 오디오를 실시간으로 저장하는 함수"""
    import pyaudio
    
    recognizer = sr.Recognizer()
    recognizer.energy_threshold = 4000
    recognizer.dynamic_energy_threshold = True
    
    # PyAudio 설정
    CHUNK = 2048  # 버퍼 크기
    FORMAT = pyaudio.paInt16  # 16-bit
    CHANNELS = 1  # 모노
    RATE = 16000  # 샘플레이트
    
    p = pyaudio.PyAudio()
    
    # 오디오 스트림 열기 - 버퍼 크기 증가
    stream = p.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        frames_per_buffer=CHUNK,
        stream_callback=None  # 콜백 사용 안 함 (블로킹 모드)
    )
    
    print("🎤 오디오 스트림 준비 완료!")
    
    # 첫 청크 읽기 시작 시간 기록
    first_chunk_time = None
    audio_start_recorded = False
    chunk_count = 0
    
    # 음성 인식을 위한 버퍼 (별도 처리)
    speech_recognition_queue = queue.Queue()
    
    # 음성 인식 스레드 시작 (녹음과 독립적으로 실행)
    def speech_recognition_worker():
        """음성 인식을 별도로 처리하는 워커"""
        while not stop_event.is_set():
            try:
                # 인식할 오디오 데이터 대기
                audio_data = speech_recognition_queue.get(timeout=1)
                if audio_data is None:
                    break
                
                try:
                    audio_data_obj = sr.AudioData(audio_data, RATE, 2)
                    text = recognizer.recognize_google(audio_data_obj, language='ko-KR')
                    if text:
                        audio_queue.put(text)
                        print(f"✅ 인식된 텍스트: {text}")
                except sr.UnknownValueError:
                    pass  # 인식 실패는 무시
                except sr.RequestError as e:
                    print(f"⚠️ 음성 인식 서비스 오류: {e}")
            except queue.Empty:
                continue
            except Exception as e:
                print(f"⚠️ 음성 인식 오류: {e}")
    
    # 음성 인식 스레드 시작
    recognition_thread = threading.Thread(target=speech_recognition_worker, daemon=True)
    recognition_thread.start()
    
    # 음성 인식용 버퍼
    speech_buffer = []
    silence_duration = 0
    SILENCE_THRESHOLD = 500
    MAX_SILENCE_CHUNKS = 15
    
    try:
        while not stop_event.is_set():
            try:
                # 청크를 읽기 직전 시간 기록
                chunk_read_time = time.time()
                
                # 실시간으로 오디오 청크 읽기 (블로킹)
                # 이 부분이 최대한 빠르게 실행되어야 함!
                audio_chunk = stream.read(CHUNK, exception_on_overflow=False)
                chunk_count += 1
                
                # 첫 청크의 정확한 시간 기록
                if first_chunk_time is None:
                    first_chunk_time = chunk_read_time
                    buffer_delay = CHUNK / RATE
                    first_chunk_time -= buffer_delay
                    print(f"🎤 오디오 녹음 시작 시간: {first_chunk_time} (버퍼 지연 {buffer_delay:.3f}초 보정)")
                
                # 오디오 프레임 저장 (최우선 작업!)
                if audio_frames_queue is not None:
                    relative_time = chunk_read_time - first_chunk_time
                    audio_frames_queue.put((first_chunk_time, relative_time, audio_chunk))
                    
                    if not audio_start_recorded:
                        audio_start_recorded = True
                
                # 진행 상황 출력
                if chunk_count % 100 == 0:
                    print(f"🎤 오디오 녹음 중: {chunk_count}개 청크 수집됨")
                
                # 음성 인식용 버퍼에 추가 (비블로킹)
                speech_buffer.append(audio_chunk)
                
                # 음량 체크
                try:
                    audio_data = np.frombuffer(audio_chunk, dtype=np.int16)
                    volume = np.abs(audio_data).mean()
                    
                    if volume < SILENCE_THRESHOLD:
                        silence_duration += 1
                    else:
                        silence_duration = 0
                    
                    # 침묵 감지 시 음성 인식 큐에 추가 (블로킹하지 않음)
                    if len(speech_buffer) > 10 and silence_duration >= MAX_SILENCE_CHUNKS:
                        combined_audio = b''.join(speech_buffer)
                        # 큐가 가득 차지 않았으면 추가
                        if speech_recognition_queue.qsize() < 5:
                            speech_recognition_queue.put(combined_audio)
                        speech_buffer = []
                        silence_duration = 0
                    
                    # 버퍼가 너무 커지면 초기화
                    if len(speech_buffer) > 300:
                        speech_buffer = speech_buffer[-150:]
                except:
                    pass  # 음량 체크 실패는 무시
                    
            except IOError as e:
                print(f"⚠️ 오디오 IO 에러 (무시): {e}")
                continue
            except Exception as e:
                print(f"⚠️ 오디오 읽기 오류: {e}")
                time.sleep(0.001)
                
    finally:
        # 음성 인식 스레드 종료
        speech_recognition_queue.put(None)
        recognition_thread.join(timeout=1)
        
        print(f"🎤 총 {chunk_count}개 오디오 청크 수집됨")
        stream.stop_stream()
        stream.close()
        p.terminate()
        print("🎤 오디오 녹음 종료")

# 감정 분석 함수
def analyze_emotion_quick(image: np.ndarray, model, face_detector) -> tuple[str, float, tuple]:
    """빠른 감정 분석 (실시간용) - 얼굴 위치만 반환"""
    emotion = '담담함'
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
                    emotion = EMOTION_TRANSLATION.get(emotion_eng, '담담함')
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
    
    # H.264 코덱 사용 (브라우저 호환성 우수)
    # 여러 코덱 시도
    codecs_to_try = [
        ('avc1', cv2.VideoWriter_fourcc(*'avc1')),  # H.264
        ('H264', cv2.VideoWriter_fourcc(*'H264')),  # H.264 alternative
        ('X264', cv2.VideoWriter_fourcc(*'X264')),  # H.264 alternative
        ('mp4v', cv2.VideoWriter_fourcc(*'mp4v')),  # MPEG-4 (fallback)
    ]
    
    out = None
    for codec_name, fourcc in codecs_to_try:
        try:
            out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
            if out.isOpened():
                print(f"✅ 비디오 코덱 '{codec_name}' 사용")
                break
            else:
                out.release()
                out = None
        except Exception as e:
            print(f"⚠️ 코덱 '{codec_name}' 실패: {e}")
            if out:
                out.release()
            out = None
    
    if out is None or not out.isOpened():
        print("❌ 모든 코덱 실패, 기본 코덱 사용")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
    
    for frame in frames:
        out.write(frame)
    
    out.release()
    print(f"✅ 비디오 저장 완료: {filename}")
    return filename

def save_audio_frames(audio_frames: list, filename: str, trim_start_seconds: float = 0.0):
    """오디오 프레임을 WAV 파일로 저장 (실시간 청크를 하나로 결합, 시작 부분 제거 옵션)"""
    if not audio_frames or len(audio_frames) == 0:
        print("⚠️ 저장할 오디오 데이터가 없습니다")
        return None
    
    try:
        import wave
        
        # 모든 오디오 청크를 하나로 병합
        combined_audio = b''.join(audio_frames)
        
        print(f"📊 오디오 정보: {len(audio_frames)}개 청크, 총 {len(combined_audio)} bytes")
        
        # 시작 부분 제거가 필요한 경우
        if trim_start_seconds > 0:
            SAMPLE_RATE = 16000
            SAMPLE_WIDTH = 2  # 16-bit = 2 bytes
            
            # 제거할 바이트 수 계산
            bytes_to_trim = int(trim_start_seconds * SAMPLE_RATE * SAMPLE_WIDTH)
            
            # 2의 배수로 조정 (16-bit 샘플이므로)
            bytes_to_trim = (bytes_to_trim // 2) * 2
            
            if bytes_to_trim < len(combined_audio):
                combined_audio = combined_audio[bytes_to_trim:]
                print(f"✂️ 시작 부분 {trim_start_seconds:.3f}초 ({bytes_to_trim} bytes) 제거")
            else:
                print(f"⚠️ 제거할 시간이 전체 오디오보다 김")
        
        # WAV 파일로 저장
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # 모노
            wf.setsampwidth(2)  # 16-bit (pyaudio.paInt16)
            wf.setframerate(16000)  # 16kHz
            wf.writeframes(combined_audio)  # 전체 오디오를 한 번에 저장
        
        # 오디오 길이 계산
        duration = len(combined_audio) / (2 * 16000)  # 2 bytes per sample, 16000 samples/sec
        print(f"✅ 오디오 저장 완료: {filename} (길이: {duration:.2f}초)")
        return filename
    except Exception as e:
        print(f"❌ 오디오 저장 실패: {e}")
        import traceback
        traceback.print_exc()
        return None

def merge_video_audio(video_path: str, audio_path: str, output_path: str, video_fps: float = None):
    """비디오와 오디오를 병합 (imageio-ffmpeg 사용, 정확한 싱크)"""
    try:
        import imageio_ffmpeg as ffmpeg
        import subprocess
        import wave
        import cv2
        
        print("🎬 비디오-오디오 병합 시작...")
        
        # 비디오 정보 확인
        cap = cv2.VideoCapture(video_path)
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps_original = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        
        # 실제 FPS가 전달되지 않았으면 원본 FPS 사용
        if video_fps is None:
            video_fps = video_fps_original
        
        video_duration = video_frame_count / video_fps if video_fps > 0 else 1
        
        print(f"📊 비디오 정보: {video_frame_count}프레임, FPS={video_fps:.2f}, 길이={video_duration:.2f}초")
        
        # 오디오 길이 확인
        with wave.open(audio_path, 'rb') as wf:
            audio_frames = wf.getnframes()
            audio_rate = wf.getframerate()
            audio_duration = audio_frames / float(audio_rate)
        
        print(f"📊 오디오 정보: 길이={audio_duration:.2f}초, 샘플레이트={audio_rate}Hz")
        
        # 길이 차이 확인
        duration_diff = abs(video_duration - audio_duration)
        print(f"📊 길이 차이: {duration_diff:.2f}초")
        
        # imageio-ffmpeg를 사용한 병합
        ffmpeg_exe = ffmpeg.get_ffmpeg_exe()
        
        # ffmpeg 명령어로 병합
        # -r: 입력 비디오의 프레임레이트 명시적 설정
        # -itsoffset: 오디오 시작 시간 조정 (필요시)
        cmd = [
            ffmpeg_exe,
            '-y',  # 덮어쓰기
            '-r', str(video_fps),  # 입력 비디오 FPS 명시
            '-i', video_path,
            '-i', audio_path,
            '-c:v', 'libx264',  # 비디오 H.264 인코딩
            '-preset', 'ultrafast',  # 빠른 인코딩
            '-r', str(video_fps),  # 출력 비디오 FPS 명시
            '-c:a', 'aac',  # 오디오 AAC 인코딩
            '-b:a', '128k',  # 오디오 비트레이트
            '-strict', 'experimental',
            '-map', '0:v:0',  # 첫 번째 입력의 비디오 스트림
            '-map', '1:a:0',  # 두 번째 입력의 오디오 스트림
            '-shortest',  # 짧은 쪽에 맞춤
            '-async', '1',  # 오디오 동기화
            '-vsync', 'cfr',  # 일정한 프레임레이트 유지
            '-max_muxing_queue_size', '1024',  # 큐 크기 증가
            output_path
        ]
        
        print(f"🔧 ffmpeg 명령 실행 중...")
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode == 0:
            print(f"✅ 비디오-오디오 병합 완료: {output_path}")
            
            # 결과 파일 검증
            cap_result = cv2.VideoCapture(output_path)
            result_fps = cap_result.get(cv2.CAP_PROP_FPS)
            result_frame_count = int(cap_result.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_result.release()
            
            result_duration = result_frame_count / result_fps if result_fps > 0 else 0
            print(f"✅ 최종 결과: FPS={result_fps:.2f}, 길이={result_duration:.2f}초")
            
            # 임시 파일 삭제
            try:
                os.remove(video_path)
                os.remove(audio_path)
            except:
                pass
            
            return output_path
        else:
            print(f"❌ 병합 오류:")
            print(result.stderr)
            # 오류 시 비디오만 사용
            try:
                os.rename(video_path, output_path)
            except:
                pass
            return output_path
        
    except ImportError as ie:
        print(f"⚠️ imageio-ffmpeg가 설치되어 있지 않습니다: {ie}")
        print("설치 방법: pip install imageio-ffmpeg")
        # 비디오를 최종 경로로 이동
        try:
            os.rename(video_path, output_path)
        except:
            pass
        return output_path
    except Exception as e:
        print(f"❌ 병합 중 오류: {e}")
        import traceback
        traceback.print_exc()
        # 오류 시 비디오를 최종 경로로 이동
        try:
            os.rename(video_path, output_path)
        except:
            pass
        return output_path

# 메인 UI
st.title("📔 감정 영상 일기 - Emotion Video Diary")
st.markdown("*웹캠으로 영상 일기를 기록하고, 제미나이 AI에게 따뜻한 피드백을 받아보세요.*")

st.markdown("---")

st.subheader("📹 영상 일기 기록하기")

# 1. 녹화 상태 표시
status_placeholder = st.empty()

# 녹화 전 상태 표시
if not st.session_state.webcam_active and not st.session_state.pending_save:
    status_placeholder.info("아래 녹화 시작 버튼을 눌러주세요")

# 모델 로드
with st.spinner("AI 모델 로딩 중..."):
    emotion_model = load_emotion_model()
    face_detector = load_face_detector()

if emotion_model is None or face_detector is None:
    st.error("⚠️ AI 모델 로드에 실패했습니다. 페이지를 새로고침해주세요.")
    st.stop()

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
        st.session_state.voice_text_placeholder = voice_text_placeholder  # session_state에 저장
        
        if st.session_state.recording and st.session_state.voice_recording:
            # 녹화 중일 때만 실시간 텍스트 표시
            current_text = st.session_state.transcribed_text if st.session_state.transcribed_text else "(음성 인식 중... 말씀해주세요)"
            voice_text_placeholder.text_area(
                f"일기 내용 (음성 입력)",
                value=current_text,
                height=480,
                disabled=True,
                key=f"voice_display_{time.time()}"
            )
        else:
            # 녹화 중이 아니면 기본 메시지만 표시
            voice_text_placeholder.text_area(
                "일기 내용 (음성 입력)",
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


# 녹화 시작 처리
if 'start_recording' in locals() and start_recording:
    if not st.session_state.recording:
        print(f"🎬 녹화 시작 - 이전 transcribed_text: '{st.session_state.get('transcribed_text', '')}'")
        
        # 먼저 비디오와 오디오 상태를 동시에 초기화
        st.session_state.recording = True
        st.session_state.webcam_active = True
        st.session_state.video_frames = []
        st.session_state.video_frame_times = []
        st.session_state.emotion_timeline = []
        st.session_state.recording_start_datetime = datetime.now()
        st.session_state.transcribed_text = ""  # 명시적 초기화
        st.session_state.gemini_advice = None
        
        print(f"✅ transcribed_text 초기화 완료: '{st.session_state.transcribed_text}'")
        
        # 오디오 초기화
        st.session_state.voice_recording = True
        st.session_state.audio_queue = queue.Queue()
        st.session_state.audio_frames_queue = queue.Queue()
        st.session_state.audio_frames = []
        st.session_state.stop_event = threading.Event()
        
        # 동기화 기준 시간은 오디오 스레드 내부에서 설정
        st.session_state.recording_start_time = None  # 나중에 설정됨
        
        # 오디오 스레드 시작 (내부에서 첫 프레임과 동시에 시작)
        st.session_state.audio_thread = threading.Thread(
            target=record_audio_continuous,
            args=(st.session_state.audio_queue, st.session_state.stop_event, st.session_state.audio_frames_queue, None)
        )
        st.session_state.audio_thread.daemon = True
        st.session_state.audio_thread.start()
        
        print(f"🎬 녹화 시작: 비디오와 오디오 동시 시작")
        
        st.rerun()

# 녹화 중지 처리
if st.session_state.recording and 'stop_recording' in locals() and stop_recording:
    # 녹화 종료 시간 기록
    recording_end_time = time.time()
    
    st.session_state.recording = False
    st.session_state.webcam_active = False
    
    if st.session_state.voice_recording:
        st.session_state.stop_event.set()
        st.session_state.voice_recording = False
        time.sleep(0.3)  # 오디오 스레드가 마지막 프레임을 처리할 시간
        
        # 오디오 프레임 수집 (타임스탬프와 함께)
        audio_frames_with_time = []
        while not st.session_state.audio_frames_queue.empty():
            try:
                audio_data = st.session_state.audio_frames_queue.get_nowait()
                audio_frames_with_time.append(audio_data)
            except queue.Empty:
                break
        
        print(f"📊 수집된 오디오 청크: {len(audio_frames_with_time)}개")
        
        # 비디오-오디오 동기화를 위한 trim 시간 계산
        trim_start_seconds = 0.0
        
        # 비디오 시작 시간과 오디오 시작 시간을 비교하여 동기화
        if st.session_state.recording_start_time and audio_frames_with_time:
            video_start_time = st.session_state.recording_start_time
            
            # 새로운 형식: (audio_start_time, relative_time, chunk)
            if len(audio_frames_with_time) > 0 and isinstance(audio_frames_with_time[0], tuple) and len(audio_frames_with_time[0]) == 3:
                audio_start_time = audio_frames_with_time[0][0]  # 오디오의 실제 시작 시간
                
                print(f"⏰ 비디오 시작: {video_start_time:.6f}")
                print(f"⏰ 오디오 시작: {audio_start_time:.6f}")
                
                time_diff = video_start_time - audio_start_time
                print(f"⏰ 시간 차이: {time_diff:.6f}초 (양수=오디오가 먼저, 음수=비디오가 먼저)")
                
                # 오디오가 비디오보다 먼저 시작한 경우 (일반적인 경우)
                if time_diff > 0:
                    print(f"✂️ 오디오 시작 부분 제거 필요: {time_diff:.3f}초")
                    
                    # 모든 청크 사용하되, save_audio_frames에서 정밀하게 제거
                    st.session_state.audio_frames = [chunk for _, _, chunk in audio_frames_with_time]
                    trim_start_seconds = time_diff
                    
                    print(f"✅ 전체 {len(st.session_state.audio_frames)}개 청크 사용, 저장 시 {trim_start_seconds:.3f}초 제거 예정")
                
                # 비디오가 오디오보다 먼저 시작한 경우 (이상한 경우)
                else:
                    print(f"⚠️ 비정상: 비디오가 오디오보다 먼저 시작됨 - 전체 오디오 사용")
                    st.session_state.audio_frames = [chunk for _, _, chunk in audio_frames_with_time]
                
        # trim 시간 저장 (나중에 save_audio_frames에서 사용)
        st.session_state.audio_trim_start = trim_start_seconds
        
        # 이전 형식 처리
        if not hasattr(st.session_state, 'audio_frames') or not st.session_state.audio_frames:
            # 타입 변환 (tuple에서 chunk만 추출)
            st.session_state.audio_frames = []
            for item in audio_frames_with_time:
                if isinstance(item, tuple):
                    if len(item) == 3:
                        st.session_state.audio_frames.append(item[2])  # chunk
                    elif len(item) == 2:
                        st.session_state.audio_frames.append(item[1])  # chunk
                else:
                    st.session_state.audio_frames.append(item)
            st.session_state.audio_trim_start = 0.0
    
    final_text = st.session_state.transcribed_text if st.session_state.transcribed_text else "(음성 입력 없음)"
    print(f"📝 저장할 텍스트: '{final_text}'")
    print(f"📝 transcribed_text 상태: '{st.session_state.transcribed_text}'")
    
    if st.session_state.video_frames and len(st.session_state.video_frames) > 0:
        status_placeholder.info("💾 영상 일기 저장 중...")
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            video_filename = f"emotion_diary_{timestamp}.mp4"
            text_filename = f"emotion_diary_{timestamp}.txt"
            
            # 절대 경로로 저장 (중요!)
            video_temp_path = str(VIDEOS_DIR.absolute() / f"temp_video_{timestamp}.mp4")
            audio_temp_path = str(VIDEOS_DIR.absolute() / f"temp_audio_{timestamp}.wav")
            video_path = str(VIDEOS_DIR.absolute() / video_filename)
            text_path = str(VIDEOS_DIR.absolute() / text_filename)
            
            # 실제 녹화 시간 계산 (초 단위)
            recording_end_time = time.time()
            actual_video_duration = recording_end_time - st.session_state.recording_start_time
            
            print(f"📊 녹화 정보 (시간 기반): 프레임={len(st.session_state.video_frames)}, 비디오 시간={actual_video_duration:.2f}초")
            
            # ⭐ 먼저 오디오를 저장하여 정확한 길이 파악
            actual_fps = len(st.session_state.video_frames) / actual_video_duration if actual_video_duration > 0 else 20
            
            if st.session_state.audio_frames and len(st.session_state.audio_frames) > 0:
                # trim 시간 가져오기
                trim_start = getattr(st.session_state, 'audio_trim_start', 0.0)
                audio_saved = save_audio_frames(st.session_state.audio_frames, audio_temp_path, trim_start_seconds=trim_start)
                
                if audio_saved:
                    # 오디오 길이 확인
                    import wave
                    with wave.open(audio_temp_path, 'rb') as wf:
                        audio_frames_count = wf.getnframes()
                        audio_rate = wf.getframerate()
                        audio_duration = audio_frames_count / float(audio_rate)
                    
                    print(f"🎤 오디오 길이: {audio_duration:.2f}초")
                    
                    # ⭐ 핵심: 비디오 FPS를 오디오 길이에 정확히 맞춤
                    actual_fps = len(st.session_state.video_frames) / audio_duration if audio_duration > 0 else actual_fps
                    
                    print(f"🎬 오디오 기준 정확한 FPS: {actual_fps:.2f}")
                    print(f"📊 비디오 길이 (오디오 맞춤): {len(st.session_state.video_frames) / actual_fps:.2f}초")
                    print(f"✅ 예상 길이 차이: 0.00초 (완벽한 동기화!)")
            
            # 정확한 FPS로 비디오 저장
            print(f"🎬 비디오 저장 중 (FPS={actual_fps:.2f})...")
            save_video(st.session_state.video_frames, video_temp_path, fps=actual_fps)
            
            # 비디오와 오디오 병합
            if st.session_state.audio_frames and len(st.session_state.audio_frames) > 0:
                if audio_saved:
                    
                    # 비디오와 오디오 병합 (오디오 길이 기준 FPS 사용)
                    video_path = merge_video_audio(video_temp_path, audio_temp_path, video_path, video_fps=actual_fps)
                else:
                    # 오디오 저장 실패 시 비디오만 사용
                    os.rename(video_temp_path, video_path)
            else:
                # 오디오가 없으면 비디오만 저장
                os.rename(video_temp_path, video_path)
                print("⚠️ 녹음된 오디오가 없습니다. 비디오만 저장됩니다.")
            
            if st.session_state.emotion_timeline:
                emotions_list = [e['emotion'] for e in st.session_state.emotion_timeline]
                emotion_counts = pd.Series(emotions_list).value_counts()
                dominant_emotion = emotion_counts.index[0] if len(emotion_counts) > 0 else "담담함"
                avg_confidence = np.mean([e['confidence'] for e in st.session_state.emotion_timeline])
            else:
                dominant_emotion = "담담함"
                avg_confidence = 0.0
            
            # 맞춤형 AI 예측
            personalized_emotion, personalized_confidence, is_personalized = \
                st.session_state.personalized_model.predict(
                    st.session_state.emotion_timeline,
                    final_text,
                    dominant_emotion
                )
            
            if st.session_state.recording_start_datetime:
                elapsed = datetime.now() - st.session_state.recording_start_datetime
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
                'is_personalized': is_personalized,
                'actual_fps': actual_fps  # 실제 FPS 저장
            }
            
            st.session_state.pending_save = True
            st.session_state.emotion_confirmed = False
            st.session_state.confirmed_emotion = None
            st.session_state.advice_loading = False
            st.session_state.gemini_advice = None
            st.session_state.processing_emotion = None
            st.session_state.video_frames = []
            st.session_state.audio_frames = []
            st.session_state.recording_start_time = None
            st.session_state.recording_start_datetime = None
            # transcribed_text는 유지 (감정 선택 화면에서 표시용)
            
            st.rerun()
            
        except Exception as e:
            st.error(f"❌ 저장 중 오류가 발생했습니다: {e}")
            import traceback
            st.error(traceback.format_exc())
    else:
        st.warning("⚠️ 녹화된 프레임이 없습니다!")
        st.session_state.video_frames = []
        st.session_state.recording_start_time = None
        st.session_state.recording_start_datetime = None

# 기분 선택 UI
if st.session_state.pending_save and st.session_state.save_data:
    save_data = st.session_state.save_data
    
    if not st.session_state.emotion_confirmed:
        # 로딩 중일 때 메시지 표시
        if st.session_state.get('advice_loading', False):
            status_placeholder.info("제미나이 AI가 일기를 분석하고 조언을 작성하고 있습니다...")
        # AI 추천 표시 (로딩 중이 아닐 때)
        elif save_data['is_personalized']:
            status_placeholder.success(
                f"✨ 맞춤형 AI 추천: **{save_data['personalized_emotion']}** "
                f"(확신도: {save_data['personalized_confidence']*100:.1f}%)"
            )
            st.info("사용자님의 과거 감정 패턴을 분석한 맞춤 추천입니다!")
        else:
            status_placeholder.info(
                f"✨ AI 추천: **{save_data['dominant_emotion']}** "
                f"(기본 분석 결과)"
            )
        
        emotion_options = [
            "😊 행복함",
            "😢 슬픔",
            "😠 화남",
            "😲 놀람",
            "😐 담담함",
            "😨 두려움",
            "🤢 혐오"
        ]
        
        # 추천 감정을 기본 선택으로
        recommended_emotion = save_data['personalized_emotion'] if save_data['is_personalized'] else save_data['dominant_emotion']
        emotion_map = {
            '행복함': 0, '슬픔': 1, '화남': 2, '놀람': 3,
            '담담함': 4, '두려움': 5, '혐오': 6
        }
        
        default_index = emotion_map.get(recommended_emotion, 4)
        
        selected_emotion = st.radio(
            "🎭 오늘의 감정을 선택해주세요:",
            emotion_options,
            index=default_index,
            key="emotion_radio",
            disabled=st.session_state.get('advice_loading', False)
        )
        
        confirm_emotion = st.button(
            "✅ 감정 확정하기", 
            type="primary", 
            use_container_width=True, 
            key="confirm_bottom_btn",
            disabled=st.session_state.get('advice_loading', False)
        )
        
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
                if st.session_state.personalized_model.train():
                    pass  # 조용히 학습만 진행
            
            # 제미나이 AI 조언 생성 플래그 설정
            st.session_state.advice_loading = True
            st.session_state.processing_emotion = final_mood
            st.rerun()  # 즉시 rerun하여 버튼 비활성화
    
    # advice_loading이 True인 경우 조언 생성 (독립적으로 처리)
    if st.session_state.get('advice_loading', False) and not st.session_state.emotion_confirmed:
        # 이미 위에서 status_placeholder.info()를 호출했으므로 여기서는 생략
        
        final_mood = st.session_state.processing_emotion
        
        gemini_advice = get_gemini_advice(
            final_mood,
            save_data['final_text'],
            save_data['emotion_timeline']
        )
        st.session_state.gemini_advice = gemini_advice
        st.session_state.advice_loading = False
        
        # 텍스트 파일 저장 (제미나이 조언 포함)
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
            
            # 제미나이 조언 추가
            f.write(f"\n=== 제미나이 AI의 조언 ===\n\n")
            f.write(gemini_advice)
        
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
            'is_personalized': save_data['is_personalized'],
            'gemini_advice': gemini_advice
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
    
    # 감정이 확정되고 조언도 생성된 후 최종 결과 표시
    elif st.session_state.emotion_confirmed and st.session_state.confirmed_emotion:
        status_placeholder.success(f"✅ 영상 일기가 저장되었습니다! (감정: {st.session_state.confirmed_emotion})")
        
        # 제미나이 조언 표시
        if st.session_state.gemini_advice:
            st.info(st.session_state.gemini_advice)
        
        if save_data['emotion_timeline'] and len(save_data['emotion_timeline']) > 0:
            timeline_df = pd.DataFrame(save_data['emotion_timeline'])
            col_chart1, col_chart2 = st.columns(2)
            
            with col_chart1:
                emotion_counts = timeline_df['emotion'].value_counts()
                fig_pie = px.pie(
                    values=emotion_counts.values,
                    names=emotion_counts.index,
                    title="오늘의 감정 분포",
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
                    title="오늘의 감정 변화",
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
            dominant_emoji = emotion_emoji_map.get(save_data.get('dominant_emotion', '담담함'), '📝')
            st.markdown(
                f"<small>주요 감정: {dominant_emoji} {save_data.get('dominant_emotion', '담담함')}</small>",
                unsafe_allow_html=True
            )
            st.markdown(
                f"<small>평균 확신도: {save_data.get('avg_confidence', 0)*100:.1f}%</small>",
                unsafe_allow_html=True
            )
            display_timeline = timeline_df[['frame', 'timestamp', 'emotion', 'confidence']].copy()
            display_timeline['confidence'] = display_timeline['confidence'].apply(lambda x: f"{x*100:.1f}%")
            display_timeline.columns = ['프레임', '시간', '감정', '확신도']
            st.dataframe(display_timeline, use_container_width=True, height=200)
        
        col_dl1, col_dl2 = st.columns(2)
        
        with col_dl1:
            if os.path.exists(save_data['video_path']):
                with open(save_data['video_path'], 'rb') as f:
                    video_bytes = f.read()
                    st.download_button(
                        label="📥 영상 다운로드",
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
                        label="📄 텍스트 다운로드",
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
            st.session_state.gemini_advice = None
            st.session_state.transcribed_text = ""  # 음성 텍스트 초기화 추가
            st.session_state.audio_frames = []  # 오디오 프레임도 초기화
            st.session_state.video_frames = []  # 비디오 프레임도 초기화
            
            st.rerun()

# 익명화 맵핑
anonymize_map = {
    "원본": None,
    "블러": "blur",
    "곰 얼굴 🐻": "bear",
    "토끼 얼굴 🐰": "rabbit",
    "고양이 얼굴 🐱": "cat"
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
                        print(f"📝 음성 텍스트 업데이트: '{st.session_state.transcribed_text}'")
                except queue.Empty:
                    pass
            
            # 텍스트가 업데이트되었으면 화면에 반영
            if text_updated and hasattr(st.session_state, 'voice_text_placeholder'):
                current_text = st.session_state.transcribed_text if st.session_state.transcribed_text else "(음성 인식 중... 말씀해주세요)"
                st.session_state.voice_text_placeholder.text_area(
                    f"일기 내용 (음성 입력)",
                    value=current_text,
                    height=480,
                    disabled=True,
                    key=f"voice_update_{time.time()}"
                )
            
            anonymized_frame = frame.copy()
            if anonymize_map[anonymize_option] == "blur":
                anonymized_frame = blur_frame(anonymized_frame)
            elif anonymize_map[anonymize_option] == "bear":
                anonymized_frame = bear_face_mask(anonymized_frame, face_detector)
            elif anonymize_map[anonymize_option] == "rabbit":
                anonymized_frame = rabbit_face_mask(anonymized_frame, face_detector)
            elif anonymize_map[anonymize_option] == "cat":
                anonymized_frame = cat_face_mask(anonymized_frame, face_detector)
            
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
            
            if face_bbox:
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
                # 첫 프레임 저장 시 정확한 시작 시간 기록
                if len(st.session_state.video_frames) == 0:
                    st.session_state.recording_start_time = time.time()
                    print(f"🎬 첫 프레임 캡처: 시작 시간 = {st.session_state.recording_start_time}")
                
                st.session_state.video_frames.append(display_frame)
                # 각 프레임의 타임스탬프 기록 (동기화를 위해)
                if hasattr(st.session_state, 'video_frame_times'):
                    st.session_state.video_frame_times.append(time.time() - st.session_state.recording_start_time)
            
            frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            webcam_placeholder.image(frame_rgb, channels="RGB", width=640)
            
            # 전역 emotion_emoji_map 사용
            emoji = emotion_emoji_map.get(st.session_state.current_emotion, '😐')
            
            if st.session_state.recording:
                if st.session_state.recording_start_datetime:
                    elapsed = datetime.now() - st.session_state.recording_start_datetime
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
            
            # 프레임 간 대기 시간 추가 (일정한 FPS 유지)
            # 목표: 약 20 FPS (0.05초 간격)
            time.sleep(0.001)  # 최소 대기로 CPU 부하 감소, 실제 FPS는 자동 계산됨
        
        cap.release()

# 저장된 일기 목록
st.markdown("---")
st.subheader("📚 저장된 영상 일기")

if st.session_state.diary_entries:

    
    # 개별 일기
    for i, entry in enumerate(reversed(st.session_state.diary_entries)):
        emotion = entry.get('emotion', '미기록')
        emotion_emoji = emotion_emoji_map.get(emotion, '📝')
        emotion_display = f" - {emotion_emoji} {emotion}" if 'emotion' in entry else ""
        ai_rec = entry.get('ai_recommended', '없음')
        is_personalized = "AI 맞춤 추천" if entry.get('is_personalized', False) else "AI 기본 추천"
        
        with st.expander(f"📔 일기 #{len(st.session_state.diary_entries)-i} - {entry['timestamp']}{emotion_display}"):
            # 영상 재생 (상단, 왼쪽)
            col_video, col_text = st.columns([2, 1])
            
            with col_video:
                st.markdown("**🎬 영상 재생**")
                
                video_path = entry.get('video_path', '')
                video_filename = entry.get('video_filename', '')
                
                if video_path and os.path.exists(video_path):
                    try:
                        # 둥근 모서리 스타일 적용
                        st.markdown("""
                        <style>
                        video {
                            border-radius: 10px;
                            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
                            max-width: 640px;
                            width: 100%;
                            height: auto;
                        }
                        /* 텍스트 영역과 부모 div 높이를 영상(480px)과 동일하게 */
                        [data-baseweb="base-input"] {
                            height: 270px !important;
                        }
                        textarea {
                            height: 270px !important;
                            max-height: 480px !important;
                            overflow-y: auto !important;
                        }
                        </style>
                        """, unsafe_allow_html=True)
                        

                        # 상대 경로로 영상 재생
                        st.video(f"emotion_diary_data/videos/{video_filename}")
                        
                    except Exception as e:
                        st.error(f"❌ 영상 재생 오류: {e}")
                else:
                    st.warning(f"⚠️ 영상 파일을 찾을 수 없습니다.")
            
            with col_text:
                st.write("**🎤 음성 입력:** 사용됨 ✅")
                
                # diary_text가 없으면 텍스트 파일에서 읽어오기
                diary_text = entry.get('diary_text', '')
                
                print(f"📝 일기 #{len(st.session_state.diary_entries)-i} ({entry.get('timestamp', 'unknown')}) 텍스트 로드 시도")
                print(f"  - diary_text from entry: '{diary_text[:50] if diary_text else '(empty)'}...' (길이: {len(diary_text)})")
                print(f"  - text_path: {entry.get('text_path', 'None')}")
                
                if not diary_text and entry.get('text_path'):
                    text_path = entry.get('text_path')
                    print(f"  - 텍스트 파일 존재 여부: {os.path.exists(text_path)}")
                    
                    if os.path.exists(text_path):
                        try:
                            with open(text_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                print(f"  - 파일 내용 길이: {len(content)}")
                                
                                # "=== 일기 내용 (음성 입력) ===" 부분 추출
                                if "=== 일기 내용 (음성 입력) ===" in content:
                                    parts = content.split("=== 일기 내용 (음성 입력) ===")
                                    if len(parts) > 1:
                                        # 다음 === 까지 추출
                                        text_section = parts[1]
                                        if "===" in text_section:
                                            diary_text = text_section.split("===")[0].strip()
                                        else:
                                            diary_text = text_section.strip()
                                        
                                        print(f"  - 추출된 텍스트: '{diary_text[:50]}...'")
                                        
                                        if not diary_text:
                                            diary_text = "(음성 입력 없음)"
                                else:
                                    print(f"  - 구분자를 찾을 수 없음")
                                    diary_text = "(텍스트 파일 형식 오류)"
                        except Exception as e:
                            print(f"  - 텍스트 파일 읽기 오류: {e}")
                            import traceback
                            traceback.print_exc()
                            diary_text = f"(텍스트 파일 읽기 실패: {e})"
                
                if not diary_text:
                    diary_text = "(음성 입력 없음)"
                
                print(f"  - 최종 표시 텍스트: '{diary_text[:50] if len(diary_text) > 50 else diary_text}...' (길이: {len(diary_text)})")
                
                # timestamp를 사용한 고유 키로 Streamlit 캐싱 문제 방지
                unique_key = f"voice_display_{entry.get('timestamp', i)}"
                
                st.text_area(
                    f"일기 내용 (음성 입력)",
                    value=diary_text,
                    disabled=True,
                    key=unique_key
                )
            
            # 감정 정보
            st.markdown("---")
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                if 'emotion' in entry:
                    user_emotion_emoji = emotion_emoji_map.get(entry['emotion'], '📝')
                    st.write(f"**✨ 그날의 감정:** {user_emotion_emoji} {entry['emotion']}")
                
                ai_rec_emoji = emotion_emoji_map.get(ai_rec, '📝')
                if entry['emotion'] == ai_rec:
                    st.success("✅ AI 추천과 일치")
                else:
                    st.info(f"ℹ️ 사용자가 다른 감정 선택 (AI제안: {ai_rec_emoji} {ai_rec})")

        
                
            with col_info2:
                st.write(f"**🎬 프레임 수:** {entry['frame_count']}")
                st.write(f"**🔒 익명화:** {entry['anonymize_method']}")
                st.write(f"**⏱️ 녹화 시간:** {entry.get('recording_duration', '00:00')}")
                st.write(f"**📏 영상 길이:** 약 {entry['frame_count'] / 20:.1f}초")
            
            # 제미나이 조언 표시
            if entry.get('gemini_advice'):
                st.markdown("---")
                st.info(entry['gemini_advice'])

            if entry.get('emotion_timeline') and len(entry['emotion_timeline']) > 0:
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
                        title="감정 변화",
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

            # 감정 타임라인 표시
            if entry.get('emotion_timeline') and len(entry['emotion_timeline']) > 0:
                st.markdown("**📋 감정 타임라인 (Emotion Timeline)**")
                dominant_emoji = emotion_emoji_map.get(entry.get('dominant_emotion', '담담함'), '📝')
                st.markdown(
                    f"<small>주요 감정: {dominant_emoji} {entry.get('dominant_emotion', '담담함')}</small>",
                    unsafe_allow_html=True
                )
                st.markdown(
                    f"<small>평균 확신도: {entry.get('avg_confidence', 0)*100:.1f}%</small>",
                    unsafe_allow_html=True
                )

                timeline_df = pd.DataFrame(entry['emotion_timeline'])
                
                # 타임라인 테이블 표시
                display_timeline = timeline_df[['frame', 'timestamp', 'emotion', 'confidence']].copy()
                display_timeline['confidence'] = display_timeline['confidence'].apply(lambda x: f"{x*100:.1f}%")
                display_timeline.columns = ['프레임', '시간', '감정', '확신도']
                st.dataframe(display_timeline, use_container_width=True, height=200)
    
    
            # 다운로드 버튼
            st.markdown("---")
            col_dl1, col_dl2 = st.columns(2)
            
            with col_dl1:
                if os.path.exists(entry['video_path']):
                    with open(entry['video_path'], 'rb') as f:
                        video_bytes = f.read()
                        st.download_button(
                            label="📥 영상 다운로드",
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
                            label="📄 텍스트 다운로드",
                            data=text_content,
                            file_name=entry['text_filename'],
                            mime="text/plain",
                            key=f"download_text_{i}",
                            use_container_width=True
                        )
                else:
                    st.warning("⚠️ 텍스트 파일 없음")
    # 전체 통계
    st.markdown("---")
    st.markdown("### 📊 전체 감정 분석")
    
    col_left, col_middle, col_right = st.columns([1, 1, 2])
    
    with col_left:
        # 원형 그래프
        all_emotions = [e['emotion'] for e in st.session_state.diary_entries]
        emotion_df = pd.DataFrame({'감정': all_emotions})
        emotion_counts = emotion_df['감정'].value_counts()
        
        fig_overall = px.pie(
            values=emotion_counts.values,
            names=emotion_counts.index,
            title="전체 감정 분포",
            color_discrete_sequence=px.colors.qualitative.Pastel
        )
        st.plotly_chart(fig_overall, use_container_width=True)
    
    with col_middle:
        # 통계 정보 - 반응형 여백
        st.markdown("""
        <style>
        @media (min-width: 768px) {
            .stats-spacing {
                margin-top: 80px;
            }
        }
        </style>
        <div class="stats-spacing"></div>
        """, unsafe_allow_html=True)
        
        # AI 추천 vs 사용자 선택 비교
        ai_matches = sum(1 for e in st.session_state.diary_entries 
                        if e['emotion'] == e.get('ai_recommended', ''))
        match_rate = (ai_matches / len(st.session_state.diary_entries)) * 100
        
        personalized_count = sum(1 for e in st.session_state.diary_entries 
                                if e.get('is_personalized', False))
        
        # 작은 글씨로 통계 표시
        st.markdown(f"""
        <div style="font-size: 14px;">
        <p><strong>AI-사용자 일치율:</strong> {match_rate:.1f}%</p>
        <p><strong>맞춤형 추천 사용:</strong> {personalized_count}회</p>
        <p><strong>학습 데이터:</strong> {len(st.session_state.personalized_model.training_data)}개</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_right:
        # 시계열 감정 변화 그래프
        st.markdown("**📅 감정 타임라인 (Emotion Timeline)**")
        
        # 타임스탬프를 datetime 객체로 변환
        timeline_data = []
        for entry in st.session_state.diary_entries:
            try:
                # timestamp 형식: YYYYMMDD_HHMMSS
                timestamp_str = entry['timestamp']
                dt = datetime.strptime(timestamp_str, '%Y%m%d_%H%M%S')
                
                timeline_data.append({
                    '날짜': dt.date(),  # 날짜만 추출
                    '날짜시간': dt,
                    '감정': entry['emotion'],
                    '이모지': emotion_emoji_map.get(entry['emotion'], '😐')
                })
            except Exception as e:
                print(f"날짜 변환 오류: {e}")
                continue
        
        if timeline_data:
            timeline_df = pd.DataFrame(timeline_data)
            timeline_df = timeline_df.sort_values('날짜시간')
            
            # 감정별 색상 매핑
            emotion_colors = {
                '행복함': '#FFD700',  # 금색
                '슬픔': '#4169E1',    # 파란색
                '화남': '#FF4500',    # 빨간색
                '놀람': '#FF69B4',    # 분홍색
                '담담함': '#A9A9A9',  # 회색
                '두려움': '#800080',  # 보라색
                '혐오': '#228B22'     # 초록색
            }
            
            # 산점도 스타일 그래프 (각 감정을 색상으로 구분)
            fig_scatter = px.scatter(
                timeline_df,
                x='날짜',
                y='감정',
                color='감정',
                text='이모지',  # 이모지를 텍스트로 표시
                title='',
                labels={'날짜': '날짜', '감정': '감정'},
                color_discrete_map=emotion_colors
            )
            
            # 마커를 투명하게 하고 이모지만 표시
            fig_scatter.update_traces(
                marker=dict(size=1, opacity=0),  # 마커를 거의 보이지 않게
                textfont=dict(size=30),  # 이모지 크기
                textposition='middle center'
            )
            
            # 모든 감정 레이블을 Y축에 표시
            all_emotions = ['행복함', '슬픔', '화남', '놀람', '담담함', '두려움', '혐오']
            
            fig_scatter.update_layout(
                height=500,
                hovermode='closest',
                xaxis_title='날짜',
                yaxis_title='감정',
                showlegend=True,
                legend=dict(
                    title="감정",
                    orientation="v",
                    yanchor="top",
                    y=1,
                    xanchor="left",
                    x=1.02
                ),
                yaxis=dict(
                    categoryorder='array',
                    categoryarray=all_emotions  # 모든 감정을 순서대로 표시
                )
            )
            
            # X축을 하루 간격으로 설정
            fig_scatter.update_xaxes(
                dtick=86400000.0,  # 1일 = 86400000 밀리초
                tickformat='%Y-%m-%d'
            )
            
            st.plotly_chart(fig_scatter, use_container_width=True)
        else:
            st.info("시계열 데이터가 충분하지 않습니다.")
else:
    st.info("📭 아직 저장된 영상 일기가 없습니다. 위에서 녹화를 시작해보세요!")