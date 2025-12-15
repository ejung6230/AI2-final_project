import streamlit as st
import cv2
import numpy as np
from PIL import Image
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
if 'last_saved_entry' not in st.session_state:
    st.session_state.last_saved_entry = None
if 'voice_recording' not in st.session_state:
    st.session_state.voice_recording = False
if 'transcribed_text' not in st.session_state:
    st.session_state.transcribed_text = ""
if 'audio_queue' not in st.session_state:
    st.session_state.audio_queue = queue.Queue()
if 'show_emotion_chart' not in st.session_state:
    st.session_state.show_emotion_chart = False

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
        # 축소
        temp = cv2.resize(image, (blocks, blocks), interpolation=cv2.INTER_LINEAR)
        # 확대 (픽셀 효과)
        return cv2.resize(temp, (w, h), interpolation=cv2.INTER_NEAREST)
    return image

def cartoonize_frame(image: np.ndarray) -> np.ndarray:
    """전체 프레임 카툰 스타일 변환"""
    try:
        if image.size == 0 or image.shape[0] < 10 or image.shape[1] < 10:
            return image
        
        # 1. 색상 단순화 (bilateral filter를 여러 번 적용)
        color = cv2.bilateralFilter(image, 9, 250, 250)
        for _ in range(2):
            color = cv2.bilateralFilter(color, 9, 250, 250)
        
        # 2. 에지 검출
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.medianBlur(gray, 7)
        edges = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_MEAN_C, 
            cv2.THRESH_BINARY, 
            9, 2
        )
        
        # 3. 에지를 3채널로 변환
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # 4. 카툰 효과 생성
        cartoon = cv2.bitwise_and(color, edges_colored)
        
        # 5. 밝기 조정 (카툰 느낌 강화)
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
                # 짧은 timeout으로 자주 확인
                audio = recognizer.listen(source, timeout=1, phrase_time_limit=10)
                
                try:
                    # 한국어 인식
                    text = recognizer.recognize_google(audio, language='ko-KR')
                    if text:
                        audio_queue.put(text)
                        print(f"인식된 텍스트: {text}")
                except sr.UnknownValueError:
                    pass  # 음성을 인식하지 못함
                except sr.RequestError as e:
                    print(f"음성 인식 서비스 오류: {e}")
                    time.sleep(1)
            except sr.WaitTimeoutError:
                continue  # timeout이면 계속 진행
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
        # 얼굴 검출
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
            
            # 경계 체크
            x = max(0, x)
            y = max(0, y)
            width = min(width, w - x)
            height = min(height, h - y)
            
            face_bbox = (x, y, width, height)
            
            # 얼굴 영역 추출
            face = image[y:y+height, x:x+width]
            
            if face.size > 0:
                # 감정 분석
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

# 메인 UI
st.title("📔 감정 영상 일기 - Emotion Video Diary")
st.markdown("*웹캠으로 실시간 감정을 분석하며 영상 일기를 작성하세요*")

# 음성 인식된 텍스트 표시
if st.session_state.transcribed_text:
    st.info(f"🎤 음성으로 입력된 텍스트: {st.session_state.transcribed_text}")

st.subheader("✍️ 오늘의 일기")
diary_text = st.text_area(
    "오늘의 감정과 생각을 자유롭게 적어보세요 (또는 음성으로 입력)", 
    value=st.session_state.transcribed_text,
    height=150, 
    key="diary_input"
)

# 웹캠 스트리밍
st.markdown("---")

# 레이아웃 구성
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 녹화 설정")
    anonymize_option = st.selectbox(
        "전체 화면 익명화 방식",
        ["원본", "블러", "픽셀화", "카툰"],
        key="anonymize",
        disabled=st.session_state.recording  # 녹화 중에는 변경 불가
    )
    
    show_emotion_overlay = st.checkbox(
        "감정 정보 오버레이 표시", 
        value=True,
        disabled=st.session_state.recording  # 녹화 중에는 변경 불가
    )
    
    # 음성 입력은 항상 활성화
    use_voice_input = True
    
    if st.session_state.recording:
        st.warning("⚠️ 녹화 중에는 설정을 변경할 수 없습니다.")
    else:
        st.info("💡 음성 입력이 자동으로 활성화됩니다. 녹화 중 말한 내용이 자동으로 텍스트로 변환됩니다.")
    
    st.info("""
    💡 **사용 방법:**
    1. 익명화 방식 선택
    2. '🔴 녹화 시작' 클릭
    3. 감정 표현하며 이야기
    4. '⏹️ 녹화 중지 & 저장' 클릭
    5. 영상 & 텍스트 다운로드
    """)

with col2:
    st.subheader("📹 웹캠 화면")
    
    if not st.session_state.recording:
        start_recording = st.button("🔴 녹화 시작", type="primary", use_container_width=True)
    else:
        start_recording = False
        stop_recording = st.button("⏹️ 녹화 중지 & 저장", type="secondary", use_container_width=True)
    
    # 녹화 상태 표시
    status_placeholder = st.empty()
    emotion_display = st.empty()

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
            st.session_state.show_emotion_chart = False
            
            # 음성 인식 시작
            if use_voice_input:
                st.session_state.voice_recording = True
                st.session_state.audio_queue = queue.Queue()
                st.session_state.stop_event = threading.Event()
                st.session_state.audio_thread = threading.Thread(
                    target=record_audio_continuous,
                    args=(st.session_state.audio_queue, st.session_state.stop_event)
                )
                st.session_state.audio_thread.daemon = True
                st.session_state.audio_thread.start()
            
            st.success("🔴 녹화가 시작되었습니다!")
            st.rerun()

    # 녹화 중지 처리
    if st.session_state.recording and 'stop_recording' in locals() and stop_recording:
        st.session_state.recording = False
        st.session_state.webcam_active = False
        
        # 음성 인식 중지
        if st.session_state.voice_recording:
            with st.spinner("🎤 음성 인식 종료 중..."):
                st.session_state.stop_event.set()
                st.session_state.voice_recording = False
                time.sleep(0.5)  # 스레드 종료 대기
        
        # 최종 텍스트 업데이트 (내용이 없어도 저장 가능)
        final_text = diary_text if diary_text else st.session_state.transcribed_text
        if not final_text:
            final_text = "(일기 내용 없음)"  # 빈 내용일 경우 기본 텍스트
        
        # 즉시 비디오 저장 (프레임만 있으면 저장)
        if st.session_state.video_frames:
            # 진행률 표시 영역 생성
            save_container = st.container()
            
            with save_container:
                save_status = st.empty()
                progress_text = st.empty()
                progress_bar = st.progress(0)
                
                save_status.info("💾 영상 일기 저장 시작...")
                
                try:
                    # 비디오 저장
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    video_filename = f"emotion_diary_{timestamp}.mp4"
                    text_filename = f"emotion_diary_{timestamp}.txt"
                    
                    # 임시 디렉토리에 저장
                    temp_dir = tempfile.gettempdir()
                    video_path = os.path.join(temp_dir, video_filename)
                    text_path = os.path.join(temp_dir, text_filename)
                    
                    # 1단계: 비디오 인코딩
                    progress_text.text("📹 비디오 인코딩 중... (1/3)")
                    progress_bar.progress(10)
                    time.sleep(0.2)
                    
                    save_video(st.session_state.video_frames, video_path, fps=20)
                    progress_bar.progress(40)
                    time.sleep(0.2)
                    
                    # 2단계: 감정 분석
                    progress_text.text("🎭 감정 데이터 분석 중... (2/3)")
                    progress_bar.progress(50)
                    time.sleep(0.2)
                    
                    # 감정 통계 계산
                    if st.session_state.emotion_timeline:
                        emotions_list = [e['emotion'] for e in st.session_state.emotion_timeline]
                        emotion_counts = pd.Series(emotions_list).value_counts()
                        dominant_emotion = emotion_counts.index[0] if len(emotion_counts) > 0 else "neutral"
                        avg_confidence = np.mean([e['confidence'] for e in st.session_state.emotion_timeline])
                    else:
                        dominant_emotion = "neutral"
                        avg_confidence = 0.0
                    
                    progress_bar.progress(70)
                    time.sleep(0.2)
                    
                    # 3단계: 텍스트 파일 생성
                    progress_text.text("📄 텍스트 파일 생성 중... (3/3)")
                    progress_bar.progress(80)
                    time.sleep(0.2)
                    
                    # 텍스트 파일 저장
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(f"=== 감정 영상 일기 ===\n")
                        f.write(f"날짜: {datetime.now().strftime('%Y년 %m월 %d일 %H:%M:%S')}\n")
                        f.write(f"익명화 방식: {anonymize_option}\n")
                        f.write(f"\n=== 일기 내용 ===\n\n")
                        f.write(final_text)
                        f.write(f"\n\n=== 감정 분석 결과 ===\n")
                        if st.session_state.emotion_timeline:
                            emotions_list = [e['emotion'] for e in st.session_state.emotion_timeline]
                            emotion_counts = pd.Series(emotions_list).value_counts()
                            f.write(f"주요 감정: {emotion_counts.index[0] if len(emotion_counts) > 0 else 'neutral'}\n")
                            f.write(f"평균 확신도: {np.mean([e['confidence'] for e in st.session_state.emotion_timeline])*100:.1f}%\n")
                            f.write(f"\n감정 분포:\n")
                            for emotion, count in emotion_counts.items():
                                percentage = (count / len(emotions_list)) * 100
                                f.write(f"  - {emotion}: {count}회 ({percentage:.1f}%)\n")
                    
                    progress_bar.progress(90)
                    time.sleep(0.2)
                    
                    # 녹화 시간 계산
                    if st.session_state.recording_start_time:
                        elapsed = datetime.now() - st.session_state.recording_start_time
                        elapsed_seconds = int(elapsed.total_seconds())
                        recording_duration = f"{elapsed_seconds // 60:02d}:{elapsed_seconds % 60:02d}"
                    else:
                        recording_duration = "00:00"
                    
                    # 일기 항목 저장
                    entry = {
                        'timestamp': timestamp,
                        'diary_text': final_text,
                        'video_filename': video_filename,
                        'video_path': video_path,
                        'text_filename': text_filename,
                        'text_path': text_path,
                        'dominant_emotion': dominant_emotion,
                        'avg_confidence': avg_confidence,
                        'frame_count': len(st.session_state.video_frames),
                        'recording_duration': recording_duration,
                        'emotion_timeline': st.session_state.emotion_timeline.copy(),
                        'anonymize_method': anonymize_option,
                        'voice_input_used': use_voice_input
                    }
                    
                    st.session_state.diary_entries.append(entry)
                    st.session_state.last_saved_entry = entry
                    st.session_state.show_emotion_chart = True  # 감정 차트 표시 플래그
                    
                    progress_bar.progress(100)
                    time.sleep(0.3)
                    
                    # 저장 완료 메시지 (하나만 표시)
                    save_status.success(f"✅ 영상 일기가 저장되었습니다! ({len(st.session_state.video_frames)} 프레임, {recording_duration})")
                    progress_text.empty()
                    progress_bar.empty()
                    
                    st.balloons()
                    
                    # 녹화 상태 초기화 (감정 타임라인은 유지)
                    st.session_state.video_frames = []
                    st.session_state.recording_start_time = None
                    
                    # 잠시 대기 후 rerun (사용자가 메시지를 볼 수 있도록)
                    time.sleep(1.5)
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ 저장 중 오류가 발생했습니다: {e}")
                    import traceback
                    st.error(traceback.format_exc())
                    # 오류 발생 시에도 상태 초기화
                    st.session_state.video_frames = []
                    st.session_state.recording_start_time = None
        else:
            st.warning("⚠️ 녹화된 프레임이 없습니다! 녹화를 시작해주세요.")
            
            # 상태 초기화
            st.session_state.video_frames = []
            st.session_state.recording_start_time = None
            time.sleep(1.0)
            st.rerun()

    # 익명화 맵핑
    anonymize_map = {
        "원본": None,
        "블러": "blur",
        "픽셀화": "pixelate",
        "카툰": "cartoon"
    }

    # 웹캠 캡처 영역
    FRAME_WINDOW = st.image([])

    # 웹캠 실행 (녹화 중일 때만)
    if st.session_state.webcam_active:
        # 웹캠 로딩 메시지를 status_placeholder에 표시
        status_placeholder.info("📹 웹캠을 시작하는 중입니다...")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            status_placeholder.error("❌ 웹캠을 열 수 없습니다. 웹캠이 연결되어 있는지 확인하세요.")
            st.session_state.webcam_active = False
        else:
            # 웹캠 해상도 설정
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap.set(cv2.CAP_PROP_FPS, 30)
            
            # 첫 프레임이 준비될 때까지 대기
            status_placeholder.info("📹 웹캠 초기화 중... 첫 프레임 대기 중...")
            
            # 웹캠이 안정화될 때까지 몇 프레임 스킵
            for _ in range(5):
                ret, _ = cap.read()
                if not ret:
                    break
                time.sleep(0.1)
            
            # 준비 완료 메시지
            status_placeholder.success("✅ 웹캠 준비 완료!")
            time.sleep(0.5)
            
            frame_count = 0
            
            while st.session_state.webcam_active:
                ret, frame = cap.read()
                
                if not ret:
                    st.error("❌ 프레임을 읽을 수 없습니다.")
                    break
                
                # 음성 인식 텍스트 업데이트
                if st.session_state.voice_recording:
                    try:
                        while not st.session_state.audio_queue.empty():
                            new_text = st.session_state.audio_queue.get_nowait()
                            if st.session_state.transcribed_text:
                                st.session_state.transcribed_text += " " + new_text
                            else:
                                st.session_state.transcribed_text = new_text
                    except queue.Empty:
                        pass
                
                # 전체 프레임 익명화 적용 (감정 분석 전에)
                anonymized_frame = frame.copy()
                if anonymize_map[anonymize_option] == "blur":
                    anonymized_frame = blur_frame(anonymized_frame)
                elif anonymize_map[anonymize_option] == "pixelate":
                    anonymized_frame = pixelate_frame(anonymized_frame)
                elif anonymize_map[anonymize_option] == "cartoon":
                    anonymized_frame = cartoonize_frame(anonymized_frame)
                
                # 감정 분석 (원본 프레임으로, 3프레임마다 - 성능 최적화)
                face_bbox = None
                if frame_count % 3 == 0:
                    emotion, confidence, face_bbox = analyze_emotion_quick(
                        frame.copy(), emotion_model, face_detector
                    )
                    st.session_state.current_emotion = emotion
                    
                    # 녹화 중이면 감정 타임라인 업데이트
                    if st.session_state.recording:
                        st.session_state.emotion_timeline.append({
                            'frame': len(st.session_state.video_frames) + 1,
                            'emotion': emotion,
                            'confidence': confidence,
                            'timestamp': datetime.now().strftime('%H:%M:%S')
                        })
                
                # 최종 표시용 프레임 (익명화된 프레임)
                display_frame = anonymized_frame.copy()
                
                # 감정 오버레이 추가 (익명화된 프레임 위에)
                if show_emotion_overlay and face_bbox:
                    x, y, w, h = face_bbox
                    
                    emotion_emoji = {
                        'happy': '😊', 'sad': '😢', 'angry': '😠', 
                        'surprise': '😲', 'neutral': '😐', 'fear': '😨',
                        'disgust': '🤢', 'joy': '😄'
                    }
                    emotion = st.session_state.current_emotion
                    emoji = emotion_emoji.get(emotion.lower(), '😐')
                    
                    # 감정 분석 정보가 있으면 confidence도 표시
                    if frame_count % 3 == 0 and 'confidence' in locals():
                        text = f"{emoji} {emotion} ({confidence*100:.1f}%)"
                    else:
                        text = f"{emoji} {emotion}"
                    
                    # 오버레이 추가
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(display_frame, (x, y-30), (x + text_width, y), (0, 255, 0), -1)
                    cv2.putText(display_frame, text, (x, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # 녹화 중이면 익명화된 프레임 저장
                if st.session_state.recording:
                    st.session_state.video_frames.append(display_frame)
                
                # 화면에 표시
                frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(frame_rgb, channels="RGB")
                
                # 감정 표시
                emotion_emoji = {
                    'happy': '😊', 'sad': '😢', 'angry': '😠', 
                    'surprise': '😲', 'neutral': '😐', 'fear': '😨',
                    'disgust': '🤢', 'joy': '😄'
                }
                emoji = emotion_emoji.get(st.session_state.current_emotion.lower(), '😐')
                emotion_display.metric(
                    "현재 감정", 
                    f"{emoji} {st.session_state.current_emotion}"
                )
                
                # 녹화 상태 업데이트 (음성 인식 상태를 함께 표시)
                if st.session_state.recording:
                    if st.session_state.recording_start_time:
                        elapsed = datetime.now() - st.session_state.recording_start_time
                        elapsed_seconds = int(elapsed.total_seconds())
                        minutes = elapsed_seconds // 60
                        seconds = elapsed_seconds % 60
                        
                        # 음성 인식 상태 추가
                        voice_status = ""
                        if st.session_state.voice_recording:
                            word_count = len(st.session_state.transcribed_text.split()) if st.session_state.transcribed_text else 0
                            voice_status = f" | 🎤 음성 인식 중 (단어: {word_count}개)"
                        
                        status_placeholder.success(
                            f"🔴 녹화 중: {minutes:02d}:{seconds:02d} | 프레임: {len(st.session_state.video_frames)}{voice_status}"
                        )
                else:
                    if len(st.session_state.video_frames) > 0:
                        status_placeholder.warning(f"⏹️ 녹화 중지됨 ({len(st.session_state.video_frames)} 프레임)")
                    else:
                        status_placeholder.info("⚪ 대기 중")
                
                frame_count += 1
            
            cap.release()

# 방금 저장된 영상 다운로드
if st.session_state.last_saved_entry:
    st.markdown("---")
    st.subheader("📥 방금 저장된 영상 일기")
    
    entry = st.session_state.last_saved_entry
    
    col_result1, col_result2 = st.columns(2)
    
    with col_result1:
        st.write("**📊 녹화 정보:**")
        st.metric("⏱️ 녹화 시간", entry['recording_duration'])
        st.metric("📹 총 프레임 수", f"{entry['frame_count']} 프레임")
        st.metric("🎭 주요 감정", entry['dominant_emotion'])
        st.metric("📈 평균 확신도", f"{entry['avg_confidence']*100:.1f}%")
        if entry.get('voice_input_used'):
            st.metric("🎤 음성 입력", "사용됨")
    
    with col_result2:
        st.write("**📥 다운로드:**")
        
        # 영상 다운로드
        if os.path.exists(entry['video_path']):
            with open(entry['video_path'], 'rb') as f:
                video_bytes = f.read()
                st.download_button(
                    label="📥 영상 일기 다운로드 (MP4)",
                    data=video_bytes,
                    file_name=entry['video_filename'],
                    mime="video/mp4",
                    type="primary",
                    use_container_width=True,
                    key="download_latest_video"
                )
        
        # 텍스트 다운로드
        if os.path.exists(entry['text_path']):
            with open(entry['text_path'], 'r', encoding='utf-8') as f:
                text_content = f.read()
                st.download_button(
                    label="📄 일기 텍스트 다운로드 (TXT)",
                    data=text_content,
                    file_name=entry['text_filename'],
                    mime="text/plain",
                    type="secondary",
                    use_container_width=True,
                    key="download_latest_text"
                )
        
        if st.button("✅ 확인 완료", type="secondary", use_container_width=True):
            st.session_state.last_saved_entry = None
            st.session_state.show_emotion_chart = False
            st.rerun()

# 감정 변화 시각화 (녹화 완료 후에만 표시)
if st.session_state.show_emotion_chart and st.session_state.emotion_timeline and len(st.session_state.emotion_timeline) > 0:
    st.markdown("---")
    st.subheader("📊 현재 세션 감정 분석")
    
    timeline_df = pd.DataFrame(st.session_state.emotion_timeline)
    
    col_chart1, col_chart2 = st.columns(2)
    
    with col_chart1:
        # 감정 분포
        emotion_counts = timeline_df['emotion'].value_counts()
        fig_pie = px.pie(
            values=emotion_counts.values,
            names=emotion_counts.index,
            title="감정 분포",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_chart2:
        # 시간에 따른 감정 변화
        fig_line = px.line(
            timeline_df,
            x='frame',
            y='confidence',
            color='emotion',
            title="프레임별 감정 변화",
            markers=True
        )
        st.plotly_chart(fig_line, use_container_width=True)
    
    # 감정 타임라인 테이블
    st.subheader("📋 감정 타임라인")
    display_timeline = timeline_df[['frame', 'timestamp', 'emotion', 'confidence']].copy()
    display_timeline['confidence'] = display_timeline['confidence'].apply(lambda x: f"{x*100:.1f}%")
    st.dataframe(display_timeline, use_container_width=True, height=200)

# 저장된 일기 목록
st.markdown("---")
st.subheader("📚 저장된 영상 일기")

if st.session_state.diary_entries:
    for i, entry in enumerate(reversed(st.session_state.diary_entries)):
        with st.expander(f"📔 일기 #{len(st.session_state.diary_entries)-i} - {entry['timestamp']}"):
            col_info1, col_info2 = st.columns(2)
            
            with col_info1:
                st.write("**📝 일기 내용:**")
                st.write(entry['diary_text'] if entry['diary_text'] else "_내용 없음_")
                st.write(f"**🎭 주요 감정:** {entry['dominant_emotion']}")
                st.write(f"**📊 평균 확신도:** {entry['avg_confidence']*100:.1f}%")
                if entry.get('voice_input_used'):
                    st.write("**🎤 음성 입력:** 사용됨")
            
            with col_info2:
                st.write(f"**🎬 프레임 수:** {entry['frame_count']}")
                st.write(f"**🔒 익명화:** {entry['anonymize_method']}")
                st.write(f"**⏱️ 녹화 시간:** {entry.get('recording_duration', '00:00')}")
                st.write(f"**📏 영상 길이:** 약 {entry['frame_count'] / 20:.1f}초")
                
                # 다운로드 버튼들
                col_dl1, col_dl2 = st.columns(2)
                
                with col_dl1:
                    # 비디오 다운로드
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
                    # 텍스트 다운로드
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
else:
    st.info("📭 아직 저장된 영상 일기가 없습니다. 위에서 녹화를 시작해보세요!")

# 사이드바
with st.sidebar:
    st.header("ℹ️ 사용 방법")
    st.markdown("""
    ### 📹 영상 일기 작성 순서
    
    1. 🎨 **익명화 방식 선택**
    2. 🔴 **녹화 시작** 클릭
    3. 😊 **감정 표현하며 이야기**
    4. ⏹️ **녹화 중지 & 저장**
    5. 📊 **감정 분석 확인**
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
    - 한국어 음성 인식
    - 실시간 텍스트 변환
    - 자동 저장 및 다운로드
    
    ### 💡 촬영 팁
    - 💡 밝은 조명 사용
    - 📷 정면 얼굴 유지
    - 😀 자연스러운 표정
    - 🔇 조용한 환경 (음성 입력 시)
    - 🎤 마이크 가까이에서 말하기
    
    ### 🛠️ 기술 스택
    - Python 3.12
    - Streamlit
    - OpenCV
    - MediaPipe
    - Hugging Face
    - SpeechRecognition
    - Plotly
    
    """)
    
    st.markdown("---")
    
    # 통계
    st.subheader("📈 전체 통계")
    total_entries = len(st.session_state.diary_entries)
    total_frames = sum([e['frame_count'] for e in st.session_state.diary_entries])
    voice_entries = sum([1 for e in st.session_state.diary_entries if e.get('voice_input_used')])
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
        st.session_state.last_saved_entry = None
        st.session_state.voice_recording = False
        st.session_state.transcribed_text = ""
        st.session_state.show_emotion_chart = False
        st.success("✅ 초기화 완료!")
        st.rerun()