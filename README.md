# YOLO11n 사물 인식 웹 (classify-image)

Python + **Gradio** + **Ultralytics YOLO11n** + **OpenCV**로 만든 단순 사물 인식 웹입니다.

- **좌측 (이미지 입력)**: 이미지를 업로드하면 YOLO11n이 사물을 탐지하고, 바운딩 박스 · 범주이름 · 신뢰도를 그린 결과 이미지를 보여줍니다.
- **우측 (웹캠 입력)**: 웹캠 영상을 프레임마다 인식해 바운딩 박스가 그려진 영상을 실시간으로 표시합니다.

전체 코드는 `app.py` 한 파일입니다.

---

## 1. 요구 사항

| 항목 | 내용 |
|---|---|
| Python | 3.10 ~ 3.12 (3.11.9에서 검증) |
| Git | 저장소 내려받기용 |
| 인터넷 | 라이브러리 설치 및 **최초 실행 시 모델 가중치 자동 다운로드** |
| 웹캠 | 우측 실시간 탐지 기능에 필요 (없어도 이미지 업로드는 동작) |
| GPU | 불필요. CPU만으로 동작 (640×480 프레임당 약 60 ms, 개발 PC 기준) |

---

## 2. 개인 노트북에서 내려받아 실행하기

### Windows (PowerShell)

```powershell
# 1) 저장소 내려받기
git clone https://github.com/sngbae12/classify-image.git
cd classify-image

# 2) 가상환경 만들기 + 활성화
python -m venv .venv
.\.venv\Scripts\Activate.ps1
#   "스크립트를 실행할 수 없습니다" 오류가 나면 아래를 한 번 실행한 뒤 다시 활성화:
#   Set-ExecutionPolicy -Scope CurrentUser RemoteSigned

# 3) 라이브러리 설치 (torch 포함, 수 분 소요)
python -m pip install --upgrade pip
pip install -r requirements.txt

# 4) 웹 실행
python app.py
```

### macOS / Linux

```bash
git clone https://github.com/sngbae12/classify-image.git
cd classify-image
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

실행하면 콘솔에 다음과 같이 표시됩니다.

```
* Running on local URL:  http://127.0.0.1:7860
```

브라우저에서 **http://127.0.0.1:7860** 을 열면 됩니다. 종료는 콘솔에서 `Ctrl + C`.

> **CPU 전용 torch로 용량 줄이기 (선택)**
> Linux에서 PyPI 기본 `torch`는 CUDA 라이브러리를 포함해 수 GB를 받습니다. GPU가 없다면 아래처럼 CPU 빌드를 먼저 설치한 뒤 나머지를 설치하세요.
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
> pip install -r requirements.txt
> ```
> Windows는 PyPI 기본 `torch`가 CPU 빌드이므로 그대로 설치하면 됩니다.

---

## 3. 모델 가중치 (`yolo11n.pt`) 준비

모델 가중치 파일(`*.pt`)은 `.gitignore`로 **저장소에서 제외**되어 있습니다. 두 가지 방법 중 하나로 준비합니다.

**방법 A. 자동 다운로드 (기본)**
`python app.py`를 처음 실행하면 Ultralytics가 `yolo11n.pt`(약 5.4 MB)를 `app.py`와 같은 폴더에 자동으로 내려받습니다. 별도 작업이 필요 없습니다.

**방법 B. 수동 준비 (방화벽 등으로 자동 다운로드가 안 될 때)**
아래 URL에서 받아 `app.py`와 같은 폴더(저장소 루트)에 두세요. 앱이 자동으로 받는 파일과 동일합니다.

```powershell
# Windows PowerShell (저장소 루트에서)
Invoke-WebRequest -Uri "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt" -OutFile "yolo11n.pt"
```

```bash
# macOS / Linux
curl -L -o yolo11n.pt https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11n.pt
```

개발 PC의 프로젝트 폴더에 이미 있는 `yolo11n.pt`를 USB 등으로 복사해 저장소 루트에 넣어도 됩니다.

---

## 4. 사용 방법

**이미지 입력 (좌측)**
1. "이미지 업로드" 영역에 이미지를 드래그 앤 드롭하거나 클릭해서 선택합니다.
2. 업로드 즉시 탐지가 실행되고, 아래 "탐지 결과"에 바운딩 박스 · 범주이름 · 신뢰도가 그려진 이미지가 표시됩니다.
3. 이미지를 지우면(X) 결과도 함께 지워집니다.

**웹캠 입력 (우측)**
1. "Click to Access Webcam"을 누르고 브라우저의 카메라 권한 요청을 **허용**합니다.
2. 미리보기가 뜨면 **Record** 버튼을 누릅니다. 스트리밍이 시작되고, 탐지 결과(박스)가 웹캠 화면 위에 실시간으로 표시됩니다.
3. **Stop**을 누르면 중지됩니다.

**탐지 설정 바꾸기** — `app.py` 상단 상수

| 상수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_PATH` | `yolo11n.pt` | 사용할 모델 파일 |
| `CONF_THRESHOLD` | `0.25` | 이 값 이상의 신뢰도만 표시 |
| `IMG_SIZE` | `640` | 추론 입력 크기 (낮추면 빨라짐) |
| `STREAM_EVERY` | `0.1` | 웹캠 프레임 전송 간격(초) |

---

## 5. 환경 변수

이 프로젝트는 **API 키 · 토큰 · 비밀번호 등 비밀정보가 필요 없습니다.**
포트 등 Gradio 서버 설정을 바꾸고 싶을 때만 선택적으로 환경 변수를 사용하며, 목록은 [`.env.example`](.env.example)에 있습니다.

`app.py`는 `.env` 파일을 자동으로 읽지 않으므로 실행 전 셸에서 설정합니다.

```powershell
# Windows PowerShell 예시: 포트 변경
$env:GRADIO_SERVER_PORT = "7870"
python app.py
```

```bash
# macOS / Linux 예시
export GRADIO_SERVER_PORT=7870
python app.py
```

---

## 6. 프로젝트 구조

```
classify-image/
├── app.py             # 웹 앱 전체 (모델 로드, OpenCV 그리기, Gradio UI)
├── requirements.txt   # 의존성 목록
├── .env.example       # 선택적 환경 변수 목록 (비밀정보 없음)
├── .gitignore         # 가상환경, 캐시, 모델 가중치 등 제외
└── README.md
```

실행 후 생성되는 `yolo11n.pt`(자동 다운로드), `__pycache__/`, `.gradio/` 등은 커밋 대상에서 제외됩니다.

---

## 7. 문제 해결

| 증상 | 확인할 것 |
|---|---|
| 웹캠 화면이 뜨지 않음 | 브라우저 카메라 권한 허용 여부. 반드시 `http://127.0.0.1:7860` 또는 `http://localhost:7860`으로 접속 (다른 PC에서 IP로 접속하면 HTTPS가 아니어서 브라우저가 웹캠을 차단). Zoom 등 다른 프로그램이 카메라를 쓰고 있지 않은지 확인 |
| 7860 포트가 이미 사용 중 | Gradio가 자동으로 7861, 7862… 로 넘어갑니다. 콘솔에 표시된 주소로 접속하거나 `GRADIO_SERVER_PORT`로 지정 |
| 모델 다운로드 실패 | 3번 항목의 "방법 B"로 수동 다운로드 |
| 웹캠 탐지가 끊김 | `app.py`의 `IMG_SIZE`를 `480` 또는 `320`으로 낮추거나 `STREAM_EVERY`를 `0.2`로 늘림 |
| `Activate.ps1` 실행 정책 오류 | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` 실행 후 다시 활성화 |
| `pip install` 중 torch 설치가 오래 걸림 | 정상입니다(수백 MB). Linux는 위 "CPU 전용 torch" 방법으로 용량을 줄일 수 있음 |
