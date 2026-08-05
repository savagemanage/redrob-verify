# redrob-verify

[English](README.md)

**문서 OCR**, **위조 탐지**, **얼굴 대조**, **개발자 신원 집계**를 위한
공개 평가 하네스와 참조 마이크로서비스입니다.

각 기능은 공통 사전점검 계약(`/v1/meta`)을 가진 HTTP 서비스입니다.
호스트 측 하네스가 재현 가능한 지표를 측정하고 JSON(및 선택적 HTML 리포트)을 씁니다.

**작성자:** 이장훈 (Janghoon Lee)

## 기능

| 모듈 | 서비스 포트 | 지표 초점 |
|------|-------------|-----------|
| 문서 OCR | `:8001` | 필드 단위 문자 오류율 (`cer_field`) |
| 위조 탐지 | `:8003` | 점수 기반 TPR / F1 (임계값 스윕) |
| 얼굴 대조 | `:8002` | 민감도 / 정확도 (임계값 스윕) |
| 신원 집계 | `:8004` | 공개 프로필 기준 종단 지연 |

- OCR·위조 학습/추론용 **GPU** Docker 스택
- 출처(`origin`)·동결 검사를 포함한 매니페스트
- Apache-2.0 지향 모델 선택 (`LICENSES.md` 참고)

## 구조

```
┌────────────┐     ┌─────┐  ┌──────┐  ┌─────────┐  ┌──────────┐
│  harness   │────▶│ OCR │  │ Face │  │ Forgery │  │ Identity │
│ eval_*.py  │     │8001 │  │ 8002 │  │  8003   │  │   8004   │
└────────────┘     └─────┘  └──────┘  └─────────┘  └──────────┘
       │                ▲
       │                │  선택적 NestJS 게이트웨이 :8000
       ▼
  results/*.json  report.html
```

기본 OCR 백엔드: **PaddleOCR classic** (`ocr.backend`로 VL 사용 가능). 얼굴: **OpenCV Zoo YuNet + SFace**.
위조: **ForgeryNet** (Apache 지향).

Hugging Face 공개 가중치 (`./tools/fetch_models.sh`로 가져옴):

| Hub 저장소 | 내용 |
|------------|------|
| [`savagemanage/redrob-verify-face`](https://huggingface.co/savagemanage/redrob-verify-face) | YuNet + SFace ONNX |
| [`savagemanage/redrob-verify-forgery`](https://huggingface.co/savagemanage/redrob-verify-forgery) | ForgeryNet 체크포인트 |

OCR VL / Paddle 아티팩트는 해당 백엔드 사용 시 업스트림 Hub에서 가져옵니다.

## 요구사항

- Python 3.11+ 및 [uv](https://docs.astral.sh/uv/)
- NVIDIA Container Toolkit이 있는 Docker (GPU OCR / 위조용)
- 선택: Java 11+ 및 [Apache JMeter](https://jmeter.apache.org/) 5.6+ (신원 지연 평가)

**GPU 참고:** OCR 기본값은 **cu129** 인덱스의 `paddlepaddle-gpu==3.3.0`
(Blackwell / sm_120, 예: RTX 50 시리즈, RTX PRO 6000). 구형 카드는 compose
빌드 인자 `PADDLE_INDEX=.../cu126/`, `PADDLE_PACKAGE=paddlepaddle-gpu==3.2.1`로
재빌드하면 됩니다. `/v1/meta`에 `backend=stub`이면 `stub_reason`과 OCR 컨테이너
로그에서 `Mismatched GPU Architecture`를 확인하세요.

## 빠른 시작

```bash
git clone https://github.com/savagemanage/redrob-verify.git
cd redrob-verify
chmod +x run.sh tools/bootstrap_gpu.sh tools/fetch_models.sh

# GPU 머신 원샷: 의존성 → 모델 → MIDV 데이터 → compose up → OCR 스모크
./run.sh bootstrap-gpu
```

`data/`가 이미 채워져 있으면:

```bash
SKIP_MIDV=1 ./run.sh bootstrap-gpu
```

수동 단계:

```bash
cp .env.example .env   # 선택: GITHUB_TOKEN / OPENAI_API_KEY
./run.sh setup
./run.sh fetch-models
./run.sh fetch-midv && ./run.sh ingest-midv
./run.sh up
./run.sh smoke-ocr
./run.sh preflight
```

## 데이터셋

이미지·아카이브는 git에 넣지 않습니다 (`.gitignore` 참고). 매니페스트와 README만 커밋합니다.

| 경로 | 역할 | 확보 방법 |
|------|------|-----------|
| `data/1_ocr` | OCR 평가 | `./run.sh fetch-midv` + `ingest-midv` (MIDV-2020) |
| `data/2_forgery` | 위조 평가/학습 | MIDV 진본 + `./run.sh gen-forgery` / **문서 단위 홀드아웃** (`./run.sh split-forgery-holdout`) |
| `data/3_face` | 얼굴 쌍 | MIDV 인제스트 / 페어 도구 |
| `data/4_resume` | 신원 지연 | `./run.sh gen-resumes` |

MIDV-2020 저자 FTP: `ftp://smartengines.com/midv-2020`  
인용: Bulatov et al., 2021, [arXiv:2107.00396](https://arxiv.org/abs/2107.00396).  
각 데이터셋 `license.txt`를 따르세요. 상세: `LICENSES.md`, `data/README.md`.

## 평가

```bash
./run.sh eval-cer        # OCR cer_field
./run.sh eval-forgery
./run.sh eval-face
./run.sh eval-tc6        # JMeter 필요
./run.sh report          # → results/report.html (gitignore)
```

목표·시드는 `config.yaml`에 있습니다. 전체 OCR 평가 시
`ocr.eval_max_samples: null`을 유지하세요.

### 위조 홀드아웃 (권장)

ForgeryNet Hub 가중치는 **문서 단위로 분리된** 400/100 분할로 학습되어,
평가용 진본 ID가 학습 위조·JPEG 네거티브에 나타나지 않습니다:

```bash
./run.sh split-forgery-holdout --seed 7 --train-n 400 --eval-n 100 \
  --regenerate-train --rebuild-eval
./run.sh train-forgery   # 또는 docker GPU 학습 — services/forgery/ 참고
./run.sh eval-forgery    # config expected_counts에서 n=200 기대
```

다중 시드 최소값 (시드 7/13/42): TPR ≥ **0.88**, F1 ≥ **0.798**. 공개 Hub
임계값은 **0.87** (시드 7). 상세: `services/forgery/MODEL_CARD.md`,
`data/2_forgery/README.md`.

## 설정

| 파일 | 용도 |
|------|------|
| `config.yaml` | 엔드포인트, 지표 목표, 시드, OCR 백엔드 |
| `.env` | 선택 비밀값 (커밋 금지; `.env.example` 사용) |
| `docker-compose.yml` | 서비스별 이미지 (Paddle / Torch / OpenCV 격리) |

유용한 명령: `./run.sh help`

## 개발

```bash
./run.sh test
./run.sh freeze --strict   # 출처 / 건수 게이트
```

## 라이선스

코드: **Apache License 2.0** — `LICENSE` 참고.
저작권 표기: `NOTICE` (Copyright 2026 이장훈 / Janghoon Lee).

서드파티 모델·데이터셋(PaddleOCR-VL, OpenCV Zoo, MIDV-2020, TruFor 등)은
각각 별도 조건이 있으며, 목록과 제약은 `LICENSES.md`에 있습니다. 가중치와
원본 이미지는 로컬에 내려받으며 커밋하지 마세요.

**위조 참고:** 기본 백엔드는 **ForgeryNet**(Apache 지향, 인레포 +
torchvision ImageNet 초기화). 가중치: [`savagemanage/redrob-verify-forgery`](https://huggingface.co/savagemanage/redrob-verify-forgery).
선택 **TruFor** (`FORGERY_BACKEND=trufor`)는 연구 전용(GRIP-UNINA nonprofit
조건) — 해당 가중치는 공개하지 마세요.

## 기여

이슈와 PR을 환영합니다. 다음을 지켜 주세요:

1. `.env`, 모델 가중치, 개인 문서 이미지를 커밋하지 마세요.
2. 새 모델 의존성은 `LICENSES.md`에 기록하세요.
3. OCR·Docker를 건드리는 PR 전에는 `./run.sh test`와 `./run.sh smoke-ocr`을 권장합니다.

## 면책

이 저장소는 **연구/평가 하네스**이며, 바로 쓰는 상용 KYC 제품이 아닙니다.
배포 전 라이선스와 데이터 권리를 스스로 확인하세요.
