# ComfyUI Viggle Animate H3 — Latent Upscale & Chunked Refine

Viggle Animate H3 영상 생성에 **MiniMax H3 learned latent upscaler**와 **두 번째 H3 refine pass**를 결합한 ComfyUI workflow 모음입니다.

이 저장소는 실제 테스트에서 최종적으로 성공한 구성만 포함합니다. 개발 중 실패했던 custom node 버전과 중간 workflow는 포함하지 않습니다.

## 포함 내용

```text
custom_nodes/
└─ ComfyUI-Viggle-Chunked-Upscale/
   └─ __init__.py

workflows/
├─ viggle-animate-h3-single-latent-upscale-switch.json
└─ viggle-animate-h3-chunked-latent-upscale-refine.json
```

### 1. Single workflow — switch 방식

`workflows/viggle-animate-h3-single-latent-upscale-switch.json`

짧은 영상 또는 한 번에 처리 가능한 길이를 위한 workflow입니다.

- **Switch OFF**: 약 1.0 MP로 직접 H3 생성
- **Switch ON**: 약 0.7 MP H3 생성 → learned latent upscale → 약 1.2 MP H3 refine
- ON/OFF 결과를 쉽게 비교할 수 있도록 구성
- 기본 refine은 3-step

권장 refine sigma:

```text
0.6000, 0.4286, 0.2000, 0.0000
```

메모리가 부족하거나 속도를 우선할 경우:

```text
0.6000, 0.3333, 0.0000
```

### 2. Chunk workflow — 긴 영상

`workflows/viggle-animate-h3-chunked-latent-upscale-refine.json`

Viggle Animate H3의 긴 영상을 chunk 단위로 처리하면서 각 chunk마다:

```text
low-resolution H3 sample
→ MiniMax H3 learned latent upscale
→ high-resolution H3 refine
→ CPU로 chunk latent offload
→ 다음 chunk 처리
→ 전체 latent assemble
→ 마지막에 VAE decode
```

을 수행합니다.

기본값:

```text
chunk_frames = 107
overlap_frames = 22
continuation = five_frame_anchor
low pass = 0.7 MP
refine = 1.2 MP
refine sigma = 0.6000, 0.4286, 0.2000, 0.0000
```

MiniMax H3 시간축 grid 규칙 때문에 `chunk_frames`는 일반적으로 다음 형태를 사용합니다.

```text
17j + 5
```

예:

```text
56, 73, 90, 107, 124 ...
```

32 GB VRAM에서 107-frame / 1.2 MP가 부담되면 먼저 다음 순서로 낮추는 것을 권장합니다.

1. `chunk_frames: 107 → 90`
2. `refine resolution: 1.2 MP → 1.1 MP`
3. refine sigma를 3-step → 2-step으로 변경

## Custom Node

`ComfyUI-Viggle-Chunked-Upscale`은 stock Viggle chunk sampler를 감싸는 companion node입니다.

노드 이름:

```text
Viggle Chunked Sampler + Latent Upscale Refine
```

주요 동작:

- Switch OFF에서는 stock `ViggleChunkedSampler` 동작 사용
- Switch ON에서는 각 chunk마다 low-res generation 후 learned latent upscale 수행
- 각 chunk에 해당하는 실제 high-resolution conditioning으로 두 번째 H3 refine 수행
- video/audio H3 joint latent의 CUDA device를 일치시켜 refine
- 완료된 chunk latent와 continuation anchor를 CPU에 유지하여 VRAM 누적 최소화
- low pass 작업 tensor를 high refine 전에 제거
- latent upscaler model을 refine 전에 CPU로 offload
- VAE를 anchor 처리 후 명시적으로 CPU offload
- chunk 사이 cache 정리 및 model unload 시도
- 전체 high-resolution latent는 CPU에서 누적한 뒤 마지막에 한 번만 decode

## 필수 Custom Nodes / 의존성

이 저장소에는 외부 프로젝트 자체를 복사해 넣지 않습니다. 아래 프로젝트를 별도로 설치해야 합니다.

### Viggle Animate H3

```text
https://github.com/Saganaki22/ComfyUI-Viggle-Animate-H3
```

필수 node 중 하나:

```text
ViggleChunkedSampler
ViggleAnimateConditioningWindowed
```

### MiniMax H3 Latent Upscaler

```text
https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler
```

사용 모델 예:

```text
minimax_h3_latent_upscaler_3d_fp16.safetensors
```

### Workflow에서 추가로 사용하는 node

환경에 따라 다음 custom node들이 필요할 수 있습니다.

- ComfyUI-KJNodes (`ModelPreviewOverrideKJ` 등)
- VideoHelperSuite (`VHS_LoadVideo`, `VHS_VideoCombine`)
- `JWImageResizeToMegapixels`
- Block Sparse Attention node

ComfyUI workflow를 불러온 후 Missing Nodes가 표시되면 ComfyUI-Manager에서 해당 node를 설치하십시오.

## 설치

### Custom Node 설치

이 저장소의 다음 폴더를:

```text
custom_nodes/ComfyUI-Viggle-Chunked-Upscale
```

ComfyUI 아래로 복사합니다.

```text
ComfyUI/custom_nodes/ComfyUI-Viggle-Chunked-Upscale
```

결과적으로 다음 파일이 존재해야 합니다.

```text
ComfyUI/custom_nodes/ComfyUI-Viggle-Chunked-Upscale/__init__.py
```

그 후 **ComfyUI를 완전히 재시작**하십시오. 브라우저 refresh만으로는 custom node Python 코드가 다시 로드되지 않습니다.

### Workflow 설치

ComfyUI에서 `workflows/` 아래 JSON을 drag & drop 하거나 `Load`로 불러옵니다.

## GPU / PyTorch 권장 환경

최종 chunk + latent upscale + refine 구성은 다음 환경에서 성공 확인했습니다.

```text
GPU: NVIDIA GeForce RTX 5090 32 GB
ComfyUI: 0.35.0
Python: 3.12
PyTorch: 2.11.0+cu130
CUDA runtime: 13.0
```

RTX 50-series / Blackwell에서는 PyTorch CUDA 13.0 이상을 권장합니다.

다음 명령으로 확인할 수 있습니다.

```bash
python - <<'PY'
import torch
print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.get_device_name(0))
PY
```

검증 환경에서는 다음과 같이 출력됐습니다.

```text
2.11.0+cu130
13.0
NVIDIA GeForce RTX 5090
```

ComfyUI startup log에서도 `comfy_kitchen` CUDA backend가 활성화되어 있는지 확인하는 것을 권장합니다.

```text
Found comfy_kitchen backend cuda: ... 'available': True, 'disabled': False ...
```

CUDA backend가 비활성화되고 eager backend만 사용될 경우 MiniMax H3 INT8 연산의 임시 메모리 사용량이 커져 high-resolution refine에서 OOM이 발생할 수 있습니다.

## Latent Upscale 효과를 더 강하게 보려면

learned latent upscaler만으로 최종 영상 디테일이 완성되는 것은 아닙니다. upscale된 latent를 H3 refine pass가 실제 디테일로 정착시키는 과정이 중요합니다.

그래서 품질 우선 기본값은 3-step refine입니다.

```text
0.6000, 0.4286, 0.2000, 0.0000
```

2-step:

```text
0.6000, 0.3333, 0.0000
```

은 빠르고 메모리를 덜 사용하지만 latent upscaler 효과가 상대적으로 약하게 느껴질 수 있습니다.

비교 시에는 특히 다음 부분을 관찰하면 차이가 잘 보입니다.

- 머리카락과 얼굴 주변 경계
- 의상 질감
- 손과 손가락 경계
- 작은 패턴
- 배경 구조물의 세부 묘사

## 메모리 문제 해결 순서

OOM이 발생하면 무조건 해상도부터 크게 낮추기보다 다음 순서로 확인하는 것을 권장합니다.

1. PyTorch가 Blackwell에서 적절한 CUDA build인지 확인
2. `comfy_kitchen backend cuda`가 `disabled: False`인지 확인
3. ComfyUI 완전 재시작 후 재시험
4. `chunk_frames`를 107에서 90으로 감소
5. refine을 1.2 MP에서 1.1 MP로 감소
6. 3-step refine을 2-step으로 감소

## 모델 파일

모델 weight는 이 저장소에 포함하지 않습니다. 각 upstream 프로젝트의 안내에 따라 직접 다운로드하십시오.

## 참고

이 companion node는 upstream Viggle / MiniMax H3 프로젝트의 내부 구현을 활용합니다. upstream에서 node API나 내부 함수가 크게 변경되면 호환성 수정이 필요할 수 있습니다.
