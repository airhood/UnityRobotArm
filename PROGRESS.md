# UnityRobotArm — 프로젝트 보고서

## 1. 프로젝트 개요

Unity 로봇팔 시뮬레이터에서 자동으로 pick-and-place 시연 데이터를 수집하고,
CLIP 기반 행동 복제(Behavior Cloning) 모델을 학습해 자연어 텍스트 명령으로 로봇팔을 제어하는 VLA(Vision-Language-Action) 시스템.

```
텍스트 명령 ──→ CLIP 텍스트 인코더 ──┐
                                     │ element-wise product
카메라 이미지 ─→ CLIP 이미지 인코더 ──┘
                         │
               관절각 + EE 위치 ─→ State MLP
                         │
                    Action MLP
                         │
              Δ위치(3) + 그리퍼(1)
                         │
                  Unity 로봇팔 제어
```

---

## 2. 시스템 구성

### 2.1 포트 맵

| 포트 | 방향 | 역할 |
|------|------|------|
| 5005 | Python → Unity | IKServer → IKReceiver (관절각 전송) |
| 5006 | Unity → Python | DataCollector → DataCollectorServer (학습 데이터 수집) |
| 5007 | Python → Unity | AICommandServer (추론 시 로봇 제어) |
| 5008 | Python → Unity | ImageServer (추론 시 카메라 프레임 요청) |

### 2.2 파일 구조

```
UnityRobotArm/                    ← Python 프로젝트
├── config.py                     ← 포트, 경로, 하이퍼파라미터
├── run_inference.py              ← 추론 서버 (텍스트 입력 → 로봇 제어)
├── run_ik_server.py              ← IK 서버 진입점 + start() 함수
├── collect_data.py               ← 데이터 수집 (IK 서버 + DataCollector 동시 실행)
├── collector_server.py           ← DataCollectorServer (포트 5006)
├── manage_episodes.py            ← 에피소드 목록 조회 / 삭제 / 재번호
├── requirements.txt
├── utils/
│   ├── unity_bridge.py           ← UnityBridge(5007), ImageClient(5008)
│   ├── coordinate_transform.py
│   ├── ik_solver.py              ← IKSolver (roboticstoolbox 기반)
│   └── ik_server.py              ← IKServer 클래스 (TCP, 포트 5005)
├── model/
│   ├── architecture.py           ← CLIP 기반 RobotArmModel
│   ├── dataset.py                ← RobotArmDataset (PyTorch)
│   ├── train.py                  ← 학습 루프
│   └── inference.py              ← RobotInference
└── data/
    └── episodes/                 ← 수집된 에피소드 데이터 (그룹별 관리)
        ├── default/              ← 기본 그룹
        │   ├── episode_0000/
        │   └── ...
        └── {group_name}/         ← 추가 그룹 (--group 인자로 지정)

UnityRobotArm-Sim/Assets/Scripts/   ← Unity C# 스크립트
├── SimCore.cs                    ← Singleton tick dispatcher (FixedUpdate, 우선순위 기반 콜백 등록)
├── SimLoop.cs                    ← stub (SimCore로 rename됨)
├── Manipulator.cs                ← 하드웨어 레이어 (actuator 관리, tick 배분) — ControlMode 없음
├── ManipulatorEditor.cs          ← Custom Inspector
├── SBCController.cs              ← Dynamixel 프로파일 설정 + Joint 동기화 + SetPosition 호출
├── IKReceiver.cs                 ← TCP 통신만 담당 (x,y,z 전송 → 관절각 수신)
├── ManualManipulatorController.cs ← Inspector 슬라이더로 수동 관절 제어
├── AICommandServer.cs            ← TCP 서버 (포트 5007), Python 명령 수신
├── DataCollector.cs              ← TCP 클라이언트, 학습 데이터 전송 (포트 5006)
├── ImageServer.cs                ← TCP 서버 (포트 5008), 카메라 프레임 제공
├── PickPlaceDemo.cs              ← 자동 pick-and-place 시연 및 데이터 수집 제어
└── SliderGripper.cs              ← 그리퍼 제어 (EndEffector 서브클래스)
```

---

## 3. 모델 설계

### 3.1 전체 프레임워크: Behavior Cloning 기반 VLA

이 시스템은 **Behavior Cloning(BC)** 방식으로 학습한다. Unity 시뮬레이터에서 IK 솔버가 시연한 pick-and-place 궤적을 정답으로 삼아, 모델이 동일한 입력을 받았을 때 동일한 행동을 출력하도록 지도 학습한다.

```
전문가 시연 데이터 (IK 기반 자동 수집)
  ↓
각 시점 t: (이미지_t, 텍스트, 관절각_t, EE위치_t) → 학습 타겟: EE위치_{t+1}
  ↓
모델 학습: 위 입력을 받아 Δpos = EE위치_{t+1} - EE위치_t 를 예측
  ↓
추론: 매 스텝마다 모델이 다음 이동량을 예측 → Unity로 전송 → 반복
```

절대 위치가 아닌 **delta(상대 이동량)를 예측**하는 이유:
- 절대 위치 예측은 목표 물체의 위치가 달라질 때마다 다른 출력값을 학습해야 하므로 일반화가 어렵다
- Δpos는 "현재 위치에서 어느 방향으로 얼마나 움직일지"만 나타내므로 물체 위치와 무관하게 일반화 가능
- 이는 실제 로봇 제어에서도 흔히 쓰이는 residual/incremental control 방식

### 3.2 아키텍처 개요

```
카메라 이미지 (224×224×3)
  ↓ CLIP ViT-B/32 이미지 인코더 (frozen)
  img_feat: (B, 512)
                         ↘
                           element-wise product
                         ↗              ↓ L2 정규화 후 곱
텍스트 명령 (token IDs)               vl_feat: (B, 512)
  ↓ CLIP 텍스트 인코더 (frozen)                ↓
  text_feat: (B, 512)               concat([vl_feat, state_feat])
                                              ↓ (B, 576)
관절각 5개 + EE 위치 3개 → State MLP → state_feat: (B, 64)
                                              ↓
                                        Action MLP
                                    Linear(576→256) + ReLU
                                       Dropout(0.1)
                                    Linear(256→256) + ReLU
                                              ↓ (B, 256)
                          ┌───────────────────┤
                   position_head         gripper_head
                   Linear(256→3)         Linear(256→1)
                          ↓                    ↓
                   Δpos (dx,dy,dz)       sigmoid → open/close
                   (미터, Unity 좌표계)   (threshold: 0.5)
```

### 3.3 인코더: CLIP ViT-B/32 (Frozen)

**CLIP(Contrastive Language–Image Pretraining)** 은 대규모 이미지-텍스트 쌍으로 사전학습된 모델로, 이미지와 텍스트를 동일한 512차원 임베딩 공간에 정렬한다.

**ViT-B/32 이미지 인코더 구조:**
- 입력 이미지를 32×32 패치 단위로 분할 → 7×7 = 49개 패치 토큰 생성
- Transformer 인코더 (12 layers, 8 heads, hidden dim 512) 처리
- `[CLS]` 토큰에 대응하는 출력 벡터를 이미지 임베딩으로 사용
- 출력: (B, 512)

**텍스트 인코더 구조:**
- 입력 텍스트를 BPE 토크나이저로 77 토큰으로 변환 (패딩/트런케이션)
- Transformer 인코더 (12 layers, 8 heads) 처리
- `[EOS]` 토큰 위치의 출력 벡터를 텍스트 임베딩으로 사용
- 출력: (B, 512)

**Frozen으로 사용하는 이유:**
- 학습 파라미터를 최소화 → 소규모 BC 데이터셋 (~수천 에피소드)에서도 과적합 없이 안정적 학습
- CLIP의 cross-modal alignment가 이미 완성되어 있어 추가 alignment 학습 불필요
- 전체 파라미터 수: CLIP ViT-B/32 약 150M, 학습 파라미터(Action+State MLP)는 약 200K

### 3.4 Vision-Language Fusion: Element-wise Product

L2 정규화 후 element-wise product로 두 modality를 융합한다.

```python
img_feat  = img_feat  / img_feat.norm(dim=-1, keepdim=True)   # (B, 512), unit vector
text_feat = text_feat / text_feat.norm(dim=-1, keepdim=True)  # (B, 512), unit vector
vl_feat   = img_feat * text_feat                               # (B, 512)
```

**다른 fusion 방식과의 비교:**

| 방식 | 출력 차원 | 특징 |
|------|----------|------|
| Concat | (B, 1024) | 단순하지만 두 modality 간 상관관계를 표현하려면 더 깊은 downstream 레이어 필요 |
| Add | (B, 512) | 두 벡터의 평균적 방향, alignment가 강한 경우에 유리 |
| **Element-wise product** | **(B, 512)** | **두 벡터 모두 활성화된 차원만 살아남음 — "이 시각적 특징이 이 텍스트 의미와 얼마나 일치하는가"를 직접 표현** |
| Cross-attention | (B, 512) | 가장 표현력이 높으나 파라미터 증가, 데이터 요구량 증가 |

L2 정규화 후 두 단위벡터의 곱은 각 차원에서의 signed agreement를 나타낸다. CLIP 공간에서 "red"라는 텍스트 특징과 빨간 물체가 있는 이미지 특징이 해당 차원에서 같은 방향으로 활성화되면 그 차원의 값이 커지고, 관련 없는 차원은 상쇄된다.

### 3.5 State Encoder

현재 로봇 상태를 64차원 벡터로 압축한다.

```
입력: [joint_angles(5), ee_position(3)] → (B, 8)
  ↓ Linear(8 → 64) + ReLU
  ↓ Linear(64 → 64) + ReLU
출력: state_feat (B, 64)
```

**입력 구성:**
- `joint_angles` (5개, degree): Joint[0]~Joint[4] 각도. 현재 자세(configuration)를 나타냄
- `ee_position` (3개, meter): end-effector의 Unity 월드 좌표. 현재 공간적 위치를 나타냄

관절각과 EE 위치를 모두 제공하는 이유: 동일한 EE 위치라도 다른 관절 배치(elbow-up vs elbow-down 등)에서는 다음 가능한 이동 방향이 달라질 수 있기 때문에 두 정보를 모두 포함한다.

State feature의 차원(64)을 VL feature(512)보다 훨씬 작게 설정한 이유: 언어-시각 정보가 어디로 이동할지의 의미론적 결정을 주도하고, 상태 정보는 현재 위치에서의 fine-grained 조정에만 기여하도록 역할을 분리한다.

### 3.6 Action MLP 및 예측 헤드

VL feature(512)와 state feature(64)를 이어붙인 (B, 576) 벡터를 입력으로 받는다.

```
Linear(576 → 256) + ReLU
Dropout(0.1)
Linear(256 → 256) + ReLU
         ↓ 공유 표현 (B, 256)
         ├── position_head: Linear(256 → 3)  →  Δpos (dx, dy, dz)
         └── gripper_head:  Linear(256 → 1)  →  gripper logit
```

**position_head (회귀):**
- 출력: 현재 EE 위치 기준 상대 이동량 (dx, dy, dz), 단위 m
- 손실 함수: MSELoss
  ```python
  loss = MSELoss(delta_pred, ee_position_{t+1} - ee_position_t)
  ```

**gripper_head (이진 분류):**
- 출력: sigmoid 후 0.5 초과 시 open, 이하 시 close
- 현재 구현에서는 gripper loss를 position loss에 합산하지 않음 (별도 학습 가능)

**Dropout(0.1):**
- 소규모 데이터셋에서 과적합 억제
- VL feature와 state feature가 concat된 고차원 입력에서 특정 feature에 의존하는 것을 방지

**공유 표현(shared trunk):**
- position head와 gripper head가 같은 256차원 표현을 공유
- 이동 방향과 그리퍼 동작이 같은 맥락(task 이해)에서 결정된다는 가정을 반영

### 3.7 학습 설정

| 항목 | 값 | 근거 |
|------|-----|------|
| Optimizer | AdamW | weight decay로 implicit regularization, 소규모 파라미터에 적합 |
| weight_decay | 1e-4 | 과적합 억제, 큰 값은 학습 방해 |
| Learning rate | 1e-4 | Action MLP 규모에 적합한 안정적 범위 |
| Scheduler | CosineAnnealingLR | 후반부에 lr을 점진적으로 감소, 수렴 안정화 |
| Batch size | 32 | GPU 메모리와 gradient noise의 균형 |
| Epochs | 100 | CosineAnnealingLR의 T_max와 일치 |
| Loss | MSELoss | 연속적인 delta 예측에 적합한 회귀 손실 |
| Gradient clipping | max_norm=1.0 | 폭발적 gradient 방지 (특히 학습 초반) |
| Checkpoint | best val loss + 매 10 epoch | 과적합 감지용 best 모델 보존 |

**학습 루프 흐름:**
```
for epoch in 1..100:
    for batch in train_dl:
        # CLIP 인코딩 (no_grad, frozen)
        img_feat  = CLIP.encode_image(images)   → L2 normalize
        text_feat = CLIP.encode_text(tokens)    → L2 normalize

        # 예측
        delta_pred, gripper_logit = model(img_feat, text_feat, joint_angles, ee_pos)

        # 손실 계산
        loss = MSELoss(delta_pred, ee_pos_{t+1} - ee_pos_t)

        # 역전파 (CLIP 파라미터는 gradient 없음)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

    scheduler.step()
    # val loss < best → best_model.pt 저장
```

### 3.8 파라미터 수 분석

| 컴포넌트 | 파라미터 수 | 학습 여부 |
|----------|------------|----------|
| CLIP ViT-B/32 이미지 인코더 | ~87M | Frozen |
| CLIP 텍스트 인코더 | ~63M | Frozen |
| State MLP (8→64→64) | 8×64 + 64×64 = 4,608 | 학습 |
| Action MLP (576→256→256) | 576×256 + 256×256 = 213,504 | 학습 |
| position_head (256→3) | 768 | 학습 |
| gripper_head (256→1) | 256 | 학습 |
| **학습 파라미터 합계** | **~219K** | **학습** |
| **전체 파라미터 합계** | **~150M** | — |

학습 파라미터가 전체의 0.15%에 불과하므로, 수천 에피소드 수준의 데이터셋에서도 안정적으로 수렴한다.

---

## 4. 데이터 수집 파이프라인

### 4.1 전체 흐름

```
Unity PickPlaceDemo.cs
  → 에피소드 자동 진행 (랜덤 오브젝트 스폰 → 집기 → 놓기)
  → DataCollector.cs: 각 프레임마다 JSON(상태) + JPEG(카메라) 전송 (포트 5006)

Python DataCollectorServer
  → Unity로부터 프레임 수신
  → 에피소드 경계 자동 감지 (episode_id 변화 감지)
  → data/episodes/episode_XXXX/ 에 저장
       ├── frame_000000.jpg
       ├── frame_000001.jpg
       ├── ...
       ├── states.json          ← 모든 프레임 상태 리스트
       └── metadata.json        ← label, frame_count, 시간 등
```

### 4.2 전송 데이터 형식

**와이어 포맷** (big-endian uint32 length-prefix):
```
[4 bytes: json_len][json_bytes][4 bytes: jpg_len][jpg_bytes]
```

**JSON 상태 필드** (프레임당):
```json
{
  "episode_id": 3,
  "episode_label": "pick up the red cube",
  "phase": "approach",
  "timestamp": 1234567890.123,
  "joint_angles": [10.5, -20.3, 45.0, -15.2, 0.0],
  "ee_position": [0.15, 0.12, -0.08],
  "gripper_angle": -90.0
}
```

### 4.3 에피소드 레이블 생성 규칙

Unity `PickPlaceDemo.cs`에서 자동 생성:
```
"pick up the {color} {objectType}"
예: "pick up the red cube", "pick up the blue sphere"
```
- 색상: Prefab에 설정된 material color 이름 (영어 고정)
- CLIP 텍스트 인코더에 바로 입력 가능한 형태

### 4.4 IK 서버

Python에서 역기구학(IK) 계산 후 Unity로 관절각 전송:

```
Unity IKReceiver.cs → [x, y, z]\n → IKServer (포트 5005)
IKServer → IKSolver.solve(x, y, z) → [j1, j2, j3, j4, j5]\n → Unity
```

- `roboticstoolbox-python`: Levenberg-Marquardt IK (`ik_LM`)
- 50회 random restart (초기값 전략: 방위각 고정 + 나머지 랜덤)
- 오차 2mm 이하 시 성공, 초과 시 `IK_FAIL` 반환
- Unity 왼손 좌표계 → 오른손 좌표계 변환 후 IK 계산

### 4.5 로봇팔 구조 (링크 길이)

| 링크 | 길이 (m) | 설명 |
|------|----------|------|
| l0 | 0.036 | base → Joint[0] |
| l1 | 0.0405 | Joint[0] → Joint[1] |
| l2a | 0.128 | Joint[1] → FixedJoint (꺾이기 전) |
| l2b | 0.024 | FixedJoint → Joint[2] (꺾인 후) |
| l3 | 0.124 | Joint[2] → Joint[3] |
| l4 | 0.06404 | Joint[3] → Joint[4] |
| l5 | 0.13906 | Joint[4] → EndEffector |

FixedAngleJoint: Z축 -90° (ㄱ자 꺾임 구조)

---

## 5. 데이터셋 구조

### 5.1 RobotArmDataset

```
data/episodes/
├── episode_0000/
│   ├── frame_000000.jpg
│   ├── ...
│   ├── states.json
│   └── metadata.json
├── episode_0001/
│   └── ...
└── ...
```

### 5.2 샘플 구성 (t → t+1 pairs)

| 필드 | 형태 | 설명 |
|------|------|------|
| `image` | (3, 224, 224) | CLIP 전처리된 프레임 t 이미지 |
| `text_tokens` | (77,) | `clip.tokenize(label)` 결과 |
| `joint_angles` | (5,) | 프레임 t 관절각 (degree) |
| `ee_position` | (3,) | 프레임 t EE 위치 (m) |
| `target_position` | (3,) | 프레임 t+1 EE 위치 (m) |

- 학습 타겟: `target_position - ee_position` = Δpos
- Train/Val split: 에피소드 단위, 9:1 비율
- CLIP 전처리 미제공 시 ImageNet 정규화로 fallback

---

## 6. 추론 파이프라인

### 6.1 흐름

```
사용자 텍스트 입력
  → CLIP 텍스트 인코딩
  → Unity ImageServer (포트 5008)에서 카메라 프레임 요청
  → Unity AICommandServer (포트 5007)에서 로봇 상태 요청
  → RobotInference.predict() → Δpos + gripper
  → MOVE_REL 명령 + GRIPPER 명령 → Unity
  → |Δ| < 0.01m 또는 최대 50 스텝까지 반복
```

### 6.2 종료 조건

| 조건 | 설명 |
|------|------|
| `|Δ| < 0.01m` | 예측 delta 크기가 임계값 이하 → 목표 도달 판정 |
| `step >= 50` | 최대 스텝 초과 → 강제 종료 |
| `0.2초` 간격 | 스텝 간 대기 (물리 시뮬레이션 안정화) |

---

## 7. Unity 씬 구성 (PickPlaceDemo)

### 7.1 Workspace 및 Spawn 설정

Inspector에서 설정 가능한 직육면체 영역:

| 변수 | 설명 |
|------|------|
| `workspaceX/Y/Z` | 로봇팔 작업 가능 영역 (절대 좌표) |
| `tableY` | workspaceY.x 기준 테이블 상면 높이 (상대) |
| `tableThickness` | 테이블 두께 |
| `useExclusionZone` | Spawn 제외 영역 활성화 |
| `excludeRangeX/Y/Z` | 제외 영역 크기 (tableY 기준 상대 좌표) |
| `maxSpawnRetries` | Spawn 재시도 최대 횟수 |

물체 스폰 Y 위치: `workspaceY.x + tableY + 0.02f` (테이블 위 물리 settle 여유)

### 7.2 Gizmo 색 구분

| 색 | 의미 |
|----|------|
| 초록 (Green) | Workspace 영역 |
| 주황 (Orange) | Spawn 제외 영역 |
| 마젠타 (Magenta) | 테이블 |

### 7.3 에피소드 자동 진행

```
물체 스폰 (랜덤 위치 + 랜덤 material) → BeginEpisode
  → 접근 (approach)
  → 파지 (grasp): kinematic attach
  → 이동 (transport)
  → 놓기 (release): kinematic detach
→ EndEpisode → 다음 에피소드
```

---

## 8. 주요 버그 수정 이력

### 8.0 Joint 동기화 acceleration 반올림 오류 (→ SBCController.cs로 이동)

- **원인**: `profileAcceleration * scale` 계산 시 `RoundToInt`로 내림 반올림되는 케이스(예: `7 × 0.33 = 2.33 → 2`)에서 비례 가속도보다 낮아져 해당 joint가 다른 joint보다 늦게 도달 → arm sweep 현상
- **수정**: reference joint의 이동 시간 T_ref를 트라페조이달 프로파일로 계산한 뒤, 각 joint에 대해 T_ref에 가장 근접한 정수 acceleration tick(floor/ceil)을 선택. 늦는 것보다 약간 빨리 도달하는 방향으로 우선
- **리팩토링**: 해당 로직은 아키텍처 분리 이후 `SBCController.cs`로 이전됨 (원래 `IKReceiver.cs`에 있었음)

### 8.1 JSON 파싱 오류 (DataCollector.cs)

- **원인**: C# interpolated string이 `CultureInfo.CurrentCulture` 사용 → 일부 시스템에서 소수점 구분자가 `,`가 되어 `"gripper_angle":-90,00}` 같은 잘못된 JSON 생성
- **수정**: 모든 숫자 포맷팅에 `CultureInfo.InvariantCulture` 명시적 적용
- `EscapeJson()` 강화: `\`, `\n`, `\r`, `\t` 처리 추가

### 8.2 테이블 Material 오류 (magenta 표시)

- **원인**: `new Material(Shader.Find("Standard"))` → URP/HDRP 환경에서 shader null 반환
- **수정**: `CreatePrimitive`가 생성하는 기본 material을 그대로 사용하고 `.material.color`만 설정

---

## 9. 완료된 작업

### Unity C# 스크립트

#### 아키텍처 레이어 분리 (리팩토링)
- [x] `SimCore.cs` (신규)
  - Singleton tick dispatcher — priority 기반 콜백 등록/해제
  - `Register(Action<float> tick, int priority)` / `Unregister()`
  - Script Execution Order: SimCore=-50, Actuator=-20
- [x] `SimLoop.cs` — stub으로 유지 (`// Renamed to SimCore.cs`)
- [x] `Manipulator.cs` — ControlMode 제거, 순수 하드웨어 레이어로 단순화
  - `ControlMode` enum 및 `controlMode` 필드 제거
  - `manualAngles`, `InitializeManualMode()`, `SetManualAngle()`, `GetManualAngle()` 제거
  - `Update()`, `OnGUI()` (수동 슬라이더 UI) 제거
  - EndEffector Tick 지원 — `ConfigType.EndEffector` 항목을 SimCore Tick 루프에 포함
- [x] `SBCController.cs` (신규)
  - Dynamixel 프로파일 설정 (`profileVelocity`, `profileAcceleration`)
  - Joint 동기화 시스템 (IKReceiver.cs에서 이전): T_ref 기반 최적 acceleration tick 선택
  - `IKReceiver.LastJointAngles` 읽어 `actuatorInstances[i].SetPosition()` 호출
  - SimCore priority=0 등록 (Manipulator priority=10보다 먼저)
- [x] `IKReceiver.cs` — TCP 통신 전담으로 단순화
  - `LastJointAngles` 프로퍼티 유지
  - `Update()`: `x,y,z` 3값만 전송 (ControlMode 체크, pitch 없음)
  - Joint 동기화 및 SetPosition 로직은 SBCController로 이전됨
- [x] `ManualManipulatorController.cs` (신규)
  - Inspector 슬라이더로 수동 관절 제어
  - `Update()`: 슬라이더 값 → `actuatorInstances[i].SetPosition()` 호출
- [x] `EndEffector.cs` — `virtual Tick()` 추가
- [x] `SliderGripper.cs`
  - 크랭크-슬라이더 메커니즘 수식 완성 (degree→radian 변환, Clamp로 arcsin 안전 처리)
  - slider2 localPosition 할당 순서 버그 수정
  - `Tick()` override로 Manipulator FixedUpdate에 동기화 (Update 제거)
- [x] `MaterialSet.cs` — ScriptableObject 기반 material 묶음 (NamedMaterial: material + colorName)
- [x] `SliderGripper.cs` — `SetGripperOpen(bool)` 메서드 추가
- [x] `AICommandServer.cs` — MOVE_TO / MOVE_REL / GRIPPER / GET_STATE
- [x] `DataCollector.cs` — BeginEpisode / EndEpisode / SetPhase + CultureInfo.InvariantCulture 버그 수정
- [x] `ImageServer.cs` — 포트 5008, GET 요청 → JPEG 응답
- [x] `PickPlaceDemo.cs`
  - 런타임 오브젝트 / IK Target 동적 생성
  - **다중 오브젝트 + 다중 drop zone** 스폰 (에피소드마다 랜덤 선택)
  - `PickablePrefab` / `DropZonePrefab`: prefab + displayNames + MaterialSet
  - `SpawnedObject`: bottom(파지 높이 기준) + top(gizmo 위치) child 참조
  - `SpawnedZone`: pivot(물체 놓는 면) + bottom(테이블 정렬) child 참조
  - 물체 높이 기반 정확한 grasp/place 위치 계산 (`_objBottomOffset = endPoint.y - bottom.y`)
  - Rigidbody 물체 kinematic attach/detach
  - 테이블 동적 생성 (workspaceY 기준 상대 좌표)
  - Spawn Exclusion Zone (XZ 체크)
  - Renderer bounds 체크로 workspace 경계 밖으로 나가는 오브젝트 자동 위치 보정 (`ClampToBounds`)
  - **프롬프트 다양성**: 23개 템플릿 × 오브젝트(색/형상/둘다/generic) × zone(색/형상/둘다/generic) 조합 (~2,208 가지)
  - Gizmo: 초록(Workspace) / 주황(제외 영역) / 마젠타(테이블) / 노란 구(target 오브젝트 top) / 시안 구(target zone pivot)
  - 에피소드 자동 진행 및 DataCollector 연동

### Python 프로젝트
- [x] `config.py` — `DEFAULT_GROUP = "default"` 추가 (단, `TEXT_EMBED_DIM = 384` 잔여 코드 있음 — 미사용)
- [x] `utils/unity_bridge.py` — UnityBridge + ImageClient
- [x] `utils/coordinate_transform.py`
- [x] `utils/ik_solver.py` — IKSolver (roboticstoolbox-python, LM IK)
- [x] `utils/ik_server.py` — IKServer 클래스 (TCP, 포트 5005)
- [x] `run_ik_server.py` — `start()` 함수 + 단독 실행 진입점
- [x] `collect_data.py` — IK 서버 + DataCollector 동시 실행, `--group` 지원
- [x] `manage_episodes.py` — 그룹 기반 에피소드 관리, 병렬 처리, migration
- [x] `collector_server.py` — 에피소드 자동 감지, JPEG+JSON 저장
- [x] `model/dataset.py` — RobotArmDataset (CLIP 전처리, t→t+1 pair, gripper_open label, 상대경로 지원)
- [x] `model/architecture.py` — RobotArmModel (CLIP frozen + Action/State MLP)
- [x] `model/train.py` — AdamW + CosineAnnealingLR, best checkpoint 저장, gripper BCEWithLogitsLoss 합산, `--group` 지원
- [x] `model/inference.py` — RobotInference
- [x] `run_inference.py` — 텍스트 명령 추론 루프

---

## 10. 남은 작업

### Unity 씬 설정
- [ ] 집을 오브젝트 Prefab 제작 (Rigidbody + Collider 포함)
- [ ] `PickPlaceDemo` Inspector에 Prefab 등록 (displayName, materials 설정)
- [ ] `ImageServer` 컴포넌트 추가 + Capture Camera 연결

### 학습 파이프라인 실행

**1. Python 패키지 설치**
```bash
pip install -r requirements.txt
pip install git+https://github.com/openai/CLIP.git
```

**2. 데이터 수집**
```bash
python collect_data.py                  # default 그룹으로 수집
python collect_data.py --group my_group # 특정 그룹으로 수집
# Unity Play 모드 실행 → PickPlaceDemo 자동 진행
```

**3. 에피소드 관리**
```bash
python manage_episodes.py                                       # 전체 그룹 목록
python manage_episodes.py --group default                       # 그룹 내 에피소드 목록
python manage_episodes.py --group default --delete 0 3 5        # 특정 에피소드 삭제
python manage_episodes.py --group default --delete 2-7          # 범위 삭제
python manage_episodes.py --group default --delete-all          # 그룹 내 전체 삭제
python manage_episodes.py --group default --delete-group        # 그룹 폴더 자체 삭제
python manage_episodes.py --delete-all                          # 모든 그룹 삭제
python manage_episodes.py --group src --move 0 3 5 --to dst     # 특정 에피소드 이동
python manage_episodes.py --group src --move 2-7 --to dst       # 범위 이동
python manage_episodes.py --group src --move-all --to dst       # 그룹 전체 이동
python manage_episodes.py --migrate                             # 기존 episode_* → default 그룹으로 migration
```

**4. 모델 학습**
```bash
python -m model.train                               # default 그룹으로 학습
python -m model.train --group my_group              # 특정 그룹으로 학습
python -m model.train --resume                      # best_model.pt에서 이어서 학습
python -m model.train --resume path/to/ckpt.pt      # 특정 체크포인트에서 이어서 학습
# 체크포인트: model/checkpoints/best_model.pt (val loss 최저)
#             model/checkpoints/epoch_XXXX.pt  (10 epoch마다)
```

**5. 추론 테스트**
```bash
python run_inference.py
# > pick up the red cube
```

---

## 11. 알려진 이슈 / 개선 필요 사항

### 학습 코드

| # | 파일 | 이슈 | 우선순위 |
|---|------|------|---------|
| 1 | `model/train.py`, `model/dataset.py` | **gripper_head 학습 안 됨** — `train.py`에서 `gripper_logit` 완전히 무시. `dataset.py`에도 gripper label 미포함 (states.json의 `gripper_angle` 사용 안 함). gripper 제어 필요하면 dataset에 label 추가 + BCELoss 합산 필요 | 높음 |
| 2 | `model/dataset.py` | **image_path 절대경로** — `states.json`에 절대경로로 저장되면 환경 이전 시 전부 깨짐. 저장 시 에피소드 디렉터리 기준 상대경로로 저장하거나 로드 시 보정 필요 | 높음 |
| 3 | `config.py` | **`TEXT_EMBED_DIM = 384` 잔여 코드** — `all-MiniLM-L6-v2` 시절 설정. 현재 CLIP 사용 기준으로 512가 맞고 실제로도 어디서도 안 씀. 혼란 유발 | 낮음 |
| 4 | `model/dataset.py`, `model/architecture.py` | **state 입력 정규화 없음** — joint_angles(degree, ±180 범위)와 ee_position(meter, ±0.3 수준)을 그대로 concat. 스케일 차이로 state encoder 학습 불안정 가능. 표준화 권장 | 중간 |
| 5 | `model/dataset.py` | **"return" phase 포함** — 물체 놓고 home으로 돌아가는 구간은 task 완료 후라 유의미한 학습 데이터가 아님. phase 필터링 고려 | 낮음 |

### Unity 씬

| # | 파일 | 이슈 |
|---|------|------|
| 1 | `PickPlaceDemo.cs` | `bottom`/`top`/`pivot` child 못 찾을 때 warning 없이 root로 fallback — 문제 감지 어려움 |
| 2 | `PickPlaceDemo.cs` | `spawnSettleTime` 이내에 Rigidbody settle 안 되면 위치 틀어진 상태로 episode 진행 가능 |
| 3 | 씬 설정 | `SimCore` 컴포넌트를 씬에 추가해야 함 (빈 GameObject에 부착). 없으면 "SimCore not found" 오류 + Manipulator/SBCController 미작동. Script Execution Order: SimCore=-50, Actuator=-20 |

### IK / 제어

| # | 파일 | 이슈 |
|---|------|------|
| 1 | `ik_solver.py` | **Pitch constraint 미구현** — 물체를 바닥에 놓을 때 end-effector 방향(pitch) 보존 시도했으나 실패. 원인: (1) FK rotation matrix에서 Rz 추출 시 `arctan2(R[1,0], R[0,0])`는 Joint[4]=Rx가 있을 때 잘못된 값 반환, (2) warm-start로 constrained 해의 비정상 configuration이 다음 unconstrained IK 초기값으로 오염됨. 미해결 |
| 2 | `IKReceiver.cs`, `ik_solver.py` | **IK FK endpoint = endPoint 기준** — IK 솔버의 FK 끝점이 Unity의 `endPoint` Transform에 대응됨. `gripperPivot`을 IK 기준점으로 전환 시도했으나 동작 불안정으로 revert. 전환하려면 `ik_solver.py`의 `l5` 링크 길이를 Joint[4]→gripperPivot 실측값으로 수정 필요 |

---

## 12. 주요 설계 결정

- **CLIP 학습 레이블은 영어 고정** (`"pick up the {color} {object}"`)
  → 한국어 추론 시 multilingual 인코더 교체 또는 영어로 입력 필요
- **물체 스폰 Y** = `workspaceY.x + tableY + 0.02f` (물리 settle 여유 포함)
- **Exclusion Zone은 XZ만 체크** (스폰 Y가 항상 tableY로 고정이므로)
- **IK 솔버 독립 복사**: `VISTA-Manip/manipulator-sim`은 참조 전용 — 수정 금지, 필요한 코드는 복사해서 사용
- **단일 명령 데이터 수집**: `python collect_data.py` 하나로 IK 서버 + 데이터 수집 서버 동시 기동
- **Behavior Cloning 학습 타겟**: 절대 위치가 아닌 Δpos를 예측 → 임의의 시작 위치에서도 일반화 가능
- **아키텍처 레이어 분리**: SimCore(tick 배분) → Manipulator(하드웨어) → SBCController+IKReceiver(제어). Manipulator는 ControlMode 없이 순수 하드웨어 레이어. 외부 컴포넌트(SBCController, ManualManipulatorController 등)가 제어 방식 결정
- **IK는 매 프레임 호출이 아님**: waypoint 도달 시 한 번 호출. IKReceiver는 target 위치 변화 감지 시에만 전송 (`positionThreshold`, `sendInterval` 조건)
