# ID-SSM: Interaction-aware Deep State-Space Model for RUL Prediction

본 프로젝트는 항공기 엔진 데이터셋(C-MAPSS)을 활용하여 **잔여 수명(RUL: Remaining Useful Life)**을 예측하는 딥러닝 기반 상태 공간 모델 연구입니다.

기존 연구인 **MSDFM(Multi-Sensor Data Fusion Model)**의 한계점(Greedy 방식의 센서 선택, 고정된 비선형 함수)을 극복하기 위해, **GAT(Graph Attention Network)**와 **MNN(Monotonic Neural Network)**을 결합한 새로운 모델인 **ID-SSM**을 제안하고 구현하였습니다.

## 🚀 ID-SSM (제안 모델) 상세 소개

**ID-SSM**은 기존 MSDFM의 물리적 제약 조건을 유지하면서, 딥러닝의 표현력을 결합하여 예측 정확도를 높인 모델입니다. 크게 두 가지 핵심 모듈로 구성됩니다.

### 1. Interaction-Aware Sensor Fusion (GAT 기반)

기존 모델이 최적의 센서 조합을 찾기 위해 Greedy 방식(PSGS)을 사용했던 것과 달리, ID-SSM은 센서 간의 상호작용을 학습합니다.

* **Graph Attention Network (GAT):** 각 센서를 그래프의 노드로 간주하고, 센서 간의 관계 중요도(Attention Score)를 계산합니다.
* **특징:** RUL 예측에 중요한 센서 정보에는 높은 가중치를, 노이즈가 많은 센서에는 낮은 가중치를 자동으로 부여하여 **융합 특징 벡터(z_t)**를 추출합니다.
* **장점:** 2^P개의 조합을 탐색할 필요 없이, End-to-End 학습을 통해 최적의 센서 융합이 가능합니다.

### 2. Monotonic Neural Network (MNN 기반)

기존 모델이 상태(x)와 관측값(z) 사이의 관계를 고정된 다항식(z = x^c)으로 가정한 반면, ID-SSM은 이를 유연한 신경망으로 대체합니다. 단, 기계 부품의 **'열화(Degradation)는 비가역적'**이라는 물리적 특성을 반영해야 합니다.

* **단조성 제약 (Monotonicity Constraints):** 신경망의 모든 가중치(Weight)에 `exp` 함수를 적용하여 양수로 강제합니다.
* **구조:** z_k = \mathcal{H}(x_k, \theta_{mnn})
* **효과:** 상태(x)가 악화됨에 따라 센서 융합 값(z)이 비선형적이지만 일관된 방향(단조 증가/감소)으로 변화하도록 보장합니다. 이는 딥러닝 모델이 물리적 모순(부품이 저절로 고쳐지는 현상 등)을 학습하는 것을 방지합니다.

### 3. 하이브리드 추론 (Hybrid Inference)

학습된 딥러닝 모델을 확률적 상태 추정 프레임워크인 **파티클 필터(Particle Filter)**와 결합합니다.

* **State Transition:** 파티클 필터를 통해 시스템의 열화 상태(x_k) 전이를 확률적으로 예측합니다.
* **Measurement Update (Inverse Mapping):** 관측된 센서 데이터(z_{new})가 들어오면, MNN의 역함수(Inverse Mapping) 최적화를 통해 현재의 잠재 상태(x_{new})를 역추적하여 파티클의 가중치를 업데이트합니다.



---

## 📊 모델 비교 (MSDFM vs ID-SSM)

| 구분 | 기존 모델 (MSDFM) | **제안 모델 (ID-SSM)** |
| --- | --- | --- |
| **센서 융합** | PSGS (Greedy Selection) | **GAT (Graph Attention)** |
| **특징** | 조합 탐색 비용 발생 | 센서 간 시너지 자동 학습 |
| **관측 모델** | 고정 함수 (x^c) | **Monotonic NN** |
| **특징** | 단순 가정, 유연성 부족 | 물리적 제약(단조성)이 포함된 비선형 학습 |
| **학습 방식** | 통계적 파라미터 추정 | **End-to-End Deep Learning** |

