# cudatraining

CUDA/Triton/vLLM 학습을 `공개 artifact` 중심으로 밀어가기 위한 개인 훈련 저장소다. 첫 단계는 작은 커널을 그냥 따라 치는 게 아니라, 측정과 해석이 가능한 형태로 쌓는 것이다.

## 현재 범위

- 첫 커널: `vector add`
- 목표: GPU 메모리 대역폭 감각 만들기
- 산출물: CUDA 구현, 벤치마크 출력, 클라우드 실행 경로, 2주 계획

## 왜 vector add부터 시작하나

`vector add`는 단순하지만 GPU 관점에서는 중요한 첫 감각을 준다.

- 연산량보다 메모리 이동량이 지배적이다
- coalesced access, launch configuration, effective bandwidth 계산을 바로 연습할 수 있다
- 이후 `reduction`, `matmul`, `softmax`로 넘어갈 때 기준선 역할을 한다

## 빌드와 실행

이 저장소는 로컬 Apple Silicon 맥이 아니라 NVIDIA GPU가 있는 클라우드 VM에서 실행하는 것을 전제로 한다.

GCP에서 빠르게 시작하려면:

```bash
./scripts/gcp_create_t4_spot_vm.sh <PROJECT_ID> <ZONE> <VM_NAME>
./scripts/gcp_copy_repo_to_vm.sh <PROJECT_ID> <ZONE> <VM_NAME>
gcloud compute ssh <VM_NAME> --project <PROJECT_ID> --zone <ZONE>
```

```bash
make vector_add
./bin/vector_add --n 67108864 --block-size 256 --iterations 100
```

자주 쓰는 예시:

```bash
make run-vector-add ARGS="--n 16777216 --block-size 256 --iterations 50"
make run-vector-add ARGS="--n 67108864 --block-size 512 --iterations 100"
```

## 출력에서 볼 것

프로그램은 아래를 출력한다.

- GPU 이름과 대략적인 theoretical memory bandwidth
- kernel best/avg time
- effective bandwidth
- H2D / D2H copy time
- CPU reference와의 최대 오차

처음에는 숫자 자체보다 아래 질문이 중요하다.

- block size를 바꾸면 왜 거의 안 변하거나, 어느 지점부터만 변하는가
- effective bandwidth가 theoretical bandwidth 대비 몇 퍼센트인가
- kernel time과 H2D/D2H 복사 시간 중 어디가 더 큰가

## 크레딧 운영 원칙

자세한 계획은 [docs/phase1_plan.md](/Users/xavier/dev/cudatraining/docs/phase1_plan.md)에 정리했다. 핵심만 먼저 적으면:

- 초기 8주는 `GPU 구매` 없이 `GCP + AWS 크레딧`으로 간다
- GCP는 싼 실험 반복용, AWS는 비교 검증과 대체 수단으로 쓴다
- 첫 2주는 `vector add -> reduction`까지 가기보다 `vector add`를 여러 크기와 block size로 직접 재고 해석하는 데 시간을 쓴다
