# Lesson 01 Handoff

기준 날짜: `2026-04-18`

이 문서는 지금까지 `cudatraining`에서 한 작업과, 다음 레슨에서 바로 이어서 해야 할 일을 클로드나 미래의 나에게 넘기기 위한 핸드오프 문서다.

## 1. 이번 세션에서 한 일

### 저장소 초기 세팅

- CUDA 학습용 저장소 골격 생성
- 첫 커널을 `vector add`로 결정
- 아래 파일 작성

파일:

- [README.md](/Users/xavier/dev/cudatraining/README.md:1)
- [Makefile](/Users/xavier/dev/cudatraining/Makefile:1)
- [src/vector_add.cu](/Users/xavier/dev/cudatraining/src/vector_add.cu:1)
- [docs/phase1_plan.md](/Users/xavier/dev/cudatraining/docs/phase1_plan.md:1)
- [scripts/check_cuda_env.sh](/Users/xavier/dev/cudatraining/scripts/check_cuda_env.sh:1)
- [scripts/gcp_create_t4_spot_vm.sh](/Users/xavier/dev/cudatraining/scripts/gcp_create_t4_spot_vm.sh:1)
- [scripts/gcp_copy_repo_to_vm.sh](/Users/xavier/dev/cudatraining/scripts/gcp_copy_repo_to_vm.sh:1)
- [scripts/gcp_visible_ssh_run.sh](/Users/xavier/dev/cudatraining/scripts/gcp_visible_ssh_run.sh:1)
- [scripts/run_visible_terminal_benchmark.sh](/Users/xavier/dev/cudatraining/scripts/run_visible_terminal_benchmark.sh:1)

### 클라우드 전략 정리

- 로컬 머신이 Apple Silicon 맥이라 CUDA 실행 불가 확인
- `GCP 200달러 + AWS 200달러` 크레딧 기반으로 초기 6~8주 운영 전략 수립
- 초반은 GPU 구매 없이 `T4/L4` 위주 실험으로 가기로 정리

세부 계획:

- [docs/phase1_plan.md](/Users/xavier/dev/cudatraining/docs/phase1_plan.md:1)

### 실제 GPU 실행

- GCP 프로젝트 `nemo-488500` 사용
- `us-east1-d`에 `Tesla T4` Spot VM 생성
- DLVM 이미지로 띄운 뒤 원격에서 `build-essential` 설치
- `vector_add` 컴파일 및 실행
- 결과를 로컬로 회수
- 실행 후 VM 삭제 완료

삭제 확인:

- `gcloud compute instances describe cuda-t4-dev-131020 ...` 결과 `not found`

## 2. 이번 세션의 산출물

### 실행 로그

- [results/gcp-run-20260418-132605.log](/Users/xavier/dev/cudatraining/results/gcp-run-20260418-132605.log:1)

### 원격 결과 회수본

- [results/remote/results/check_cuda_env.txt](/Users/xavier/dev/cudatraining/results/remote/results/check_cuda_env.txt:1)
- [results/remote/results/vector_add_t4.txt](/Users/xavier/dev/cudatraining/results/remote/results/vector_add_t4.txt:1)

## 3. 이번 벤치마크 결과

GPU:

- `Tesla T4`

메인 실행 (`block_size=256`):

- `n = 67,108,864`
- `best kernel time = 3.488 ms`
- `avg kernel time = 3.508 ms`
- `effective bandwidth = 230.909 GB/s`
- `theoretical bandwidth = 320.064 GB/s`
- `efficiency = 72.145%`
- `H2D copy = 44.276 ms`
- `D2H copy = 20.439 ms`
- `max abs error = 0.0`

block size sweep:

- `128`: `219.228 GB/s`, 효율 `68.495%`
- `256`: `230.909 GB/s`, 효율 `72.145%`
- `512`: `236.592 GB/s`, 효율 `73.920%`

원본:

- [vector_add_t4.txt](/Users/xavier/dev/cudatraining/results/remote/results/vector_add_t4.txt:1)

## 4. 이번 세션에서 확인한 핵심 해석

- 첫 커널이 정상 동작했고 정확도 문제는 없었다.
- 이 커널은 전형적인 `memory-bound` 패턴을 보였다.
- `block size`를 바꿔도 성능이 극적으로 변하지 않았다.
- 오히려 end-to-end 관점에서는 `kernel time`보다 `H2D/D2H 복사 시간`이 훨씬 컸다.
- 첫 교훈은 `연산보다 데이터 이동이 더 비싸다`는 것이다.

직관적으로 보면:

- 커널 계산 자체는 `3.4 ms` 수준
- 복사 시간은 합쳐서 `60 ms+`
- 즉, 진짜 병목은 아직 커널 내부 연산보다 `PCIe 이동`이다

## 5. 코드 핵심 구조

대상 파일:

- [src/vector_add.cu](/Users/xavier/dev/cudatraining/src/vector_add.cu:1)

핵심 포인트:

- [27-33](/Users/xavier/dev/cudatraining/src/vector_add.cu:27): 실험 파라미터를 바꾸기 위한 `Config`
- [35-42](/Users/xavier/dev/cudatraining/src/vector_add.cu:35): 이론 메모리 대역폭 계산
- [44-82](/Users/xavier/dev/cudatraining/src/vector_add.cu:44): CLI 파라미터 파싱
- [84-94](/Users/xavier/dev/cudatraining/src/vector_add.cu:84): `grid-stride loop` 기반 `vector_add_kernel`
- [121-133](/Users/xavier/dev/cudatraining/src/vector_add.cu:121): pinned host memory + device memory 할당
- [157-164](/Users/xavier/dev/cudatraining/src/vector_add.cu:157): H2D timing
- [166-184](/Users/xavier/dev/cudatraining/src/vector_add.cu:166): warmup 후 kernel timing 반복 측정
- [187-193](/Users/xavier/dev/cudatraining/src/vector_add.cu:187): D2H timing
- [206-213](/Users/xavier/dev/cudatraining/src/vector_add.cu:206): effective bandwidth 계산
- [215-241](/Users/xavier/dev/cudatraining/src/vector_add.cu:215): text / CSV 출력

## 6. 다음 레슨 목표

다음 레슨의 주제는 `복사 비용과 메모리 종류`다.

이번 레슨에서 이미:

- pinned memory를 사용했다
- kernel time과 copy time을 분리해서 봤다
- copy가 훨씬 비싸다는 사실을 확인했다

따라서 다음 레슨은 아래 순서가 좋다.

### Next Lesson A: pageable vs pinned 비교

목표:

- 왜 pinned memory를 썼는지 몸으로 이해
- H2D / D2H 차이가 실제로 얼마나 나는지 측정

할 일:

- 현재 코드에 `--pageable` 옵션 추가
- `cudaMallocHost` 대신 `std::vector<float>` 또는 `new` 기반 host buffer 사용 경로 추가
- 같은 `n`에서 `pageable vs pinned` 복사 시간 비교
- 결과를 CSV로 남기기

예상 학습 포인트:

- pinned memory가 DMA에 왜 유리한지
- end-to-end 성능에서 host memory 선택이 왜 중요한지

### Next Lesson B: n sweep

목표:

- 작은 문제와 큰 문제에서 병목이 어떻게 달라지는지 보기

추천 sweep:

- `2^20`
- `2^24`
- `2^26`

볼 것:

- 작은 `n`에서는 launch overhead 비중이 커지는지
- 큰 `n`에서는 bandwidth 효율이 더 안정되는지

### Next Lesson C: 코드 설명을 말로 해보기

목표:

- 아래 질문을 말로 답할 수 있는지 점검

질문:

- 왜 `grid-stride loop`를 쓰는가?
- 왜 `bytes_moved = n * 4 * 3`인가?
- 왜 `vector add`는 compute-bound가 아니라 memory-bound인가?
- 왜 block size를 크게 바꿔도 성능 변화가 제한적인가?
- 왜 실제 시스템에서는 커널보다 복사 최적화나 fusion이 더 중요할 수 있는가?

## 7. 다음 레슨에서 바로 실행할 체크리스트

1. `src/vector_add.cu`에 pageable host memory 경로 추가
2. `--pageable` 플래그 설계
3. `pinned`와 `pageable` 각각에 대해 `n=2^20, 2^24, 2^26` 측정
4. 결과를 `results/` 아래 CSV로 저장
5. 결과 해석 메모를 짧게 작성

## 8. 클로드에게 바로 넘길 프롬프트 초안

아래처럼 넘기면 된다.

```text
/Users/xavier/dev/cudatraining/docs/lesson_01_handoff_2026-04-18.md 를 읽고 이어서 작업해줘.

현재 상태:
- vector_add CUDA 커널은 T4에서 실행 완료
- 결과는 /Users/xavier/dev/cudatraining/results/remote/results/vector_add_t4.txt 에 있음
- 다음 레슨은 pageable vs pinned host memory 비교를 추가하는 것

원하는 것:
- src/vector_add.cu 에 --pageable 옵션을 추가하고
- pinned vs pageable 복사 시간을 비교하는 벤치마크를 구현하고
- 결과를 CSV로 남기고
- 마지막에 해석까지 정리해줘
```

## 9. 한 줄 요약

이번 레슨은 성공이다. `첫 CUDA 커널을 실제 GPU에서 돌리고`, `memory-bound 감각`을 얻었고, `다음엔 복사 비용을 더 깊게 파는 단계`로 넘어가면 된다.
