# Phase 1 Plan

기준 날짜는 `2026-04-18`이다. 아래 계획은 현재 확인 가능한 공식 가격 문서를 바탕으로 짰고, 실제 과금은 리전과 스팟 가용성에 따라 달라질 수 있다.

## 1. 크레딧 400달러 운영 전략

### 결론

지금은 `GPU를 사지 않는 쪽`이 맞다. 첫 6~8주는 `클라우드 크레딧으로 학습 속도`를 최대화하고, 그 뒤에도 확신이 남을 때만 중고 3090/4090 구매를 다시 검토하면 된다.

### 왜 이렇게 가나

- 현재 로컬 머신은 Apple Silicon 맥이라 CUDA 실행이 불가능하다
- 이미 `GCP 200달러 + AWS 200달러` 크레딧이 있어 초기 실험 비용을 사실상 선불로 확보한 상태다
- Month 0~2에는 H100급이 필요 없다. `T4/L4/A10G`면 커널 감각과 vLLM 진입에 충분하다

## 2. 클라우드 역할 분담

### GCP: 주력 실험 환경

권장 역할:

- CUDA 기초 커널
- Triton 튜토리얼
- 짧은 반복 벤치마크

공식 가격 자료 기준으로, GCP GPU 가격 페이지에는 다음 예시가 보인다.

- NVIDIA T4: 온디맨드 `0.35 USD/hour`, Spot `0.16 USD/hour`
- NVIDIA L4: 온디맨드 `0.56004024 USD/hour`, Spot `0.252018108 USD/hour`

주의:

- 위 숫자는 GPU 가격이다. VM의 vCPU/메모리/디스크 비용은 별도다
- 그래도 초기 학습 단계에서는 전체 비용이 여전히 충분히 낮다

실전 운영:

- 기본값은 `T4 또는 L4 Spot`
- Spot이 안 잡히는 날만 온디맨드로 잠깐 전환
- 장시간 방치 금지, 세션 끝날 때 즉시 종료

### AWS: 비교 검증 + 백업 환경

권장 역할:

- GCP Spot이 안 잡힐 때 대체
- 서로 다른 GPU 세대에서 같은 코드를 재실행
- 나중에 vLLM/serving 실험 시 EC2 감각 익히기

AWS Pricing API 기준 `us-east-1` Linux on-demand:

- `g4dn.xlarge` (T4): `0.526 USD/hour`
- `g6.xlarge` (L4): `0.8048 USD/hour`
- `g5.xlarge` (A10G): `1.006 USD/hour`
- `g6e.xlarge` (L40S): `1.861 USD/hour`

AWS Pricing API 기준 `ap-northeast-2` Linux on-demand:

- `g4dn.xlarge` (T4): `0.647 USD/hour`
- `g6.xlarge` (L4): `0.9896 USD/hour`
- `g5.xlarge` (A10G): `1.237 USD/hour`
- `g6e.xlarge` (L40S): `2.288 USD/hour`

해석:

- 값이 싸고 리전 제약이 덜하면 `us-east-1`이 유리하다
- 서울 리전은 편하지만 학습 초반엔 비용 우위가 크지 않다

## 3. 예산 배분 제안

### GCP 200달러

- `120달러`: T4/L4 Spot 반복 실험
- `40달러`: Spot이 막힐 때 온디맨드 fallback
- `40달러`: 후반 1~2회 A100/L4 검증, 스토리지, 실수 완충

### AWS 200달러

- `120달러`: `g4dn.xlarge` 또는 `g6.xlarge`에서 CUDA/vLLM 재현
- `50달러`: `g5.xlarge` 단기 사용
- `30달러`: EBS, AMI 실수, 리전 실험 완충

### 이 배분이 의미하는 것

보수적으로 잡아도:

- GCP Spot 위주면 수백 시간의 T4/L4 실험이 가능하다
- AWS에서도 수십~100시간 단위로 T4/L4/A10G 테스트가 가능하다

즉, 첫 2개월은 `장비 부족`이 아니라 `측정 rigor 부족`이 병목이다.

## 4. 첫 2주 상세 계획

### Week 1

Day 1:

- GCP 또는 AWS에 첫 GPU VM 1대 만든다
- `nvidia-smi`, `nvcc --version`, 샘플 빌드 확인
- 이 저장소의 `vector_add`를 빌드/실행

Day 2:

- `--n`을 `2^20`, `2^24`, `2^26`으로 바꿔 측정
- `--block-size`를 `128`, `256`, `512`로 바꿔 측정
- 결과를 표로 적는다

Day 3:

- effective bandwidth와 theoretical bandwidth를 비교
- 왜 차이가 나는지 한 단락으로 정리
- 작은 README 메모 또는 블로그 초안 작성

Day 4:

- pageable host memory와 pinned host memory 차이를 조사할 준비
- 현재 코드 읽고 launch/grid 계산을 손으로 설명해본다

Day 5:

- 같은 코드를 다른 GPU 한 번 더 실행
- T4 vs L4 또는 T4 vs A10G 한 번만 비교해도 충분하다

Weekend:

- PMPP 1~3장 읽기
- CS149 첫 강의 보기
- 벤치마크 결과를 짧게 정리

### Week 2

- PMPP 4~6장
- `reduction` 착수 전, `vector add`를 직접 설명할 수 있는지 점검
- 가능하면 pinned memory 실험 또는 memcpy timing 분리 추가
- 결과를 짧은 글 1편으로 묶기

## 5. 지금 저장소에서의 성공 기준

첫 단계 성공 조건은 아래다.

- `vector add`를 클라우드 GPU에서 직접 돌렸다
- effective bandwidth 계산이 무슨 뜻인지 설명할 수 있다
- block size를 바꿨을 때 숫자가 왜 그렇게 나오는지 가설을 세울 수 있다
- 결과를 외부에 보여줄 수 있는 형태로 남겼다

## 6. 공식 자료

- GCP GPU pricing: [cloud.google.com/compute/gpus-pricing](https://cloud.google.com/compute/gpus-pricing)
- GCP VM pricing overview: [cloud.google.com/products/compute/pricing](https://cloud.google.com/products/compute/pricing)
- AWS EC2 on-demand pricing: [aws.amazon.com/ec2/pricing/on-demand](https://aws.amazon.com/ec2/pricing/on-demand/)
- AWS EC2 pricing API regional offer index example:
  - [us-east-1](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/us-east-1/index.json)
  - [ap-northeast-2](https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonEC2/current/ap-northeast-2/index.json)

