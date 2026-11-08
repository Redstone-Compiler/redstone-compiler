# Rust placer의 결과 복원과 단일 worker 최적화

이 변경은 실험 도구가 아니라 실제 `ExactLocalPlacer::place` 경로에 적용했다.
CNF나 허용하는 배치 집합, solver 설정은 바꾸지 않았다.

## 이전 분기 재사용이 느렸던 이유

CaDiCaL은 기존 blocking-clause 전체 열거에서도 내부 탐색 정보를 재사용한다.
그 위에 별도의 물리 배치 분할을 넣으면 많은 접두 분기를 다시 SAT에 묻게 된다.
Rust에서는 모델의 리터럴을 읽는 비용도 이미 작아서 Python의 처리 비용 감소가
같이 따라오지 않는다. 실제 inv5 측정은 기존 약 13.3ms, 분기 재사용 25.5ms였다.
서로 다른 문제의 경계를 재사용하는 캐시와 solver 내부 재사용은 구분해야 한다.

## 적용한 변경

`verify::decode_kinds`:

- 기존에는 각 칸에서 Solid/Dust/Torch/Repeater/Switch 후보를 새 Vec에 넣었다.
  이제 iterator로 검사해서 칸마다 후보 배열을 할당하지 않는다.
- Kind는 exactly-one이므로 Air가 참이면 그 칸은 즉시 완료한다.
- Air가 아니면 첫 참 후보에서 끝낸다. 이전처럼 나머지 후보까지 읽지 않는다.
- 결과와 차단 절에 쓰는 Kind 리터럴 벡터는 필요한 용량을 처음부터 확보한다.

`ExactLocalPlacer::place`:

- worker가 한 개이면 호출한 스레드에서 바로 실행한다. 작은 construction 및
  compaction 시도마다 새 스레드를 만들고 join하는 비용을 줄인다.
- 두 개 이상이면 기존 portfolio 실행을 유지한다. timeout, restart, core-guided
  경로 및 결과 처리는 같은 `run_worker`를 사용한다.
- 비교를 위해 private `place_with_worker_dispatch`에서 이전 dispatch도 실행할
  수 있게 했다. 공개 API는 항상 새 경로를 쓴다.

## 측정

전부 release 빌드다. decoder는 실제 production encoding에서 얻은 서로 다른
90개 모델에 대해 기존/새 구현의 **블록 종류와 차단 리터럴 전체가 같은지** 검사했다.
각 모델에서 순서를 바꾸어 2,000회씩 실행한 평균 시간을 모으고 중앙값을 비교했다.

| Kind 복원 | 기존 | 새 구현 | 속도 비율 |
|---|---:|---:|---:|
| 6칸 inverter | 0.839µs | 0.103µs | 8.13배 |
| 8칸 NOR | 1.297µs | 0.163µs | 7.94배 |
| 30칸 NOR | 7.098µs | 0.436µs | 16.28배 |

이는 **Kind 복원 부분만의 개선**이며 전체 SAT 풀이가 8~16배 빨라진다는 의미가 아니다.
일반적인 첫 해 탐색에서는 전체 시간의 작은 부분이다. 후보를 많이 복원·거부하는
경로에서 누적 이득을 기대할 수 있지만 그 워크로드의 전체 향상률은 아직 측정하지 않았다.

worker dispatch는 각 사례 30쌍을 순서를 바꾸어 측정했다. 제약 생성부터 solver,
시뮬레이터 검증, 결과 생성까지 포함하며 쌍마다 최종 RCELL 문서가 같음을 검사했다.
두 경로 모두 새 decoder를 사용하므로 dispatch만의 차이를 비교한다.

| 실제 전체 배치 | 새 스레드 | 직접 실행 | 판단 |
|---|---:|---:|---|
| inverter | 1.909ms | 1.866ms | 약 2.3% 감소 |
| NOR | 6.641ms | 6.610ms | 차이가 작음 |
| 30칸 NOR | 42.032ms | 42.232ms | 유의미한 개선 확인 못 함 |

inverter의 쌍별 절약 시간 중앙값은 47µs였다. NOR와 큰 NOR는 측정 변동에 비해
차이가 작으므로 전체 성능 향상을 주장하지 않는다. 기존 순차 baseline과 새 실행의
시간 차이를 전부 최적화 효과로 계산하지 않고 위 교차 측정 결과를 사용한다.

## 제외한 시도와 검증

CNF 마지막 단위 전파를 occurrence queue로 바꾸는 방법과, 열린 리터럴 두 개에서
검사를 종료하는 방법도 시험했다. 대표 production 경로에서 전체 성능 개선이
확인되지 않아 둘 다 제거했다. `crates/rsdsl/src/formula.rs`는 원래 상태다.

inverter, NOR, 큰 NOR, XOR의 변경 전후 CNF가 **바이트 단위로 동일**하다.
실제 인버터·NOR·XOR·AND/OR 배치 및 시뮬레이터 검증 release 테스트 4개도 통과했다.
decoder A/B는 90개 모델, dispatch A/B는 총 180회의 실제 배치를 검사했다.

재현:

```sh
cargo test --release --lib benchmark_kind_decoder -- --ignored --nocapture
cargo test --release --lib benchmark_worker_dispatch -- --ignored --nocapture
cargo test --release --lib exact_placer_builds_ -- --test-threads=1
```

제약 생성 분리 측정과 CNF 스냅샷:

```sh
RUST_PIPELINE_LABEL=current cargo test --release --lib benchmark_production_pipeline -- --ignored --nocapture
```

원시 측정값과 CNF 해시:
[rust_production_optimization_results.json](../tools/pruning_lab/rust_production_optimization_results.json).
