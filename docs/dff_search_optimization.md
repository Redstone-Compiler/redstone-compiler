# DFF 최초 탐색: 보조 순위 정규화

2026-11-08. 같은 2×7×4 상자, 6게이트 모델, rank/stage=8, seed=11,
worker=4, 고정 블록·고정 입력 위치·초기 배치 없이 측정했다.
최소화 후처리는 껐다. 컴파일 시간은 탐색 시간에 포함하지 않는다.

| 모델 | 최초 검증된 해까지 | 발견 블록 수 (스위치 제외) |
|---|---:|---:|
| 기존 기록 | 163.53초 | 45 |
| 기존 모델 재측정 | 144.51초 | 45 |
| 보조 순위 정규화, 첫 측정 | 43.41초 | 42 |
| 보조 순위 정규화, 반복 측정 | 43.79초 | 42 |

같은 seed의 재측정 기준 약 3.3배, 과거 기록 기준 약 3.7배다.
SAT 시간은 seed·환경에 민감하므로 모든 회로·seed에 대한 보장은 아니다.
**seed 12와 13은 모두 120초 내 해를 찾지 못했다.** 해당 seed의 기준 모델과는
비교하지 않았으므로 개선·퇴보를 판단하지 않는다. 추가 seed 점검 중 짧은
회귀 테스트도 실행했으므로 엄밀한 속도 비교 자료로 사용하지 않는다.
전체 기록은 [측정 JSON](../tools/pruning_lab/dff_search_optimization/results.json)에 있다.

## 병목

기존 모델 제약 생성은 0.177초였다. 성공 worker는 SAT에 144.30초,
해독 약 20μs, 검증 약 694μs를 썼다. 다른 worker는 토치 소진 후보
2,001개를 거절했다. 이 worker는 SAT 41.43초, 해독 0.016초,
검증 0.560초 후 refinement 상한에 도달했다. worker들은 동시에
실행되므로 시간 합계를 벽시계 시간으로 해석하면 안 된다.

병목은 SAT 탐색이다. 변수 셔플, chronological backtracking 비활성화,
중복 wire 식 제거, 공간 대칭, 기존 보조 변수 대칭 규칙, pairwise
cardinality도 시험했지만 60초 내 검증된 해를 얻지 못했다.
stable/positive 통일은 90초 내 찾지 못했다. 이는 각 예산 내 성공 여부이며
완전한 성능 순위가 아니다. 채택하지 않은 후보를 개선 효과로 세지 않았다.

## 바꾼 식과 보존 근거

블록의 종류·위치·신호를 고정하지 않고 보조 숫자만 정규화한다.

~~~text
공기 · 스위치 · 토치 · 리피터 ⇒ Rank = 0
공기 · 스위치                  ⇒ Stage = 0
~~~

현재 모델에서 Rank를 받는 쪽으로 비교하는 관계는 가루·고체를 향한다.
리피터로 들어가는 관계는 Rank 대신 Stage를 비교하며, 토치의 받침 관계도
Stage만 비교한다. 따라서 위 종류의 Rank를 0으로 내려도 들어오는 Rank
제약을 위반하지 않는다. 나가는 비교는 원천 숫자가 작아져 더 쉬워진다.
공기·스위치에는 들어오는 Stage 비교가 없다. 이들의 Stage를 0으로 낮추어도
나가는 비교와 출력의 상한을 만족한다. 다른 값은 그대로 둔다.

즉 기존 해마다 동일한 물리 배치·신호를 유지하는 정규화된 해가 존재한다.
역방향은 원래 식에 제약을 더했으므로 자명하다. 물리 배치를 버리지 않고
불필요한 보조 변수 할당을 제거한다. 현재 순위·단계 규칙이 바뀌면 이
보존 근거도 재검토해야 한다.

기존 6게이트 토폴로지 제한은 그대로다. 모든 Boolean 등가 DFF 구조까지
허용하는 모델로 바뀐 것은 아니다.

## 반영과 재현

기본 Rust 모델의 canonical_source_ranks 매개변수는 기본 false다.
조합회로별 ON/OFF 결과는 [별도 다중 시드 측정](combinational_source_rank_benchmark.md)을 참고한다.
[DFF 빠른 모델](rsdsl/dff_global_fast.rsdsl)은 기본 true다.
[원본 모델](rsdsl/dff_global_six_gate.rsdsl)은 재현용으로 보존했다.
벤치마크 스크립트는 빠른 모델을 기본 사용한다.

EXACT_PROFILE=1은 일반 SAT worker의 탐색·해독·검증 시간과 실패 이유를
기록한다. 꺼져 있으면 단계별 타이머를 읽지 않는다.
SEQ_SOLVER_OPTIONS는 비교 실험용 CaDiCaL 옵션이다.
채택된 모델은 기본 solver 옵션을 그대로 사용한다.

~~~sh
python3 tools/pruning_lab/benchmark_dff_search.py --label fast --seeds 11 --seconds 180
python3 tools/pruning_lab/benchmark_dff_search.py --label control --seeds 11 --seconds 180 --params canonical_source_ranks=false
~~~

옵션을 끈 빠른 모델의 CNF 리터럴 배열은 원래 모델과 완전히 동일함을
회귀 테스트로 확인한다. 작은 inverter/NOR에서는 물리 배치 집합을
양쪽 모두 UNSAT까지 전부 열거하여 일치하는지 검사한다.

새 42블록 DFF는 추가 규칙이 없는 원래 모델과 전체 순차 검증, 저장 NBT의
2,048회 입력 변화 검사를 통과했다. 42는 최초 발견 블록 수이며 최소화
후의 블록 수가 아니다.

~~~sh
cargo test --release --lib canonical_source_ranks_preserve_small_layout_sets -- --nocapture
cargo test --release --lib dff_fast_model_disabled_matches_original_cnf
DFF_VERIFY_STEM=../target/dff-search-bench/fast/11 cargo test --release --lib verify_dff_search_artifact -- --ignored --nocapture
~~~


물리 배치 집합 비교는 inverter 두 크기(136개·1,206개), NOR(100개),
총 1,442개를 양쪽 모두 완전 열거했다. 리피터가 포함된 배치도 검사한다.
새 최초 발견 결과: [RCELL](../test/dff-global-fast-2x7x4.rcell),
[NBT](../test/dff-global-fast-2x7x4.nbt).


## 모델 내 위치 정리

매개변수는 상단의 대칭 제거 옵션 옆에, 규칙은 대칭 제거 절의 기존
순위·단계 정규화 규칙 바로 뒤에 배치했다. 제약의 의미와 기본값은 같다.
위 시간 측정과 모델 해시는 이 위치 정리 전의 기록이다. 규칙 순서에 따른
CNF 생성 순서가 탐색 시간에 영향을 줄 수 있어, 이동 후 시간은 재측정하지 않았다.


후속 [조합회로 비교](combinational_source_rank_benchmark.md)에서는 XOR의 개선을
확인했지만 NOR/AND는 느려졌다. 일반 모델의 기본값은 false로 유지한다.
