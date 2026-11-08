# 정확한 전체 열거의 비용 줄이기

경계 요약을 직접 만드는 실험에서 이어진 Python/Z3 최적화 기록이다.
대상은 rsdsl의 실제 CNF에서 **물리 배치 전체를 열거하는 실험 도구**다.
Rust production placer의 첫 해 탐색이나 CaDiCaL이 빨라졌다는 결과는 아니다.

## 적용한 방법

1. **CNF를 Z3에 직접 읽힌다.** Python으로 모든 절의 AST를 만들지 않는다.
   Python에서 다룰 Boolean 객체도 실제 Kind 선택 리터럴에 대해서만 만든다.
   DIMACS의 변수 번호는 Z3 정수 심볼과 연결한다. 예를 들어 `Bool(1)`을 쓰며,
   출력 이름이 비슷한 문자열 `Bool('k!1')`과 혼동하면 안 된다.
2. **배치를 한 번에 읽는다.** 셀마다 선택된 Kind의 번호를 자릿값으로 묶은
   정수 식을 모델에서 한 번 평가하고, Python에서 각 자리로 나눈다. 이 식은
   solver에 새 조건으로 추가하지 않는다. 원래 Kind의 exactly-one 조건 때문에
   선택된 배치와 이 정수는 일대일로 대응한다.
3. **동일한 CNF를 QF_FD solver에 넣는다.** 일반 `Solver()` 대신
   `SolverFor('QF_FD')`를 사용한다. 물리 모델의 제약을 삭제하거나 바꾸지 않는다.

해를 하나 찾으면 그 물리 배치만 제외하는 blocking clause를 추가한다.
보조 신호·순서 변수의 값이 다르더라도 같은 배치를 중복 출력하지 않는다.
마지막 UNSAT까지 도달해야 전체 열거 완료로 인정한다.

## 비교한 다른 방법

- 증분 전파로 불가능한 Kind를 먼저 제외하고, 이미 강제된 선택에서는 SAT 검사를
  생략하는 `forward`를 구현했다. `lazy`는 SAT 검사를 완전한 배치까지 미룬다.
  경계 캐시가 있는 버전도 비교했다. 작은 사례에서는 Python 전파·복원 비용이
  크고, 검사를 늦추면 더 많은 불가능한 분기를 방문하여 느렸다.
- `partition`은 SAT가 준 배치 하나를 그 경로 전체에서 재사용한다. 각 셀에서
  그 선택을 처리하고 제외한 뒤 다른 가능한 선택을 찾으므로 분기 집합은 서로
  겹치지 않고 모든 해를 포함한다. 일부 사례에서는 빨랐지만 일관되지는 않았다.
- `Tactic('sat').solver()`도 시험했지만 이 반복 열거에서는 더 느렸다.
  이를 일반적으로 SAT가 SMT보다 느리다는 주장으로 해석하면 안 된다.
- 경계 잔여 조건 공유는 해를 보존했지만, 이번 비교에서 가장 빠른 방법이 아니었다.

따라서 이번 성과는 새로운 경계 표현으로 조합 수를 획기적으로 줄인 것이 아니다.
동일한 해 공간을 더 적은 준비·결과 처리 비용과 더 적합한 solver 설정으로 탐색한다.

## 검증과 측정

기존 5개 작은 회로에 입력 스위치와 출력 관측 위치가 고정된 8개 사례를 추가했다.
고정 핀 사례는 12~24칸이며, 블록 수 제한을 달리한 SAT 및 UNSAT 사례를 포함한다.
참조 해는 기존 방식의 직접 AllSAT로 독립적으로 열거했다. 새 탐색은 참조 목록을
읽어 가지치기하지 않으며, 끝난 뒤 배치 집합 전체를 대조한다.

비교 구성은 다음 다섯 가지다.

| 이름 | CNF 읽기 | 배치 읽기 | Solver |
|---|---|---|---|
| python_smt | Python AST 구성 | Kind별 개별 평가 | Solver |
| native_smt | 직접 읽기 | Kind별 개별 평가 | Solver |
| packed_smt | 직접 읽기 | 정수 식 한 번 평가 | Solver |
| packed_fd | 직접 읽기 | 정수 식 한 번 평가 | QF_FD |
| partition_fd | 직접 읽기 | 정수 식 및 배치 경로 재사용 | QF_FD |

13개 사례 × 5개 구성 × seed 5개를 순차 실행하고, seed마다 구성 실행 순서를
회전한다. 매번 새 solver를 만든다. 시간에는 파일 읽기, solver 구성, 결과 추출,
전체 탐색과 결과 목록 생성이 포함된다. 모든 구성에 공통인 rsdsl→CNF 생성과
사후 정답 대조는 제외한다. 이전 버전과 비교할 때도 변경된 준비 비용을 구분한다.

재현:

```sh
cargo build --release -p rsdsl --example boundary_export
/private/tmp/redstone-pruning-venv/bin/python tools/pruning_lab/boundary.py --seconds 45 --limit 10000
/private/tmp/redstone-pruning-venv/bin/python tools/pruning_lab/extended_boundary.py
/private/tmp/redstone-pruning-venv/bin/python tools/pruning_lab/benchmark_search.py --seeds 5
```

기존 경계 표현 및 증분 갱신 기록은
[exact_boundary_representation.md](exact_boundary_representation.md)에 있다.

## 결과

각 5회 중앙값, 단위는 초. 모든 325회가 완료되고 참조 해 집합과 정확히 일치했다.

| 사례 | 해 | Python+SMT | 직접 읽기+SMT | 묶어 읽기+SMT | 묶어 읽기+QF_FD | 경로 재사용+QF_FD |
|---|---:|---:|---:|---:|---:|---:|
| inv3 | 12 | 0.0709 | 0.0176 | 0.0171 | 0.0094 | 0.0114 |
| inv4 | 130 | 0.2359 | 0.1545 | 0.1408 | 0.0746 | 0.0930 |
| inv5 | 406 | 0.5630 | 0.4693 | 0.4322 | 0.2202 | 0.2620 |
| nor4 | 12 | 0.1679 | 0.0512 | 0.0497 | 0.0257 | 0.0304 |
| nor5 | 60 | 0.2573 | 0.1430 | 0.1352 | 0.0651 | 0.0835 |
| inv_fixed6 | 2 | 0.1090 | 0.0169 | 0.0182 | 0.0092 | 0.0106 |
| inv_fixed7 | 0 | 0.1521 | 0.0118 | 0.0135 | 0.0113 | 0.0111 |
| nor_fixed7 | 0 | 0.1435 | 0.0104 | 0.0124 | 0.0112 | 0.0104 |
| inv_fixed8 | 70 | 0.2196 | 0.1197 | 0.1049 | 0.0540 | 0.0661 |
| inv_long9 | 64 | 0.2817 | 0.1565 | 0.1422 | 0.0725 | 0.0909 |
| nor_fixed10 | 0 | 0.3403 | 0.0365 | 0.0402 | 0.0428 | 0.0426 |
| inv_long10 | 297 | 0.6957 | 0.5709 | 0.5001 | 0.2477 | 0.3055 |
| inv_wide8 | 528 | 1.9981 | 1.7204 | 1.5895 | 0.7202 | 1.0343 |

406개 해인 inv5: Python 기준선 0.563→0.220초, **2.56배**.
24칸 고정 핀 inv_wide8의 528개 해: 1.998→0.720초, **2.77배**.
준비 비용 개선을 이미 반영한 native_smt 기준으로도 각각 **2.13배, 2.39배**다.
즉 효과 전부가 Python CNF 구성 시간 제거에서 나온 것은 아니다.

해가 있는 10개 사례에서 packed_fd가 다른 비교 구성보다 중앙값이 낮았다.
UNSAT 3개 중에는 native_smt가 더 빠른 사례가 있으므로 모든 문제에서 항상
최적이라고 주장하지 않는다. UNSAT 사례의 큰 Python 대비 배율은 주로 준비 비용
차이에서 나오므로 대표 가속 수치로 쓰지 않는다.

실험 runner의 기본값을 `--modes allsat_packed --loader native --engine fd`로 변경했다.
과거 경계 캐시 방식은 비교용 옵션으로 유지한다. Rust production 경로는 변경하지 않았다.

현재 가장 유리한 방법으로 실행:

```sh
/private/tmp/redstone-pruning-venv/bin/python tools/pruning_lab/online_boundary.py \
  --instances inv5,inv_fixed8,inv_long10,inv_wide8 --seeds 3
```

원시 기록:

- [최종 325회 비교](../tools/pruning_lab/search_ablation_results.json)
- [고정 핀 참조 해 생성](../tools/pruning_lab/extended_reference_results.json)
- [전파·캐시·SAT tactic 탐색 기록](../tools/pruning_lab/search_exploration_results.json)

정답 fixture와 생성 스크립트를 함께 보존했다. 새 fixture를 추가할 때는 출력 위치와
입력 사이트를 고정했는지, 최종 UNSAT까지 열거했는지 구분해야 한다. 고정 핀만으로
모든 배치에서 사용하지 않는 부품까지 제거되는 것은 아니다.

## Rust / CaDiCaL 재검증

Python의 개선을 Rust 본체 성능으로 오해하지 않도록, production이 실제 사용하는
`SatSolver` 래퍼를 호출하는 opt-in release 벤치마크를 추가했다.
`src/transform/place_and_route/local_placer/exact/search_benchmark.rs`가 그 구현이다.

Rust는 이미 CNF의 정수 리터럴을 CaDiCaL에 직접 전달하고, 모델 값도 C API에서
읽는다. Python AST 생성과 Python↔Z3 모델 평가 비용이 그대로 존재하지 않는다.
또한 QF_FD는 Z3 설정이므로 CaDiCaL에 적용할 수 없다.

같은 13개 CNF에서 다섯 방법을 seed 1–5로 실행한다.

- `flat_phase0`: production worker 0처럼 phase=0, stabilizeonly=0.
  물리 배치를 찾고 해당 배치만 blocking clause로 제외한다.
- `flat_phase1`, `flat_stable0`, `flat_stable1`: production portfolio의
  다른 worker들과 같은 phase/stable 옵션 조합이다.
- `partition_phase0`: 찾은 배치를 그 하위 경로에서 재사용하고, 완료한 Kind를
  제외한 형제 분기만 다시 SAT에 묻는다. 추가 조건은 assumptions로 전달하고
  재귀에서 복원한다. Python의 `partition`과 같은 완전한 분할 방식이다.

각 실행의 배치 집합을 Python 참조 해와 정확히 대조하고, 중복 출력도 검사한다.
해 개수만 맞는 것으로 성공을 판정하지 않는다. 30초 제한에 걸리면 실패시키며
부분 열거를 완료된 결과로 기록하지 않는다.

시간에는 CNF/Kind 파일 읽기, solver 생성과 CNF 주입, 전체 열거 및 결과 읽기가
포함된다. 참조 해 파일 읽기와 사후 집합 비교, 공통 rsdsl grounding은 제외한다.
Python과 같은 작업 범위이지만 언어뿐 아니라 solver도 Z3와 CaDiCaL로 다르므로
언어 자체의 성능 차이로 해석하면 안 된다. 반복마다 새 solver를 만들고, 설정
실행 순서는 seed마다 회전한다.

```sh
cargo test --release --lib benchmark_projected_search -- --ignored --nocapture
```

앞의 Python 재현 명령으로 CNF와 참조 fixture를 먼저 생성해야 한다.
`RUST_SEARCH_SEEDS`로 반복 수를 바꿀 수 있으며 기본값은 5다.

### Rust 측정 결과

각 5회 중앙값, 단위 ms. 325회 모두 참조 해 집합과 정확히 일치했다.

| 사례 | Rust 기본 | 분기 재사용 | phase=1 | stable/phase=0 | stable/phase=1 |
|---|---:|---:|---:|---:|---:|
| inv3 | 1.57 | 1.93 | 1.59 | 1.55 | 1.50 |
| inv4 | 6.25 | 11.33 | 6.33 | 8.22 | 8.09 |
| inv5 | 13.34 | 25.52 | 15.44 | 20.03 | 21.24 |
| nor4 | 6.15 | 6.03 | 5.59 | 4.72 | 5.60 |
| nor5 | 14.65 | 17.70 | 11.14 | 10.96 | 12.57 |
| inv_fixed6 | 0.82 | 0.97 | 0.90 | 0.89 | 0.87 |
| inv_fixed7 | 2.15 | 2.15 | 1.87 | 2.02 | 1.66 |
| nor_fixed7 | 1.01 | 1.05 | 0.81 | 0.89 | 0.85 |
| inv_fixed8 | 2.46 | 5.42 | 2.37 | 3.84 | 4.97 |
| inv_long9 | 5.99 | 13.86 | 5.70 | 8.87 | 9.14 |
| nor_fixed10 | 10.70 | 10.91 | 14.13 | 18.57 | 13.97 |
| inv_long10 | 11.19 | 37.14 | 12.25 | 21.37 | 26.40 |
| inv_wide8 | 40.51 | 186.39 | 39.49 | 58.61 | 64.79 |

Rust 기본 경로는 이미 매우 빠르며, 새 분기 재사용은 inv5와 inv_wide8에서
오히려 느려졌다. phase/stable 옵션의 유불리도 사례마다 달라 작은 전체 열거
벤치마크만으로 production portfolio를 한 가지 설정으로 바꾸지 않았다.

**추가한 것은 production solver를 재사용하는 Rust 벤치마크와 분기 재사용의
검증 구현이다. 기존 Rust placer 자체가 이번 변경으로 빨라졌다는 결과는 아니다.**
시뮬레이터 검증과 refinement, 목적함수 최적화까지 포함한 전체 배치 시간은 별도다.

원시 결과: [rust_search_results.json](../tools/pruning_lab/rust_search_results.json).
