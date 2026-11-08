# 조합회로의 전원 시작점 순위 정규화 비교

2026-11-08. canonical_source_ranks만 켜고 끈 paired cold search 실험이다.
순차회로 특화 규칙인지, 조합회로에서도 실제 시간을 줄이는지 확인한다.

## 조건

- 시드: 1, 2, 3, 11, 23. 각 시드에서 OFF/ON 모두 실행.
- 1차: 회로 6종 × 시드 5개 × 설정 2개 = 60회, worker 4개.
- SAT 제한 15초, rank/stage=24, 기본 토치 mirror 대칭 설정.
- 고정 블록·고정 핀·초기 배치 없음. 블록 수 최소화는 끔.
- 정상 전체 시뮬레이터 검증을 통과한 최초 배치까지 측정.
- 제약 생성 포함, Rust 컴파일 제외. 실행은 직렬, OFF/ON 순서는 교대.
- 작은 회로와 XOR를 역순으로 재측정하고, XOR는 worker 8개로도 확인.
- 측정 모델은 규칙을 대칭 제거 절로 옮긴 현재 버전이다. 앞선 DFF 43초
  기록의 파일 순서와 다르므로 DFF 시간과 직접 비교하지 않는다.

## 1차 결과

시간은 설정별 5개 시드의 중앙값이다. 성공 수를 반드시 함께 본다.
15초에서 중단된 실행의 실제 완료 시간이나 배치 불가능 여부는 알 수 없다.

| 회로 | 상자 | OFF | ON | 성공 OFF → ON |
|---|---|---:|---:|---:|
| inverter | 1×3×2 | 3.66ms | 3.67ms | 5/5 → 5/5 |
| NOR | 1×4×2 | 5.65ms | 7.96ms | 5/5 → 5/5 |
| AND | 2×4×3 | 34.77ms | 56.94ms | 5/5 → 5/5 |
| OR | 2×4×3 | 22.26ms | 23.28ms | 5/5 → 5/5 |
| XOR | 2×6×4 | 전부 15초 제한 | 0.213초 | 0/5 → 5/5 |
| half adder | 2×7×5 | 4회 제한 | 4회 제한 | 1/5 → 1/5 |

반가산기는 OFF에서 seed 2가 13.04초, ON에서 seed 23이 14.18초에 성공했다.
서로 다른 시드의 성공 시간만 비교해 속도 향상·퇴보라고 판단하면 안 된다.
같은 성공률이고, 대부분 완료되지 않아 이 예산으로는 우열을 정할 수 없다.

XOR의 OFF는 중단된 실행이므로 정확히 몇 배 빨라졌는지는 계산하지 않는다.
작은 회로는 절대 차이가 ms 단위이며, inverter/OR의 작은 차이는 특히
측정 잡음과 구분해야 한다. 블록 수를 최적화하지 않으므로 ON/OFF가 찾는
첫 배치의 블록 수는 다를 수 있다.

## 반복 및 8-worker 확인

추가 60회를 포함해 총 120회 실행했다. 단순히 첫 측정의 유리한 시드만
골라 재측정하지 않고, 동일한 시드 5개를 모두 유지했다.

- 4-worker 역순 반복: inverter 3.30→3.80ms, NOR 5.51→7.70ms,
  AND 34.38→56.19ms, OR 21.97→22.82ms (각 설정 5회 중앙값).
- XOR 4-worker 반복: OFF 0/5 성공, ON 5/5 성공·중앙값 0.215초.
  첫 측정까지 합치면 OFF 0/10, ON 10/10 성공이다.
- XOR 8-worker: OFF 1/5, ON 5/5 성공, ON 중앙값 약 0.225초.
  두 설정 모두 성공한 seed 1에서는 9.631→0.252초, 약 38.2배였다.
  나머지 OFF 4회는 15초 제한이므로 5개 시드 전체의 정확한 배율은 모른다.

작업자 수를 늘려 기준 모델에 더 유리한 조건을 주어도 이 XOR의 효과는
남았다. 다만 시드별·회로별·상자별 일반화는 별도 실험이 필요하다.

## 해석

순차회로에만 유효한 최적화는 아니다. 이 XOR에서는 조합회로에서도 크게
도움이 된다. 하지만 NOR/AND는 느려졌고 반가산기에서는 개선을 확인하지 못했다.
따라서 일반 모델의 기본값 false를 유지한다.

해를 보존하는 규칙이어도 SAT 실행 시간이 단조롭게 줄지는 않는다.
보조 값의 자유도를 줄이는 대신 절이 추가되고, 충돌·변수 선택의 순서도
달라진다. 안전성 증명과 속도 측정은 별개의 판단이다.

## 재현 및 원자료

~~~sh
cargo test --release --lib benchmark_combinational_source_ranks -- --ignored --nocapture
RANK_BENCH_LABEL=repeat RANK_BENCH_REVERSE=1 RANK_BENCH_CASES=inverter,nor,and,or,xor cargo test --release --lib benchmark_combinational_source_ranks -- --ignored --nocapture
RANK_BENCH_LABEL=xor-8workers RANK_BENCH_REVERSE=1 RANK_BENCH_CASES=xor RANK_BENCH_WORKERS=8 cargo test --release --lib benchmark_combinational_source_ranks -- --ignored --nocapture
python3 tools/pruning_lab/summarize_combinational_ranks.py target/combinational-ranks/first/results.json
~~~

[원자료 디렉터리](../tools/pruning_lab/combinational_rank_results/)에 시드별
시간, 성공 여부, CNF 크기, solver/encode 시간과 모델 해시를 저장했다.
요약기는 실패를 숨기지 않으며 PAR2(실패에 제한 시간의 두 배 부과)도 기록한다.
