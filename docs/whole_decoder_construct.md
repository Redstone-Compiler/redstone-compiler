# 전체 decoder의 construct → compact 실험

`examples/whole_decoder_construct.rs`는 기존 최소 cover Verilog와 `blank=0` wrapper를 공유 NOT/OR 그래프로 내리고, 기존 `ExactLocalPlacer::construct()` 다음 `compact()`를 호출한다. 정형 물리 셀20개나 Global PnR를 사용하지 않는다. 전체 논리의16개 입력×7개 출력을 독립 segment mask와 비교한다.

바깥은 gate order를 선택하고 이전 창 대부분을 고정하며 진행하는 휴리스틱 순차 구성이고, 각 창 내부의 배치·배선과 단계별 최소화는 SAT다. overlap/되돌리기/재시작은 잘못된 국소 선택을 복구하는 수단이다. 각 창의 SAT를 사용한다는 사실이 전체 회로의 전역 최적성이나 탐색 완전성을 뜻하지 않는다.

완성된 초기 회로는 `constructed/`에 NBT로 저장해 다시 읽고,16개 fresh 입력·모든256개 순서쌍 전이·burnout 부재를 검사한다. 이를 통과한 후에만 compact를 실행하고 `compacted/`를 별도로 저장·검증한다. 압축 실패 시 검증된 초기 결과를 보존하도록 구성했다.

초기 세 시도와 추가 재시작 시험은 **모두 construct를 완료하지 못했다**. 따라서 완성 NBT와 압축 전후 크기 비교가 없고, compact를 실행했다고 해석하면 안 된다.

## 측정

base seed1, 자동 seed 재시작0회, given frozen signals/early outputs 사용. 각 창의 Unknown은 해당 시간 안에 해를 찾지 못했다는 뜻이며 decoder 전체의 불가능 증명이 아니다.

| 설정 | 단계별 최소화 | 진행 | 중단 지점 | 시간 |
| --- | --- | --- | --- | ---: |
|width4/height6, MinLive, overlap1, 1 worker, 창당10초|없음|2/28|g99=NOR(g5,g6), 창3~6 Unknown|45.234초|
|width2/height10, SmallestConeFirst, overlap1, 4 workers, 창당15초|5초|6/28|g103=NOR(g6,g11), 창3~6 Unknown|106.655초|
|width2/height24, SmallestConeFirst, overlap2, 4 workers, 창당15초, chain2|5초|1/45|g11=NOT(digit_1), 창3~6 Unknown|64.946초|
|width2/height10, SmallestConeFirst, overlap1, 4 workers, 창당15초, 내장 재시작1회|5초|첫 시도6/28 → 재시작7/28|재시작 후 g103 통과, seg_4에서 시도별120초 예산 소진|232.462초|

마지막 시험만 `CONSTRUCT_RESTARTS=1 CONSTRUCT_BUDGET_SECONDS=120`을 사용했다. 이 예산은 전체 실행이 아닌 각 구성 시도에 적용된다. 첫 시도(seed1)는 g103의 창3~6에서 Unknown이었다. 두 번째(seed104730)는 g103을 통과했지만 seg_4의 창3/4에서 Unknown, 창5에서 남은13.359초를 사용해 예산이 끝났다. 창6은 시도하지 않았다. 일반 오류 문구의 창6 제한과 실제 예산 중단을 구분해 `restart-analysis.json`에 기록했다. 재시작이 한 국소 정체를 벗어나는 사례지만 전체 구성 성공이나 seed 변경만의 인과 효과를 입증하지 않는다. 4-worker 탐색과 시간 제한 최소화는 같은 seed에서도 완전히 결정적이지 않다.

두 번째 실행의 성공 단계는 g5/g79/g19/g83/g6/g11이고 마지막 부분 배치가197블록이었다. 이는 전체 decoder 크기가 아니다. 첫 게이트가4블록으로 생성됐으나 첫 실험과 geometry/order/worker도 달라 최소화 옵션만의 A/B 비교는 아니다.

원래28-NOR 표현은 최대 live 신호가 NetIndex13/SmallestConeFirst14/MinLive14였다. `chain_wide_gates(2)`는 NOR28개를 유지하고 OR17개를 추가해45단계로 바꾸며, SmallestConeFirst의 peak를13, continuing fan-in 최대를2로 낮췄다. 변환된 전체 netlist도16×7 출력을 독립 검증했다. 논리 동등성을 확인한 다른 물리 합성 표현이며, 물리 배치 탐색 공간까지 동일하다는 주장은 아니다.

출력 계약은 `anywhere-observable`이다. 기존 monolithic 실험의 박스 경계 driving 출력7개와 구분한다. `CONSTRUCT_OUTPUT_MODE=max-y`로 출력7개를 MaxYFace에 둘 수 있으며 같은 계약을 compact에도 전달하지만 이번 측정에서는 사용하지 않았다.

각 실행의 `attempt.json`, 설정 원문, `step-observations.json`과 solver 로그는 `target/alu-display/whole-decoder-construct-*` 및 임시 로그에 보존한다. simulator refinement는 실패한 창마다0회여서, 발견한 모델의 물리 검증 거절보다 최초 단계 배치 탐색에서 막혔다.

## 재현

```sh
CONSTRUCT_WIDTH=2 CONSTRUCT_HEIGHT=10 CONSTRUCT_ORDER=smallest-cone \
CONSTRUCT_STEP_OPT_MS=5000 CONSTRUCT_STEP_SECONDS=15 \
CONSTRUCT_WORKERS=4 CONSTRUCT_BUDGET_SECONDS=180 \
  cargo run --release --example whole_decoder_construct -- \
  target/alu-display/whole-decoder-construct-new

CONSTRUCT_WIDTH=2 CONSTRUCT_HEIGHT=24 CONSTRUCT_ORDER=smallest-cone \
CONSTRUCT_OVERLAP=2 CONSTRUCT_MAX_FANIN=2 CONSTRUCT_STEP_OPT_MS=5000 \
CONSTRUCT_STEP_SECONDS=15 CONSTRUCT_WORKERS=4 CONSTRUCT_BUDGET_SECONDS=360 \
  cargo run --release --example whole_decoder_construct -- \
  target/alu-display/whole-decoder-construct-chain-new
```

새 디렉터리를 사용한다. 초기 회로가 검증되면 compact는 최대60초/창당5초로 실행한다. 아직 이 decoder에서는 그 단계에 도달하지 못했다. 기존4,187블록의 검증된 회로는 별도의 개별 Exact 후보+Global PnR 결과로 유지한다.
