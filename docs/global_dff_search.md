# 기존 래치를 고정하지 않는 DFF 전체 탐색

2026-11-08, **DFF 전체 배치 탐색에 성공했다.** 마스터 래치 fixture를 읽거나
블록 좌표를 미리 고정하지 않았다. 6개 게이트와 모든 배선을 하나의 전역 SAT
모델에서 찾았고, 실제 순차 시뮬레이터 검증을 통과했다.

| 항목 | 결과 |
|---|---|
| 동작 | 하강 에지 D flip-flop (`dff-neg`) |
| 크기 | 2×7×4, 56칸 |
| 블록 | 45개 + 입력 스위치 2개 |
| 토치 | 6개 |
| 탐색 시간 | 약 163.5초, 제약 생성 약 0.17초 포함 |
| 고정 블록 / 고정 입력 위치 | 0 / 0 |
| 초기 배치 / 기존 latch fixture | 사용 안 함 |
| 실행 | seed=11, worker 4개, rank=8, stage=8 |
| 최적성 | 첫 검증된 해. 최소 크기·블록 수 증명은 아님 |

각 worker는 전체 회로를 풀었다. 게이트를 따로 풀어서 조립하거나, 기존 29블록
DFF를 변형한 결과가 아니다. 이전 29블록 fixture는 여전히 더 작지만 생성 경로가
마스터 래치 고정 후 compaction이므로 이번 목표와 다르다.

## 무엇을 바꿨나

원래 netlist는 다음과 같다. `m`과 `q`를 읽는 곳은 해당 state의 저장된 값을 읽는다.

```text
state(m,q); out(q);
nen = NOR(clk)
a   = NOR(d,nen)
b   = NOR(m,clk)
m   = NOR(a,b)
f   = NOR(q,nen)
q   = NOR(b,f)
```

기존 모델의 기능 조건에 다음을 더했다.

1. 위 6개 게이트의 신호를 출력하는 토치를 각각 하나씩 사용한다.
2. 각 토치 받침에 기여하는 신호를 그 게이트의 원래 피연산자로 제한하고,
   필요한 피연산자마다 실제 기여 연결이 있게 한다.
3. 전기 조건에서 나오는 wire 신호 클래스 일치와 torch의 보수 관계를 직접 표현한다.
4. rank/stage 상한은 8, box mirror symmetry는 끈다.

**게이트의 논리적 연결은 지정하지만 물리 좌표·부착 방향·배선·스위치 위치는
지정하지 않는다.** 마스터와 슬레이브의 공간 구역도 성공한 실행에는 지정하지 않았다.

1–2는 가능한 모든 Boolean 등가 DFF 구현을 보존하는 lemma가 아니다.
주어진 6게이트 netlist를 직접 구현하는 물리 배치로 탐색을 제한한다. 따라서 이
모델의 UNSAT로 모든 DFF 배치가 불가능하다고 말할 수 없다. 성공한 배치는 원래
전기·순차 검증을 그대로 통과해야 한다.

이번 실험은 이 설정들의 조합이 성공했음을 보여준다. 상자 크기와 rank/stage도
달라졌으므로 개별 규칙의 단독 속도 향상률을 측정한 결과는 아니다.

## 검증

- 탐색 중 기존 `verify_sequential`이 모든 안정 상태와 단일 입력 변화의 순차
  동작을 검사했다. 병렬 worker 전체에서 거부된 후보 수는 2,001개였다.
- 저장된 NBT를 다시 로드한 독립 테스트에서 **2,048회의 단일 입력 변화**를
  수행한다. 하강 에지에서만 `q=d`로 바뀌고 다른 때는 유지되는지, 토치 소진이
  없는지 확인한다. 변화 사이에는 기존 수동 입력의 idle/settle 시간을 둔다.
- 찾은 결과의 물리 블록을 고정한 사후 검사로, 탐색용 추가 규칙이 없는 **원래
  exact placer 모델**과 기존 시뮬레이터에서도 받아들여지는지 확인한다. 여기서의
  고정은 결과 검증용이며 discovery 실행의 고정 블록 수는 0이다.

```sh
cargo test --release --lib exact_global_dff_fixture -- --nocapture
```

## 재현

성공한 실행의 정확한 모델 스냅샷은
[dff_global_six_gate.rsdsl](rsdsl/dff_global_six_gate.rsdsl)이다.
이 파일은 당시 기본 모델 전체와 위 추가 규칙을 보존한다.

```sh
mkdir -p target/dff-global-repro
SEQ_CIRCUIT=dff-neg SEQ_DIMS=2x7x4 SEQ_SECONDS=300 \
SEQ_WORKERS=4 SEQ_OPTIMIZE=0 SEQ_RANK=8 SEQ_STAGE=8 \
SEQ_SYM=off SEQ_SEED=11 SEQ_PRUNE=0 \
SEQ_MODEL_FILE=docs/rsdsl/dff_global_six_gate.rsdsl \
SEQ_WRITE=target/dff-global-repro/dff \
cargo test --release --lib explore_sequential_cells -- --ignored --nocapture
```

`SEQ_FIX`, `SEQ_FIX_RCELL`을 설정하지 않는다. 병렬 실행 시간과 처음 발견되는 배치는
달라질 수 있다. 진단 테스트의 종료 코드만 보지 말고 `SEQ placed` 결과를 확인한다.
`SEQ unknown`은 시간 내 발견하지 못했다는 뜻이다.

탐색에서 `SEQ_RANK`, `SEQ_STAGE`, `SEQ_MODEL_PARAMS`를 바꿀 수 있게 했고,
고정 셀 수, CNF 크기, 순환 refinement 수와 전체 거부 수를 로그에 남긴다.

## 결과 파일

- [RCELL](../test/dff-global-2x7x4.rcell)
- [NBT](../test/dff-global-2x7x4.nbt)
- [입출력 위치](../test/dff-global-2x7x4.outputs.json)
- [성공 실행 로그](../tools/pruning_lab/dff_global_success.log)
- [설정과 파일 해시](../tools/pruning_lab/dff_global_result.json)

NBT viewer의 `dff-global-2x7x4.nbt` 예제에도 추가했다.
`clk`는 (0,3,3), `d`는 (0,6,1), `q`는 (1,1,3)이다.

## 첫 해에 남은 중복 리피터

viewer의 NBT 좌표 `(1,1,0)`은 compiler 좌표 `(0,1,1)`이다.
이 위치의 리피터를 Air로 바꾼 별도 후보는 다음 검사를 모두 통과했다.

- 원래 NBT의 다른 블록 및 토치 상태를 유지한 채 제거 후 2,048회 입력 변화 검사.
- 원래 exact 모델에 대한 SAT 검사와 전체 순차 시뮬레이터 검증.
- 블록 수 45→44 (입력 스위치 제외).

따라서 이 리피터는 검증된 동작에 필수적이지 않다. `powered=true`는 현재 전원이
들어온다는 뜻이지 필요한 부품임을 뜻하지 않는다. 발견 실행은 `optimize=false`였으므로
이러한 중복 부품이 남아도 정상이다. 원래 discovery fixture는 변경하지 않았다.

```sh
cargo test --release --lib diagnose_global_dff_repeater_removal -- --ignored --nocapture
```

삭제한 후보는 `target/dff-repeater-probe/removed.{rcell,nbt}`로 생성한다.

두 번째로 지적된 NBT 좌표 `(2,3,0)` (compiler `(0,2,3)`)의 리피터도,
첫 리피터를 이미 제거한 44블록 후보에서 추가로 제거할 수 있었다.
**두 리피터를 모두 제거한 43블록 후보**가 2,048회 입력 변화와 원래 모델의
전체 순차 검증을 모두 통과했다. 후보는
`target/dff-second-repeater-probe/removed.{rcell,nbt}`에 생성한다.

```sh
cargo test --release --lib diagnose_global_dff_second_repeater_removal -- --ignored --nocapture
```

이 두 번의 삭제 검증은 나머지 모든 부품이 필수라는 증명은 아니다.


## 전체 삭제 조합의 자동 최소화（2026-11-08）

수동으로 두 리피터를 지웠던 진단을 대체하는 Rust 경로를 추가했다.
원본 45블록을 그대로 입력하면 **33블록 + 스위치 2개**가 나온다.
측정된 최소화 시간은 약 0.081초이며 discovery 163.5초와 별도다.
원래 모델 재검사 및 NBT 2,048회 입력 변화 검사도 통과했다.

- API: `ExactLocalPlacer::minimize_existing`.
- 모든 기존 부품을 동시에 `원래 종류 또는 Air`로 제한한다. 빈칸은 Air로
  고정하고 입력·출력 위치도 유지한다. 특정 좌표를 하드코딩하지 않는다.
- 각 후보의 모든 신호를 다시 풀고 기존 전체 시뮬레이터 검증을 거친다.
- 블록 수를 목적함수로 최적화한다. 단일 삭제를 탐욕적으로 반복하는 방식이 아니다.
- `optimal=true`: 이 모델과 rank/stage 등 설정 아래, 기존 배치에서 삭제만
  허용한 후보들 중 더 작은 해가 없다는 뜻이다. 다른 배치까지 포함하는 전역
  최소 DFF라는 뜻은 아니다. 타임아웃 결과를 최소라고 표시하지 않는다.
- 일반 `compact`는 `minimize_blocks=true`일 때 창별 수정 전에 이 단계를
  자동 실행한다. 시간 제한이나 실패 시 기존 검증된 배치를 유지한다.
- `explore_sequential_cells`도 발견 후 저장 전에 기본으로 자동 실행한다.
  `SEQ_PRUNE_SECONDS`는 별도 최소화 예산(기본 30초), `SEQ_PRUNE=0`은
  과거 discovery 그대로를 재현할 때만 사용하는 비활성화 옵션이다.

일반 회귀 테스트는 원본 45블록에서 출발해 자동 최소화와 `compact` 연결,
입출력 위치 보존, 부품 이동·교체 없음, 순차 동작을 확인한다.

```sh
cargo test --release --lib minimize_global_dff_automatically -- --nocapture
```

자동 생성 결과: [RCELL](../test/dff-global-optimized.rcell),
[NBT](../test/dff-global-optimized.nbt),
[입출력](../test/dff-global-optimized.outputs.json).
viewer 예제 이름은 `dff-global-optimized.nbt`다. 원본 discovery 파일은 보존한다.
