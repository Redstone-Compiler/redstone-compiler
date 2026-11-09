# 전체 decoder의 construct → compact 실험

## 현재 성공한 결과

일반 Verilog에서 시작해 전체 7seg decoder의 초기 NBT를 생성했다. **4,132블록, Minecraft 크기220×7×33**, 입력4개·출력7개다. `blank=0`인 decoder이며10~15는 모두 꺼진다. 새 출력 fixture는 `test/alu_display/nbt/alu-display-decoder-whole-exact.nbt`, 독립 검증 요약은 `test/alu_display/whole_decoder_exact_verification.json`이다.

중간 checkpoint 없이 실행한 cold 구성은185.937초였고, 저장한 NBT를 다시 읽어16개 정적 입력·모든256개 순서쌍/512회 drive·burnout 부재를 통과했다. seed1/4 workers, 재시작0회다. 형제 실행/회귀 검사 부하가 있으므로 보편적인 속도 벤치마크로 해석하지 않는다.

**성공 방식은 범용 NOR 템플릿으로 초기 후보의 기하를 제한한 Exact 구성이다.** 자유로운 단일 SAT 박스에서 전역 최적해를 찾은 결과가 아니다. Global PnR와 기존20개 물리 셀 분할은 사용하지 않는다. 이후 `compact`는 공간 창을 다시 자유롭게 푼다. 첫 압축은4,132블록을 유지했고, 별도 창 압축은 길이220→218을 얻었으나 블록4,594개로 늘어 초기해를 배포한다. 출력은 `anywhere-observable`이며 이전 monolithic 실험의 경계 driving 출력 계약과 다르다.

재현은 저장된 NBT나 checkpoint 없이 다음 명령을 쓴다.

```sh
sh tools/alu_display/whole_decoder_exact.sh target/alu-display/whole-decoder-new
```

## 성공 경로의 표현과 제약

- `flatten_positive_logic()`가 이중 부정과 양의 OR 중간 단계를 접어 원래28-NOR를24-NOR로 바꾼다. 전체16×7 진리값을 변환 뒤 다시 검증한다.
- 경계에서 같은 함수·반대 함수를 하나의 대표로 운반한다. 예를 들어 입력과 그 반전 네 쌍을 모두 들고 가던 경계를 줄이고, 필요한 반전은 다음 창에서 다시 만든다. 가능한 경우 원래 입력과 이미 운반 중인 극성을 유지한다. 완료 출력은 별도로 계속 관측한다.
- 살아 있는 기존 포트를 먼저 예약한 다음 새 결과의 포트를 배정한다. 그렇지 않으면 새 결과가 아직 필요한 옛 신호의 포트를 빼앗았다.
- 경계 포트는 바깥 방향 repeater와 지지 블록이다. 간격3으로 토치 약전원의 다른 rail 간섭을 막는다. 새 창에서는 같은 위치의 블록·repeater 열로 신호를 운반하고 블록에서 신호를 꺼낸다.
- `nor_scaffold.rs`는 입력 극성 버퍼, 상부 OR 버스, NOR 토치와 출력 하강 경로를 만드는 범용 후보 생성기다. OR 버스의 repeater가 신호 세기를 갱신하며, 별도 pass rail을 포함한 작은 전수 검증도 한다. Exact는 이 후보의 블록별 유지/삭제 부분집합을 탐색하고 전체 simulator 검증을 수행한다.
- 중간 OR의 함수도 vocabulary에 넣는다. `case_rank_levels > 0`은 입력 경우마다 전원이 실제 원천에서 오는 경로를 순위로 증명해, 서로 다른 신호가 합쳐지는 dust OR를 허용한다. 예전 모델이 거절하던 합법적인 작은 OR 배선을 새 모델이 받아들이는 회귀를 추가했다.

상보 대표는 **논리 정보**를 보존하지만 물리적으로 같은 오른쪽 배치를 붙일 수 있다는 증명은 아니다. 템플릿·포트 격자·vocabulary 제한은 의도적인 합성 제한이며, 그 범위 밖의 모든 해를 보존하는 가지치기라고 주장하지 않는다. 기본 `place()`와 기존 construct의 선택적 설정은 유지한다.

## 모델 생성 병목

`Cut[s,t]`는 Rust에서 읽지 않는 내부 판정식인데 export되어 모든 칸 쌍을 생성했다. 큰 prefix에서 약50GB를 쓰며 `Model::ground/get_def`에 머무는 스택을 확인했다. `@internal`로 바꿔 실제 참조하는 feed 쌍만 생성한다. 이는 물리 배치 조건을 바꾸지 않는다.

hard Air·인증된 given 신호의 선택지를 미리 줄이고, Air의 전원 판정을 생략했다. soft/proof-core 셀은 이 제한에서 제외한다. 경우별 순위는 별도 정수 인덱스로 필요한 셀에만 할당한다. 예전 standalone 모델에 새 fact를 강제로 주입하지 않도록 모델 선언을 확인하며, 기존 DFF 모델의 CNF 동등성 회귀도 유지한다.

긴 초기해에는 simulator 한도를4,096 cycles/2,000,000 events로 둔다. SAT 시간 제한은 모델 생성 시간을 포함하지 않아 전체 벽시계 제한과 같지 않다. 전역 삭제 최적화는 이 긴 초기해에서 모델 생성 비용이 컸으므로 재현 runner는 창별 slice 압축부터 진행하며, 블록 수가 늘어나는 압축 후보를 최종 결과로 선택하지 않는다.

검증은 release core456개 통과/실패0/ignored56개, RSDsl9개, native WASM3개와 기존 integration을 포함한다. 오래된 DFF standalone 모델과 RSDsl reference instance의 새 domain 호환성도 확인했다. 기존 legacy 테스트가 NBT fixture를 덮어쓰므로 검증 후 원래 파일을 복원했다.

## 앞선 실패 실험

`examples/whole_decoder_construct.rs`는 기존 최소 cover Verilog와 `blank=0` wrapper를 공유 NOT/OR 그래프로 내리고, 기존 `ExactLocalPlacer::construct()` 다음 `compact()`를 호출한다. 정형 물리 셀20개나 Global PnR를 사용하지 않는다. 전체 논리의16개 입력×7개 출력을 독립 segment mask와 비교한다.

바깥은 gate order를 선택하고 이전 창 대부분을 고정하며 진행하는 휴리스틱 순차 구성이고, 각 창 내부의 배치·배선과 단계별 최소화는 SAT다. overlap/되돌리기/재시작은 잘못된 국소 선택을 복구하는 수단이다. 각 창의 SAT를 사용한다는 사실이 전체 회로의 전역 최적성이나 탐색 완전성을 뜻하지 않는다.

완성된 초기 회로는 `constructed/`에 NBT로 저장해 다시 읽고,16개 fresh 입력·모든256개 순서쌍 전이·burnout 부재를 검사한다. 이를 통과한 후에만 compact를 실행하고 `compacted/`를 별도로 저장·검증한다. 압축 실패 시 검증된 초기 결과를 보존하도록 구성했다.

아래 초기 세 시도와 추가 재시작 시험은 **모두 construct를 완료하지 못했다**. 이 초기 실험들에는 완성 NBT와 압축 결과가 없다. 위의 성공 경로와 구분한다.

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

새 디렉터리를 사용한다. 초기 회로가 검증되면 compact는60초/창당5초의 탐색 예산으로 실행하며 모델 생성 때문에 실제 벽시계 시간은 더 길 수 있다. 기존4,187블록의 회로는 별도의 개별 Exact 후보+Global PnR 결과로 유지한다.
