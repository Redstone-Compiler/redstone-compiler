# Verilog → 8-bit ALU → 십진수 표시판 개발 계획

작성: 2026-11-08. 기준: master `008197f`, 작업 브랜치 `codex/8bit-alu-display`.
이 문서는 구현 계획이다. 아래 완료 기준은 아직 달성했다고 주장하는 항목이 아니다.

이 브랜치의 draft PR에는 7-segment 디코더부터 unsigned 8-bit ALU와 BCD/표시 연결까지 계속 구현을 쌓는다. 현재 검증된 체크포인트와 남은 물리 문제는 [구현 기록](alu_display_compiler_progress.md)에 구분해서 남긴다. 작은 디코더의 성공만으로 ALU 작업 완료나 merge 준비 상태로 표시하지 않는다.

## 목표와 범위

일반 Verilog 모듈로 작성한 unsigned 8-bit ALU와 이진수→BCD→7-segment 회로를 논리 IR, 물리 배치/배선, NBT까지 연결한다. 처음에는 CPU나 ISA 없이 두 입력과 연산 선택 스위치를 사용한다. 표시 범위는 000~255, 세 자리 직접 구동으로 시작한다.

BCD 변환기와 디코더는 재사용 가능한 RTL 예제로 정의한다. 컴파일러가 특정 모듈 이름을 알아보고 특별 취급하지 않는다. 검증된 물리 셀을 재사용하는 최적화는 기능과 포트 계약을 확인하는 target mapping 계층에 둔다.

## 현재 확인된 공백

| 영역 | 현재 상태 | 필요한 작업 |
|---|---|---|
| Verilog 표현식 | `~`, `&`, `^`, `\|`, `+` 중심. `a[3]`은 `a_3` 이름으로 파싱됨 | 실제 선택 AST, 폭 계산, 상수, 연결, 조건식 |
| 폭과 인덱스 | RTL에서 선언 범위를 폭으로 축약 | unsigned 확장/절단과 지원 범위 명시 |
| 일반 조합 process | `always @*` 처리에 latch 패턴 중심 경로가 있음 | 완전 할당 조합 process와 latch 구분 |
| 논리→물리 | 일반 조합 leaf는 scalar 중심. `Add`, `Inc`, `Mux` 일반 mapping 미완성 | bus 분해, 작은 leaf, 연산 mapping |
| 계층 | cell과 instance 혼합 및 중첩 계층 제약 | flat composite와 작은 leaf 정의로 정규화 |
| 가산기 | CarryTiling 실험/검증 존재 | production IR/PnR 연결과 배선 가능한 포트 |
| 표시 출력 | 이름과 위치를 담는 `outputs.v1` 메타데이터 있음 | 관찰 가능한 21개 endpoint와 표시판 위치 연결 |
| 실제 램프 | `BlockKind`에 lamp 없음 | block/NBT/simulator/WASM/viewer 지원 |

근거 경로: `src/verilog/{ast,parser,rtl,synth}.rs`, `src/ir/{logical_adapter,logical_lowering}.rs`, `src/transform/place_and_route/local_placer/exact/tiling.rs`, `src/transform/place_and_route/global_pnr/`, `src/world/block.rs`, `src/output.rs`, `tools/nbt-viewer/src/main.ts`.

기존 `test/alu.v`는 완성된 8-bit ALU fixture가 아니다. 새 예제는 `test/alu_display/`에 둔다. CLI에서 출력 경로 없이 Verilog를 읽는 성공은 IR 로드 성공이며 NBT 컴파일 성공과 구분한다.

## 고정할 모듈 계약

### ALU

`alu8(a[7:0], b[7:0], op[2:0]) → y[7:0], z, n, c, v`

| op | 동작 | c | v |
|---|---|---|---|
| 000 | a + b | 9번째 carry | signed 덧셈 overflow |
| 001 | a − b | borrow가 없으면 1 | signed 뺄셈 overflow |
| 010 | a AND b | 0 | 0 |
| 011 | a OR b | 0 | 0 |
| 100 | a XOR b | 0 | 0 |
| 101 | a 전달 | 0 | 0 |
| 110 | a를 왼쪽으로 1bit, 빈자리 0 | a[7] | 0 |
| 111 | a를 오른쪽으로 1bit, 빈자리 0 | a[0] | 0 |

결과는 하위 8bit, `z=(y==0)`, `n=y[7]`. ADD overflow는 입력 부호가 같고 결과 부호가 다를 때, SUB overflow는 입력 부호가 다르고 결과 부호가 a와 다를 때다. unsigned 인터페이스에도 signed 해석용 v를 제공한다.

첫 물리 단계는 ADD/SUB와 y, 다음은 나머지 연산과 z/n, 마지막은 c/v다. ADD/SUB는 `b XOR {8{sub}}`, 초기 carry=sub인 하나의 carry chain을 공유하도록 RTL을 작성한다. 반복 연결 문법 지원 전에는 명시적 연결을 사용한다. 컴파일러가 별개 ADD/SUB 회로를 자동 공유한다고 가정하지 않는다.

### 표시 논리

| 모듈 | 입력 | 출력 | 계약 |
|---|---|---|---|
| `add3_if_ge5` | nibble[3:0] | corrected[3:0] | 5 이상이면 3을 더한 하위 4bit, 아니면 그대로 |
| `bin8_to_bcd` | value[7:0] | hundreds/tens/ones 각각 [3:0] | 100h+10t+o=value, 각 자리 0~9 |
| `bcd_to_7seg` | digit[3:0], blank | seg[6:0] | active-high, bit 0부터 a,b,c,d,e,f,g |
| `display_u8` | value[7:0] | 세 자리 seg 각각 [6:0] | 기본 000~255, 21개 독립 출력 |
| `alu8_display` | a,b,op | ALU 결과/flags와 세 자리 seg | y를 unsigned 십진수로 표시 |

디코더 마스크는 0~9 순서로 `3f,06,5b,4f,66,6d,7d,07,7f,6f` (hex). blank=1 또는 digit=10~15이면 모두 0이다. don't-care를 사용하지 않는다. 선행 0 생략은 후속 옵션으로 둔다.

Double Dabble은 BCD=0에서 시작해 각 자리의 add3 보정 후 다음 입력 bit를 MSB부터 shift-in하는 8단계를 명시적으로 펼친다. 초기 구현에는 나눗셈, 나머지, 배열, generate, for, parameter가 필요 없다. 중간 보조 함수는 전체 16개 입력에서 정의하고, 실제 BCD 단계의 도달 가능한 값은 별도로 확인한다.

물리 endpoint 이름은 `hundreds_a`~`ones_g`로 통일한다. 첫 단계는 viewer가 상태를 읽을 수 있는 dust/repeater 등의 출력을 사용한다. 현재 endpoint 메타데이터는 위치를 기록할 뿐, 원하는 표시판 좌표까지 자동 배선하는 기능이 아니다. 실제 표시판에는 고정 sink 위치/접근 방향/점유 공간 계약을 추가한다. viewer의 기존 compiler→NBT 좌표 변환도 재사용한다.

## 프런트엔드와 IR 설계 순서

1. 첫 subset은 unsigned `[N-1:0]` packed vector다. 다른 범위와 signed, x/z, 동적 선택은 명확한 오류로 거부한다. 미지원 표현식을 조용히 다른 의미로 해석하지 않는다.
2. sized binary/hex/decimal 상수, 정적 bit/part select, concatenation, `?:`, `==`, `>=`, `-`를 추가한다. `{a,b}`에서 a는 상위 bit다. 고정 shift는 select와 concat으로 표현한다.
3. 폭 추론과 operand 확장, 결과 절단을 명시적으로 elaboration한다. unsized 상수는 Verilog 규칙을 구현하기 전까지 새 subset에서 거부하고 fixture는 sized 상수로 작성한다. 기존 최소 폭 상수 처리와의 호환성 변경은 테스트로 드러낸다.
4. carry는 `{1'b0,a}+{1'b0,b}`처럼 9bit 연산으로 드러낸다. 같은 폭 unsigned 덧셈의 결과 폭, 조건식의 분기 폭, 비교의 1bit 결과, assignment 절단을 각각 검증한다.
5. `.digit(bcd[3:0])` 같은 입력 포트의 표현식 바인딩을 지원한다. 출력 연결은 첫 단계에서 이름 있는 wire로 제한한다.
6. typed LogicalDesign을 유지한 채 hierarchy를 정규화하고 bus를 target mapping에서 작은 leaf로 분해한다. scalar LogicGraph로 전체 설계를 우회하지 않는다.
7. 연속 할당 기반 예제가 관통한 다음 `always @*`, blocking assignment, if/else, case/default를 확장한다. 기본값과 순차적인 덮어쓰기를 보존하고 의도치 않은 불완전 할당을 진단한다. 기존 latch 경로의 회귀를 확인한다.

공유 IR enum/API 변경은 통합 담당자가 먼저 확정한다. subtraction/comparison/select를 기존 IR 연산으로 정규화할지 새 연산으로 남길지도 이 단계에서 결정하고 frontend와 mapper가 각자 다른 계약을 만들지 않게 한다.

## 물리 구현 전략

- 전체 19-input ALU를 하나의 exact leaf로 풀지 않는다. decoder, bitwise 연산, mux, carry slice 등 작은 단위로 나누고 op decode를 공유한다.
- CarryTiling은 바로 연결할 수 있는 완제품으로 취급하지 않는다. 현재 chain 조립에는 스위치, carry polarity adapter, 출력 torch가 포함된다. `LayoutCandidate`로 옮기면서 외부 입력 스위치를 router 구동 포트로 바꾸고 carry 극성, 접근점, support/blocked cell을 보존한다.
- 기존 XOR/half-adder macro와 candidate cache를 활용한다. 새 물리 recipe의 revision을 cache key/version에 반영한다.
- 1→2→4→8bit에서 논리 정확성, 배선 성공, 안정화 시간, torch burnout을 함께 확인한다. 기존 tile의 8/16bit 일부 입력 검증을 전수 검증으로 확대 해석하지 않는다.
- broadcast fanout과 bus routing 비용을 먼저 계측한다. local solver 캐시가 빨라져도 전역 candidate 조합/배치/배선 비용은 남는다.
- 큰 회로용 NBT vector runner는 케이스마다 truth table을 재생성하지 않도록 예상값 계산과 실행을 분리한다.

## 현재 bit mapping 구현과 실험 경계

`LogicalDesign::lower_to_routable()`의 일반 조합회로 mapping은 기본적으로 Boolean 연산 1개, 경계 입력 최대 2개의 primitive leaf를 사용한다. fanout-free cone 병합은 `lower_to_routable_with_bit_mapping_limits(4, 3)`으로 명시적으로 켠다. 이 제한의 gate 수는 NOR 변환 전 Boolean 연산 수이며, 물리 블록 수나 변환 후 gate 수와 같지 않다.

조건식 decoder에서 4/3 병합은 instance를 112개에서 68개로 줄였고 32개 입력에서 primitive mapping 및 독립 oracle과 일치했다. 그러나 3-input AND cone의 local candidate 생성이 제한된 탐색 설정에서 실패했다. 따라서 이 수치로 물리 컴파일 성공이나 속도 개선을 주장하지 않는다. 경계 입력 2개로 제한한 SOP decoder는 70개 instance이며, 세 종류의 primitive가 각각 local candidate 4개를 생성했다. 전체 배선과 exported NBT 검증은 별도 단계다.

계층은 개별 cell의 입력 준비 상태를 검사하기 전에 flat netlist로 펼친다. 서로 다른 child의 독립 출력끼리 교차 연결된 합법적인 조합 DAG를 모듈 단위 순환으로 오판하지 않기 위함이다. 일반 bus 포트 이름은 `digit_0`처럼 유지하지만, 이미 같은 이름의 scalar 포트가 있으면 해당 bus bit만 `digit[0]`으로 표기한다. 그 이름도 사용 중이면 `$` 접두사를 붙여 충돌을 피한다. scalar 포트 이름은 보존하며 생성된 모든 이름은 유일하다.

상수 접기, select/concat/resize 배선, 구조 공유를 적용한다. 사용되지 않는 top input bit는 interface에 남기고 배선하지 않는다. 상수 출력은 기존 입력을 읽는 항진식/모순식 leaf로 표현하며, 입력이 전혀 없는 상수 전용 회로는 아직 명확한 오류를 반환한다. pass-through 출력에는 관찰 가능한 buffer leaf를 둔다.

## 단계별 완료 기준

| 단계 | 결과물 | 통과 기준 |
|---|---|---|
| M0 계약/재현성 | 포트/폭 계약, 독립 oracle, fixture 골격 | clean checkout에서 필요한 release test가 재현됨 |
| M1 작은 관통 | decoder RTL→IR→NBT→7개 출력 관찰 | digit×blank 32개 논리 전수, 물리 안정값과 전이 검사 |
| M2 bus 산술 | ADD/SUB mapping과 carry adapter | 1,2,4bit 단계 통과 후 8bit 물리 성공; 논리 전수 oracle 일치 |
| M3 ALU | 8개 op와 flags, snapshot | 524,288개 논리 입력 전수; NBT 경계/연속 전이 검사 |
| M4 표시 논리 | double dabble와 decoder 3개 | 256개 변환 논리 전수, 표시 출력 21개 물리 검증 |
| M5 결합 | alu8_display NBT와 재현 명령 | y/flags/표시값 일치, 시간·크기·안정화 보고서 |
| M6 표시판 | 실제 lamp와 고정 panel sink | simulator/NBT/viewer 일치, 인접 segment 간섭 검증 |

M2/M3의 산술 작업과 M4의 논리 fixture는 M0 이후 병렬 가능하다. M4의 물리 실행은 M1과 bus mapping을 사용한다. M5는 M3/M4 완료 후 진행한다. 작은 단계를 통과하지 않은 상태에서 큰 ALU 탐색 시간을 무작정 늘리지 않는다.

M0에서 `src/ir/debug.rs`의 `include_str!`가 참조하는 `test/counter.snapshot/counter.v`, `test/d-flip-flop.snapshot/d-flip-flop.v`의 tracked fixture 여부를 정리한다. 현재 로컬에는 있지만 Git에 없는 입력에 의존하는 테스트가 있으므로 기존 snapshot 디렉터리를 통째로 추가하지 말고 최소 입력 fixture로 고친다.

## 병렬 작업 소유권

첫 구현 묶음은 통합 담당 1명 + 에이전트 3명으로 진행한다. 같은 checkout에서 각 파일의 작성자를 한 명으로 고정한다.

| 담당 | 소유 범위 | 다른 담당자에게 제공 |
|---|---|---|
| 통합 | 공유 IR 계약, 테스트 연결, 단계별 실행/보고서 | 합의된 연산/폭 API와 검증 명령 |
| Frontend | `src/verilog/`, parser/elaboration 테스트 | typed IR, 미지원 문법 진단 |
| Mapping | `src/ir/logical_adapter.rs`, `logical_lowering.rs`, mapping 테스트 | flat composite와 scalar leaf/산술 mapping |
| Fixtures/reference | 새 `test/alu_display/`, 독립 oracle 테스트 파일 | decoder/BCD/ALU RTL와 기대값 |

후속 묶음은 physical carry adapter, reusable vector runner, viewer/panel을 나눈다. `global_pnr`와 cache 변경은 physical 담당 한 명이 맡는다. lamp 도입은 block enum/NBT/simulation/WASM에 걸친 작업이므로 공유 파일 변경을 묶어서 순차 통합한다. agent마다 임의 커밋하거나 다른 담당자의 파일을 동시에 수정하지 않는다.

## 검증과 성능 판단

논리 검증은 독립 정수 모델로 수행한다. decoder 32개, add3 16개, converter 256개, ALU 256×256×8=524,288개를 전수 비교한다. 가능하면 이후 외부 Verilog simulator를 두 번째 oracle로 추가한다. 현재 환경에서 iverilog/verilator/yosys 설치는 확인되지 않았으며 이를 이미 사용한 검증으로 기록하지 않는다.

물리 검증은 별도다. 작은 leaf에는 전수 입력과 전수 전이, 큰 회로에는 경계값과 seed 고정 연속 전이를 사용한다. 현재 모든 입력 순서쌍 검사기를 19-input ALU에 적용하면 약 2.75×10¹¹ 전이가 되어 사용할 수 없다. 무작위 물리 검사를 완전한 증명이라고 부르지 않는다.

필수 물리 사례: 긴 carry 전파, 0/255 경계, SUB borrow, op 변경, `9↔10`, `99↔100`, `199↔200`, `255↔0`. 입력 변경 중 조합 glitch는 허용하고 안정화 이후 결과로 판정한다. timeout, event limit, 미안정화, burnout은 모두 별도로 실패/미확정으로 기록한다. 탐색 timeout을 UNSAT으로 해석하지 않는다.

각 단계에서 cold/warm cache, local candidate 생성 시간, global placement/routing 시간, 실패 원인, 블록 수, bounding box, 안정화 tick을 기록한다. 동일 입력/설정/seed로 비교하고, 최적화는 여러 seed에서 성공률과 시간 분포를 함께 본다. 첫 작은 회로 측정 후 시간/event 예산을 정하고 초과 시 결과를 남겨 다음 단계 진입을 막는다.

빌드/테스트/실행은 모두 release 모드다. 기존 compilation snapshot과 candidate cache를 재사용한다. 저장소에는 최소 RTL/골든 fixture/측정 요약을 남기고 대형 탐색 trace는 기본적으로 로컬 산출물로 둔다.

## 후속 확장: 순차 변환기와 CPU

8-cycle Double Dabble은 조합 버전과 별도의 후속 목표다. 여러 레지스터, reset, FSM, nonblocking assignment 지원이 필요하다. 계약은 `clk/reset/start/value → busy/done/bcd`로 한다. idle에서 start를 받은 edge에 입력을 저장하고, 이후 8개의 edge에서 처리한다. 마지막 edge에서 결과를 공개하고 done을 한 cycle 올린다. 결과는 다음 완료까지 유지한다. reset은 동기식 최우선, busy 중 start는 무시하고 완료 edge에서 새 요청은 받지 않는다. clock 주기는 물리 안정화 측정으로 정한다.

CPU를 붙일 때는 accumulator, 입력/결과 register, 작은 instruction ROM부터 확장한다. MIPS 호환성이나 명령어 집합 구현을 이번 ALU/표시 컴파일러 완료 조건에 포함하지 않는다.

## 산술 물리 경로 진척 — 2026-11-08

`examples/adder_nbt.rs`에서 unsigned `a,b → sum[N:0]`를 Verilog부터 NBT까지 연결했다. 이 결과는 ADD 경로이며 ADD/SUB 공유, 전체 ALU의 8개 op/flags 또는 BCD/표시판 완료를 뜻하지 않는다. 물리 설정은 `10x10x5`, ranked beam 256, A* expansion 20,000이며 seed 조정 없이 box/beam을 바꾸었다. 작은 box의 XOR는 출력 diode reservation에서 탈락했고, 입력 처리 전의 실패였음을 debug stage 로그로 확인했다.

- 2bit cold: PnR 6.507초, 645블록. 저장 NBT의 정적 16개 입력과 256개 ordered transition 전수 통과.
- 4bit warm: PnR 7.648초, 1,925블록. local candidate 17개 instance 모두 cache 재사용. 정적 256개 입력 전수와 연속 walk 256개 통과.
- 8bit warm: PnR 38.709초, 3,433블록. directed carry/boundary 33개와 고정 seed 무작위 256개, 총 289개 정적 사례와 같은 연속 walk 통과. 8bit 전체 입력/전이의 완전 검증은 아니다.

8bit 초기 walk 실패는 simulator의 redstone normalization에서 OFF만 통지하고 새 ON을 누락한 문제로 재현되었다. fixpoint 전후 ON/OFF 통지를 수정한 뒤 원래 NBT를 다시 읽어 검증했다. NBT 및 입력 순서는 바꾸지 않았다. `--verify-only`로 2/4/8bit를 모두 재검증했고, 기존 실패/성공/status 기록을 `diagnostics/prior-*`로 보존한 뒤 `verification.json`에 simulator 수정 후 재검증 여부와 파일 fingerprint를 남겼다. 검증 원본은 독립 정수 `a+b`이며 graph 자기평가를 사용하지 않는다.

명령과 coverage 범위는 `test/alu_display/README.md`의 unsigned 가산기 절을 따른다. artifact는 `target/alu-display/` 아래 두며, `verification.json`이 실제 통과를 나타낸다. source/IR 로딩 성공이나 NBT 생성만으로 검증 완료를 선언하지 않는다.
