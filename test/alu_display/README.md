# ALU / 표시 회로 실험

`bcd_to_7seg.v`는 조건식 기반, `bcd_to_7seg_sop.v`는 공유 숫자 판정 기반의 동일한 디코더다. `seg[0]`부터 a,b,c,d,e,f,g이고 active-high다. blank와 잘못된 BCD(10~15)는 모두 OFF다.

`bin8_to_bcd.v`는 8단계 조합 Double Dabble이며 입력 0~255를 세 자리 BCD로 변환한다. `alu8.v`의 연산/flag 계약은 [계획 문서](../../docs/alu_display_compiler_plan.md)에 있다.

독립 정답과 실제 parser/RTL 결과 비교:

```sh
cargo test --release --test alu_display
```

기본 숫자 표시용 디코더(`blank=0`)의 물리 컴파일, NBT 재로딩, 16개 상태와256개 순서쌍 전이 검증:

```sh
BIT_LITERAL=1 NARY_STAGGER=1 PNR_PIN_ACCESS=1 \
PNR_ASTAR_EXPANSIONS=20000 PNR_MAX_STEPS=256 \
  cargo run --release --example decoder_nbt -- \
  target/alu-display/decoder-digit-verified \
  test/alu_display/bcd_to_7seg_minimized.v \
  test/alu_display/bcd_digit_to_7seg.v
```

실제 성공 기록은23.308초,20개 cell,62개 배선,5,703블록이다. `verification.json`은 NBT 재읽기 및 모든256개 전이를 연속512회 입력 적용으로 검증한 뒤 생성한다. invalid BCD10~15도 모두 OFF다. `bcd_to_7seg_minimized.v`는 전체 진리표를 보존하는 일반 POS식이며 `python3 tools/alu_display/minimize_decoder.py`로 재생성할 수 있다.

마지막 wrapper 인자를 생략하면 blank 제어를 포함한32상태/1,024전이 실험이다. 이 모드의 전체 전이 검증은 미완료다. 정적32개를 통과한 배치에서도 burnout이 검출되어 최종 성공으로 인정하지 않았다. 실패 시 RTL 검증 성공과 물리 검증 성공을 구분한다. 생성된 NBT, 포트 위치, IR, candidate cache는 지정한 산출물 디렉터리에 남는다. NBT 출력은7개의 논리 segment 신호이며 실제 lamp 표시판은 후속 단계다.

## Unsigned 가산기 물리 실험

Exact SAT 셀 후보와 자유3D를 함께 사용하는 decoder는4,187블록,36×64×56이며 모든256개 전이를 통과했다. 같은 자유3D에서 recipe만 쓰는 대조군은4,312블록이다. 재현 명령, cold 시간과 메모리 상태를 정규화하는 cache 계약은 [Exact 전역 후보 실험](../../docs/exact_global_candidates.md)에 있다.

`adder_nbt`는 Verilog `{1'b0,a}+{1'b0,b}`에서 시작해 Logical IR, associative cell mapping, 전역 배치/배선, NBT 내보내기까지 실행한다. `1`, `2`, `4`, `8` bit를 받는다. 검증 정답은 graph 평가가 아닌 독립 정수 덧셈 `a+b`다.

작동을 확인한 설정은 local box `10x10x5`, ranked beam `256`, A* expansion budget `20,000`, 예약된 입출력 diode와 verified n-ary cell이다. seed를 바꾸어 얻은 결과가 아니다. 기본 `8x8x4` box에서는 XOR의 출력 diode escape 확보가 실패할 수 있으므로 아래 설정을 명시한다.

```sh
# 첫 실행에 cache가 비어 있을 때만 cold 측정이다.
ADDER_DIM=10x10x5 ADDER_BEAM=256 \
  ADDER_CACHE_DIR=target/alu-display/arithmetic-candidate-cache \
  cargo run --release --example adder_nbt -- 2 target/alu-display/adder-2

# 같은 설정의 검증된 local candidate를 재사용한다.
ADDER_DIM=10x10x5 ADDER_BEAM=256 \
  ADDER_CACHE_DIR=target/alu-display/arithmetic-candidate-cache \
  cargo run --release --example adder_nbt -- 4 target/alu-display/adder-4
ADDER_DIM=10x10x5 ADDER_BEAM=256 \
  ADDER_CACHE_DIR=target/alu-display/arithmetic-candidate-cache \
  cargo run --release --example adder_nbt -- 8 target/alu-display/adder-8
```

실제 기록(2026-11-08):

| 폭 | 원래 PnR 시간 | candidate cache | 블록 수 | 수정된 simulator로 저장 NBT 재검증 |
| --- | ---: | --- | ---: | --- |
| 2bit | 6.507초 | cold | 645 | 정적 16개 + 모든 ordered transition 256쌍 |
| 4bit | 7.648초 | warm, 17개 instance 모두 재사용 | 1,925 | 정적 256개 전수 + 연속 walk 256개 |
| 8bit | 38.709초 | warm | 3,433 | 정적 289개 + 같은 순서의 연속 walk 289개 |

8bit는 directed carry/boundary 33개와 seed `0xadd8202611080001`의 무작위 256개다. directed 항목에는 중복이 있으며 **전체 65,536개 입력 또는 모든 전이를 증명한 결과가 아니다**. 4bit walk도 전체 입력 순서쌍 검사가 아니다.

저장된 파일만 재검증하면 PnR를 다시 실행하거나 NBT를 수정하지 않는다. 현재 실제 성공 산출물 경로는 다음과 같다.

```sh
cargo run --release --example adder_nbt -- --verify-only 2 target/alu-display/adder-2-large-io-diagnostic
cargo run --release --example adder_nbt -- --verify-only 4 target/alu-display/adder-4
cargo run --release --example adder_nbt -- --verify-only 8 target/alu-display/adder-8
```

8bit는 처음 연속 walk 47번째 `(252,52) → (149,247)`에서 `sum_6`이 틀렸다. 원인은 carry 수식이나 저장 NBT가 아니라 simulator의 dust 강도 정규화 후 상승 이벤트 누락이었다. fixpoint 전후 ON/OFF 통지를 수정한 뒤 **같은 NBT 파일**로 위 검사를 통과했다. 원래 실패 기록은 산출물의 `diagnostics/prior-*/failure.json`에 보존한다.

`verification.json`에는 `verification_after_simfix`, `nbt_unchanged`, NBT/interface fingerprint, 검증 범위와 시간을 기록한다. verify-only의 `compile_ms`는 원래 컴파일 기록이고 `verification_ms`/`total_ms`는 이번 재검증 시간이다. `experiment.json`의 `cache_entries_before`와 실제 로그의 reused 수로 cold/warm을 구분한다. 현재 runner는 PnR의 assembled-layout verifier에서도 정적 입력과 연속 walk를 검사하며, 최종 NBT 재읽기 검증을 별도로 수행한다.

## 최소 cover 디코더 실험

`bcd_to_7seg_minimized.v`는 동일한 32개 입력 계약을 ordinary Verilog로 표현한 대안이다. `tools/alu_display/minimize_decoder.py`가 4-input prime implicant를 전부 열거하고, 각 segment의 최소 항 수 SOP/POS cover 조합 648개를 비교해 공유 recipe 입력 연결 수가 가장 작은 구성을 선택한다. 10~15는 OFF이며 don't-care를 사용하지 않는다. 임의 Boolean network 전체에 대한 전역 최적성을 주장하지 않는다.

```sh
python3 tools/alu_display/minimize_decoder.py
cargo test --release --lib minimized_decoder_covers -- --nocapture
```

현재 literal-cell mapping에서 blank=0 버전은 18→20개 cell로 늘지만 입력 연결은 90→62개로 줄어든다. blank 입력을 포함하면 25→20개 cell, 연결 104→69개다. 최대 fanout은 10→11이므로 이 역시 tradeoff이며, 연결 수 감소만으로 배선 성공이나 속도 향상을 단정하지 않는다. 전체 16/32개 값 및 각 leaf 진리표는 검증했고 NBT와 전이 검증은 별도 실험이다. `bcd_digit_to_7seg.v` wrapper와 함께 사용하면 기존 blank=0 물리 runner 입력 계약을 유지한다.

## 하나의 carry chain을 공유하는 ADD/SUB

`add_sub1.v`, `add_sub2.v`, `add_sub4.v`, `add_sub8.v`는 일반 Verilog로 `adjusted_b = b XOR subtract`, `carry_0 = subtract`와 bit별 sum/carry를 명시한다. ADD/SUB를 따로 합성한 뒤 자동으로 합쳐 주는 최적화에 의존하지 않는다. `y`는 하위 Nbit, `c`는 ADD에서 carry, SUB에서 **borrow가 없으면 1**이다.

```sh
python3 tools/alu_display/generate_add_sub.py
cargo test --release --lib explicit_shared_add_sub -- --nocapture
ADD_SUB_CACHE_DIR=target/alu-display/add-sub-candidate-cache \
  cargo run --release --example add_sub_nbt -- 2 target/alu-display/add-sub-2-transition-filter
# 저장된 실제 성공 NBT만 다시 검증:
cargo run --release --example add_sub_nbt -- --verify-only 2 target/alu-display/add-sub-2-transition-filter
```

독립 정수 ADD/SUB와 carry/no-borrow oracle로 1/2/4bit 논리 전수 검사를 통과했다. **2bit 물리 결과**는 12개 cell/24개 route/1,333블록, warm cache PnR 6.040초이며 NBT 재읽기 후 32개 정적 입력과 1,024개 ordered transition을 모두 통과했다. 첫 배치의 특정 전이에서 torch burnout이 발생했지만, PnR 내부의 전수 전이 verifier가 이를 거부하고 두 번째 배치를 선택했다. 결과를 통과시키기 위해 검증을 완화하거나 블록을 손으로 수정하지 않았다.

4bit 첫 물리 실행은 12개 배치/배선 시도 중 10개 routing 실패와 2개 verifier 실패로 69.460초 후 종료했다. 마지막 후보는 `a=0, b=1, subtract=0`에서 `c=0`이어야 하는 조건을 만족하지 못했다. `target/alu-display/add-sub-4/failure.json`에 보존했으며 이는 논리 전수 검사 성공과 별개인 물리 실패다. 최신 routing contract를 적용한 단 한 번의 재실행은 성공했다. 24개 cell/48개 route/2,906블록이며 PnR 77.012초, NBT 재검증을 포함해 82.135초였다. 512개 정적 입력 전수와 연속 walk 512개를 통과했다. 10번째 시도에서 선택되었으며 앞선 8개 routing 실패와 1개 verifier 실패를 통과로 세지 않았다. candidate cache version 변경으로 3개 정의를 새로 생성하고 21개 같은 instance에서 재사용했으므로 이 측정은 완전한 warm-cache 실행이 아니다. 성공 산출물은 `target/alu-display/add-sub-4-restored-direct/verification.json`이다.

8bit ADD/SUB는 같은 seed와 예산으로 한 번 실행했으나 12개 시도가 모두 배선 단계에서 실패했다. 48개 cell의 candidate는 모두 재사용했고 준비 시간은 1.09ms였으며, 총 136.385초 후 종료했다. 마지막 실패 연결은 `add_sub_bit_50.y → add_sub_bit_52.a0`이고 `target/alu-display/add-sub-8/failure.json`에 기록했다. 성공 NBT는 생성되지 않았으므로 계획한 정적 578개와 연속 walk 578개 물리 검증도 수행되지 않았다. 다음 병목은 전역 배선이며, 앞 절의 8bit ADD 성공을 ADD/SUB 성공으로 해석하면 안 된다.

```sh
# 성공한 4bit 저장 NBT 재검증:
cargo run --release --example add_sub_nbt -- --verify-only 4 target/alu-display/add-sub-4-restored-direct
# 8bit의 현재 실패를 재현하는 동일 설정(성공 명령이 아님):
ADD_SUB_CACHE_DIR=target/alu-display/add-sub-candidate-cache \
  cargo run --release --example add_sub_nbt -- 8 target/alu-display/add-sub-8
```

`add_sub_nbt`는 `ADD_SUB_DIM`(기본 `10x10x5`), `ADD_SUB_BEAM`(기본 256), `ADD_SUB_CACHE_DIR`를 사용한다. 기존 `adder_nbt`의 CLI와 파일 계약은 바꾸지 않았다. 1/2bit는 정적 전수+모든 ordered transition, 4bit는 정적 512개 전수+512개 walk, 8bit를 실행할 경우 directed/random operand pair를 두 연산으로 검사하는 bounded coverage다.
