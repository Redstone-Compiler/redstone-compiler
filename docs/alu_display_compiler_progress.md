# ALU / 표시 컴파일러 구현 기록

기준 브랜치: `codex/8bit-alu-display`. 목표와 인터페이스는 [개발 계획](alu_display_compiler_plan.md), 재현 명령은 [fixture 안내](../test/alu_display/README.md)에 있다.

## 논리 경로

- unsigned sized b/h/d 상수, bit/part select, concat, 조건식, `==`, `>=`, `-`, 입력 포트 표현식을 파싱하고 RTL에서 폭 확장/절단을 명시한다.
- Logical IR에도 비교/선택/연결/resize를 보존한다. 계층은 먼저 펼친 뒤 cell 의존 순서로 처리한다. 독립 child 출력의 교차 연결을 모듈 단위의 가짜 cycle로 거절하던 문제를 회귀 검사한다.
- 버스 bit와 scalar 이름이 충돌하면 생성한 bit 이름을 escape한다. `a[0]`과 scalar `a_0`가 같은 입력으로 합쳐지지 않는다.
- 상수 접기와 구조 공유 후 기본적으로 1gate/2input leaf로 내린다. 작은 cone grouping은 명시적 API의 실험 옵션이다. 조건식 decoder에서 4gate/3input grouping은 112→68 instances지만 일부 leaf의 물리 후보를 찾지 못해 기본값으로 사용하지 않는다.

RTL 독립 정답 검사: decoder 두 표현 각각 32개, Double Dabble 256개, ALU/flags 524,288개 전수 비교가 통과했다. 별도로 생성된 Routable 논리는 decoder32개, 4bit 산술/비교 등512개, 계층16개, 상수/전달4개, 검토 중 발견한 계층·이름 충돌 회귀를 검사한다. `display_u8`는 8개 scalar 입력과21개 출력으로 lowering하고 Logical text 왕복을 검사한다.

## 물리 경로에서 발견한 문제

1. 외부 Cobble 입력의 self hard-power source가 전원 정규화와 attached torch 재평가에서 빠졌다. 외부 ON이 지워져 NOT 후보가 모두 truth 검사에서 탈락했다. 두 계산에 해당 전원을 포함하고 반복 ON/OFF 회귀를 추가했다.
2. input diode의 첫 fanout 가지가 아직 연결하지 않은 다른 sink까지 ON이어야 한다고 요구했다. 합법적인 접촉 위치와 현재 가지의 전원 의무를 분리했다. 두 가지를 차례로 연결하고 ON/OFF를 검사한다.
3. diode의 repeater와 target이 route path에서 빠져 최종 contact 검사에서 정상 연결을 거절했다. 두 adapter 생성 경로에 실제 경로를 기록한다.
4. input support에 붙은 torch의 정상 반응을 foreign short로 판정했다. 허용 범위를 그 support에 물리적으로 붙은 torch로 한정했다. 생성 NOT/AND/OR 후보의 입력 접근과 독립 NOT의 실제 반전, 무관한 인접 torch의 거절을 검사한다.
5. floor-level diode 포트에는 지지 블록을 둘 아래 공간이 필요하다. 생성한 전역 배치에 support layer를 예약한 뒤 명시적 physical intent를 적용한다.

6. 상세 배선의 `ConnectPosition` 판정 방향이 뒤집혀 있었다. terminal→sink 방향으로 고쳤으며 방향성이 있는 repeater 회귀로 검사한다.
7. 전역 배선은 기존 Cobble support를 재사용할 수 있어야 한다. `SupportReusePolicy`를 명시하고 전역 경로에서만 켠다. 기존 local placer의 기본 `FreshOnly`는 유지한다.
8. dust 강도 정규화의 중간 OFF 상태가 이벤트로 전달되고 마지막 ON 복구가 누락됐다. fixpoint 시작/끝의 실제 ON/OFF 변화만 전달한다. 저장된 8bit 가산기의 특정 입력 전이 실패가 NBT 변경 없이 해결됐다.
9. Cobble 전원 카운트를 정정할 때도 consumer에 변화가 전달되지 않았다. 꺼진 Cobble 옆 리피터가 ON으로 남는 작은 회귀를 추가하고, 전체/강한 전원 각각의 ON/OFF 변화만 통지한다. 기존 decoder NBT의 48개 전이 실패가 파일 수정 없이 모두 해결됐다.

실험 옵션인 입출력 diode 예약과 verified n-ary recipe는 생성 후 진리표를 검사한다. 의미상의 단일 minterm/maxterm을 식별하므로 모듈 이름에 의존하지 않는다. literal absorption은 별도 lowering API이며 기본 1gate/2input mapping을 바꾸지 않는다. recipe 출력 위 진입 공간은 `blocked_cells`로 예약하고 cache/복원에도 보존한다. 배선의 불변 `RoutingSearchContext`가 이 예약을 전달한다.

동일 경계에 일렬로 붙은 입력은 먼저 연결한 fanout bus가 나머지 입력을 가둘 수 있다. 실험 옵션 `nary_stagger_inputs`는 입력을 계단식으로 노출하고 repeater로 강도를 복원한다. 4bit minterm을 2/4/10개 공유하는 probe가 각각 8/16/40개 배선을 완료하고 NBT 재읽기 후16개 입력을 검사했다. `PinAccess`는 같은 방향의 깊은 핀을 먼저 연결하고, 입력 앞 접근 공간은 해당 sink만 사용하도록 예약하는 선택적 전략이다. 기본 순서는 유지한다.

내부 fanout은 근처 tap을 먼저 시도하되 실패하면 원래 출력/접근점으로 돌아간다. tap을 제한할 때는 완료한 sink를 제외하고 아직 연결하지 않은 sink까지의 거리로 순위를 매긴다. 두 보완 모두 별도의 실제 신호 회귀가 있다.

꺼진 dust 출력은 기존에 시작 세기를2로 보수적으로 잡았다. 같은 실패 prefix에서2는 막히고15는124칸 경로를 찾았다. `RestoredDirect`는 활성 상태에서15를 보장하는 출력만의 명시적 계약이다. n-ary recipe와 준비된 IO driver는 진리표에서 이를 인증한 뒤 표시한다. 일반 dust/다른 접근점/중간 tap에는 이 약속을 적용하지 않는다. cache v10과 candidate library v2는 계약을 보존한다. 리피터를 기존 dust 옆에 배치할 때 이전 dust의 연결 모양을 갱신하지 않던 상세 배선 오류도 직각 배선 회귀로 수정했다.

## 디코더 관통 성공

`bcd_to_7seg_minimized.v`와 `blank=0`인 일반 RTL wrapper를 컴파일해 `target/alu-display/decoder-digit-final/decoder.nbt`를 생성했다. 실제 기록은 PnR23.308초,20개 cell,62개 배선,5,703블록이다. 16개 정적 입력을 검사하고 저장한 NBT를 다시 읽어 모든256개 입력 전이(연속512회 입력 적용), invalid BCD10~15 OFF, burnout 부재를 확인했다. `verification.json`이 성공 증거다. 전수 검사가 작은 probe에만 한정된 결과가 아니다.

원래 shared-minterm식은18개 cell/90개 연결이었다. 재현 가능한 prime implicant/최소 cover 탐색으로 동등한 POS식(20개 cell/62개 연결)을 만들었다. 최대 fanout은10→11로 증가하므로 모든 지표가 좋아진 것은 아니다. 생성 도구는 `tools/alu_display/minimize_decoder.py`이며 결과는 일반 Verilog이고 컴파일러 이름 기반 특례가 아니다. invalid10~15를 don't-care로 없애지 않았다.

blank 제어를 포함한 전체32상태/1,024전이 NBT는 미완료다. 한 배치는32개 정적 입력과69개 배선을 통과했지만 연속 전이 중 burnout이 검출됐다. runner의 배치 선택 callback에도 전체 전이 검증을 추가해 잘못된 배치를 거절한다. 실제 램프 표시판도 후속 단계다. 현재 성공은 기본 숫자 표시용7개 논리 segment 출력이다.

## 실제 가산기 NBT

Verilog에서 시작해 공통 lowering/PnR/NBT 경로로 2/4/8bit 가산기를 생성했다. 측정값과 재현 명령은 fixture 안내에 있다. 2bit는 cold PnR6.507초,16개 정적 입력과256개 모든 입력 전이, 4bit는 warm7.648초,256개 정적 입력과256개 연속 walk가 통과했다. 8bit는 warm38.709초,3,433블록이며289개 정적 입력과289개 연속 walk를 검사했다. 8bit 검사는 경계 사례와 seed가 고정된 무작위256개를 포함하며 전수 증명은 아니다.

8bit는 처음 연속 walk에서 실패했다. 원인을 simulator의 정규화 이벤트에서 고친 뒤 기존 `adder.nbt`를 수정하거나 재컴파일하지 않고 같은 검사를 통과했다. `verification.json`에는 NBT fingerprint와 `nbt_unchanged`를 남긴다. 전체 ALU와 Double Dabble의 물리 컴파일 성공을 뜻하지는 않는다.

## 공유 ADD/SUB

`test/alu_display/add_sub{1,2,4,8}.v`는 `b XOR subtract`, 초기 carry=`subtract`를 사용하는 하나의 명시적인 carry chain이다. 별개 ADD/SUB의 자동 공유를 가정하지 않는다. 출력 `c`는 덧셈 carry 또는 뺄셈에서 borrow 없음이다. `examples/add_sub_nbt.rs`는 독립 정수 연산으로 검증한다.

- 2bit:12개 cell/24개 배선,1,333블록. PnR6.040초,32개 입력 및 모든1,024개 입력 전이/NBT 재읽기 통과. transition callback이 burnout 배치를 거절한 뒤 다음 배치를 선택했다.
- 4bit:24개 cell/48개 배선,2,906블록. PnR77.012초,512개 입력 전수 및512회 연속 walk/NBT 재읽기 통과. cache v10의3개 unique 후보는 재생성하고21개 instance를 재사용했으므로 완전 warm 측정은 아니다. 모든262,144개 입력 순서쌍 검사는 아니다.
- 8bit:48개 cell, 후보48개 재사용. 동일20,000 expansion 예산과12개 배치/순서 시도에서136.36초 후 배선 실패했다. 성공 NBT/검증 결과가 없으며,8bit 덧셈 단독 성공과 구분한다.

다음 물리 병목은 ADD/SUB의8bit 전역 배선과 decoder의 blank 포함 전이 안정성이다. 전체 ALU의8개 연산/flags 및 Double Dabble/세 자리 표시 연결은 아직 물리 완료가 아니다.

## 배선 복사 비용

실행 중 primitive decoder의 1초 스택 샘플에서85개 중70개가 상세 배선의 World3D deep clone 경로였다. World3D를 plane/행의 두 단계 Arc로 바꾸고 수정한 plane의 행 포인터와 해당 행의 block만 복사한다. clone 격리와 sparse 왕복을 검사했다.

`cargo run --release --example world_clone_benchmark`의 동일64×64×8 world,1,000 clone,각 clone에서 서로 다른 두 행 수정,5회 중앙값 측정은 기존49.039ms / COW1.029ms였다. checksum은 동일하다. 47.68배는 이 복사 microbenchmark 수치이며 전체 PnR 배속이 아니다.

## 검증 범위

release mode만 사용한다. 최종 `cargo test --release --workspace --all-targets`는 core444개 통과/실패0/ignored56개(216.54초)와 workspace 및 integration 검사를 통과했다. `PinAccess`, 출력 강도 계약, 직각 repeater, 두 정규화 알림 회귀를 포함한다. workspace native check에는 nbt-sim-wasm도 포함하며 viewer용 WASM release와 viewer production build도 통과했다. 검색이 무거운 ignored 측정을 전부 실행한 결과로 해석하지 않는다.

전체 탐색 trace와 실패 NBT는 기본적으로 저장소에 추가하지 않는다. 작은 RTL/검증 코드와 요약을 남기고 `target/alu-display/`에 IR, NBT, cache, 검증 결과를 생성한다.

## Exact 후보를 연결한 압축

이후 Global PnR에 opt-in Exact SAT 후보 공급을 연결했다. 실제 SAT 셀6개를 채택한36×64×56 NBT는4,187블록이며16개 입력과 모든256개 전이를 통과했다. 기존2층 대비25.7%, 같은 자유3D의 recipe 대조군4,312블록 대비2.9% 감소다. 최초 cold 실행은89.252초로 더 오래 걸렸고, 모델 최적성이나 전반적인 속도 향상을 주장하지 않는다. 입력 어댑터, provenance, 실패 시 recipe fallback과 비교 조건은 [별도 기록](exact_global_candidates.md)에 있다.

Exact 물리 후보의 내부 Cobble/dust 전력 count는 NBT에 저장되지 않으므로, persistent cache의 world 해시를 NBT 왕복으로 정규화했다(cache v13). geometry/포트/reservation/cost와 실제 블록 변조 검증은 유지한다. 실제 mixed-polarity SAT 후보의 캐시 재로딩 후 전이 검증도 통과했다. 이 확장 직후 전체 release workspace는 core447개/실패0/ignored56개와 나머지 검사를 통과했으며, 이후 cache 해시 보정은 별도 회귀로 검사했다.

## 디코더 전체 monolithic Exact

Verilog를20개 셀로 나누지 않는 `lower_flat_logic_graph()`와 `whole_decoder_exact` 실험을 추가했다. 입력4개·출력7개·NOR28개를 하나의 SAT 모델로 넣고 독립 mask16개를 먼저 확인한다. 자유핀3개 박스/단일worker30초,5×10×5/4worker60초,입력면고정·출력면제한/4worker60초까지 다섯 시도는 모두 UNKNOWN이었다. encoding은0.2~0.5초였고 simulator refinement는0회여서 최초 SAT witness 탐색이 병목이었다. 성공 NBT가 없으며, 불가능 증명으로 해석하지 않는다. [조건·측정·재현 명령](whole_decoder_exact.md)에 별도로 기록했다.
