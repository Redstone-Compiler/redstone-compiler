# Exact 후보를 사용하는 전역 PnR 압축

기준: `codex/8bit-alu-display`, draft PR #69. 2026-11-09.

## 구현

`UnitCandidateConfig.exact: Option<ExactCandidateConfig>`를 켜면 각 작은 조합 leaf에 실제 `ExactLocalPlacer` SAT 탐색과 블록 최소화를 수행한다. 기본값은 `None`이며 기존 경로를 유지한다. 기본 탐색 상자는3×5×3, 입력 최대4개, 원본 logic node 최대16개, solver 제한은 셀당5초다. worker와 seed는 각각1이다. 시간 제한은 solver 시간이며 인코딩·어댑터·검증 시간까지 포함하는 총 실행 제한은 아니다.

배치용 double-NOT 버퍼를 논리적으로 제거한 뒤 NOR로 내린다. Exact의 출력은 박스 밖을 구동할 수 있어야 한다. 원래 스위치가 강하게 구동하던 solid를 외부 입력 대상으로 보존하고, 입력마다 격리 diode를 붙인다. 원본 스위치 좌표의 단락만 보고 공유 support를 부당하게 거절하지 않고, 격리된 최종 경계에서 입력 독립성을 검사한다.

완성된 후보는 NBT 재읽기 후 진리표, 작은 셀의 모든 입력 전이, 입력 역류, 출력 역구동과 `RestoredDirect` 강도를 독립 simulator로 검사한다. 실패·시간 초과·범위 초과는 기존 검증된 recipe를 사용한다. 후보끼리는 어댑터를 포함한 비용으로 경쟁한다. `exact_core`는 실제 채택된 SAT 후보의 출처, 박스, 스위치를 포함한 물리 원본 블록 수, 시간과 최적성 여부를 보존한다. 최종 `selected-candidates.json`과 `verification.json.exact_selected`로 SAT 실행과 최종 채택을 구분한다.

`ExactPlacement.block_count`는 입력 스위치를 제외하므로, recipe의 전체 물리 블록 수와 직접 비교하지 않는다. 보고하는 최종 후보 비용에는 스위치 제거와 외부 인터페이스가 반영된다. 제한 시간 내 해를 찾았다는 사실은 최소 크기 증명이 아니다.

## 실제 전체 decoder 측정

동일한 최소 cover RTL과 `blank=0` wrapper, 입출력 예약, PinAccess, A*20,000, 경로 길이256을 사용했다. 전부 최종 NBT의16개 입력, invalid BCD OFF, 모든256개 전이(연속512회 입력 적용), burnout 부재를 통과했다.

| 설정 | NBT 크기(X×높이×Z) | 블록 | 경계 상자 부피 | Exact 채택 |
| --- | --- | ---: | ---: | ---: |
| 기존2층 recipe |95×16×95|5,638|144,400|0|
| 자유3D recipe 대조군 |41×64×51|4,312|133,824|0|
| 자유3D + Exact |36×64×56|4,187|129,024|6/20|

기존2층 대비 블록25.7%, 바닥 면적77.7%, 부피10.6% 감소다. 같은 자유3D에서 Exact만 비교하면 블록2.9%, 부피3.6% 감소다. 높이는16→64로 증가하므로 모든 축이 짧아진 결과는 아니다. 더 이른 평면95×7×95 결과와 비교하면 총 경계 상자 부피는 증가한다.

자유3D 대조군은20.023초, Exact는 cold89.252초(후보 준비70.949초, 전역 단계18.30초)였다. 후보 재사용과 한 번씩의 측정이며 반복 평균이나 보편적인 속도 향상을 주장하지 않는다. Exact를 켠 최초2층 실험은 셀이 작아졌는데도 전체5,823블록으로 늘었다. 따라서 내부 셀 크기만으로 전체 압축 효과를 판단하지 않는다.

## 재현

```sh
EXACT_CANDIDATES=1 EXACT_BOX=3x5x3 EXACT_MS=5000 \
BIT_LITERAL=1 NARY_STAGGER=1 PNR_PIN_ACCESS=1 PNR_FREE3D=1 \
PNR_LAYOUT_COMBINATIONS=4 PNR_ASTAR_EXPANSIONS=20000 PNR_MAX_STEPS=256 \
  cargo run --release --example decoder_nbt -- \
  target/alu-display/decoder-exact-final \
  test/alu_display/bcd_to_7seg_minimized.v \
  test/alu_display/bcd_digit_to_7seg.v
```

`PNR_CANDIDATE_CACHE`로 후보 cache 경로를 따로 지정할 수 있다. 같은 출력 경로로 재실행해도 cache를 사용한다. `PNR_LAYERED=1 PNR_LAYER_SPACING=2`는 자유3D 대신 작은 층 간격을 실험하는 옵션이다. 대조군은 `EXACT_CANDIDATES`만 생략한다.

검증된 최종 fixture는 `test/alu_display/nbt/alu-display-decoder-exact-verified.nbt`에 보존한다. 아직 blank 제어를 포함한 전체 decoder와 ALU 물리 완료를 뜻하지 않는다.
