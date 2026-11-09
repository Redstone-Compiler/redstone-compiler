# Decoder 전체를 한 번의 Exact 모델로 푸는 실험

`examples/whole_decoder_exact.rs`는 기존 최소 cover Verilog와 `blank=0` wrapper에서 시작한다. `LogicalDesign::lower_flat_logic_graph()`가 계층·버스·상수 접기를 기존 bit mapper와 공유하면서 하나의 NOT/OR 그래프를 만든다. 물리 셀 분할, 정형 recipe, Global PnR, window construction은 사용하지 않는다.

입력4개·출력7개를 동시에 모델링하며, 독립적인 segment mask로16개 입력의7개 출력을 먼저 전수 검증한다. 결과는 NOR28개/net32개/서로 다른 net 함수31개이며, 기존 Exact 모델의 함수와 보수 vocabulary는 zero를 포함해51종이다. 임의의 모든 Boolean 중간함수를 발명하는 무제한 회로 합성이나 전역 최적성 증명을 뜻하지 않는다.

## 조건

- seed1, `optimize=false`, rank/stage24, 블록 수 추가 상한 없음
- 모든 출력은 박스 바깥을 구동하는 관찰 지점이어야 함
- 기본 실험은 입력·출력 위치를 자유롭게 선택
- solver 제한은 encoding 이후부터 적용되므로 인코딩/탐색/전체 wall 시간을 별도로 기록
- 외부 실행기는30초 solver 실험에90초,60초 실험에120초 wall 상한을 적용
- 해를 찾으면 NBT 재읽기 후16개 fresh 입력·모든256개 순서쌍 전이·burnout 부재를 독립 검증

`UNKNOWN`은 시간 내 해도 불가능 증명도 얻지 못했다는 뜻이다. `INFEASIBLE`과 구분하며, 성공 NBT로 표시하지 않는다. 각 시도는 새 산출물 디렉터리를 사용한다. `attempt.json`은 상태·변수·절·시간·인터페이스 제약을 보존한다.

## 초기 자유 배치 측정

| 박스 | worker | solver 제한 | 변수 | 절 | encoding | 결과 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
|3×8×4|1|30초|66,907|451,097|196ms|UNKNOWN, wall30.201초|
|4×8×4|1|30초|86,352|597,336|235ms|UNKNOWN, wall30.239초|
|4×10×5|1|30초|139,896|995,256|404ms|UNKNOWN, wall30.466초|
|5×10×5|4|60초|171,354|1,235,624|529ms|UNKNOWN, wall60.647초|

모두 simulator refinement0회, last rejection 없음이었다. 즉 찾은 배치를 simulator가 거절한 것이 아니라, 제한 시간 내 최초 SAT 배치를 찾지 못했다. 이 결과로 박스가 반드시 작다거나, 전체 decoder의 monolithic 합성이 불가능하다고 판단하지 않는다.

## 재현

```sh
cargo run --release --example whole_decoder_exact -- \
  target/alu-display/whole-decoder-exact-new 3x8x4 30

WHOLE_WORKERS=4 cargo run --release --example whole_decoder_exact -- \
  target/alu-display/whole-decoder-exact-portfolio-new 5x10x5 60

WHOLE_WORKERS=4 WHOLE_PIN_CONTRACT=faces \
  cargo run --release --example whole_decoder_exact -- \
  target/alu-display/whole-decoder-exact-faces-new 5x10x5 60
```

`faces`는 입력4개를 `(1,0,1),(3,0,1),(1,0,3),(3,0,3)`에 North 방향 부착으로 고정하고, 출력7개는 모두 y-max 면에서 위치를 고른다. 내부 전체는 여전히 한 SAT 모델이다. 자유 배치보다 추가적인 인터페이스 제약이므로 서로 다른 문제로 명시한다.

최종 `5×10×5`/4-worker/60초 `faces` 실험도 UNKNOWN이었다. 변수122,273개·절894,912개, encoding425ms·solve60.028초·wall60.457초, refinement0회/last rejection 없음이다. 자유 핀보다 변수·절은 줄었지만 최초 witness를 얻지는 못했다. 다섯 실험 모두 성공 NBT가 없다.

기존4,187블록 결과는 개별 Exact 후보와 Global PnR로 만든 별도 체크포인트이며, 이 전체 모델에서 찾은 결과가 아니다.
