# Exact placer: 유도한 제약과 반복 실험

기본 명세: [exact_placer_constraint_formula.md](exact_placer_constraint_formula.md).
기준 모델: `3e9a4cc`의 `exact_placer.rsdsl`; 명세 문서 커밋: `bbb4d13`.
이 파일에는 기본 물리 규칙이 아니라 **기존 모델에서 따라오는 제약**, 증명 근거,
반례, 실제 측정 결과를 기록한다. 현재 실험 조건을 production 모델에 추가하지 않았다.

## 검증 기준

```text
B(I,E) ≔ 기존 rsdsl 모델의 조건
L(I,E) ≔ 추가하려는 조건

B ∧ ¬L = UNSAT  ⇒  B ⇒ L  ⇒  Models(B) = Models(B ∧ L)
B ∧ ¬L = SAT    ⇒  반례를 조사하고 조건을 수정하거나 폐기
B ∧ ¬L = UNKNOWN ⇒ 해당 검사로는 판정하지 못함
```

보존 대상은 부품 배치뿐 아니라 기존 보조 변수까지 포함한 기존 모델의 해다.
새 CNF의 보조 변수는 존재 양화해서 비교한다. Minecraft 전체 배치에 대한
완전성은 이 검증으로 증명되지 않는다. 모든 soundness guard가 활성화된 모델을 전제로 한다.

## L1 — 블록으로 들어가는 신호의 함의

```text
a ≼ b  ⇔  ∀q∈Case : on(a,q) ⇒ on(b,q)

Feeds(u,v,f,SolidSink) ∧ Sig(u)=a ∧ Sig(v)=b  ⇒  a ≼ b
```

블록에 전원을 주는 쪽이 켜졌는데 블록이 꺼지는 입력 조합이 하나라도 있으면,
그 두 신호를 해당 연결 양끝에 배정할 수 없다. 블록은 여러 입력을 OR로 합칠 수
있으므로 `a=b`까지 요구하면 안 된다.

유도: 기존 soundness가 모든 q에 대해 `Feeds ∧ Powered(q,u) ⇒ Powered(q,v)`를
요구한다. 양끝의 신호 정의를 대입하고 q 전체를 묶으면 위 식이다.
SolidSink에는 Cut 예외가 없고, 블록→가루의 Strong 특례도 해당하지 않는다.

Z3 검증 스키마: 임의의 Boolean `F,a,b`에 대해
`(F∧a⇒b) ∧ F∧a∧¬b`는 UNSAT. 이 결과를 각 q에 적용하는 유도는 입력 그래프나
상자의 크기에 의존하지 않는다. Z3 proof 텍스트도 출력하지만 독립 proof checker나
Lean으로 검사한 결과는 아니다.

## L2 — 일반 배선 양끝의 함수 동치

```text
a ≡ b  ⇔  ∀q∈Case : on(a,q) = on(b,q)

Feeds(u,v,f,j) ∧ j≠SolidSink
  ∧ ¬(f=FromSolid ∧ j=DustSink)
  ∧ ¬Cut(u,v) ∧ Sig(u)=a ∧ Sig(v)=b
  ⇒ a ≡ b
```

일반적인 가루·리피터 수신 연결은 보내는 쪽과 받는 쪽의 함수가 같아야 한다.
Class 이름이 다른 것만으로 배제하지 않고 **각 입력 조합의 값**을 비교한다.

유도: 기존 soundness의 정방향과 역방향 함의를 합치면 입력 조합마다 동치다.
Cut에서는 그 조건이 적용되지 않는다. 블록→가루에서는 보내는 값이 블록의 전체
전원 대신 Strong이므로 반드시 제외해야 한다.

Z3 검증 스키마: `(F∧¬cut⇒a=b) ∧ F∧¬cut∧a≠b`는 UNSAT.

## 폐기한 일반화 — 예외를 없애면 안전하지 않다

두 반례는 국소 Boolean 전제의 반례이며, 완성된 Minecraft 배치를 생성한 결과는 아니다.

```text
Cut 무시:
  F=true, cut=true, source=true, target=false
  기존 조건은 만족하지만 “연결되면 함수가 같다”는 거짓.

블록의 전체 전원을 강전원과 동일시:
  F=true, blockPower=true, strong=false, dustPower=false
  strong=dustPower, strong⇒blockPower를 만족한다.
  하지만 blockPower=dustPower는 거짓.
```

특히 순차회로를 다루면서 Cut을 빼먹으면 유효한 상태 표현을 잘라낼 수 있다.

## 빠른 반복 방법

Python이 `.rsdsl` 후보를 만들어 기존 release `ground` 실행 파일에 전달한다.
Rust를 반복 수정하지 않고, 후보 식은 `tools/pruning_lab/run.py`의 `BAD`에 추가한다.
`BAD`에는 금지하고 싶은 상황을 작성한다. 검증 모델은 `B ∧ BAD`, 추가 모델은
`B ∧ ¬BAD`로 자동 생성된다.

```sh
cargo build --release -p rsdsl --example ground
python3 -m venv /private/tmp/redstone-pruning-venv
/private/tmp/redstone-pruning-venv/bin/pip install -r tools/pruning_lab/requirements.txt
/private/tmp/redstone-pruning-venv/bin/python tools/pruning_lab/run.py \
  --seeds 5 --timeout-ms 5000 --proof-timeout-ms 30000
```

이미 만든 venv와 release binary는 다음 실행에서 그대로 재사용한다.
모델을 바꾸고 Python 명령만 다시 실행하면 된다. 결과·생성 모델·CNF·ground 로그·
스키마 proof는 `target/pruning-lab/`에 쌓인다. SAT 반례가 나오면 중단하고,
검증 타임아웃은 UNKNOWN으로 기록한다.

## 실험 범위

- Z3로 **ground된 CNF의 첫 SAT/UNSAT 판정**을 측정했다. production CaDiCaL의
  병렬 탐색·최적화·시뮬레이터 재검증을 포함한 end-to-end 성능이 아니다.
- 입력은 Python으로 생성한 inverter, 2블록 상한 inverter, NOR, RS latch다.
  RS latch는 `nq=¬(s∨stored)`, `next=¬(r∨nq)`, `cut(next,stored)`로 만들었다.
  이 인스턴스 생성기는 production의 vocabulary/instance builder를 그대로 재현하지 않는다.
- 각 모델에 입력 함수와 보수를 포함한다. 같은 진리표의 이름이 중복될 수 있어
  Class 이름의 동일성과 함수의 동치를 혼동하는 오류도 노출한다.
- 먼저 일반 Boolean 스키마를 검사하고, 각 실제 생성 인스턴스에도 `B∧¬L`을
  따로 ground해서 검사했다. 작은 인스턴스의 UNSAT만으로 모든 입력을 일반화하지 않았다.
- seed 1–5, 1개 Z3 프로세스/실행. 증명 검사나 앞선 실행의 내부 상태가 다음 측정에
  영향을 주지 않게 새 프로세스에서 시작한다. solve 시간과 CNF 로드 시간은 분리한다.
- 5개 seed의 중앙값은 탐색적 측정이다. 작은 시간 차이나 한 회로의 결과를 전체
  placer의 성능 개선으로 주장하지 않는다.

## 측정 결과

<!-- results:start -->
Z3 5.3.3; 기존 모델 SHA-256: `1ac2de3341256496f20edda514c91ee195208f3c64e80a9a548af5ef131b4ac6`. 원시 결과는
[results.json](../tools/pruning_lab/results.json)에 저장했다. 단위는 초이며 5개 seed의 solve 중앙값이다.

| 인스턴스 | 기존 | L1 추가 | L2 추가 | 둘 다 추가 | 기존 절 수 → 둘 다 |
|---|---:|---:|---:|---:|---:|
| inv | 0.0017 | 0.0018 | 0.0019 | 0.0021 | 2,242 → 2,918 |
| inv_tight | 0.0017 | 0.0019 | 0.0020 | 0.0021 | 2,266 → 2,942 |
| nor | 0.0389 | 0.0387 | 0.0419 | 0.0453 | 18,020 → 29,060 |
| rs_latch | 0.4341 | 1.1005 | 0.8101 | 0.4606 | 24,962 → 40,894 |

기존·추가 모델 모두 inv/NOR/RS latch는 SAT, inv_tight는 UNSAT였다.
4개 인스턴스 × 2개 lemma의 부정 검사는 모두 UNSAT였다. 이미 UNSAT인 inv_tight의
검사는 공허하게 성립하므로, SAT인 나머지 3개 인스턴스의 결과와 구분해야 한다.
가장 느린 반례 검사는 RS latch의 L1로 14.17초였다. 검증 시간은 위 성능 표에 포함하지 않았다.

**판정: 해 보존 근거는 확보했지만, 중복 추가 방식은 채택하지 않는다.**
특히 RS latch는 L1 단독 약 2.53배, L2 단독 약 1.87배 느려졌다.
둘 다 추가해도 약 1.06배이며, 절은 약 64% 증가했다. seed별 변동도 크다.
현재 자료에서는 속도 개선을 주장할 수 없고 production 모델은 수정하지 않았다.
<!-- results:end -->

## 다음 실험

두 조건은 기존 per-case soundness를 신호 함수 수준에서 다시 묶은 것이다.
따라서 “추가해도 안전함”과 “추가하면 빨라짐”은 별개다. 같은 정보를 CNF로
중복 추가하면 오히려 느려질 수 있다.

후속 후보는 작은 국소 구조의 모순을 짧은 절로 요약하는 방향이 적절하다.
예를 들어 토치의 출력이 특정 경로로 자기 받침을 구동하는 구조를, Cut·Given·
강전원 예외를 유지하며 조기에 배제하는 식을 찾을 수 있다. 아직 검증된 lemma로
취급하지 않는다. 원래 soundness를 제거하는 대체 인코딩 실험도 별도로 양방향
동치를 검증해야 한다.


## 후보 묶음 — 48개, 아직 검증·측정하지 않음

기존 L1/L2와 별도로 새 실험 후보를 48개 모았다. 식은 연구용 축약 표기이며
바로 실행되는 rsdsl은 아니다. **C01–C32도 아직 proven lemma가 아니다.**
일반 증명·전체 모델 반례 검사·성능 검사를 통과해야 채택한다.
기계가 읽을 수 있는 동일 목록은 [candidates.json](../tools/pruning_lab/candidates.json)에 있다.

- C01–C32: 기존 모델에서 함의되는지 검증할 추가 제약 32개.
- C33–C40: 표현·분해·대칭을 바꾸는 변환 8개. `B⇒L`만으로는 검증할 수 없다.
- C41–C48: 반례를 찾아야 하는 위험 조건 8개. 성능이 좋아도 검증 없이 채택하지 않는다.

우선순위 P0는 먼저 실험할 가치가 있다는 판단이다. 안전성이나 성능 보장이 아니다.

축약: `sig(u)`는 u의 신호 함수, `P(q,u)`는 q에서의 전원 상태,
`EqEdge`는 L2의 모든 예외 조건을 만족하는 활성 연결이다.
`RankEdge`와 `StagePath`는 실제로 기존 순서 제약이 적용되는 간선만 포함한다.
`DomainSig`, `Possible…`, `Forced…`는 향후 구현할 부분 배정 분석 결과이며,
원래 가능한 값을 임의로 버리지 않는다는 검증도 필요하다.

### 먼저 돌릴 순서

1. **C17 → C18 → C13:** 짧은 순환을 작은 금지절로 요약. 우선 길이 2–3만.
2. **C12 → C32 → C04 → C07:** 토치 받침·여러 원천이 만나는 국소 충돌.
3. **C27 → C30:** 핀 자리와 블록 예산이 부족한 compaction 창.
4. **C35:** 첫 실험의 교훈을 반영해 중복 추가 대신 일부 인코딩을 대체.
5. **C40 → C39:** 경계 가정에 대한 UNSAT 학습, 이후 완전 분해.

동일한 결과를 내는 후보를 모두 한꺼번에 추가하지 않는다. 개별 측정 후
성공한 후보만 조합하고, CNF 크기·ground 시간·solve 시간·conflicts·timeouts를 함께 본다.
작은 SAT, 작은 UNSAT, RS latch, D latch, DFF, fixed/Given 창, timing, tiling,
중복 truth-table Class 인스턴스로 검증 범위를 넓힌다. 특히 이미 UNSAT인 문제만으로
검증하지 않는다.

### 추가 제약 후보

#### C01 · 토치·받침 함수 호환 · P0

```text
T(t,a) ∧ Sig(t)=x ∧ Sig(support(t))=y ⇒ ∀q:on(x,q)=¬on(y,q)
```

- 전제: 모든 Case를 비교한다. 이름이 다른 동일 함수도 허용한다.
- 검증: 토치 반전 규칙에 Class 정의를 대입; B∧¬L 검사.
- 기대: 토치를 놓는 순간 불가능한 신호 쌍을 제거.

#### C02 · 받침을 공유하는 토치 · P0

```text
T(t₁)∧T(t₂)∧support(t₁)=support(t₂) ⇒ sig(t₁)≡sig(t₂)
```

- 전제: 고정된 두 받침 좌표가 실제로 같아야 한다.
- 검증: C01을 두 번 적용; 두 토치 배치 반례 검사.
- 기대: 공유 받침 주변의 서로 다른 토치 신호를 조기 배제.

#### C03 · 같은 원천의 두 배선 · P1

```text
EqEdge(u,v)∧EqEdge(u,w) ⇒ sig(v)≡sig(w)
```

- 전제: EqEdge는 비블록 수신, 비Cut, 블록→가루 제외.
- 검증: L2의 전이성; 모티프 SAT 검사.
- 기대: 분기된 배선의 신호 선택을 동기화.

#### C04 · 한 배선에 들어오는 충돌 · P0

```text
EqEdge(u,w)∧EqEdge(v,w) ⇒ sig(u)≡sig(v)
```

- 전제: 실제로 활성인 두 연결에만 적용한다.
- 검증: L2로 공통 수신 함수와 같음을 유도.
- 기대: 서로 다른 신호를 가루 한 칸에 합치는 배치를 일찍 차단.

#### C05 · 강전원은 전체 전원의 부분 · P1

```text
Hq(b) ⇒ P(q,b)
```

- 전제: Strong의 각 원천에 대응하는 실제 Feeds가 생기는지 기하 규칙까지 확인.
- 검증: Strong 정의를 원천별 분해하고 SolidSink soundness 적용.
- 기대: Strong과 Powered 사이의 직접 전파 추가.

#### C06 · 블록→가루의 약한 함수 하한 · P0

```text
F(b,d,FB,DS)∧¬Cut(b,d) ⇒ ∀q:P(q,d)⇒P(q,b)
```

- 전제: 가루와 블록이 같다는 강한 조건은 쓰지 않는다.
- 검증: 가루=Strong, C05 적용; weak-only 반례를 통제 사례로 사용.
- 기대: L2에서 제외했던 블록→가루에도 안전한 부분 제약 제공.

#### C07 · 같은 블록이 구동하는 가루 · P0

```text
F(b,d₁,FB,DS)∧F(b,d₂,FB,DS)∧¬Cut(b,d₁)∧¬Cut(b,d₂) ⇒ sig(d₁)≡sig(d₂)
```

- 전제: 두 가루 모두 동일한 Strong(b)를 읽는다.
- 검증: Strong soundness의 두 동치를 연결.
- 기대: 공통 블록 주위 여러 가루의 함수 선택을 함께 줄임.

#### C08 · 꺼진 강전원 원천 집합 · P1

```text
F(b,d,FB,DS)∧¬Cut(b,d)∧(∀u∈StrongSources(b):¬P(q,u)) ⇒ ¬P(q,d)
```

- 전제: StrongSources는 전체 전원 원천이 아니라 Strong 정의의 원천만 포함.
- 검증: Strong 정의와 power 규칙의 역방향 적용.
- 기대: 필요한 강전원 공급이 불가능한 배치를 조기 차단.

#### C09 · 블록 전원의 정확한 OR · P1

```text
B(b)∧¬Given(b) ⇒ P(q,b) ⇔ ⋁{F(u,b,f,BS)∧P(q,u)}
```

- 전제: Given에는 coverage가 없으므로 제외.
- 검증: 정방향은 coverage와 J⇒F, 역방향은 soundness.
- 기대: 선택된 Contrib를 거치지 않는 전원 전파 보조식.

#### C10 · 원천 하나뿐인 블록 · P0

```text
B(b)∧¬Given(b)∧F(u,b,f,BS)∧NoOtherFeed(b,u) ⇒ sig(b)≡sig(u)
```

- 전제: NoOtherFeed는 다른 Feeds가 모두 비활성이라는 명시적 조건.
- 검증: C09에서 OR 항을 하나로 축소.
- 기대: 고립된 받침의 신호를 즉시 고정.

#### C11 · 켜짐을 설명할 마지막 후보 · P1

```text
B(b)∧¬Given(b)∧P(q,b)∧AllOtherWitnessesFalse(e,q) ⇒ J(e)∧P(q,src(e))
```

- 전제: e는 b로 들어오는 BS 후보이며 다른 J∧P 항들을 모두 포함해야 한다.
- 검증: coverage의 단위 전파를 짧은 조건부 절로 재표현.
- 기대: 큰 coverage OR의 마지막 선택을 빠르게 전파.

#### C12 · 켜진 토치의 받침에는 전원 금지 · P0

```text
T(t)∧F(u,support(t),f,BS) ⇒ ∀q:¬(P(q,t)∧P(q,u))
```

- 전제: 받침은 실제 토치 부착 방향으로 계산.
- 검증: 토치 반전과 SolidSink soundness를 합성.
- 기대: 토치와 그 받침 공급 신호의 겹치는 참 비트를 조기 차단.

#### C13 · 토치→가루→자기 받침 · P0

```text
T(t)∧EqEdge(t,d)∧F(d,support(t),FD,BS) ⇒ ∀q:¬P(q,t)
```

- 전제: 무조건 배치 금지가 아니라 먼저 항상 꺼짐을 유도. 금지는 ∃q:P(q,t)가 추가로 필요.
- 검증: C12와 L2 합성; Cut가 있는 변형을 반드시 비교.
- 기대: 작은 자기 피드백 모티프를 몇 개의 절로 차단.

#### C14 · 홀수 반전 고리 · P1

```text
OddInversionCycle(t₁,…,tₖ) ⇒ false
```

- 전제: 각 support(tᵢ)의 전원이 이전 토치 출력과 정확히 같아야 함. Cut 경계 제외. OR 경로는 동치로 취급 금지.
- 검증: x₁=¬x₂,…의 홀수 순환 Boolean 증명 후 모티프 검사.
- 기대: 3개 토치 등 짧은 반전 고리를 조기 제거.

#### C15 · 보수 함수가 없는 토치 자리 · P1

```text
T(t,a) ⇒ ∃x∈DomainSig(t),y∈DomainSig(support(t)):x≡¬y
```

- 전제: DomainSig는 현재 분기의 가능한 함수들을 과대근사한 집합이어야 함.
- 검증: C01의 존재 양화와 도메인 삭제의 건전성 증명.
- 기대: 배치 위치의 함수 호환성이 비면 토치 선택을 제거.

#### C16 · 설명 가능한 전원 후보가 없는 배선 · P1

```text
(D(c)∨R(c))∧¬Given(c)∧¬Zero(c) ⇒ ∃e∈CandidateIncoming(c):J(e)
```

- 전제: 후보 제거는 Order·함수 호환 조건의 증명된 결과만 사용. Cut sink도 포함.
- 검증: coverage에 후보별 불가능성 증명을 대입.
- 기대: 전원 연결의 가능성 자체가 없는 가루·리피터 위치 제거.

#### C17 · 두 기여 간선의 rank 순환 · P0

```text
RankEdge(u,v)∧RankEdge(v,u) ⇒ false
```

- 전제: RankEdge는 J 활성, 수신 비Given, 수신≠RepeaterSink, rankMax>0인 간선.
- 검증: ru<rv ∧ rv<ru의 모순.
- 기대: 가루 양방향 기여를 단일 짧은 금지절로 표현.

#### C18 · 엄격한 stage 간선을 포함한 순환 · P0

```text
StageCycle ∧ HasStrictEdge ⇒ false
```

- 전제: 모든 간선에 실제 stage 순서 제약이 있어야 함. Cut·Given 예외를 그대로 반영.
- 검증: ≤의 전이성과 적어도 하나의 <를 결합.
- 기대: 리피터·토치를 포함한 짧은 순환의 조기 차단.

#### C19 · rank 상한에 닿은 칸의 기여 · P2

```text
RankEdge(u,v) ⇒ ru<rankMax ∧ rv>0
```

- 전제: rankMax>0. rank를 올리지 않는 리피터 수신에는 적용 금지.
- 검증: 기존 유계 정수 부등식의 끝점 조건.
- 기대: 상한/하한 리터럴로 간선 선택을 빠르게 막기.

#### C20 · 긴 rank 경로 금지 · P1

```text
RankPath(v₀,…,vₖ) ∧ k>rankMax ⇒ false
```

- 전제: 경로의 각 간선이 RankEdge.
- 검증: 엄격 증가 정수열 길이 하한.
- 기대: 구불구불한 기여 경로를 상한 초과 전에 차단.

#### C21 · stage 비용 경로 금지 · P1

```text
StagePath(p) ∧ Σδ(e)>stageMax ⇒ false
```

- 전제: δ는 기존 제약에서 <면 1, ≤면 0. 순서 제약 없는 경계는 경로에서 제외.
- 검증: 정수 차분 제약 합산.
- 기대: 깊은 우회 경로의 배치 선택을 조기 제거.

#### C22 · 짧은 경로의 transitive shortcut · P2

```text
RankEdge(u,v)∧RankEdge(v,w) ⇒ ru+2≤rw
```

- 전제: 선택된 두 간선의 조건을 모두 antecedent에 유지.
- 검증: 차분 제약 합산.
- 기대: 연속된 작은 순서 조건을 직접 연결.

#### C23 · 부분 배치의 전방 단계 하한 · P1

```text
ForcedStagePath(root,c,w) ⇒ stage(c)≥w
```

- 전제: root의 하한은 최소 0. 현재 분기의 강제된 간선만 사용.
- 검증: 경로 합산; 분기 가정을 절에 보존.
- 기대: 고정 셀 주변의 stage 하한을 미리 전파.

#### C24 · 출력 기한에서 역방향 상한 · P1

```text
timing∧Obs(o,v)∧delay(o,d)∧StagePath(u,v,w) ⇒ stage(u)≤d−w
```

- 전제: 해당 Obs와 경로가 실제 활성일 때만.
- 검증: 출력 기한과 경로 차분 제약 합성.
- 기대: 출력에서 멀리 떨어진 느린 후보를 빠르게 차단.

#### C25 · 최단 지연으로 출력 불가능 판정 · P1

```text
timing∧RequiredOutput(o)∧AdmissibleDelayLB(o)>deadline(o) ⇒ false
```

- 전제: LB가 모든 합법 경로에 대한 하한이어야 함. 기존 휴리스틱 거리 값을 증명 없이 사용 금지.
- 검증: 낙관적 완화 문제의 하한을 증명 후 full 모델 반례 검사.
- 기대: 타이밍상 불가능한 영역·핀 선택을 묶어서 제거.

#### C26 · 타이밍 모드의 전체 연결 순환 · P1

```text
timing∧Cycle(allFeedsAndTorchEdges)∧Σδ>0 ⇒ false
```

- 전제: timing은 선택된 J뿐 아니라 모든 Feeds를 순서화한다. 현재 timing 규칙에는 Cut 예외가 없음.
- 검증: Timing과 TorchRule/Given 토치 보강에서 차분 합산.
- 기대: J가 선택되기 전부터 불가능한 순환 배치 제거.

#### C27 · 입력 스위치 자리의 Hall 하한 · P0

```text
∀A⊆RequiredInputs: |⋃{PossibleSwitchCells(k):k∈A}|≥|A|
```

- 전제: 서로 다른 입력 Class의 스위치는 같은 칸을 공유할 수 없음. 위치 도메인을 과소평가하지 말 것.
- 검증: 입력 정확히 하나와 Kind의 상호 배타성에서 matching 조건 유도.
- 기대: 특정 면에 입력이 몰린 배치·창의 불가능성을 조기 증명.

#### C28 · 서로 다른 출력의 자리 하한 · P1

```text
∀A⊆PairwiseIncompatibleOutputs: |⋃{PossibleOutputCells(o):o∈A}|≥|A|
```

- 전제: 같은 셀에서 함께 Obs가 될 수 없는 출력들에만. 같은 함수의 출력은 공유 가능.
- 검증: Obs의 동시 가능성을 먼저 검증하고 Hall 하한 적용.
- 기대: 출력 핀 공간이 부족한 창을 조기 배제.

#### C29 · 서로 다른 토치 함수의 받침 수 · P1

```text
DistinctRequiredTorchFunctions(S) ⇒ #Solid≥|S|
```

- 전제: 그 함수들의 토치가 실제로 반드시 필요하다는 근거가 있어야 함. 입력 netlist 노드 수를 그대로 사용 금지.
- 검증: C02와 서로 다른 함수 토치의 존재에서 단사 대응 유도.
- 기대: 정당화된 구조 하한으로 더 작은 블록 예산을 증명.

#### C30 · 남은 블록 예산과 필수 자리 · P0

```text
FixedOccupied(n)∧DistinctMandatoryNewCells(m) ⇒ n+m≤maxBlocks
```

- 전제: 새 필수 셀 집합의 중복과 고정 셀 겹침을 제거. 조건부 필수성의 가정을 유지.
- 검증: 블록 수 상한의 부분합.
- 기대: compaction 창에서 남은 예산이 부족한 선택 차단.

#### C31 · 함수 호환성의 arc consistency · P1

```text
SelectedEdge(u,v) ⇒ DomainSig(u)⊆{a:∃b∈DomainSig(v),Compatible(a,b)}
```

- 전제: Compatible은 L1/L2/C01/C06 등 증명된 관계만 사용. Cut·Strong별 구분 필수.
- 검증: 각 제거값에 B∧선택∧Sig=a의 UNSAT 근거를 생성.
- 기대: CNF에 가기 전 또는 lazy 단계에서 불가능한 Class 값 축소.

#### C32 · 스위치가 붙은 받침의 토치 · P0

```text
W(u,a,k)∧T(t,b)∧support(u)=support(t) ⇒ ∀q:¬(on(k,q)∧P(q,t))
```

- 전제: 두 부품이 같은 실제 블록에 붙은 경우.
- 검증: 스위치→받침 soundness와 토치 반전 합성.
- 기대: 입력 위치 선택 직후 토치 함수의 많은 후보 제거.

### 변환·분해 후보

#### C33 · 동일 함수 Class 합치기 · P1

```text
Class a,b를 semantic-equivalence별 대표로 치환
```

- 전제: on만 같으면 부족. cut, input, output, given, inc/dec 및 모든 Class 참조를 일관되게 변환.
- 검증: 원래→새 표현과 새→원래 표현의 사상 증명; observable 배치 투영 비교.
- 기대: 함수 중복이 많은 인스턴스의 변수·절 감소.

#### C34 · 신호 함수의 비트벡터 표현 · P2

```text
Sig의 one-hot 선택 대신 truth-table 또는 class-index bitvector 사용
```

- 전제: 허용 vocabulary 밖의 함수가 추가되지 않도록 membership 조건 필요.
- 검증: 인코딩 간 양방향 대응과 decode 검증.
- 기대: 큰 vocabulary에서 변수 수 절감 가능; propagation 악화 가능.

#### C35 · L1/L2로 soundness 일부 대체 · P0

```text
ordinary soundness ↔ class compatibility; Strong/Cut 부분은 보존
```

- 전제: 추가가 아니라 기존 부분 제거. 같은 신호 배정에 대해 정확한 동치 필요.
- 검증: B전체∧¬B새, B새∧¬B전체 모두 검사; 보조 변수는 존재 양화.
- 기대: 직전 실험의 중복 절 증가를 피할 가능성.

#### C36 · 고정된 equality 배선 성분 합치기 · P1

```text
ForcedEqEdges의 연결 성분에 신호 변수를 하나로 공유
```

- 전제: 선택 가능한 연결을 고정된 연결처럼 합치면 안 됨. Cut/Strong 예외 제외.
- 검증: 성분 내 함수 동치와 원래 라벨로의 확장 가능성 증명.
- 기대: 고정된 배선의 반복 신호 변수를 제거.

#### C37 · 대칭인 입력의 교환 대표 · P2

```text
입력 순열 π에 대해 lex-leader 추가
```

- 전제: 회로 함수, 입출력 이름 계약, 핀, fixed, 상태 의미까지 보존하는 automorphism만.
- 검증: 모든 해의 orbit에 대표가 남고 비용이 보존됨을 증명.
- 기대: 교환 가능한 입력이 많은 회로의 중복 탐색 감소.

#### C38 · 평행이동 대표 · P2

```text
허용되는 평행이동 orbit에서 최소 좌표 대표를 선택
```

- 전제: 경계·입력 사이트·고정 셀·타일링 조건이 이동 후에도 유효한 경우만.
- 검증: 해의 위치 이동 사상과 비용 보존 증명.
- 기대: 빈 공간에서 같은 회로를 여러 위치에 놓는 중복 감소.

#### C39 · 경계 상태 완전 열거 분해 · P1

```text
전체 문제 ↔ ⋁boundary (왼쪽(boundary)∧오른쪽(boundary))
```

- 전제: 전원/강전원/가루 모양/상태 cut/순위·단계 등 교차 의존을 빠짐없이 경계에 포함.
- 검증: 분리된 식의 결합이 원래 식과 동치인지 변수 제거 관점에서 증명.
- 기대: DFF의 두 루프를 따로 풀면서 모든 가능성 보존.

#### C40 · 불가능한 경계 배정 학습 · P0

```text
BoundaryAssumptions A ∧ Subproblem = UNSAT ⇒ ¬⋀Core(A)
```

- 전제: core의 모든 literal을 포함하며 다른 인스턴스·고정 조건으로 무단 재사용 금지.
- 검증: core 단독 재검사와 full problem이 nogood를 함의하는지 확인.
- 기대: 부분 문제에서 배운 짧은 금지절을 전체 탐색에 재사용.

### 반례 탐색용 통제 후보

#### C41 · 모든 가루는 반드시 유용 · CONTROL

```text
D(c)⇒ContributesOut(c)∨Observed(c)
```

- 전제: unused dust가 연결 모양을 바꿀 수 있다는 알려진 위험.
- 검증: shape 변화가 있는 작은 배치에서 B∧¬L 반례 탐색.
- 기대: 안전하지 않은 usefulness 가지치기를 걸러내는 통제 사례.

#### C42 · 모든 경로는 거리 감소 · CONTROL

```text
wire(u,v)⇒dist(v,output)<dist(u,output)
```

- 전제: 장애물과 방향 조건 때문에 우회가 필요할 수 있음.
- 검증: 막힌 공간과 고정 셀을 둔 반례 탐색.
- 기대: 거리 휴리스틱을 hard constraint로 바꾸는 실수 탐지.

#### C43 · 논리 노드 하나당 토치 하나 · CONTROL

```text
각 netlist node에 정확히 하나의 토치
```

- 전제: 현재 모델은 함수 중복 구현과 다른 조합을 허용함.
- 검증: 동일 함수 공유·복제 배치의 배제 여부 확인.
- 기대: netlist 구조를 과도하게 고정하는 실수 탐지.

#### C44 · 래치 fixture를 항상 고정 · CONTROL

```text
MasterLatch = 저장된 단일 fixture
```

- 전제: 존재하는 전체 해를 놓칠 수 있는 구조 고정.
- 검증: 다른 핀·박스·비용의 해와 비교; 동일 배치 보존 주장은 즉시 불가.
- 기대: construction 힌트와 완전 탐색의 차이를 검증.

#### C45 · state cut은 토치 직후만 · CONTROL

```text
Cut(u,v)⇒Torch(u)
```

- 전제: stored state 경계가 다른 위치에 있을 가능성을 제한.
- 검증: cut 위치를 옮긴 모델 비교, projection과 state 의미를 모두 검사.
- 기대: 편한 cut 위치 강제가 완전성을 해치는지 확인.

#### C46 · stage를 전역 최단거리로 고정 · CONTROL

```text
stage(c)=shortestDistance(input,c)
```

- 전제: stage는 모든 의존 경로를 만족해야 함; 최단경로가 충분하지 않음.
- 검증: 짧은 경로와 긴 경로가 합쳐지는 모티프 반례.
- 기대: 타이밍 하한을 등식으로 오해하는 실수 탐지.

#### C47 · Cut에서도 양끝 함수 동일 · CONTROL

```text
Feeds(u,v)⇒sig(u)≡sig(v)
```

- 전제: 이미 Boolean 반례가 나온 일반화. 특히 순차회로에서 위험.
- 검증: 기존 schema unsafe_ignore_cut 재사용; 순차 full 모델에서도 확인.
- 기대: 상태 루프 보존 여부를 확인하는 회귀 통제.

#### C48 · 약전원도 가루로 전달 · CONTROL

```text
F(b,d,FB,DS)⇒P(q,b)=P(q,d)
```

- 전제: 이미 Strong=false, total=true인 Boolean 반례가 존재.
- 검증: 기존 schema unsafe_use_total_block_power 재사용; weak-power 배치 검사.
- 기대: Strong 특례를 누락하는 인코딩 오류 탐지.
