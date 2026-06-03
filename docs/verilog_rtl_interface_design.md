# Verilog RTL 인터페이스 설계 검토

## 목적

이 문서는 Verilog `always` 문법을 단발성 패턴 매칭으로 계속 추가하지 않고, `DFF + incrementer` counter 같은 RTL을 정석적으로 받아들이기 위해 어떤 하위 인터페이스가 필요한지 정리한다.

현재 프로젝트는 이미 작은 조합 회로, RS latch, D latch, D flip-flop global PnR smoke를 다루기 시작했다. 하지만 Verilog sequential 구문을 더 넓게 지원하려면 지금의 `recognized_d_latch()` 같은 함수가 계속 늘어나는 방식은 위험하다. `always` 문법은 단순 문법이 아니라 procedural hardware semantics이기 때문에, Verilog AST와 `GraphModuleDesign` 사이에 별도 RTL/synthesis 계층이 필요하다.

이 문서는 구현 계획이 아니라 하이레벨 인터페이스 디자인 문서다. 목표는 현재 상태와 앞으로 도입할 중간 계층의 책임을 명확히 이해하는 것이다.

## 현재 상태

### Verilog frontend

현재 관련 파일:

- `src/verilog/lexer.rs`
- `src/verilog/parser.rs`
- `src/verilog/ast.rs`
- `src/verilog/lower.rs`
- `src/verilog/design.rs`

현재 지원되는 주요 문법:

- `module ... endmodule`
- `input`, `output`, `output reg`, `wire`
- 단순 vector declaration과 bit select 일부
- `assign`
- `~`, `&`, `^`, `|`
- named module instance
- 제한적인 `always @(*) begin if (...) begin q <= d; end end`

현재 lowering 경로는 두 갈래다.

```text
Verilog AST
  -> lower.rs
  -> LogicGraph
```

`lower.rs`는 named instance를 assignment로 inline해서 하나의 조합 `LogicGraph`를 만든다. 조합 Verilog에는 유용하지만 sequential/hierarchy 유지에는 맞지 않다.

```text
Verilog AST
  -> design.rs
  -> GraphModuleDesign
```

`design.rs`는 module hierarchy를 어느 정도 보존해서 `GraphModuleDesign`을 만든다. 하지만 현재 `recognized_d_latch()`가 `AlwaysBlock` AST 모양을 보고 바로 `SequentialPrimitive::DLatch` graph module을 생성한다. 이 부분이 지금 가장 취약한 확장 지점이다.

### GraphModuleDesign

현재 `GraphModuleDesign`은 다음 구조다.

```rust
GraphModuleDesign {
    context: GraphModuleContext,
    top: String,
}
```

`GraphModule`은 크게 두 종류를 표현한다.

- `graph: Some(Graph)`: leaf graph-backed module
- `graph: None`: child instance와 var/port 연결을 가진 hierarchical module

현재 `GraphModule`의 한계:

- `instances: Vec<String>`라 instance name만 있고 module type 정보가 분리되어 있지 않다.
- `GraphModuleVariable`은 `(source module, source port) -> (target module, target port)` 단일 sink 연결이다.
- top port는 `GraphModulePortTarget::Module` 또는 `Wire`로 child port를 가리킨다.
- width/bit-vector 개념이 없다.
- net identity가 명시적이지 않고, source/target 관계로만 연결된다.
- clock/reset/data net 같은 의미 정보가 없다.

이 구조는 현재 DFF smoke처럼 소수의 child module을 배치하고 point-to-point route하는 데는 충분하다. 그러나 counter/register/FSM처럼 vector state와 next-state logic이 섞이면 바로 부족해진다.

### SequentialPrimitive

현재 sequential primitive:

```rust
enum SequentialType {
    RsLatch,
    DLatch,
}
```

현황:

- RS latch, D latch는 local placer 지원 경로가 있다.
- DFF는 `SequentialPrimitive`가 아니라 `SynthCell::Dff`를 `not_clk + master D latch + slave D latch` 구조의 `GraphModule`로 낮추는 mapping 대상이다.
- 현재 DFF global smoke도 monolithic sequential primitive가 아니라 `not_clk + master D latch + slave D latch` 구조를 global PnR로 조합한다.

Counter를 `DFF + incrementer`로 가려면 DFF/register cell을 실제 PnR 단위로 표현하는 인터페이스가 필요하다. 지금처럼 D latch 두 개로 DFF를 구성하는 module을 생성할 수도 있고, 나중에는 prebuilt DFF cell 또는 state cell로 캐시할 수도 있다.

## 목표 예시

초기 counter 목표는 reset을 제외한 최소 형태가 좋다.

```verilog
module counter(clk, q);
  input clk;
  output reg [3:0] q;

  always @(posedge clk) begin
    q <= q + 1;
  end
endmodule
```

의미는 다음과 같다.

```text
on posedge(clk):
  q_next = q + 1
  q <= q_next
```

합성 결과는 다음처럼 볼 수 있다.

```text
combinational:
  q_next = inc(q)

state:
  DFF q[0] <= q_next[0]
  DFF q[1] <= q_next[1]
  DFF q[2] <= q_next[2]
  DFF q[3] <= q_next[3]
```

reset이 포함되면:

```verilog
module counter(clk, rst, q);
  input clk, rst;
  output reg [3:0] q;

  always @(posedge clk) begin
    if (rst) begin
      q <= 4'b0000;
    end else begin
      q <= q + 1;
    end
  end
endmodule
```

정규화된 의미는:

```text
q_next = mux(rst, 4'b0000, q + 1)
on posedge(clk): q <= q_next
```

이 예시는 `always`, edge event, if/else, nonblocking assignment, vector, constant, add, mux, DFF/register를 모두 요구한다. 따라서 “문법 모양 하나를 매칭”하는 접근으로는 확장하기 어렵고, 의미 중심의 중간 IR이 필요하다.

## 필요한 하위 인터페이스

### 1. Signal/Width 인터페이스

현재는 signal이 대부분 `String`이다. Counter부터는 width가 필수다.

필요한 모델:

```rust
struct RtlSignal {
    id: RtlSignalId,
    name: String,
    width: usize,
    kind: RtlSignalKind,
}

enum RtlSignalKind {
    Input,
    Output,
    Wire,
    Reg,
}

struct RtlBitRef {
    signal: RtlSignalId,
    bit: usize,
}
```

선택지:

- **비트 단위 flatten 우선**
  - `q[3:0]`을 `q_0`, `q_1`, `q_2`, `q_3`로 초기에 펼친다.
  - 기존 `LogicGraph`와 잘 맞는다.
  - vector operation을 구현할 때 bit-blasting이 빠르다.
  - 단점: Verilog width/type 정보를 잃기 쉽고, debug 출력이 지저분해진다.

- **vector signal 유지 후 나중에 bit-blast**
  - RTL 단계에서는 `q: width=4`로 유지한다.
  - synthesis 단계에서 add/mux/register를 bit 단위로 낮춘다.
  - counter, register, bus, memory 쪽 확장성이 좋다.
  - 단점: `LogicGraph`로 내려가기 전에 별도 vector expression lowering이 필요하다.

추천:

초기 RTL IR에서는 vector를 유지하고, `SynthNetlist -> GraphModuleDesign` 변환 직전에 bit-blast하는 것이 낫다. Verilog 의미를 오래 보존할수록 counter/debug가 쉬워진다.

### 2. Expression 인터페이스

현재 `Expr`는 `Ident`, `Not`, `Binary { And/Xor/Or }` 정도다. Counter에는 `Const`, `Add`, bit select, mux가 필요하다.

필요한 모델:

```rust
enum RtlExpr {
    Signal(RtlSignalRef),
    Const { width: usize, value: u128 },
    Not(Box<RtlExpr>),
    And(Box<RtlExpr>, Box<RtlExpr>),
    Or(Box<RtlExpr>, Box<RtlExpr>),
    Xor(Box<RtlExpr>, Box<RtlExpr>),
    Add(Box<RtlExpr>, Box<RtlExpr>),
    Mux {
        select: Box<RtlExpr>,
        when_true: Box<RtlExpr>,
        when_false: Box<RtlExpr>,
    },
    BitSelect {
        expr: Box<RtlExpr>,
        bit: usize,
    },
}
```

선택지:

- **Verilog AST Expr를 그대로 확장**
  - parser와 RTL lowering 사이 구조가 적다.
  - 작은 기능 추가는 빠르다.
  - 단점: Verilog syntax와 hardware semantics가 섞인다. width inference, name resolution, signedness 처리 위치가 애매해진다.

- **RtlExpr를 별도 도입**
  - Verilog AST는 syntax, RtlExpr는 elaborated semantic expression으로 분리된다.
  - name resolution과 width inference 결과를 담을 수 있다.
  - 단점: 변환 단계가 하나 더 생긴다.

추천:

`RtlExpr`를 별도로 둔다. `Verilog Expr -> RtlExpr` 변환에서 signal id, width, constant width를 확정해야 한다.

### 3. Procedural AST / Process 인터페이스

`always`는 continuous assign이 아니라 process다. parser AST와 semantic process를 분리해야 한다.

Verilog syntax AST:

```rust
struct AlwaysBlock {
    sensitivity: AlwaysSensitivity,
    body: AlwaysStmt,
}

enum AlwaysStmt {
    Block(Vec<AlwaysStmt>),
    If {
        condition: Expr,
        then_branch: Box<AlwaysStmt>,
        else_branch: Option<Box<AlwaysStmt>>,
    },
    Case { ... },
    BlockingAssign { lhs: SignalRefSyntax, rhs: Expr },
    NonBlockingAssign { lhs: SignalRefSyntax, rhs: Expr },
}
```

Elaborated RTL process:

```rust
struct RtlProcess {
    sensitivity: RtlSensitivity,
    statements: Vec<RtlStmt>,
}

enum RtlSensitivity {
    Combinational,
    Posedge(RtlSignalId),
    Negedge(RtlSignalId),
    EdgeList(Vec<RtlEdgeEvent>),
}

enum RtlStmt {
    Assign {
        kind: RtlAssignKind,
        lhs: RtlSignalRef,
        rhs: RtlExpr,
    },
    If {
        condition: RtlExpr,
        then_branch: Vec<RtlStmt>,
        else_branch: Vec<RtlStmt>,
    },
    Case { ... },
}
```

중요한 점:

- parser는 문법을 보존한다.
- RTL process는 signal/width/name이 resolve된 상태여야 한다.
- latch/DFF 추론은 `RtlProcess` 이후 단계에서 한다.

### 4. Assignment Coverage Analysis

이 계층이 `always` 지원의 핵심이다.

예:

```verilog
always @(*) begin
  if (en) q <= d;
end
```

분석 결과:

```text
q assigned when en
q unassigned when !en
=> q_next = mux(en, d, q_current)
=> latch
```

예:

```verilog
always @(*) begin
  if (sel) y = a;
  else y = b;
end
```

분석 결과:

```text
y assigned on all paths
=> y = mux(sel, a, b)
=> combinational
```

필요한 인터페이스:

```rust
struct AssignmentEffect {
    target: RtlSignalRef,
    value: RtlExpr,
    condition: RtlExpr,
    priority: usize,
    kind: RtlAssignKind,
}

struct ProcessAnalysis {
    sensitivity: RtlSensitivity,
    targets: Vec<RtlSignalRef>,
    next_values: HashMap<RtlSignalRef, RtlNextValue>,
}

enum RtlNextValue {
    FullyAssigned(RtlExpr),
    HoldsUnless {
        assigned_value: RtlExpr,
        hold_condition: RtlExpr,
    },
}
```

더 단순한 초기 형태:

```rust
struct StateUpdate {
    target: RtlSignalRef,
    next: RtlExpr,
    clock: Option<RtlClock>,
}
```

초기에는 다음만 지원해도 된다.

- single target
- if/else tree
- nonblocking assignment
- posedge clock
- no multiple assignment priority

하지만 인터페이스 이름과 위치는 나중 확장 가능한 형태로 잡는 것이 중요하다.

### 5. State Element 인터페이스

Process analysis 결과는 state element로 정규화되어야 한다.

```rust
enum StateElement {
    Latch {
        output: RtlSignalRef,
        enable: RtlExpr,
        data: RtlExpr,
    },
    Dff {
        output: RtlSignalRef,
        clock: RtlSignalRef,
        edge: ClockEdge,
        data: RtlExpr,
        reset: Option<ResetSpec>,
    },
    Register {
        output: RtlSignalRef,
        clock: RtlSignalRef,
        width: usize,
        data: RtlExpr,
        reset: Option<ResetSpec>,
    },
}

enum ClockEdge {
    Posedge,
    Negedge,
}
```

`DFF + incrementer` counter는 `Register`로 보존할 수도 있고, bit별 `Dff`로 쪼갤 수도 있다.

선택지:

- **즉시 bit별 DFF로 분해**
  - PnR leaf primitive와 빨리 연결된다.
  - `GraphModuleDesign`으로 변환이 단순해진다.
  - 단점: register/counter 단위 최적화 기회를 잃는다.

- **Register cell 유지**
  - counter/register layout 최적화, shared clock routing, compact PnR에 유리하다.
  - 단점: local/global PnR이 multi-bit state cell을 이해해야 한다.

- **둘 다 지원**
  - RTL/synthesis 단계에서는 `Register`를 유지한다.
  - GraphModule 변환 전략에서 bit별 DFF 분해 또는 macro register 선택을 한다.

추천:

초기에는 `Register`를 IR에 보존하되, GraphModule 변환에서는 bit별 DFF module로 낮춘다. 나중에 compact redstone layout이 필요해지면 `Register -> TFF counter` 또는 `Register macro` mapping pass를 추가한다.

### 6. SynthNetlist 인터페이스

`RtlModule`은 Verilog 의미를 표현한다. PnR에 바로 넣기 전에는 technology-independent netlist가 필요하다.

```rust
struct SynthNetlist {
    ports: Vec<SynthPort>,
    nets: Vec<SynthNet>,
    cells: Vec<SynthCell>,
}

enum SynthCell {
    Comb {
        name: String,
        graph: LogicGraph,
        inputs: Vec<SynthNetId>,
        outputs: Vec<SynthNetId>,
    },
    Mux {
        output: SynthNetId,
        select: SynthNetId,
        when_true: SynthNetId,
        when_false: SynthNetId,
    },
    Inc {
        output: SynthBus,
        input: SynthBus,
    },
    Dff {
        q: SynthNetId,
        d: SynthNetId,
        clk: SynthNetId,
        edge: ClockEdge,
    },
    DLatch {
        q: SynthNetId,
        d: SynthNetId,
        en: SynthNetId,
    },
}
```

Counter 예시:

```text
SynthCell::Inc {
  input: q[3:0],
  output: q_next[3:0],
}

SynthCell::Dff { q: q[0], d: q_next[0], clk }
SynthCell::Dff { q: q[1], d: q_next[1], clk }
SynthCell::Dff { q: q[2], d: q_next[2], clk }
SynthCell::Dff { q: q[3], d: q_next[3], clk }
```

`Inc`는 나중에 다음 둘 중 하나로 낮출 수 있다.

- `LogicGraph` ripple incrementer
- `TFF counter` optimization

중요한 점:

`SynthCell::Inc`를 바로 `LogicGraph`로 bit-blast하지 않고 잠깐 보존하면 optimization 선택지가 남는다. 하지만 PnR 전에 반드시 실제 routable module/cell로 낮아져야 한다.

### 7. GraphModuleDesign 변환 인터페이스

`GraphModuleDesign`은 global PnR 입구다. 따라서 `SynthNetlist -> GraphModuleDesign` 변환이 필요하다.

필요한 변화:

```rust
struct GraphModuleInstance {
    instance_name: String,
    module_name: String,
}
```

현재 `GraphModule.instances: Vec<String>`는 instance name만 있으므로, 같은 module type을 여러 instance로 쓰거나 library cell을 재사용하는 구조가 약하다.

가능한 개선:

```rust
pub struct GraphModule {
    pub name: String,
    pub graph: Option<Graph>,
    pub instances: Vec<GraphModuleInstance>,
    pub nets: Vec<GraphModuleNet>,
    pub ports: Vec<GraphModulePort>,
}

pub struct GraphModuleNet {
    pub name: String,
    pub source: GraphModuleEndpoint,
    pub sinks: Vec<GraphModuleEndpoint>,
    pub kind: GraphModuleNetKind,
}

pub struct GraphModuleEndpoint {
    pub instance: Option<String>,
    pub port: String,
}
```

선택지:

- **현재 GraphModule 유지**
  - 구현량이 적다.
  - DFF smoke 수준에서는 충분하다.
  - counter처럼 `q[0..3]`, shared clock, repeated DFF를 표현하면 vars/ports가 지저분해진다.

- **GraphModule에 typed instance/net 추가**
  - global PnR과 RTL lowering 사이 인터페이스가 명확해진다.
  - fanout/shared net, clock net, bus bit를 명시하기 쉽다.
  - 기존 global PnR router 일부 리팩토링 필요.

- **GraphModule은 그대로 두고 SynthNetlist가 netlist 역할을 담당**
  - GraphModule 변경을 미룰 수 있다.
  - `SynthNetlist -> GraphModuleDesign` 변환에서 현재 구조로 어댑트한다.
  - 단점: GraphModule의 한계가 계속 어댑터에 누적된다.

추천:

초기에는 `SynthNetlist`를 명확히 두고, `GraphModuleDesign`은 현재 구조로 어댑트한다. 다만 `GraphModuleInstance`와 `GraphModuleNet` 도입은 장기적으로 필요하다. 특히 counter/register를 global PnR에서 안정적으로 다루려면 net identity와 fanout을 명시해야 한다.

## 주요 설계 선택지

### 선택지 A: 현재 방식 확장

```text
Verilog AST
  -> design.rs pattern matcher
  -> GraphModuleDesign
```

장점:

- 구현이 빠르다.
- D latch, DFF 같은 소수 패턴은 금방 붙일 수 있다.
- 현재 global PnR API를 거의 건드리지 않는다.

단점:

- `recognized_d_latch`, `recognized_dff`, `recognized_counter`처럼 패턴이 계속 늘어난다.
- `always` 의미 분석이 함수 이름과 if-let 중첩에 묻힌다.
- `q <= q + 1`, reset, enable, case, multiple assignment가 추가되면 곧 깨진다.
- 문법 모양이 조금만 달라져도 지원되지 않는다.

판단:

단기 smoke에는 괜찮지만 counter 목표에는 부적절하다.

### 선택지 B: RtlModule만 도입

```text
Verilog AST
  -> RtlModule
  -> GraphModuleDesign
```

장점:

- Verilog syntax와 resolved signal/width를 분리할 수 있다.
- `always` parser와 name resolution을 정리할 수 있다.
- 구현량이 선택지 C보다 적다.

단점:

- `RtlModule -> GraphModuleDesign`에서 여전히 state inference, bit-blasting, cell mapping이 한 번에 일어날 수 있다.
- synthesis 결과를 테스트하기 어렵다.
- counter의 `Inc`, `Dff`, `Mux` 같은 중간 결과가 명시적으로 남지 않는다.

판단:

현재보다는 낫지만, `DFF + incrementer`까지 가면 중간 netlist가 필요해질 가능성이 높다.

### 선택지 C: RtlModule + SynthNetlist 도입

```text
Verilog AST
  -> ElaboratedDesign / RtlModule
  -> SynthNetlist
  -> GraphModuleDesign
```

장점:

- 각 단계 책임이 명확하다.
- Verilog parser, elaboration, synthesis, PnR interface를 분리한다.
- `q <= q + 1`을 `Inc + DFF`로 검증할 수 있다.
- 나중에 `Inc -> TFF counter` 같은 optimization pass를 넣기 좋다.
- 상용 EDA의 큰 구조와 가장 유사하다.

단점:

- 초기 구현량이 가장 크다.
- 타입과 테스트를 많이 만들어야 한다.
- 기존 `lower.rs`와 `design.rs`를 재배치해야 한다.

판단:

추천 경로다. Counter를 제대로 목표로 잡으면 이 구조가 가장 안전하다.

### 선택지 D: 외부 합성기 활용

예를 들어 Verilog를 외부 합성기로 netlist화한 뒤 우리 PnR로 가져오는 방식이다.

장점:

- Verilog 문법 전체 지원 부담을 줄일 수 있다.
- 복잡한 RTL inference를 직접 구현하지 않아도 된다.

단점:

- 외부 dependency가 생긴다.
- 우리가 원하는 redstone-specific primitive와 hierarchy를 통제하기 어렵다.
- 현재 프로젝트의 학습/제어 목적과 맞지 않을 수 있다.
- generated netlist format을 다시 import해야 한다.

판단:

장기 보조 도구로는 가능하지만, 현재는 내부 IR을 만드는 편이 낫다.

## 추천 구조

추천 파이프라인:

```text
Verilog source
  -> Verilog AST
  -> ElaboratedDesign
  -> RtlModule
  -> SynthNetlist
  -> GraphModuleDesign
  -> Global PnR
```

각 단계 책임:

### Verilog AST

문법을 그대로 보존한다.

담당:

- token/parse
- source-level module/declaration/assignment/always/instance 표현
- 아직 width/name resolution을 완전히 하지 않음

담당하지 말아야 할 것:

- latch/DFF inference
- GraphModule 생성
- PnR cell 선택

### ElaboratedDesign

module hierarchy와 signal table을 확정한다.

담당:

- top module 선택
- module definition lookup
- instance binding
- port direction/width resolve
- vector declaration resolve
- signal id allocation
- bit select/constant width 정리

### RtlModule

hardware semantics를 표현한다.

담당:

- continuous assign
- procedural process
- resolved signal references
- width-aware expression

예:

```rust
struct RtlModule {
    name: String,
    signals: Vec<RtlSignal>,
    ports: Vec<RtlPort>,
    continuous_assigns: Vec<RtlAssign>,
    processes: Vec<RtlProcess>,
    instances: Vec<RtlInstance>,
}
```

### SynthNetlist

합성된 technology-independent cell graph다.

담당:

- mux/add/inc/not/and/or/xor
- latch/dff/register
- net fanout
- bit-blasted 또는 bus-aware cell

이 단계에서 counter는 다음처럼 보여야 한다.

```text
cell inc0: Inc(q[3:0]) -> q_next[3:0]
cell dff0: Dff(q[0], q_next[0], clk)
cell dff1: Dff(q[1], q_next[1], clk)
cell dff2: Dff(q[2], q_next[2], clk)
cell dff3: Dff(q[3], q_next[3], clk)
```

### GraphModuleDesign

PnR 입구다.

담당:

- local placer가 만들 수 있는 leaf module 정의
- child instance 배치 대상 정의
- net/port 연결을 global router가 이해할 수 있게 변환

담당하지 말아야 할 것:

- Verilog procedural semantics 분석
- `always` assignment coverage 분석
- incrementer 최적화

## Counter 기준 최소 기능 목록

Reset 없는 counter:

```verilog
module counter(clk, q);
  input clk;
  output reg [3:0] q;

  always @(posedge clk) begin
    q <= q + 1;
  end
endmodule
```

필수:

- lexer/parser
  - `posedge`
  - vector output reg
  - based constant 또는 unsized constant `1`
  - `+`
  - `always @(posedge clk)`
  - statement list
- AST
  - sensitivity edge
  - nonblocking assignment
  - vector signal ref
  - add expression
- elaboration
  - `q` width = 4
  - `1` width extension to 4
  - `q + 1` result width = 4
- synthesis
  - edge process -> DFF/register
  - add-by-one -> incrementer
  - register output feedback allowed
- GraphModuleDesign
  - four q output bits or bus metadata
  - shared clk fanout
  - DFF/register cells
  - incrementer combinational cell

Reset 있는 counter:

```verilog
always @(posedge clk) begin
  if (rst) begin
    q <= 4'b0000;
  end else begin
    q <= q + 1;
  end
end
```

추가 필수:

- parser
  - `else`
  - based constants `4'b0000`
- synthesis
  - if/else -> mux
  - reset-as-data-mux
  - optional reset metadata

초기에는 reset을 data mux로 처리하는 것이 단순하다.

```text
q_next = mux(rst, 0, q + 1)
DFF(q, clk, q_next)
```

async reset primitive는 나중에 추가한다.

## TFF 최적화와의 관계

Counter를 compact하게 만들려면 TFF counter가 매력적이다.

하지만 Verilog:

```verilog
q <= q + 1;
```

의 기본 semantic lowering은:

```text
DFF + incrementer
```

이다.

TFF는 semantic lowering이 아니라 optimization/mapping 단계로 보는 것이 좋다.

```text
Register + Inc
  -> optional optimization
  -> TFF chain
```

이렇게 분리해야 correctness와 compactness를 동시에 잡을 수 있다.

## 현재 코드에서 바꿔야 할 지점

### `src/verilog/ast.rs`

필요:

- `AlwaysStmt::Block(Vec<AlwaysStmt>)`
- `AlwaysStmt::If { else_branch }`
- `AlwaysStmt::BlockingAssign`
- `AlwaysSensitivity::Posedge(String)`
- `AlwaysSensitivity::Negedge(String)`
- `Expr::Const`
- `Expr::Add`
- `Expr::Mux` 또는 ternary
- `SignalRef` syntax type

주의:

AST는 syntax용이다. resolved signal id를 넣지 않는 편이 낫다.

### `src/verilog/parser.rs`

필요:

- statement list parser
- nested begin/end
- if/else parser
- posedge/negedge sensitivity parser
- based number parser
- `+` expression parser
- blocking/nonblocking assignment parser

주의:

parser에서 latch/DFF를 판단하지 않는다.

### `src/verilog/lower.rs`

현재 조합 flatten용이다.

선택:

- 기존 `load_logic_graph` compatibility용으로 유지한다.
- 새 RTL 경로와 분리한다.
- 나중에 `lower.rs` 이름을 `comb_lower.rs`로 바꾸는 것도 고려한다.

### `src/verilog/design.rs`

현재는 `GraphModuleDesign` 생성과 latch inference가 섞여 있다.

추천:

- `recognized_d_latch()` 제거 또는 `rtl/synth`로 이동
- `design.rs`는 `SynthNetlist -> GraphModuleDesign` adapter 역할로 축소
- Verilog AST에서 바로 GraphModule을 만들지 않게 한다.

### 새 파일 후보

```text
src/verilog/elaborate.rs
src/verilog/rtl.rs
src/verilog/synth.rs
src/verilog/synth/netlist.rs
src/verilog/synth/process.rs
src/verilog/synth/graph_module.rs
```

역할:

- `elaborate.rs`: signal table, widths, instance binding
- `rtl.rs`: RtlModule/RtlExpr/RtlProcess 타입
- `synth/process.rs`: process analysis, DFF/latch inference
- `synth/netlist.rs`: SynthCell/SynthNetlist 타입
- `synth/graph_module.rs`: SynthNetlist -> GraphModuleDesign

### `src/graph/module.rs`

단기:

- 지금 구조 유지 가능.
- generated DFF/incrementer module을 instance name으로 context에 넣어 어댑트한다.

장기:

- `GraphModuleInstance { instance_name, module_name }`
- `GraphModuleNet { source, sinks, kind }`
- net fanout 명시
- bus/bit naming convention 문서화

### `src/sequential/mod.rs`

필요:

- DFF는 `SequentialPrimitive`에 두지 않고 `SynthCell::Dff -> composed GraphModule` mapping으로 처리
- Register primitive를 둘지 결정

선택:

- **DFF를 composed GraphModule로 생성**
  - 현재 global PnR 구조와 맞다.
  - 기존 D latch local placer를 재사용한다.
  - 느릴 수 있다.

- **DFF leaf primitive/macro 도입**
  - counter/register 확장에 유리하다.
  - local placer/layout 지원이 필요하다.

추천:

초기 counter는 DFF를 composed module로 생성한다. 이후 DFF layout이 안정화되면 macro candidate로 캐시한다.

## 단계별 검증 예시

### 1단계: DFF process

```verilog
module dff(clk, d, q);
  input clk, d;
  output reg q;

  always @(posedge clk) begin
    q <= d;
  end
endmodule
```

기대:

```text
StateElement::Dff { q, clk, data: d }
```

### 2단계: enable DFF

```verilog
module dff_en(clk, en, d, q);
  input clk, en, d;
  output reg q;

  always @(posedge clk) begin
    if (en) begin
      q <= d;
    end
  end
endmodule
```

기대:

```text
q_next = mux(en, d, q)
DFF(q, clk, q_next)
```

### 3단계: 1-bit toggle counter

```verilog
module toggle(clk, q);
  input clk;
  output reg q;

  always @(posedge clk) begin
    q <= ~q;
  end
endmodule
```

기대:

```text
q_next = not(q)
DFF(q, clk, q_next)
```

### 4단계: 4-bit increment counter

```verilog
module counter(clk, q);
  input clk;
  output reg [3:0] q;

  always @(posedge clk) begin
    q <= q + 1;
  end
endmodule
```

기대:

```text
q_next = inc(q)
DFF(q[0], clk, q_next[0])
DFF(q[1], clk, q_next[1])
DFF(q[2], clk, q_next[2])
DFF(q[3], clk, q_next[3])
```

### 5단계: reset counter

```verilog
module counter(clk, rst, q);
  input clk, rst;
  output reg [3:0] q;

  always @(posedge clk) begin
    if (rst) begin
      q <= 4'b0000;
    end else begin
      q <= q + 1;
    end
  end
endmodule
```

기대:

```text
q_next = mux(rst, 0, inc(q))
DFF(q[*], clk, q_next[*])
```

## 추천 결론

`DFF + incrementer` counter를 타겟으로 삼는다면 다음 방향을 추천한다.

1. `recognized_d_latch()`를 계속 확장하지 않는다.
2. `Verilog AST -> RtlModule -> SynthNetlist -> GraphModuleDesign` 계층을 도입한다.
3. 초기에는 vector를 RTL에서 유지하고, GraphModule 변환 직전에 bit-blast한다.
4. `q <= q + 1`은 `Register + Inc` 또는 `Inc + DFF bits`로 낮춘다.
5. TFF counter는 semantic lowering이 아니라 optimization/mapping pass로 둔다.
6. `GraphModule`은 당장 유지하되, `GraphModuleNet`/typed instance 도입을 장기 인터페이스 개선으로 잡는다.

이렇게 가면 Verilog 문법이 늘어도 parser/RTL/synthesis/PnR 책임이 분리된다. 반대로 지금 `design.rs`에 패턴을 계속 추가하면 D latch, DFF, enable DFF, reset DFF, counter, FSM이 모두 같은 함수 계층에 섞이면서 인터페이스가 금방 깨질 가능성이 높다.

## 가장 먼저 만들 인터페이스 초안

구현을 시작한다면 첫 타입은 이 정도가 적당하다.

```rust
pub struct RtlModule {
    pub name: String,
    pub signals: Vec<RtlSignal>,
    pub ports: Vec<RtlPort>,
    pub continuous_assigns: Vec<RtlAssign>,
    pub processes: Vec<RtlProcess>,
}

pub struct RtlSignal {
    pub id: RtlSignalId,
    pub name: String,
    pub width: usize,
    pub kind: RtlSignalKind,
}

pub struct RtlProcess {
    pub sensitivity: RtlSensitivity,
    pub statements: Vec<RtlStmt>,
}

pub enum RtlStmt {
    Assign {
        kind: RtlAssignKind,
        lhs: RtlSignalRef,
        rhs: RtlExpr,
    },
    If {
        condition: RtlExpr,
        then_branch: Vec<RtlStmt>,
        else_branch: Vec<RtlStmt>,
    },
}

pub enum RtlExpr {
    Signal(RtlSignalRef),
    Const { width: usize, value: u128 },
    Not(Box<RtlExpr>),
    And(Box<RtlExpr>, Box<RtlExpr>),
    Or(Box<RtlExpr>, Box<RtlExpr>),
    Xor(Box<RtlExpr>, Box<RtlExpr>),
    Add(Box<RtlExpr>, Box<RtlExpr>),
    Mux {
        select: Box<RtlExpr>,
        when_true: Box<RtlExpr>,
        when_false: Box<RtlExpr>,
    },
}
```

그 다음 타입:

```rust
pub struct SynthNetlist {
    pub nets: Vec<SynthNet>,
    pub cells: Vec<SynthCell>,
    pub ports: Vec<SynthPort>,
}

pub enum SynthCell {
    CombGraph { graph: LogicGraph, inputs: Vec<SynthNetId>, outputs: Vec<SynthNetId> },
    Inc { input: SynthBus, output: SynthBus },
    Mux { select: SynthNetId, when_true: SynthBus, when_false: SynthBus, output: SynthBus },
    Dff { clk: SynthNetId, d: SynthNetId, q: SynthNetId, edge: ClockEdge },
}
```

이 두 계층이 생기면 counter의 의미를 GraphModule/PnR 전에 테스트할 수 있다. 그게 핵심이다.
