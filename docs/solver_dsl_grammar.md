# rsdsl v2: grammar and language specification

Status: implemented in `crates/rsdsl` (parser, grounder, CNF encoder, DIMACS
writer). The syntax is also machine-checked: `docs/rsdsl/rsdsl.lark` builds an
LALR(1) table with no conflicts in Lark's strict mode, and it parses both
examples:

- `src/transform/place_and_route/local_placer/exact/exact_placer.rsdsl`: the complete exact-placer model. The exact placer grounds it by
  default; `local_placer/exact/encode.rs` is the hand-written original.
- `docs/rsdsl/inverter_1x3x2.instance.rsdsl`: a small instance.

The architecture and migration plan are in `solver_dsl_design.md`. This
document fixes the language itself.

## 1. Design principles

Each principle came from a concrete problem in the exact placer.

1. **One expression language, two stages.**
   - Every expression is either *compile-time* (known while grounding) or
     *solver-time* (a formula over solver variables).
   - The type checker tracks the stage, and guards must be compile-time.
   - Writing both stages the same way keeps rules short: a compile-time
     `true`/`false` simply folds away inside a formula.
2. **Brackets are dense, parentheses are sparse.**
   - `Kind[c]`, `Powered[k, c]`, and `Rank[c]` index families defined
     everywhere on their domains.
   - `on(j, k)`, `Feeds(s, t, f, to)`, and `step(c, d)` are tuple predicates
     or function calls; they hold only where something put them.
   - The brackets tell the reader which kind of access it is.
3. **Words for logic, symbols for structure.**
   - Connectives are `not`, `and`, `or`, `xor`, `->`, and `<->`, which read
     like the rule labels.
   - `&&`, `||`, and `!` do not exist, so there is no second way to write
     the same thing.
4. **Nothing implicit that a typo could trigger.**
   - Patterns never bind names; tuple iteration binds them explicitly.
   - Shadowing is an error.
   - An enum variant name that is ambiguous needs a qualifier.
5. **The edge of the world is declared, not assumed.**
   - `step(c, Up)` can leave the box. What a family means outside is part of
     its declaration (`@outside(Air)`), not a hidden convention.
6. **Every clause has a reason.**
   - Constraints live in labeled `rule`s, and every family has a display
     template.
   - Explained DIMACS, UNSAT cores, and debugging all speak in model terms,
     never `aux#123`.
7. **Encoding is a choice, not a hard-coded part of the compiler.**
   - Cardinality and integer encodings, polarity preference, and relaxation
     guards are annotations.
   - Every trick in `encode.rs` is reachable from the model file.
8. **Declarative and order-independent.**
   - Declarations and rules can appear in any order.
   - The compiler topologically sorts definitions and rejects cycles that
     have no meaning.

## 2. Lexical structure

| Item | Form |
| --- | --- |
| Encoding | UTF-8. Identifiers are ASCII; strings and comments may use any Unicode, such as Korean labels. |
| Whitespace | Insignificant. |
| Comments | `// ...` to end of line; `/* ... */` (not nested). |
| Identifier | `[A-Za-z][A-Za-z0-9_]*` or `_[A-Za-z0-9_]+`. A lone `_` is the hole token. |
| Integer | `[0-9]+`; negative values use unary `-`. |
| String | `"..."` with `\"`, `\\`, `\n`, `\t` escapes; `{name}` placeholders inside `@display`. |
| Punctuation | `; , : := = |= ( ) [ ] { } . .. ..= @ ? | => -> <-> == != < <= > >= + - * / %` |

Reserved words (cannot be identifiers):

```
rsdsl model instance of include
grid dirs enum subset domain fact param fn
choice var over def relation int bool set rule minimize maximize
require forall where if else let match for in is
not and or xor true false none
any all count exactly_one at_most_one
```

Conventions, which the linter warns about rather than errors:
- Types, families, and enum variants are `UpperCamel`.
- Binders, facts, params, and functions are `lower_snake`.

## 3. Grammar

EBNF, informally. `rsdsl.lark` is the checked version.
`X?` means optional, `X*` repetition, `X ("," X)*` comma lists, and a
trailing comma is allowed where shown as `","?`.

```ebnf
file          = header item* ;
header        = "rsdsl" INT ";" ( "model" NAME ";" | "instance" NAME "of" NAME ";" ) ;

item          = include | grid | enum | subset | domain | fact | param | fn
              | annotation* ( choice | var | def | relation | int_family | rule )
              | ( "minimize" | "maximize" ) expr ";" ;

include       = "include" STRING ";" ;

(* compile-time world *)
grid          = "grid" NAME "(" NAME "," NAME "," NAME ")" "dirs" NAME
                "{" axis ("," axis)* ","? "}"
              | "grid" NAME "=" "(" INT "," INT "," INT ")" ";" ;      (* instance *)
axis          = NAME "=" ("+" | "-") NAME ;
enum          = "enum" NAME "{" NAME ("," NAME)* ","? "}" ;
subset        = "subset" NAME "of" NAME "=" list ";" ;
domain        = "domain" NAME (":" type)? ("=" expr)? ";" ;
fact          = "fact" NAME "(" fparams? ")" (":=" expr)? ";"
              | "fact" NAME "=" expr ";" ;                               (* instance *)
param         = "param" NAME ":" type ("=" expr)? ";"
              | "param" NAME "=" expr ";" ;                              (* instance *)
fn            = "fn" NAME "(" fparams? ")" "->" type "=" expr ";" ;
fparams       = fparam ("," fparam)* ;
fparam        = (NAME ":")? type ;
type          = NAME | "int" | "bool" | "set" "<" type ">" | type "?"
              | "(" type ("," type)+ ")" ;

(* solver-level declarations *)
annotation    = "@" NAME ( "(" ( ann_arg ("," ann_arg)* )? ")" )? ;
ann_arg       = NAME "=" expr | expr ;
choice        = "choice" NAME index "{" member ("," member)* ","? "}" ;
member        = NAME ( "(" fparams ")" )? ( "if" expr )? ;
index         = "[" binders "]" ;
var           = "var" NAME index ";" | "var" NAME "over" NAME ";" ;
def           = "def" NAME index? ":=" expr ";" ;
relation      = "relation" NAME "(" fparams ")" ";" ;
int_family    = "int" NAME index "in" range ";" ;
rule          = "rule" STRING block ;

(* statements *)
block         = "{" stmt* "}" ;
stmt          = annotation* ( "require" expr ";"
                            | "forall" binders ("where" expr)? block
                            | if_stmt
                            | "let" NAME "=" expr ";"
                            | NAME "(" args ")" "|=" expr ";" ) ;
if_stmt       = "if" expr block ( "else" ( if_stmt | block ) )? ;
binders       = binder ("," binder)* ;
binder        = NAME ":" type
              | NAME "in" ( range | expr )
              | "(" (NAME | "_") ("," (NAME | "_"))* ")" "in" NAME ;
range         = sum ".." sum | sum "..=" sum ;

(* expressions, lowest precedence first *)
expr          = imp ( "<->" imp )? ;                      (* non-associative *)
imp           = or ( "->" imp )? ;                        (* right-associative *)
or            = xor ( "or" xor )* ;
xor           = and ( "xor" and )* ;
and           = not ( "and" not )* ;
not           = "not" not | cmp ;
cmp           = sum ( cmp_op sum | "is" pattern | "in" sum )? ;   (* non-chaining *)
cmp_op        = "==" | "!=" | "<" | "<=" | ">" | ">=" ;
sum           = term ( ("+" | "-") term )* ;
term          = unary ( ("*" | "/" | "%") unary )* ;
unary         = "-" unary | postfix ;
postfix       = primary ( "[" args "]" | "(" args? ")" | "." NAME )* ;
primary       = INT | STRING | "true" | "false" | "none" | NAME
              | "(" expr ")" | "(" expr "," args ")"            (* tuple *)
              | list | aggregate | if_expr | match_expr ;
list          = "[" args? "]" | "[" expr "for" binders ("where" expr)? "]" ;
aggregate     = ("any" | "all" | "count" | "exactly_one" | "at_most_one")
                "(" expr "for" binders ("where" expr)? ")" ;
if_expr       = "if" expr "{" expr "}" "else" ( if_expr | "{" expr "}" ) ;
match_expr    = "match" expr "{" arm ("," arm)* ","? "}" ;
arm           = pattern "=>" expr ;
pattern       = alt ("|" alt)* ;
alt           = NAME | NAME "(" pat_arg ("," pat_arg)* ")" | "_" ;
pat_arg       = "_" | expr ;
args          = expr ("," expr)* ;
```

### 3.1 Precedence and associativity

| Level (low → high) | Operators | Associativity |
| --- | --- | --- |
| 1 | `<->` | none: `a <-> b <-> c` is a syntax error |
| 2 | `->` | right: `a -> b -> c` = `a -> (b -> c)` |
| 3 | `or` | left |
| 4 | `xor` | left |
| 5 | `and` | left |
| 6 | `not` | prefix |
| 7 | `==` `!=` `<` `<=` `>` `>=`, `is`, `in` | none: comparisons do not chain |
| 8 | `+` `-` | left |
| 9 | `*` `/` `%` | left |
| 10 | unary `-` | prefix |
| 11 | `[...]`, `(...)`, `.name` | postfix |

Consequences that rules rely on (all checked against the reference grammar):

- `a and b -> c` is `(a and b) -> c`. Premises read naturally.
- `not K[c] is Solid` is `not (K[c] is Solid)`.
- `K[c] is Dust | Torch(_) and x` is `(K[c] is (Dust | Torch(_))) and x`.
  The pattern operator `|` lives inside `is` and never competes with `or`.
- `R[a] + 1 <= R[b]` compares integer terms.

### 3.2 Ambiguities designed out

| Hazard | Decision |
| --- | --- |
| `if x in {a, b} { ... }`: a set literal or a block? | Set and list literals use brackets: `[a, b]`. Braces are always blocks or bodies. |
| `->` versus `=>` | `->` is implication and the return type of `fn`; `=>` appears only in `match` arms. |
| `|` versus `or` | `|` only separates pattern alternatives; `|=` is its own token. |
| `=`, `:=`, `==` | `=` binds (let, params, instance values); `:=` defines (`def`, derived facts); `==` compares. |
| A statement starting with a name | The only such statement is a relation contribution `R(...) |= f;`. There are no expression statements. |
| `in` in binders versus `in` as membership | In binder lists, `x in S` introduces `x`. Anywhere else, `x in S` tests membership. |

## 4. Stages and types

### 4.1 Stages

| Stage | Produced by | Allowed in |
| --- | --- | --- |
| **const** | literals, enum variants, domain values, binders, params, facts, `fn` calls, `match`, `has`, `step`, comparisons of consts | everywhere |
| **formula** | choice tests, `var`/`def`/`relation` atoms, integer comparisons, aggregates over formulas | `require`, `def` bodies, contributions, aggregates, `minimize` |

Rules:

- Connectives accept both stages. If every operand is const, the result is
  const; otherwise it is a formula, with const operands folded.
- **Const-only positions:** `where` guards, `if` conditions (statement and
  expression), member guards, family indices, `fn` bodies, derived facts,
  ranges, and param defaults.
- Solver values never index anything; there are no element constraints by
  design. Using a formula in a const-only position is an error (E0304,
  section 10).

### 4.2 Types

| Type | Values |
| --- | --- |
| `bool` | `true`, `false` (const) |
| `int` | 64-bit signed (const) |
| enum `E` | its variants |
| subset `S of E` | the listed variants. `S` is also a const `set<E>`, so `d in Dir4` works. |
| `domain D` | instance-supplied symbols; `domain D: int` for integer-valued domains |
| grid `G` | cells; `c.x`, `c.y`, `c.z` are `int` |
| choice `K` | the members of choice `K`, including payloads such as `Torch(East)` |
| `(T1, ..., Tn)` | tuples |
| `set<T>` | finite sets (list literals, comprehensions, subsets) |
| `T?` | `T` or `none` |
| formula | solver-time boolean |
| int term | solver-time integer (`int` families, `count(...)`, `+`/`-` with consts) |

Subsets are subtypes: a `Dir4` value is accepted where `Dir6` is expected.
Grid builtins (section 5.1) preserve a subset when it is closed under the
operation. `opposite(d: Dir4)` is a `Dir4`, which the compiler checks when
the subset is declared.

### 4.3 Enum variant resolution

Variant names are resolved by the expected type (bidirectional checking):

- In `step(c, East)`, `step` expects `Dir6`, so this is `Dir6.East`.
- In `Torch(East)`, the payload is `Attach`, so this is `Attach.East`.
- In `match a { Floor => Down, East => East }`, each arm's pattern is typed
  by the scrutinee (`Attach`) and its result by the return type (`Dir6`).

When no expected type exists and several enums define the name, qualify it:
`Dir6.East`. The compiler reports the candidates (E0211).

### 4.4 `none` and the edge of the box

`step(c, d)` returns `Cell?`. Rules for `none`:

- Const functions given `none` return `none`; facts given `none` are `false`.
- `has(x)` is `true` exactly when `x` is not `none`.
- `x == none` is not allowed; use `has(x)`.
- **A family indexed by `none` takes the family's outside value:**

| Declaration | Default outside value | Override |
| --- | --- | --- |
| `choice` | none; the access is an error unless declared | `@outside(Member)`, e.g. `@outside(Air)`: `K[none] is Air` is `true`, every other member `false` |
| `var`, `def` | `false` | `@outside(true)` |
| `relation` atom `R(..., none, ...)` | `false` | none; contributions with a `none` key are dropped |
| `int` family | error | `@outside(v)` with a const `v` |

This is why the dust stair rule needs no bounds checks:
`not Kind[step(c, Up)] is Solid` is `true` at the top layer because outside
is air. That matches the physics, and it holds only because `Kind` declares
`@outside(Air)`. Without that declaration the access is a compile error, not
a silent `false`.

Optional consts coerce to their payload where a plain value is needed
(`count(...) <= max_blocks`). If the value is `none` at grounding time, it is
an error (E0412), so guard such uses with `has(max_blocks)`.

## 5. Declarations

### 5.1 `grid`

```
grid Cell(x, y, z) dirs Dir6 {
    East = +x, West = -x, North = +y, South = -y, Up = +z, Down = -z,
}
```

This declares the cell type `Cell`, the enum `Dir6`, and the coordinate names.
Each direction maps to one signed axis. The mapping is explicit because the
compiler's convention (`North` = +y, z is height) is easy to get wrong.

Builtins:
- `step(c: Cell?, d: Dir6) -> Cell?`
- `opposite(d) -> same type`
- `has(x: T?) -> bool`
- `c.x`, `c.y`, `c.z`
- `Cell(x, y, z)` literal, for instance files and tests

The instance gives the size: `grid Cell = (2, 14, 10);` in an instance file,
or `Instance::grid` from Rust.

### 5.2 `enum`, `subset`, `domain`

```
enum Attach { Floor, East, West, North, South }
subset Dir4 of Dir6 = [East, West, North, South];
domain Class;              // values from the instance
domain Case: int;          // integer-valued domain
domain Output = [sum, cout];   // inline values (tests, small models)
```

Domain order is the order the instance supplies. It is preserved everywhere,
which makes variable numbering deterministic.

### 5.3 `fact`, `param`, `fn`

```
fact on(Class, Case);                                  // supplied by the instance
fact outward_ok(c: Cell) := any(output_site(o, c) for o: Output);   // derived
param rank_levels: int = 24;
param max_blocks: int? = none;
fn perp(d: Dir4) -> set<Dir4> = match d { East | West => [North, South], North | South => [East, West] };
```

- An extern fact is a set of tuples; `on(j, k)` is a const bool.
- A derived fact (`:=`) is evaluated once per instance. It may reference other
  facts, params, and functions, but not solver families.
- Derived facts must not depend on themselves (stratified, E0502).
- `fn` bodies are single const expressions. Recursion is not allowed, so
  grounding always terminates.

### 5.4 `choice`

```
@prefer(Air) @outside(Air) @display("{c} {member}")
choice Kind[c: Cell] {
    Air,
    Dust if has(step(c, Down)),
    Torch(a: Attach) if has(step(c, dir(a))),
    ...
}
```

- For each index, exactly one *existing* member holds.
- A member exists for a given index and payload only if its `if` guard holds.
  Guards are const and may use the index binders and the member's own
  payload binders.
- A member that does not exist is the constant `false`, with no variable and
  no clause.
- The members form the type `Kind`, so facts and binders can range over them
  (`fact fixed(Cell, Kind)`, `forall m: Kind`).

### 5.5 `var`, `def`, `relation`, `int`

| Form | Variables | Notes |
| --- | --- | --- |
| `var V[i: D];` | one free boolean per index | For pure decisions with no definition. |
| `var V over R;` | one free boolean per grounded tuple of relation `R` | Accessed as `V(t...)`, sparse like `R`. |
| `def L[i: D] := f;` | a named definition | Hash-consed; full equivalence unless `@internal` allows one-sided encoding (section 7.2). |
| `relation R(p: T, ...);` | one atom per tuple that received a non-false contribution | See below. |
| `int I[i: D] in lo..=hi;` | order-encoded bounded integer | Comparisons as in section 6.4. |

**Relations.**
- `R(e1, ..., en) |= f;` inside a rule adds `f` to tuple `(e1, ..., en)`. The
  tuple's atom is the OR of all contributions to it.
- A tuple whose contributions all fold to `false`, or whose key contains
  `none`, does not exist.
- `R(...)` used as an atom is `false` for tuples that do not exist.
- Iterating `forall (a, b, ...) in R` happens after every contribution to `R`
  has been collected.
- A rule may not iterate over `R` and also contribute to `R`, directly or
  through other relations (E0503). This stratification keeps grounding
  order-independent.

## 6. Statements and formulas

### 6.1 Rules

```
@guarded(per = c)
rule "켜진 것은 켜진 원천이 있어야 함" {
    forall c: Cell { require ...; }
}
```

- Every `require` and contribution must sit inside a rule.
- The label is free text. Each clause records its rule label and its binder
  valuation (for example `c=(0,0,1), d=East`). That provenance drives the
  explained DIMACS output and UNSAT-core reports.
- Clauses that come from definitions (`def`, choices, integers) carry the
  label `정의: <Name>` or a custom `@label("...")` on the declaration.

### 6.2 `forall`, `if`, `let`

- `forall binders where g { ... }` expands over the cartesian product of the
  binders in order, keeping only valuations where `g` holds. `g` is const.
- Binder forms:
  - `x: D` ranges over a domain, enum, subset, choice, or grid.
  - `x in S` ranges over a const set, list, or range.
  - `(a, _, b) in R` ranges over the grounded tuples of a relation or fact.
- `if g { ... } else { ... }` picks one branch at grounding time.
- `let x = e;` names a const or a formula for the rest of the block. It
  creates no variable by itself (formulas are hash-consed anyway).

### 6.3 Patterns (`is`)

- `e is P` tests a choice value against pattern `P`.
- `P` is one or more alternatives `A | B | ...`. Each alternative is:
  - a member name;
  - a member with payload arguments, where each argument is a const
    expression (equality) or `_` (any payload);
  - a const value of the choice type (`Kind[c] is m`).
- `e` may be a family access, which gives a formula, or a const choice
  value, which gives a const (`m is Repeater(_)`).
- Patterns never introduce names. To quantify over a payload, write an
  aggregate: `any(Kind[c] is Torch(a) for a: Attach)`.

### 6.4 Integers

- Int terms are `I[...]`, `count(...)`, and either of those plus or minus a
  const.
- Comparisons between int terms and consts are formulas:
  - `I[a] < I[b]`, `I[a] + 1 <= I[b]`, `I[c] >= 3`, `count(...) <= n`.
- Products of two solver terms are not allowed (E0441). Linear sums of more
  than one solver term are reserved for a future pseudo-Boolean backend.
- Order encoding:
  - `I[c] >= v` is a literal for `v` in `lo+1..=hi`, with the chain
    `I >= v+1 -> I >= v`.
  - Comparisons expand to clauses. For example, `p -> I[a] < I[b]` becomes
    `p -> I[b] >= lo+1` plus `p and I[a] >= v -> I[b] >= v+1`, which is
    exactly `Encoding::strictly_below`.

### 6.5 Aggregates

| Aggregate | Meaning |
| --- | --- |
| `any(f for B where g)` | OR over the valuations |
| `all(f for B where g)` | AND over the valuations |
| `count(f for B where g)` | int term: the number of true `f` |
| `exactly_one(...)`, `at_most_one(...)` | sugar for `count(...) == 1` and `count(...) <= 1` |

- `where` filters at grounding time.
- An aggregate over zero valuations is the identity: `any` is `false`,
  `all` is `true`, `count` is `0`.
- At the top of a `require`, cardinality uses the cheap one-sided encodings:
  - pairwise up to 6 literals, otherwise a sequential counter, as in `cnf.rs`;
  - `@encoding(...)` selects another one.
- Nested inside other formulas, cardinality is reified with a totalizer.

### 6.6 Objectives

`minimize e;` and `maximize e;` take an integer term built from counts,
order-encoded integers, constants, `+`, `-`, and multiplication by a constant
(weights may come from params, so `block_cost * count(...)` with
`block_cost = 0` drops the term). All objective items are summed, `maximize`
negated, into `offset + Σ weight · [literal]` with positive weights; negative
terms are rewritten as `w + |w| · [not f]`. Terms are encoded in both
directions, so the cost of a model can be read from it.

Grounding only records the cost. A driver bounds it through a totalizer
(`Program::objective_counter`), whose outputs it assumes false to ask for a
cheaper model on the same incremental solver, so a run is anytime: it keeps
the best model so far and is optimal once a bound is unsatisfiable.
`Program::write_wcnf` writes the same problem as weighted partial MaxSAT
(hard clauses plus one soft unit per term) for external MaxSAT solvers.

Weighted sums can only be optimized; comparing one with a constant is an
error (E0441), since that needs a pseudo-Boolean encoding. The exact placer's
model minimizes non-air blocks plus optional repeater and torch weights.

## 7. Annotations

### 7.1 Reference

| Annotation | Applies to | Meaning |
| --- | --- | --- |
| `@display("template")` | choice, var, def, relation, int | Name used in the legend, explained DIMACS, and diagnostics. Placeholders are binder names, `{member}` for choices, and tuple field names for relations. |
| `@prefer(Member)` / `@prefer(false)` | choice / var | Polarity so CaDiCaL's `phase=0` starts at the preferred value. For choices, the preferred member is the negation of an "occupied" variable. |
| `@outside(v)` | choice, var, def, int | Value of a `none`-indexed access (section 4.4). |
| `@internal` | def | May be eliminated or encoded one-sided; it is not readable from Rust. |
| `@label("text")` | declarations | Rule label for clauses produced by the declaration itself. |
| `@encoding(name)` | int (`order`), cardinality statements (`pairwise`, `seqcounter`, `totalizer`, `auto`) | Encoding choice. |
| `@guarded(per = binders)` | rule | Adds a selector literal per distinct valuation of the listed binders, or one per rule without `per`. Rust assumes them true by default and can drop any for UNSAT cores, relaxation, or toggling. This generalizes `relax_soundness`. |
| `@fold` | rule | Runs the rule before all others; its `require`s must be literals (or conjunctions of them), which become fixed. A choice option fixed true fixes its siblings false. Later formulas fold fixed literals to constants, so constraints over fixed parts of the instance never reach the solver (the units are still emitted). `GroundOptions::no_fold` runs such rules as ordinary ones. It cannot contribute to relations. |

Unknown annotations are errors, so a misspelled `@encodng` cannot silently do
nothing.

### 7.2 Polarity and the `encode.rs` equivalence

`encode.rs` builds every `and`/`or` with full equivalence and writes the
per-case witnesses one-sided by hand. In rsdsl:

- Top-level `def`s are full equivalences, so Rust can read them reliably.
- `@internal` definitions and subformulas inside `require` are encoded by
  polarity (Plaisted–Greenbaum).
- So
  `require ... -> any(Contrib(s, t, f, to) and Powered[k, s] for ...)` yields
  exactly the hand-written witness variables (`w -> contributes`,
  `w -> powered`).

## 8. Instance files and Rust

An instance supplies grid sizes, domain values, facts, and params:

```
rsdsl 2;
instance inverter_1x3x2 of exact_placer;
grid Cell = (1, 3, 2);
domain Class = [off, a, out];
fact on = [(a, 1), (out, 0)];
fact switch_site = [(c, att, a) for c: Cell, att: Attach];
param rank_levels = 8;
```

- Instance values are checked against the model's declarations: names,
  arities, and types (E0601).
- List comprehensions keep large facts short.
- Rust builds the same thing through `Instance::builder()`
  (`solver_dsl_design.md` §5). Files are for tests, DIMACS exports, and
  reproducing bug reports without code.

`include "physics.rsdsl";` splices another file into the same namespace, and
duplicate names are an error. This lets the redstone physics live in one file
while each placer model adds its own interface rules.

## 9. Grounding semantics

1. Resolve names and check types and stages for the whole model.
2. Load the instance and evaluate derived facts in dependency order.
3. Create choice members, `var`s, and `int` literals by evaluating member
   guards.
4. Collect relation contributions from every rule that does not iterate over
   relations. Then, in stratified order, process the rules that do. Freeze
   each relation's tuple set before any rule iterates over it.
5. Ground the remaining rule statements into Boolean IR, folding consts and
   `none` and hash-consing.
6. Encode (section 7.2) and number the variables deterministically: by
   declaration order, then index order, then member order.

The order of declarations and rules in the file does not affect the result
beyond numbering and clause order. The rule order in the file is kept for
readability of the explained output.

## 10. Diagnostics

All errors point at model source spans. Some examples:

```
error[E0304]: guard must be known at grounding time
  --> exact_placer.rsdsl:42:30
   |
42 |     forall c: Cell where Kind[c] is Dust {
   |                          ^^^^^^^^^^^^^^^ solver variable `Kind` in a guard
   = help: move it into the formula: `require Kind[c] is Dust -> ...`

error[E0211]: `East` is ambiguous here
   = note: candidates: `Dir6.East`, `Attach.East`
   = help: write `Dir6.East`

error[E0421]: `Kind[step(c, Up)]` may be outside the grid
   = help: declare `@outside(Air)` on `choice Kind`, or guard with `has(step(c, Up))`

error[E0503]: rule "x" iterates over `Feeds` and contributes to it

warning[W0101]: rule "벽 토치·스위치는 붙은 쪽 칸이 블록이어야 함" produced no clauses
   = note: every guard folded to false for instance `inverter_1x3x2`
```

## 11. Alternatives considered

| Alternative | Why not |
| --- | --- |
| Keep rsdsl v0.1 as is (proc-macro, SCIP LP) | Domains are fixed at compile time, there are no guards or outside values, no integers, cardinality, or acyclicity, and the ILP lowering does not fit a Boolean problem (`solver_dsl_design.md` §2). |
| Rust builder API or proc-macro only | Verbose, needs a recompile per change, and macro errors are poor. A macro wrapper over the text parser can still be added. |
| ASP (clingo) syntax | Very concise for facts and rules, but stable-model semantics hide the acyclicity encoding. The clingo prototype with proper latch exclusion was no faster. `:-` syntax is also unfamiliar to the project. Facts and derived facts are borrowed from it. |
| MiniZinc | Integer-centric, offers little control over CNF encodings, needs an external toolchain, and its grid and relation patterns would be emulated. |
| Python embedding (PySAT) | Adds a runtime dependency, has no static stage checking, and grounding is slower. |
| Symbolic operators (`&&`, `||`, `!`) | Less readable next to Korean rule labels, and having two spellings invites mixing. |
| Braces for set literals | Collide with blocks (`if x in {a} { }`). Brackets avoid this. |
| Binding inside patterns (`Feeds(?s, c, ...)` or Datalog capital letters) | A typo could silently create a binder. Explicit tuple binders with `where` filters are as short and safe. |
| Silent `false` outside the grid | Wrong for `not K[outside] is Solid`. Declaring `@outside` makes the physics explicit. |

## 12. Mapping from rsdsl v0.1

| v0.1 | v2 |
| --- | --- |
| `place`, `shape`, `state` vars | `choice` (exclusive placements), `def` (derived shape and state), `var` (free) |
| `def X <-> e` | `def X := e;` |
| `sources S[...]` + `add S += e where c` + `OR(S)` | `relation R(...)` + `R(...) \|= e;` + atom `R(...)` |
| `exclude S += e` | Removed: guard the contribution with `if` instead. |
| `force A == B` | `require A <-> B;` or `require A == v;` |
| `scenario s in {0,1}` | `domain Case: int;` from the instance |
| `Observe(PIN, s=...)` | facts plus `def Observe[...]` |
| `feature NAME { ... }` | `param NAME: bool` with `if NAME { ... }` |
| `objective minimize { ... }` | `minimize e;` |
| `sum((i,j) in A * B) e` | `count(e for i: A, j: B)` |

## 13. Open questions

1. Should `match` allow guards (`East if cond =>`)? Not needed by the exact
   placer; it can be added without breaking anything.
2. Should there be parameterized rule templates (macros) for repeated
   patterns, such as the four `Feeds` contributions per neighbor? Plain
   `forall` has been enough so far.
3. Should a nested aggregate be encoded as a totalizer, or rejected in favor
   of explicit `def`s? This draft reifies it.
4. Should there be a pseudo-Boolean backend for weighted sums (`2*a + 3*b <=
   4`)? The grammar already parses it; the stage checker rejects it today.
