```text
// 이 조건들을 모두 만족하는 부품 배치와 신호 배정이 하나라도 있는지 묻는다.
// ∃는 “하나라도 존재”, ∀는 “모두”, ∧는 “그리고”, ⇒는 “이면”, ⇔는 “정확히 같은 조건”이다.
∃κ,σ,r,s,J,Same : Φ(I,κ,σ,r,s,J,Same)

// Φ(파이)는 아래 조건들을 전부 묶은 판정식이다. 한 조건이라도 어기면 탈락한다.
Φ ≔ Domain ∧ Support ∧ Labels ∧ Power ∧ Order ∧ Coverage ∧ TorchRule
   ∧ NoLock ∧ SwitchGap ∧ Fixed ∧ Inputs ∧ Outputs ∧ Bounds
   ∧ Symmetry ∧ Mirror ∧ Useful ∧ Ghost ∧ Carry ∧ Monotone ∧ Seam ∧ Timing

// 조건을 통과한 배치 중 비용이 가장 작은 것을 찾는다.
// 기본 비용은 공기가 아닌 칸의 수이며, 리피터·토치에 추가 비용을 줄 수도 있다.
min (wB·Σc[κc≠Air] + wR·Σc[R(c)] + wT·Σc[T(c)])  subject to Φ

where

  // I는 주어진 문제(상자 크기, 입력 함수, 핀 위치, 옵션 등), E는 솔버가 찾을 답이다.
  E ≔ (κ,σ,r,s,J,Same); I ≔ (C,K,Q,O,Facts,Params)
  // C는 배치할 수 있는 칸들, K는 사용을 허용한 신호 함수들이다.
  // Q는 검사할 입력·저장 상태 조합들, O는 만들어야 할 출력들이다.
  C ≔ [0,X) × [0,Y) × [0,Z) ⊂ ℤ³;  K ≔ Class;  Q ≔ Case;  O ≔ Output
  // E/W/N/S는 동서남북, U/D는 위아래다. D₄는 수평 네 방향, D₆는 여섯 방향이다.
  // a는 토치·스위치가 붙은 방향이다. δ(a)는 받침 쪽, d̄는 d의 반대쪽이다.
  d ∈ D₄ ≔ {E,W,N,S};  e ∈ D₆ ≔ D₄∪{U,D};  a ∈ A ≔ {Floor}∪D₄
  δ(Floor)=D; δ(a)=a (a≠Floor);  d̄ ≔ opposite(d);  c+d ≔ step(c,d)
  // ⊥(d)는 d에 직각인 수평 방향들이다. [조건]은 참이면 1이어서 개수를 셀 때 쓴다.
  ⊥(E)=⊥(W)={N,S}; ⊥(N)=⊥(S)={E,W};  [p] ≔ if p then 1 else 0
  ⋁∅=false; ⋀∅=true;  자유 첨자는 해당 영역에서 전칭 양화

  // κc는 c 칸의 부품, σc는 그 칸이 어떤 입력에서 켜지는지 나타내는 신호 이름이다.
  // r은 전원이 생긴 근거를 추적하는 순서, s는 단계다. timing 옵션에서는 s를 도착 시각으로 쓴다.
  κc ∈ {Air,Solid,Dust} ⊎ {Torch(a)} ⊎ {Repeater(d)} ⊎ {Switch(a,k) : k∈K}
  σc ∈ {Empty}⊎K;  rc∈[0,rankMax]; sc∈[0,stageMax]
  // B/D/T/R/W는 각각 고체 블록/가루/토치/리피터/스위치라는 짧은 이름이다.
  // R(c,d)의 d는 리피터가 입력을 읽는 쪽이다. 출력은 반대쪽 d̄다.
  B(c)⇔κc=Solid; D(c)⇔κc=Dust; T(c,a)⇔κc=Torch(a)
  R(c,d)⇔κc=Repeater(d); W(c,a,k)⇔κc=Switch(a,k)
  T(c)⇔∃a:T(c,a); R(c)⇔∃d:R(c,d); W(c)⇔∃a,k:W(c,a,k)
  c∉C ⇒ κc=Air ∧ σc=Empty
  // P(q,c)는 입력·상태 조합 q에서 c가 켜진다는 뜻이다.
  // Z(c)는 현재 꺼져 있다는 뜻이 아니라, 모든 조합에서 꺼지는 신호라는 뜻이다.
  P(q,c)⇔⋁{σc=k : on(k,q)}; Z(c)⇔⋁{σc=k : unpowered(k)}
  // Given은 앞선 배치에서 이미 검증해 넘겨준 신호다. 일부 전원 근거 검사를 생략한다.
  Given(c)⇔∃k:givenSig(c,k)
  // 래치의 피드백을 풀기 위해, 다음 상태를 만드는 쪽과 저장 상태를 읽는 쪽의 경계를 표시한다.
  // 실제 배선을 없애는 것이 아니라, 그 경계에서 일부 신호 일치·순서 제약을 예외 처리한다.
  Cut(u,v)⇔(D(v)∨R(v))∧⋁{σu=n∧σv=m : cut(n,m)}
  // 리피터가 상자 밖을 향해도 되는 위치다. 외부 출력 위치나 이미 고정된 리피터가 해당한다.
  Outward(c)⇔(∃o:outSite(o,c)∧driving(o))∨(∃d:fixed(c,Repeater(d)))

  // 부품을 놓는 데 필요한 이웃 칸이 상자 안에 있어야 한다. 스위치는 지정된 후보 위치에만 둔다.
  Domain ≔ ∀c∈C,a,d,k :
    (D(c)⇒c+D∈C) ∧ (T(c,a)⇒c+δ(a)∈C)
    ∧ (R(c,d)⇒c+D∈C ∧ c+d∈C ∧ (c+d̄∈C∨Outward(c)))
    ∧ (W(c,a,k)⇒swSite(c,a,k))

  // 가루·리피터·바닥 토치 아래에는 블록이 있어야 한다. 벽 토치·스위치도 붙일 블록이 필요하다.
  Support ≔ ∀c,a,k :
    ((D(c)∨R(c)∨T(c,Floor))⇒B(c+D))
    ∧ (T(c,a)∧a≠Floor⇒B(c+δ(a))) ∧ (W(c,a,k)⇒B(c+δ(a)))

  // 부품과 신호 이름을 일치시킨다. 공기는 신호가 없고, 스위치는 자기 입력 신호를 갖는다.
  // 항상 꺼진 토치 등을 제외하는 현재 모델의 제한도 여기 포함된다.
  Labels ≔ ∀c,a,k :
    (κc=Air⇔σc=Empty)
    ∧ (Z(c)⇒B(c)∨(allowUnpoweredWires∧(D(c)∨R(c))))
    ∧ (W(c,a,k)⇒σc=k) ∧ (T(c)⇒σc≠Empty∧¬Z(c))

  // 가루 선이 어느 방향으로 이어지는지 계산한다. 평지뿐 아니라 한 칸 위·아래의 가루도 본다.
  Conn(c,d) ⇔ D(c) ∧ (D(c+d)∨T(c+d)∨W(c+d)∨R(c+d,d)∨R(c+d,d̄)
                        ∨(¬B(c+U)∧D(c+d+U))∨(¬B(c+d)∧D(c+d+D)))
  // 가루 선이 실제로 어느 쪽을 향하는지 정한다. 그 방향이 옆 블록에 전원을 줄 수 있는지 좌우한다.
  Points(c,d) ⇔ D(c) ∧ (Conn(c,d)∨⋀{¬Conn(c,p):p∈⊥(d)})
  // H는 강전원을 줄 부품이 연결돼 있는지, Hq는 그 부품이 q에서 실제로 켜져 있는지다.
  // 약전원만 받은 블록을 통해 옆 가루까지 켜지는 것으로 잘못 계산하지 않기 위해 둘을 구분한다.
  H(c) ⇔ B(c) ∧ (T(c+D)∨⋁d R(c+d,d)∨⋁a,k W(c+opposite(δ(a)),a,k))
  Hq(c) ⇔ B(c) ∧ ((T(c+D)∧P(q,c+D))∨⋁d(R(c+d,d)∧P(q,c+d))
                       ∨⋁a,k(W(c+opposite(δ(a)),a,k)∧P(q,c+opposite(δ(a)))))

  // 받는 부품 v가 어느 쪽에서 전원을 읽을 수 있는지다.
  // DS/RS/BS는 받는 쪽이 가루/리피터/블록임을 뜻한다. 리피터는 입력 쪽 하나만 읽는다.
  In(v,j) ≔ D₆       (j=DS ∧ D(v), 또는 j=BS ∧ B(v))
           ≔ {d}      (j=RS ∧ R(v,d))
           ≔ ∅        (그 외)

  // 보내는 부품 u가 “어떤 종류의 이웃에게, 어느 방향으로” 전원을 줄 수 있는지다.
  // 예: 토치는 위쪽 블록에는 전원을 주지만, 가루에는 위쪽을 제외한 방향으로 준다.
  // ∅는 전달 방향이 없음, ∖는 해당 방향을 제외한다는 뜻이다. 켜짐 여부는 뒤의 Power에서 검사한다.
  Out(u,j) ≔ 아래 표의 방향 집합

    κu              DS                  RS               BS
    ──────────────────────────────────────────────────────────────
    Dust            D₄                  D₄               {D}∪{d:Points(u,d)}
    Torch(a)        D₆∖{U}              D₄               {U}
    Repeater(d)     {d̄}                 {d̄}              {d̄}
    Solid           if H(u) then D₆     D₄               ∅
                    else ∅
    Switch(a,k)     D₆∖{δ(a)}           D₄∖{δ(a)}        {δ(a)}
    Air             ∅                   ∅                ∅

  // u와 v가 바로 이웃하고, u의 출력 방향과 v의 입력 방향이 마주 보면 연결된다.
  // 예: u 동쪽의 리피터가 서쪽을 입력으로 읽으면, u→v의 방향이 맞는다.
  Face(u,v,j) ⇔ ∃e∈D₆ : v=u+e ∧ e∈Out(u,j) ∧ opposite(e)∈In(v,j)
  // 가루끼리는 대각선으로 한 칸 올라가는 계단 연결도 된다. 낮은 가루 바로 위가 막히면 안 된다.
  // {u,v}처럼 순서 없는 쌍으로 써서 올라가는 방향과 내려가는 방향을 한 식에 담았다.
  Stair(u,v) ⇔ D(u)∧D(v) ∧
    ∃c∈C,d∈D₄ : {u,v}={c,c+d+U} ∧ ¬B(c+U)

  // src는 전원을 보내는 부품 종류를 붙이는 이름표다. 이후 블록→가루 관계를 따로 처리할 때 쓴다.
  src(Dust)=FD; src(Torch(a))=FT; src(Repeater(d))=FR
  src(Solid)=FB; src(Switch(a,k))=FW; src(Air)=∅

  // F는 최종 전원 연결 관계다. 면끼리 맞닿은 연결과 가루 계단 연결을 합친 것이다.
  F(u,v,f,j) ⇔ u,v∈C ∧ f=src(κu) ∧
    (Face(u,v,j) ∨ (f=FD ∧ j=DS ∧ Stair(u,v)))

  // 유효한 위치·받침을 전제로 하면, 이 표와 공통식이 기존에 나열한 전원 관계를 표현한다.
  Domain∧Support ⇒ (F기존(u,v,f,j) ⇔ F(u,v,f,j))

  FD=FromDust; FT=FromTorch; FR=FromRepeater; FB=FromSolid; FW=FromSwitch
  DS=DustSink; RS=RepeaterSink; BS=SolidSink
  // ε는 연결 하나, Jε는 그 연결을 “이 칸이 켜지는 근거”로 선택했다는 뜻이다.
  // Open은 상태 루프를 끊어 계산하는 경계다.
  ε=(u,v,f,j); Fε⇔ε∈F; Jε∈{false,true}; Openε⇔j≠BS∧Cut(u,v)
  // 블록→가루에서는 블록의 전체 전원이 아니라 강전원만 읽는다. 다른 연결은 보내는 칸의 전원을 읽는다.
  V(q,ε) ≔ if f=FB∧j=DS then Hq(u) else P(q,u)

  // 보내는 쪽이 켜지면 받는 쪽도 켜져야 한다. 가루·리피터는 해당 원천과 켜짐이 같아야 한다.
  // 블록은 여러 원천을 OR처럼 합칠 수 있으므로 역방향 조건을 걸지 않는다. Cut 경계는 예외다.
  Power ≔ ∀ε,q : Fε∧¬Openε ⇒
    (V(q,ε)⇒P(q,v)) ∧ (j≠BS⇒(P(q,v)⇒V(q,ε)))

  // 선택한 전원 근거는 실제 연결이어야 하고, 정해진 순서도 따라야 한다.
  // 가루끼리 서로를 근거로 삼아 전원 없이 켜지는 식의 순환을 막는다. 리피터는 단계를 올린다.
  Order ≔ ∀ε : ¬Given(v) ⇒
    (Jε⇒Fε) ∧ (Jε∧j≠BS⇒¬Z(u))
    ∧ (Jε∧j=RS∧stageMax>0∧¬Cut(u,v)⇒su<sv)
    ∧ (Jε∧j≠RS∧rankMax>0⇒ru<rv)
    ∧ (Jε∧j≠RS∧stageMax>0∧¬(j=DS∧Cut(u,v))⇒su≤sv)

  // 켜진다고 표시한 칸에는 켜질 근거가 있어야 한다.
  // 특히 블록은 각 입력 조합에서 실제로 켜진 원천이 하나 이상 있어야 한다.
  Coverage ≔ ∀c,q : ¬Given(c) ⇒
    ((D(c)∨R(c))∧¬Z(c)⇒⋁{Jε:ε=(u,c,f,j)∈F∧j≠BS})
    ∧ (P(q,c)∧B(c)⇒⋁{Jε∧P(q,u):ε=(u,c,f,BS)∈F})

  // 토치는 붙어 있는 블록이 꺼지면 켜지고, 켜지면 꺼진다. 단계도 받침보다 뒤여야 한다.
  TorchRule ≔ ∀c,a,q :
    (T(c,a)⇒(P(q,c)⇔¬P(q,c+δ(a))))
    ∧ (T(c,a)∧stageMax>0∧¬Given(c)⇒s(c+δ(a))<sc)

  // 다른 리피터의 옆을 구동해 잠그는 배치는 현재 모델에서 금지한다. Minecraft에서 불가능하다는 뜻은 아니다.
  NoLock ≔ ∀c,d,p∈⊥(d) : ¬(R(c,d)∧R(c+d̄,p))
  // 스위치 옆 블록 주변의 토치·리피터 배치 중, 현재 모델이 안전하게 다루지 않는 조합을 금지한다.
  SwitchGap ≔ ∀c,a,k,e≠δ(a),b∈A,d : swSite(c,a,k) ⇒
    ¬(W(c,a,k)∧T(c+e+opposite(δ(b)),b))
    ∧ ¬(W(c,a,k)∧B(c+e)∧R(c+e+d,d̄))

  // 사용자나 앞선 배치 단계가 지정한 부품·신호를 그대로 유지한다.
  Fixed ≔ (∀fixed(c,m):κc=m) ∧ (∀givenSig(c,k):σc=k)
  // 이번 배치에 필요한 입력마다 스위치를 정확히 하나 둔다.
  Inputs ≔ ∀k : input(k)∧present(k) ⇒ Σ{[W(c,a,k)]:swSite(c,a,k)}=1
  // 출력 o를 c에서 읽을 수 있는지 판정한다. 신호가 맞고 출력 종류에 맞는 부품이어야 한다.
  // 상자 밖을 구동하는 출력은 토치 또는 바깥을 향한 리피터로 제한한다.
  Obs(o,c) ⇔ outSite(o,c) ∧ ⋁{σc=k:outClass(o,k)} ∧
    (if driving(o) then T(c)∨⋁{R(c,d):c+d̄∉C}
     else if solidOut(o) then B(c) else D(c)∨R(c)∨T(c))
  // 모든 출력은 허용된 위치 중 적어도 한 곳에서 관측할 수 있어야 한다.
  Outputs ≔ ∀o∈O : ⋁{Obs(o,c):outSite(o,c)}
  // 설정한 최대 블록·리피터 수를 지킨다. minTorches가 주어지면 토치와 받침 블록의 최소 수도 강제한다.
  Bounds ≔ (maxBlocks≠∅⇒Σc[κc≠Air]≤maxBlocks)
          ∧ (maxRepeaters≠∅⇒Σc[R(c)]≤maxRepeaters)
          ∧ (minTorches>0⇒Σc[T(c)]≥minTorches ∧ Σc[B(c)]≥minTorches)

  // Fed는 c로 들어오는 연결 중 전원 근거로 선택한 것이 있다는 뜻이다.
  Fed(c)⇔⋁{Jε:ε=(u,c,f,j)∈F}
  // 옵션이 켜져 있으면 중복된 근거 선택을 줄인다.
  // 가루·리피터의 근거는 최대 하나로, 들어오는 근거가 없는 칸의 불필요한 순위는 0으로 둔다.
  Symmetry ≔ symmetryBreaking ⇒ ∀c :
    Σ{[Jε]:ε=(u,c,f,j)∈F∧j≠BS}≤1
    ∧ (rankMax>0∧rc≥1⇒Fed(c)) ∧ (stageMax>0∧sc≥1⇒Fed(c)∨T(c))
  // 거울 비교에서 토치 위치만 볼지, 공기가 아닌 모든 칸을 볼지 고른다.
  x(c) ≔ if mirrorTorches then T(c) else κc≠Air
  // 앞부분이 같은 동안 다음 칸을 비교해, 거울상 중 정해진 순서의 대표를 남긴다.
  // Same은 “여기까지 같음”이다. 입력·고정 조건도 대칭인 경우에만 host가 mirror를 제공해야 한다.
  Mirror ≔ ∀mirror(g,i,c,m) :
    (i=0⇒Same(g,i)) ∧ (Same(g,i)∧x(c)⇒x(m))
    ∧ (Same(g,i)∧(x(c)⇔x(m))⇒Same(g,i+1))
  // 옵션이 켜지면 가루·리피터·토치는 다른 칸의 전원 근거가 되거나 출력으로 쓰여야 한다.
  // 쓰이지 않는 가루도 주변 선 모양을 바꿀 수 있으므로, 이 옵션은 모든 해를 보존한다고 볼 수 없다.
  Useful ≔ ∀c : requireUseful∧¬Given(c)∧(T(c)∨R(c)∨D(c)) ⇒
    ⋁{Jε:ε=(c,v,f,j)∈F} ∨ ⋁{Obs(o,c):outSite(o,c)}

  // 가산기 타일을 반복 연결할 때 앞 타일을 대신하는 가상 영역이다. 캐리 입력 스위치와 블록만 허용한다.
  Ghost ≔ ∀c : ghost(c)∧κc≠Air ⇒
    ⋁{W(l):carrySite(l,s,t,p)∧l=c} ∨ ⋁{W(l)∧B(c):carrySite(l,s,t,p)∧s=c}
  // 선택한 캐리 입력 위치에 맞춰 입력 토치와 출력 블록 위치를 맞춘다.
  Carry ≔ ∀carrySite(l,s,t,p) : W(l)⇒T(t,W)∧⋀{Obs(o,p):solidOut(o)}
  // 한 방향으로만 변해야 하는 신호는 같은 방향으로 변하는 신호에서만 받게 한다.
  // 타일을 연쇄 연결할 때 캐리에 일시적인 펄스가 생기는 것을 억제하려는 조건이다.
  Monotone ≔ ∀(u,v,f,j)∈F, M∈{inc,dec} : monotoneSignals ⇒
    (⋁{σv=k:M(k)}⇒⋁{σu=k:M(k)})
  // 옆 타일의 신호가 섞이지 않도록 경계의 불필요한 가루·토치·리피터 상호작용을 막는다.
  Seam ≔ ∀seam(c,d,n) : seamIsolation ⇒
    ¬(D(c)∧(D(n)∨T(n)∨W(n))) ∧ ¬(D(c)∧(R(n,d)∨R(n,d̄)))
    ∧ ¬(D(c)∧D(n+U)∧¬B(c+U)) ∧ ¬(Points(c,d)∧B(n))
    ∧ ¬(T(c)∧(D(n)∨R(n,d̄))) ∧ ((R(c,d)∨R(c,d̄))⇒κn=Air)
    ∧ ¬(H(c)∧D(n)) ∧ ¬(B(c)∧R(n,d̄)) ∧ (W(c)⇒κn=Air)

  // 옵션이 켜지면 모든 전원 연결에 시간 순서를 건다. 리피터는 1틱, 가루·블록은 0틱이다.
  // 토치도 1틱 이상 뒤여야 하고, 출력은 지정된 시간 안에 도달해야 한다. 이 식에는 Cut 예외가 없다.
  Timing ≔ timing ⇒
    (∀(u,v,f,j)∈F : su+[j=RS]≤sv)
    ∧ (∀c,a : Given(c)∧T(c,a)⇒s(c+δ(a))<sc)
    ∧ (∀delay(o,d),c : outSite(o,c)∧Obs(o,c)⇒sc≤d)

  // 실제 탐색에서는 Φ에 더해, 이미 거절한 배치를 막는 절과 발견한 순환을 막는 절을 추가한다.
  Ψₙ ≔ Φ ∧ ⋀i≤n(⋁{κc≠m:(c,m)∈Blockedᵢ∧c∈C}) ∧ ⋀Λₙ
  Λₙ ≔ acyclic.rs의 루프 절
  // 솔버가 찾은 배치를 실제 시뮬레이터 검사까지 통과시켜야 최종 결과로 받아들인다.
  Accepted(E) ⇔ Ψₙ(I,E) ∧ Verifier(decode(E))

  // 이 파일은 현재 구현을 읽기 쉽게 적은 명세다. 실행 입력이나 Minecraft 전체의 정식 증명은 아니다.
  Φ : exact_placer.rsdsl · 3e9a4cc · 일반 실행의 모든 guard 활성
  Φ ≠ Minecraft 전체 의미론;  이 파일 ≠ SMT-LIB;  timeout ⇒ unknown
  // 마지막 식은 “모든 유효한 Minecraft 배치를 이 표현으로 담을 수 있다”는 주장이다. 아직 증명되지 않았다.
  // G는 입력 그래프, L은 물리 배치, Inst는 입력 변환, Rep는 표현과 배치의 대응, Validₘ은 물리적 유효성이다.
  ∀G,I,L : Inst(G,I)∧Validₘ(G,L) ⇒ ∃E:Rep(E,L)∧Φ(I,E)   [미증명]
```
