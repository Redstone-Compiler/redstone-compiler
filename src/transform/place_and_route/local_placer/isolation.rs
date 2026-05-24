use std::collections::{HashMap, HashSet};

use crate::graph::GraphNodeId;
use crate::world::position::Position;
use crate::world::World3D;

// OR 라우팅은 redstone path를 새로 놓으면서 기존 신호망에 물리적으로 합류할 수 있다.
// 이번 OR의 입력으로 선언된 신호만 합류해야 하고, 이미 외부 출력으로 확정된 신호망은
// 다른 route 때문에 upstream/downstream 영역이 넓어지면 안 된다.
pub(super) struct RouteIsolation {
    // 이번 route terminal에서 보여야 하는 원천 입력 위치들이다.
    // terminal upstream source가 이 set과 같아야 OR가 정확히 두 입력만 합친 것이다.
    allowed_sources: HashSet<Position>,
    // 보호해야 하는 기존 신호망의 route 전 경계다.
    // route 후에도 이 경계가 확장되지 않아야 기존 출력망이 오염되지 않는다.
    protected_boundaries: HashMap<Position, SignalBoundary>,
}

impl RouteIsolation {
    // route를 만들기 전 world에서 isolation 기준을 캡처한다.
    // allowed_inputs는 이번 OR의 논리 입력 위치이고, protected_positions는
    // 이후 route가 건드리면 안 되는 이미 확정된 출력 신호 위치다.
    pub(super) fn new(
        world: &World3D,
        allowed_inputs: impl IntoIterator<Item = Position>,
        protected_positions: HashSet<Position>,
    ) -> Self {
        let signals = SignalGraph::new(world);
        let allowed_sources = allowed_inputs
            .into_iter()
            .flat_map(|position| signals.upstream_sources(position))
            .collect();
        let protected_boundaries = protected_positions
            .into_iter()
            .map(|position| (position, signals.boundary(position)))
            .collect();

        Self {
            allowed_sources,
            protected_boundaries,
        }
    }

    // route 후보가 논리적으로 고립되어 있는지 검사한다.
    // 1. route terminal의 upstream source가 이번 OR 입력들과 정확히 같아야 한다.
    // 2. protected signal boundary가 route 전보다 넓어지면 안 된다.
    pub(super) fn accepts_or_route(&self, world: &World3D, route_path: &[Position]) -> bool {
        let Some(&terminal) = route_path.last() else {
            return false;
        };
        let signals = SignalGraph::new(world);
        signals.upstream_sources(terminal) == self.allowed_sources
            && self
                .protected_boundaries
                .iter()
                .all(|(position, before)| signals.boundary(*position).does_not_expand_from(before))
    }
}

// WorldGraphBuilder 결과를 신호 격리 검증에 쓰기 좋게 감싼 얇은 helper다.
// local placer state가 아직 모든 신호 의미를 직접 들고 있지 않기 때문에,
// 현재는 실제 world에서 물리 신호 그래프를 다시 만들어 correctness oracle로 사용한다.
struct SignalGraph {
    graph: crate::graph::world::WorldGraph,
    position_to_node: HashMap<Position, GraphNodeId>,
}

impl SignalGraph {
    // World3D를 WorldGraph로 변환하고, block position에서 graph node id로
    // 역조회할 수 있는 map을 만든다.
    fn new(world: &World3D) -> Self {
        let world = crate::world::World::from(world);
        let graph = crate::graph::world::WorldGraphBuilder::new(&world).build();
        let position_to_node = graph
            .positions
            .iter()
            .map(|(node_id, position)| (*position, *node_id))
            .collect();

        Self {
            graph,
            position_to_node,
        }
    }

    // 특정 position이 속한 신호망의 upstream/downstream 물리 영역을 함께 캡처한다.
    // 이 값은 route 전후 비교에 쓰인다.
    fn boundary(&self, position: Position) -> SignalBoundary {
        SignalBoundary {
            upstream: self.upstream_positions(position),
            downstream: self.downstream_positions(position),
        }
    }

    // position으로 들어오는 신호를 끝까지 거슬러 올라가 source 위치만 모은다.
    // OR terminal 검증에서는 이 source set이 이번 OR 입력 source set과 같아야 한다.
    fn upstream_sources(&self, position: Position) -> HashSet<Position> {
        let Some(node_id) = self.node_id(position) else {
            return HashSet::new();
        };

        let mut visited = HashSet::new();
        let mut sources = HashSet::new();
        self.collect_upstream_sources(node_id, &mut visited, &mut sources);
        sources
    }

    // position의 upstream에 있는 모든 물리 block 위치를 모은다.
    // protected boundary가 route 후에 새 upstream을 얻었는지 비교하기 위한 값이다.
    fn upstream_positions(&self, position: Position) -> HashSet<Position> {
        let Some(node_id) = self.node_id(position) else {
            return HashSet::new();
        };

        let mut visited = HashSet::new();
        self.collect_upstream_positions(node_id, &mut visited);
        self.positions(visited)
    }

    // position의 downstream에 있는 모든 물리 block 위치를 모은다.
    // protected output망에 새 consumer가 붙는 경우를 잡기 위한 값이다.
    fn downstream_positions(&self, position: Position) -> HashSet<Position> {
        let Some(node_id) = self.node_id(position) else {
            return HashSet::new();
        };

        let mut visited = HashSet::new();
        self.collect_downstream_positions(node_id, &mut visited);
        self.positions(visited)
    }

    // WorldGraph는 node id 기준으로 edge를 들고 있으므로 position을 node id로 바꾼다.
    fn node_id(&self, position: Position) -> Option<GraphNodeId> {
        self.position_to_node.get(&position).copied()
    }

    // graph node id set을 다시 물리 position set으로 바꾼다.
    fn positions(&self, node_ids: HashSet<GraphNodeId>) -> HashSet<Position> {
        node_ids
            .into_iter()
            .filter_map(|node_id| self.graph.positions.get(&node_id).copied())
            .collect()
    }

    // producer edge를 따라 올라가다가 더 이상 producer가 없는 graph node를 source로 본다.
    // source는 논리 input 이름이 아니라 현재 world에서의 물리 source 위치다.
    fn collect_upstream_sources(
        &self,
        node_id: GraphNodeId,
        visited: &mut HashSet<GraphNodeId>,
        sources: &mut HashSet<Position>,
    ) {
        if !visited.insert(node_id) {
            return;
        }

        let producers = self
            .graph
            .graph
            .producers
            .get(&node_id)
            .cloned()
            .unwrap_or_default();
        if producers.is_empty() {
            if let Some(position) = self.graph.positions.get(&node_id) {
                sources.insert(*position);
            }
            return;
        }

        for producer in producers {
            self.collect_upstream_sources(producer, visited, sources);
        }
    }

    // producer edge 전체를 따라가며 upstream 영역을 수집한다.
    fn collect_upstream_positions(&self, node_id: GraphNodeId, visited: &mut HashSet<GraphNodeId>) {
        if !visited.insert(node_id) {
            return;
        }

        for producer in self
            .graph
            .graph
            .producers
            .get(&node_id)
            .into_iter()
            .flatten()
        {
            self.collect_upstream_positions(*producer, visited);
        }
    }

    // consumer edge 전체를 따라가며 downstream 영역을 수집한다.
    fn collect_downstream_positions(
        &self,
        node_id: GraphNodeId,
        visited: &mut HashSet<GraphNodeId>,
    ) {
        if !visited.insert(node_id) {
            return;
        }

        for consumer in self
            .graph
            .graph
            .consumers
            .get(&node_id)
            .into_iter()
            .flatten()
        {
            self.collect_downstream_positions(*consumer, visited);
        }
    }
}

// 특정 신호 위치 주변의 관측 가능한 물리 신호 영역이다.
// upstream은 그 위치를 구동하는 쪽, downstream은 그 위치가 구동하는 쪽이다.
struct SignalBoundary {
    upstream: HashSet<Position>,
    downstream: HashSet<Position>,
}

impl SignalBoundary {
    // route 후 boundary가 route 전 boundary보다 넓어졌는지 확인한다.
    // 새 upstream/downstream이 생겼다면 protected signal에 다른 신호가 합류했거나
    // protected signal이 새 consumer를 구동하게 된 것이므로 route를 reject한다.
    fn does_not_expand_from(&self, before: &Self) -> bool {
        self.upstream.difference(&before.upstream).next().is_none()
            && self
                .downstream
                .difference(&before.downstream)
                .next()
                .is_none()
    }
}
