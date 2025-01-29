use std::collections::{HashMap, HashSet};
use std::usize;

use eyre::ensure;
use indicatif::{ParallelProgressIterator, ProgressStyle};
use itertools::{iproduct, Itertools};
use rand::rngs::StdRng;
use rand::seq::IteratorRandom;
use rand::SeedableRng;
use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::graph::logic::LogicGraph;
use crate::graph::world::builder::{PlaceBound, PropagateType};
use crate::graph::world::WorldGraph;
use crate::graph::{GraphNode, GraphNodeId, GraphNodeKind};
use crate::logic::LogicType;
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::{DimSize, Position};
use crate::world::world::World3D;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PlacedNode {
    position: Position,
    block: Block,
}

impl PlacedNode {
    pub fn new(position: Position, block: Block) -> Self {
        Self { position, block }
    }

    pub fn new_cobble(position: Position) -> Self {
        Self {
            position,
            block: Block {
                kind: BlockKind::Cobble {
                    on_count: 0,
                    on_base_count: 0,
                },
                direction: Direction::None,
            },
        }
    }

    pub fn new_redstone(position: Position) -> Self {
        Self {
            position,
            block: Block {
                kind: BlockKind::Redstone {
                    on_count: 0,
                    state: 0,
                    strength: 0,
                },
                direction: Direction::None,
            },
        }
    }

    pub fn is_propagation_target(&self) -> bool {
        self.block.kind.is_stick_to_redstone() || self.block.kind.is_repeater()
    }

    // signal을 보낼 수 있는 부분들의 위치를 반환합니다.
    pub fn propagation_bound(&self, world: Option<&World3D>) -> Vec<PlaceBound> {
        PlaceBound(PropagateType::Soft, self.position, self.block.direction)
            .propagation_bound(&self.block.kind, world)
    }

    pub fn has_conflict(&self, world: &World3D, except: &HashSet<Position>) -> bool {
        if !world.size.bound_on(self.position) {
            return true;
        }

        if self.block.kind.is_cobble() {
            return self.has_cobble_conflict(world, except);
        }

        if !world[self.position].kind.is_air() {
            return true;
        }

        // 다른 블록에 signal을 보낼 수 있는 경우
        let bounds = self.propagation_bound(Some(world));
        bounds
            .into_iter()
            .filter(|bound| bound.is_bound_on(world) && !except.contains(&bound.position()))
            .any(|bound| !bound.propagate_to(world).is_empty())
    }

    fn has_cobble_conflict(&self, world: &World3D, except: &HashSet<Position>) -> bool {
        if world[self.position].kind.is_cobble() {
            return false;
        }
        if !world[self.position].kind.is_air() {
            return true;
        }

        if let Some(bottom) = self.position.walk(Direction::Bottom) {
            // 다른 레드스톤 연결을 끊어버리는 경우
            if world[bottom].kind.is_redstone()
                && (self.position.cardinal().iter())
                    .any(|&pos| world.size.bound_on(pos) && world[pos].kind.is_redstone())
            {
                return true;
            }

            // 레드스톤을 끊어버리는 경우는 예외 케이스로 반영하고 싶이 않아서 except는 여기서 체크
            if except.contains(&bottom) {
                return false;
            }

            // 바로 아래쪽에 Torch가 있는 경우
            if world[bottom].kind.is_torch() {
                return true;
            }
        }

        return false;
    }

    // 다른 블록의 signal을 받을 수 있는 경우
    fn has_short(&self, world: &World3D, except: &HashSet<Position>) -> bool {
        assert!(self.block.kind.is_redstone());

        PlaceBound::propagated_from(self.position, &self.block.kind, world)
            .into_iter()
            .filter(|bound| !except.contains(&bound.position()))
            .next()
            .is_some()
    }

    fn has_connection_with(&self, world: &World3D, target: Position) -> bool {
        assert!(self.block.kind.is_redstone());
        assert!(world[target].kind.is_stick_to_redstone());

        self.position
            .cardinal()
            .into_iter()
            .chain(Some(self.position.up()))
            .any(|pos| pos == target)
    }
}

pub struct LocalPlacer {
    graph: LogicGraph,
    config: LocalPlacerConfig,
    visit_orders: Vec<GraphNodeId>,
}

#[derive(Default)]
pub struct LocalPlacerConfig {
    greedy_input_generation: bool,
    step_sampling_policy: SamplingPolicy,
    // torch place시 input과 direct로 연결되도록 강제한다.
    route_torch_directly: bool,
    // 최대 routing 거리를 지정한다.
    max_route_step: usize,
    route_step_sampling_policy: SamplingPolicy,
}

#[derive(Default, Copy, Clone)]
pub enum SamplingPolicy {
    #[default]
    None,
    Take(usize),
    Random(usize),
}

impl SamplingPolicy {
    pub fn sample<T>(self, src: Vec<T>) -> Vec<T> {
        match self {
            SamplingPolicy::None => src,
            SamplingPolicy::Take(count) => src.into_iter().take(count).collect(),
            SamplingPolicy::Random(count) => src
                .into_iter()
                .choose_multiple(&mut Self::placer_rng(), count),
        }
    }

    fn placer_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }
}

pub const K_MAX_LOCAL_PLACE_NODE_COUNT: usize = 25;

type PlacerQueue = Vec<(World3D, HashMap<GraphNodeId, Position>)>;

impl LocalPlacer {
    pub fn new(graph: LogicGraph, config: LocalPlacerConfig) -> eyre::Result<Self> {
        let visit_orders = graph.topological_order();
        let result = Self {
            graph,
            config,
            visit_orders,
        };
        result.verify()?;
        Ok(result)
    }

    fn verify(&self) -> eyre::Result<()> {
        ensure!(self.graph.nodes.len() > 0, "");
        ensure!(
            self.graph.nodes.len() <= K_MAX_LOCAL_PLACE_NODE_COUNT,
            "too large graph"
        );

        for node_id in &self.graph.nodes {
            let kind = &self.graph.find_node_by_id(node_id.id).unwrap().kind;
            ensure!(
                kind.is_input() || kind.is_output() || kind.is_logic(),
                "cannot place"
            );
            if let Some(logic) = kind.as_logic() {
                ensure!(logic.is_not() || logic.is_or(), "cannot place");
            }
        }

        Ok(())
    }

    pub fn generate(&self, dim: DimSize, finish_step: Option<usize>) -> Vec<World3D> {
        tracing::info!("generate starts");

        let mut queue = PlacerQueue::new();
        queue.push((World3D::new(dim), Default::default()));

        let mut step = 0;
        while step < self.visit_orders.len() && Some(step) != finish_step {
            let prev_len = queue.len();

            // 1. Generate places and routes
            let next_queue = self.do_step(step, queue);
            let next_len = next_queue.len();

            // 2. Sampling
            queue = self.config.step_sampling_policy.sample(next_queue);

            step += 1;
            tracing::info!(
                "from {prev_len} -> generated {next_len} -> sampled {}",
                queue.len()
            );
        }

        tracing::info!("generate complete");
        queue.into_iter().map(|(world, _)| world).collect()
    }

    fn do_step(&self, step: usize, queue: PlacerQueue) -> PlacerQueue {
        let node = self.graph.find_node_by_id(self.visit_orders[step]).unwrap();
        tracing::info!("[{}/{}] {node}", step + 1, self.visit_orders.len());

        queue
            .into_par_iter()
            .panic_fuse()
            .progress_with_style(progress_style())
            .flat_map(|(world, pos)| {
                self.generate_place_and_route(node, world, &pos)
                    .into_iter()
                    .map(|(world, place_position)| {
                        let mut nodes_position = pos.clone();
                        nodes_position.insert(node.id, place_position);
                        (world, nodes_position)
                    })
                    .collect_vec()
            })
            .collect()
    }

    fn generate_place_and_route(
        &self,
        node: &GraphNode,
        world: World3D,
        positions: &HashMap<GraphNodeId, Position>,
    ) -> Vec<(World3D, Position)> {
        match node.kind {
            GraphNodeKind::Input(_) => input_node_kind()
                .into_iter()
                .flat_map(|kind| generate_inputs(&self.config, &world, kind))
                .collect(),
            GraphNodeKind::Output(_) => {
                // Noop
                vec![(world.clone(), positions[&node.inputs[0]])]
            }
            GraphNodeKind::Logic(logic) => match logic.logic_type {
                LogicType::Not => not_node_kind()
                    .into_iter()
                    .flat_map(|kind| {
                        generate_place_and_routes(
                            &self.config,
                            &world,
                            positions[&node.inputs[0]],
                            kind,
                        )
                    })
                    .collect(),
                LogicType::Or => {
                    assert_eq!(node.inputs.len(), 2);
                    generate_or_routes(
                        &self.config,
                        &world,
                        positions[&node.inputs[0]],
                        positions[&node.inputs[1]],
                    )
                    .into_iter()
                    .map(|(worlds, positions)| (worlds, positions.last().copied().unwrap()))
                    .collect()
                }
                _ => unreachable!(),
            },
            _ => unreachable!(),
        }
    }
}

fn progress_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.green} [{eta_precise}] [{bar:40.cyan/blue}] {pos:>7}/{len:7} {msg}",
    )
    .unwrap()
    .progress_chars("#>-")
}

fn input_node_kind() -> Vec<BlockKind> {
    vec![
        BlockKind::Switch { is_on: true },
        // BlockKind::Cobble {
        //     on_count: 0,
        //     on_base_count: 0,
        // },
        // BlockKind::Redstone {
        //     on_count: 0,
        //     state: RedstoneState::None as usize,
        //     strength: 0,
        // },
        // BlockKind::RedstoneBlock,
    ]
}

fn not_node_kind() -> Vec<BlockKind> {
    vec![BlockKind::Torch { is_on: false }]
}

fn place_node(world: &mut World3D, node: PlacedNode) {
    if world[node.position] == node.block {
        // TODO: Relax for no-op
        assert!(world[node.position].kind.is_cobble());
        return;
    }

    world[node.position] = node.block;
    if node.block.kind.is_redstone() {
        world.update_redstone_states(node.position);
    }
}

fn generate_inputs(
    config: &LocalPlacerConfig,
    world: &World3D,
    kind: BlockKind,
) -> Vec<(World3D, Position)> {
    let mut input_strategy = Direction::iter_direction_without_top()
        .map(|direction| Block { kind, direction })
        .collect_vec();
    if config.greedy_input_generation {
        input_strategy = input_strategy.into_iter().take(1).collect();
    }

    let place_strategy = iproduct!(0..1, 0..world.size.1, 0..world.size.2)
        .chain(iproduct!(0..world.size.0, 0..1, 0..world.size.2))
        .map(|(x, y, z)| Position(x, y, z));

    input_strategy
        .into_iter()
        .cartesian_product(place_strategy)
        // Place Input Node
        .flat_map(|(block, position)| {
            let placed_node = PlacedNode { position, block };
            if placed_node.has_conflict(world, &Default::default()) {
                return None;
            }

            let mut new_world = world.clone();
            place_node(&mut new_world, placed_node);
            Some((new_world, position))
        })
        .collect_vec()
}

fn generate_place_and_routes(
    config: &LocalPlacerConfig,
    world: &World3D,
    start: Position,
    kind: BlockKind,
) -> Vec<(World3D, Position)> {
    match kind {
        BlockKind::Torch { .. } => generate_torch_place_and_routes(config, world, start, kind),
        _ => unimplemented!(),
    }
}

fn generate_torch_place_and_routes(
    config: &LocalPlacerConfig,
    world: &World3D,
    source: Position,
    kind: BlockKind,
) -> Vec<(World3D, Position)> {
    let torch_strategy =
        Direction::iter_direction_without_top().map(|direction| Block { kind, direction });

    let place_strategy = iproduct!(0..world.size.0, 0..world.size.1, 0..world.size.2)
        .map(|(x, y, z)| Position(x, y, z))
        // start에서 최소 두 칸 떨어진 곳에 위치시킨다.
        .filter(|pos| source.manhattan_distance(pos) > 1)
        .filter(|pos| !config.route_torch_directly || source.manhattan_distance(pos) == 2);

    torch_strategy
        .cartesian_product(place_strategy)
        // 1. Place Torch and Cobble
        .flat_map(|(torch, torch_pos)| place_torch_with_cobble(world, torch, torch_pos))
        // 2. Route Source with Torch Place Target Position
        .flat_map(|(world, torch_pos, cobble_pos)| {
            generate_routes_to_cobble(config, &world, source, torch_pos, cobble_pos)
                .into_iter()
                .map(|(world, _)| (world, torch_pos))
                .collect_vec()
        })
        .collect()
}

fn place_torch_with_cobble(
    world: &World3D,
    torch: Block,
    torch_pos: Position,
) -> Option<(World3D, Position, Position)> {
    let cobble_pos = torch_pos.walk(torch.direction)?;
    let cobble_node = PlacedNode::new_cobble(cobble_pos);
    let torch_node = PlacedNode::new(torch_pos, torch);
    if cobble_node.has_conflict(&world, &Default::default())
        || torch_node.has_conflict(world, &Default::default())
    {
        return None;
    }

    let mut new_world = world.clone();
    place_node(&mut new_world, cobble_node);
    place_node(&mut new_world, torch_node);
    Some((new_world, torch_pos, cobble_pos))
}

fn generate_routes_to_cobble(
    _config: &LocalPlacerConfig,
    world: &World3D,
    source: Position,
    _torch_pos: Position,
    cobble_pos: Position,
) -> Vec<(World3D, Position)> {
    let source_node = PlacedNode::new(source, world[source]);
    assert!(source_node.is_propagation_target() && world[cobble_pos].kind.is_cobble());

    // (world, pos, cost)
    let mut route_candidates = Vec::new();
    for start in source_node.propagation_bound(Some(world)) {
        let directly_connected = start.position() == cobble_pos
            && matches!(
                start.propagation_type(),
                // TODO: 조건 삭제 가능한지 확인
                // Hard: Switch -> Torch 연결
                // Soft: Redstone -> Torch 연결
                PropagateType::Hard | PropagateType::Soft
            );

        if directly_connected {
            route_candidates.push((world.clone(), start.position(), 0usize));
            continue;
        }
    }

    // route_candidates.extend(
    //     generate_or_routes(config, world, source, torch_pos)
    //         .into_iter()
    //         .map(|(world, positions)| (world, positions.last().copied().unwrap(), 0usize)),
    // );

    route_candidates
        .into_iter()
        .sorted_by_key(|(_, _, cost)| *cost)
        .map(|(world, pos, _)| (world, pos))
        .collect()
}

// 두 torch를 연결하는 redstone routes를 생성한다.
fn generate_or_routes(
    config: &LocalPlacerConfig,
    world: &World3D,
    from: Position,
    to: Position,
) -> Vec<(World3D, Vec<Position>)> {
    let from_node = PlacedNode::new(from, world[from]);
    let to_node = PlacedNode::new(to, world[to]);
    assert!(from_node.is_propagation_target() && to_node.is_propagation_target());

    let first_bound = from_node.propagation_bound(Some(world));

    let mut queue = vec![(world.clone(), vec![from], first_bound)];
    let mut candidates = Vec::new();
    let mut step = 0;

    // TODO: torch의 위쪽으로 전파할 수 있는 경우도 고려
    while step < config.max_route_step && !queue.is_empty() {
        let mut next_queue = vec![];
        for (world, prevs, bounds) in queue {
            let prev_pos = prevs.last().copied().unwrap();
            for (new_world, redstone_node) in bounds
                .into_iter()
                .flat_map(|bound| place_redstone_with_cobble(&world, bound, prev_pos, to))
            {
                let new_prevs = prevs
                    .iter()
                    .copied()
                    .chain([redstone_node.position])
                    .collect();

                if redstone_node.has_connection_with(&world, to) {
                    candidates.push((new_world, new_prevs));
                } else {
                    let nexts = redstone_node.propagation_bound(Some(&new_world));
                    next_queue.push((new_world, new_prevs, nexts));
                }
            }
        }
        queue = config.route_step_sampling_policy.sample(next_queue);
        step += 1;
    }

    candidates
}

fn place_redstone_with_cobble(
    world: &World3D,
    bound: PlaceBound,
    prev: Position,
    to: Position,
) -> Option<(World3D, PlacedNode)> {
    let mut new_world = world.clone();
    let cobble_pos = bound.position().walk(Direction::Bottom)?;
    let cobble_node = PlacedNode::new_cobble(cobble_pos);
    if cobble_node.has_conflict(&new_world, &Default::default()) {
        return None;
    }
    place_node(&mut new_world, cobble_node);

    let bound_pos = bound.position();
    let bound_back_pos = bound_pos.walk(bound.direction()).unwrap();
    let redstone_node = PlacedNode::new_redstone(bound_pos);
    let except = [prev, bound_back_pos, bound_pos, to].into_iter().collect();
    if redstone_node.has_conflict(&new_world, &except) || redstone_node.has_short(&world, &except) {
        return None;
    }
    place_node(&mut new_world, redstone_node);
    new_world.update_redstone_states(prev);

    Some((new_world, redstone_node))
}

pub struct LocalPlacerCostEstimator<'a> {
    graph: &'a WorldGraph,
}

impl<'a> LocalPlacerCostEstimator<'a> {
    pub fn new(graph: &'a WorldGraph) -> Self {
        Self { graph }
    }

    pub fn cost(&self) -> usize {
        let _buffer_depth = self.graph.graph.critical_path().len();

        todo!()
    }
}

#[cfg(test)]
mod tests {

    use crate::graph::graphviz::ToGraphvizGraph;
    use crate::graph::logic::builder::LogicGraphBuilder;
    use crate::graph::logic::LogicGraph;
    use crate::nbt::{NBTRoot, ToNBT};
    use crate::transform::placer::{LocalPlacer, LocalPlacerConfig, SamplingPolicy};
    use crate::world::position::DimSize;
    use crate::world::world::World3D;

    fn build_graph_from_stmt(stmt: &str, output: &str) -> eyre::Result<LogicGraph> {
        LogicGraphBuilder::new(stmt.to_string()).build(output.to_string())
    }

    #[test]
    fn test_generate_component_and() -> eyre::Result<()> {
        tracing_subscriber::fmt::init();
        // rayon::ThreadPoolBuilder::new()
        //     .num_threads(1)
        //     .build_global()
        //     .unwrap();

        let logic_graph = build_graph_from_stmt("a&b", "c")?.prepare_place()?;
        println!("{}", logic_graph.to_graphviz());

        let config = LocalPlacerConfig {
            greedy_input_generation: true,
            step_sampling_policy: SamplingPolicy::Random(100),
            route_torch_directly: true,
            max_route_step: 3,
            route_step_sampling_policy: SamplingPolicy::Random(100),
        };
        let placer = LocalPlacer::new(logic_graph, config)?;
        let worlds = placer.generate(DimSize(10, 10, 5), None);

        let sampled_worlds = SamplingPolicy::Random(100).sample(worlds);

        let ww = World3D::concat_tiled(sampled_worlds);

        let nbt: NBTRoot = ww.to_nbt();
        nbt.save("test/and-gate-new.nbt");

        Ok(())
    }
}
