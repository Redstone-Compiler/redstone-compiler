use std::collections::HashSet;

use super::place_bound::{PlaceBound, PropagateType};
use crate::world::block::{Block, BlockKind, Direction};
use crate::world::position::Position;
use crate::world::world::World3D;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct PlacedNode {
    pub position: Position,
    pub block: Block,
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

    pub fn is_diode(&self) -> bool {
        self.block.kind.is_switch() || self.block.kind.is_torch() || self.block.kind.is_repeater()
    }

    // signal을 보낼 수 있는 부분들의 위치를 반환합니다.
    pub fn propagation_bound(&self, world: Option<&World3D>) -> Vec<PlaceBound> {
        PlaceBound(PropagateType::Soft, self.position, self.block.direction)
            .propagation_bound(&self.block.kind, world)
    }

    fn propagated_from(&self, world: &World3D) -> Vec<PlaceBound> {
        PlaceBound::propagated_from(self.position, &self.block.kind, world)
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

            // 재귀를 이르키는 경우
            if world[bottom].kind.is_torch()
                && !world[bottom].direction.is_bottom()
                && self.position.cardinal().into_iter().any(|pos| {
                    world.size.bound_on(pos)
                        && world[pos].kind.is_redstone()
                        && self.position.diff(pos) == world[bottom].direction
                })
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
    pub fn has_short(&self, world: &World3D, except: &HashSet<Position>) -> bool {
        assert!(self.block.kind.is_redstone());

        self.propagated_from(world)
            .into_iter()
            .filter(|bound| !except.contains(&bound.position()))
            .next()
            .is_some()
    }

    pub fn has_connection_with(&self, world: &World3D, target: Position) -> bool {
        assert!(self.block.kind.is_redstone());
        assert!(world[target].kind.is_stick_to_redstone());

        self.propagated_from(world)
            .into_iter()
            .any(|bound| bound.position() == target)
    }
}
