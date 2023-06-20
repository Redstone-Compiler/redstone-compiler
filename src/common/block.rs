use crate::common::{DimSize, Position};

use super::world::World;

#[derive(Debug, Default, Copy, Clone)]
pub enum Direction {
    #[default]
    None,
    Bottom,
    Top,
    East,
    West,
    South,
    North,
}

impl Direction {
    pub fn is_cardinal(&self) -> bool {
        matches!(self, Self::East | Self::West | Self::South | Self::North)
    }
}

pub type RedstoneStateType = usize;

#[derive(Default, Clone, Debug, Copy)]
pub enum RedstoneState {
    #[default]
    None = 0,
    East = 1,
    West = 2,
    South = 4,
    North = 8,
    Horizontal = 1 | 2,
    Vertical = 4 | 8,
}

// 블럭의 종류
#[derive(Default, Clone, Debug, Copy)]
pub enum BlockKind {
    #[default]
    Air,
    Cobble {
        // redstone
        on_count: usize,
        // repeater, redstone
        on_base_count: usize,
    },
    Switch {
        is_on: bool,
    },
    Redstone {
        on_count: usize,
        state: RedstoneStateType,
        strength: usize,
    },
    Torch {
        is_on: bool,
    },
    Repeater {
        is_on: bool,
        is_locked: bool,
    },
    RedstoneBlock,
}

impl BlockKind {
    pub fn is_stick_to_redstone(&self) -> bool {
        matches!(
            self,
            BlockKind::Redstone { .. }
                | BlockKind::Repeater { .. }
                | BlockKind::Switch { .. }
                | BlockKind::Torch { .. }
                | BlockKind::RedstoneBlock
        )
    }

    pub fn is_redstone(&self) -> bool {
        matches!(self, BlockKind::Redstone { .. })
    }

    pub fn is_cobble(&self) -> bool {
        matches!(self, BlockKind::Cobble { .. })
    }
}

// 모든 물리적 소자의 최소 단위
#[derive(Debug, Copy, Clone)]
pub struct Block {
    pub kind: BlockKind,
    pub direction: Direction,
}

impl Block {
    pub fn count_up(&mut self, is_base: bool) -> eyre::Result<usize> {
        let cnt;

        self.kind = match self.kind {
            BlockKind::Cobble {
                on_count,
                on_base_count,
            } => {
                cnt = on_count + 1;

                let on_base_count = if is_base {
                    on_base_count + 1
                } else {
                    on_base_count
                };

                BlockKind::Cobble {
                    on_count: cnt,
                    on_base_count,
                }
            }
            BlockKind::Redstone {
                on_count,
                strength,
                state,
            } => {
                cnt = on_count + 1;

                BlockKind::Redstone {
                    on_count: cnt,
                    state,
                    strength,
                }
            }
            _ => eyre::bail!("You cannot count up on {:?}!", self.kind),
        };

        Ok(cnt)
    }

    pub fn count_down(&mut self, is_base: bool) -> eyre::Result<usize> {
        let cnt;

        self.kind = match self.kind {
            BlockKind::Cobble {
                on_count,
                on_base_count,
            } => {
                eyre::ensure!(on_count > 0, "On count must higher than zero");

                cnt = on_count - 1;

                let on_base_count = if is_base {
                    on_base_count + 1
                } else {
                    on_base_count
                };

                BlockKind::Cobble {
                    on_count: cnt,
                    on_base_count,
                }
            }
            BlockKind::Redstone {
                on_count,
                strength,
                state,
            } => {
                eyre::ensure!(on_count > 0, "On count must higher than zero");

                cnt = on_count - 1;

                BlockKind::Redstone {
                    on_count: cnt,
                    state,
                    strength,
                }
            }
            _ => eyre::bail!("You cannot count down on {:?}!", self.kind),
        };

        Ok(cnt)
    }
}

impl Default for Block {
    fn default() -> Self {
        Self {
            kind: Default::default(),
            direction: Default::default(),
        }
    }
}

impl From<Block> for World {
    fn from(value: Block) -> Self {
        Self {
            size: DimSize(1, 1, 1),
            blocks: vec![(Position(1, 1, 1), value)],
        }
    }
}
