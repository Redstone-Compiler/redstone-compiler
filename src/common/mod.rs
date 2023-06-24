use self::block::{Direction, RedstoneState, RedstoneStateType};

pub mod block;
pub mod gate;
pub mod world;

// 위치
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Position(pub usize, pub usize, pub usize);

impl Position {
    pub fn index(&self, size: &DimSize) -> usize {
        self.0 + self.1 * size.0 + self.2 * size.0 * size.1
    }

    pub fn forwards(&self) -> Vec<Position> {
        let mut result = vec![
            Position(self.0 + 1, self.1, self.2),
            Position(self.0, self.1 + 1, self.2),
            Position(self.0, self.1, self.2 + 1),
        ];

        if self.0 > 0 {
            result.push(Position(self.0 - 1, self.1, self.2));
        }

        if self.1 > 0 {
            result.push(Position(self.0, self.1 - 1, self.2));
        }

        if self.2 > 0 {
            result.push(Position(self.0, self.1, self.2 - 1));
        }

        result
    }

    pub fn forwards_except(&self, dir: &Direction) -> Vec<Position> {
        if let Some(pos) = self.walk(dir) {
            return self
                .forwards()
                .into_iter()
                .filter(|pos_src| *pos_src != pos)
                .collect();
        }

        self.forwards()
    }

    pub fn cardinal(&self) -> Vec<Position> {
        let mut result = vec![
            Position(self.0 + 1, self.1, self.2),
            Position(self.0, self.1 + 1, self.2),
        ];

        if self.0 > 0 {
            result.push(Position(self.0 - 1, self.1, self.2));
        }

        if self.1 > 0 {
            result.push(Position(self.0, self.1 - 1, self.2));
        }

        result
    }

    pub fn cardinal_redstone(&self, state: RedstoneStateType) -> Vec<Position> {
        let mut result = Vec::new();
        let all = state == 0;

        if all || (state & RedstoneState::East as usize) > 0 {
            result.push(Position(self.0 + 1, self.1, self.2));
        }

        if self.0 > 0 && (all || (state & RedstoneState::West as usize) > 0) {
            result.push(Position(self.0 - 1, self.1, self.2));
        }

        if all || ((state & RedstoneState::North as usize) > 0) {
            result.push(Position(self.0, self.1 + 1, self.2));
        }

        if self.1 > 0 && (all || (state & RedstoneState::South as usize) > 0) {
            result.push(Position(self.0, self.1 - 1, self.2));
        }

        result
    }

    pub fn cardinal_except(&self, dir: &Direction) -> Vec<Position> {
        let mut result = Vec::new();

        if !matches!(dir, Direction::East) {
            result.push(Position(self.0 + 1, self.1, self.2));
        }

        if !matches!(dir, Direction::North) {
            result.push(Position(self.0, self.1 + 1, self.2));
        }

        if self.0 > 0 && !matches!(dir, Direction::West) {
            result.push(Position(self.0 - 1, self.1, self.2));
        }

        if self.1 > 0 && !matches!(dir, Direction::South) {
            result.push(Position(self.0, self.1 - 1, self.2));
        }

        result
    }

    pub fn up(&self) -> Position {
        Position(self.0, self.1, self.2 + 1)
    }

    pub fn down(&self) -> Option<Position> {
        if self.2 == 0 {
            return None;
        }

        Some(Position(self.0, self.1, self.2 - 1))
    }

    pub fn walk(&self, dir: &Direction) -> Option<Position> {
        match dir {
            Direction::None => Some(*self),
            Direction::Bottom => {
                if self.2 == 0 {
                    return None;
                }

                Some(Position(self.0, self.1, self.2 - 1))
            }
            Direction::Top => Some(Position(self.0, self.1, self.2 + 1)),
            Direction::East => Some(Position(self.0 + 1, self.1, self.2)),
            Direction::West => {
                if self.0 == 0 {
                    return None;
                }

                Some(Position(self.0 - 1, self.1, self.2))
            }
            Direction::South => {
                if self.1 == 0 {
                    return None;
                }

                Some(Position(self.0, self.1 - 1, self.2))
            }
            Direction::North => Some(Position(self.0, self.1 + 1, self.2)),
        }
    }

    pub fn diff(&self, tar: &Position) -> Direction {
        if tar.0 > self.0 {
            Direction::East
        } else if tar.0 < self.0 {
            Direction::West
        } else if tar.1 > self.1 {
            Direction::North
        } else if tar.1 < self.1 {
            Direction::South
        } else if tar.2 > self.2 {
            Direction::Top
        } else if tar.2 < self.2 {
            Direction::Bottom
        } else {
            unreachable!()
        }
    }
}

// 사이즈

#[derive(Debug, Copy, Clone)]
pub struct DimSize(pub usize, pub usize, pub usize);

pub trait Verifier {
    fn verify(&self) -> bool;
}
