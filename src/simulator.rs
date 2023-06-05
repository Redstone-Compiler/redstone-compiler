use std::collections::VecDeque;

use crate::{common::Position, world::World};

enum EventType {
    SignalOff,
    SignalOn,
}

struct Event {
    event_type: EventType,
    target_position: Position,
}

pub struct Simulator {
    queue: VecDeque<Vec<Event>>,
    world: World,
}

impl Simulator {
    pub fn from(world: World) -> Self {
        let mut sim = Self {
            queue: VecDeque::new(),
            world: world,
        };

        sim.init();
        sim.consume_events();
        sim
    }

    pub fn change_state(&mut self, states: Vec<(Position, bool)>) {}

    pub fn is_empty(&mut self) -> bool {
        self.queue.is_empty()
    }

    fn init(&mut self) {}

    fn consume_events(&mut self) {}
}
