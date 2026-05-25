use crate::world::block::BlockKind;
use crate::world::position::Position;
use crate::world::simulator::Simulator;
use crate::world::World3D;

#[derive(Clone, Copy)]
pub(super) enum SequentialScenarioStep<'a, Input: Copy, Output: Copy> {
    SetInput { input: Input, value: bool },
    ExpectOutputs(&'a [(Output, bool)]),
}

pub(super) fn run_sequential_scenario<'a, Input, Output>(
    sim: &mut Simulator,
    steps: &[SequentialScenarioStep<'a, Input, Output>],
    input_position: impl Fn(Input) -> Position,
    output_value: impl Fn(&World3D, Output) -> Option<bool>,
    max_cycles: usize,
    max_events: usize,
) -> bool
where
    Input: Copy,
    Output: Copy,
{
    for step in steps {
        match *step {
            SequentialScenarioStep::SetInput { input, value } => {
                if sim
                    .change_state_with_limits(
                        vec![(input_position(input), value)],
                        max_cycles,
                        max_events,
                    )
                    .is_err()
                {
                    return false;
                }
            }
            SequentialScenarioStep::ExpectOutputs(expectations) => {
                if expectations
                    .iter()
                    .any(|&(output, expected)| output_value(sim.world(), output) != Some(expected))
                {
                    return false;
                }
            }
        }
    }

    true
}

pub(super) fn torch_output_value(world: &World3D, position: Position) -> Option<bool> {
    match world[position].kind {
        BlockKind::Torch { is_on } => Some(is_on),
        _ => None,
    }
}
