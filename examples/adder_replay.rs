//! Replay saved NBT only: this never invokes compilation or placement.
//! cargo run --release --example adder_replay -- target/alu-display/adder-8
#[allow(dead_code)]
#[path = "adder_nbt.rs"]
mod original;

use std::path::PathBuf;
use std::time::Instant;

use redstone_compiler::nbt::NBTRoot;
use redstone_compiler::output::OutputEndpoint;
use redstone_compiler::world::simulator::Simulator;
use redstone_compiler::world::World;
use serde::Deserialize;
use serde_json::json;

#[derive(Deserialize)]
struct Interface {
    inputs: Vec<OutputEndpoint>,
    outputs: Vec<OutputEndpoint>,
}

fn drive(sim: &mut Simulator, interface: &Interface, a: usize, b: usize) -> eyre::Result<()> {
    sim.advance_idle_cycles(61)?;
    let states = interface
        .inputs
        .iter()
        .map(|p| {
            let (bus, bit) = p.name.rsplit_once('_').expect("scalar bus pin");
            let bit = bit.parse::<usize>().expect("bit index");
            (
                p.position(),
                (if bus == "a" { a } else { b }) & (1 << bit) != 0,
            )
        })
        .collect();
    sim.drive_inputs_with_limits(states, 4096, 2_000_000)
}

fn observed(sim: &Simulator, interface: &Interface) -> usize {
    interface.outputs.iter().fold(0, |sum, p| {
        let bit = p.name.rsplit_once('_').unwrap().1.parse::<usize>().unwrap();
        sum | (usize::from(sim.world()[p.position()].kind.is_powered()) << bit)
    })
}

fn burned(sim: &Simulator) -> Vec<[usize; 3]> {
    sim.world()
        .iter_block()
        .into_iter()
        .filter(|(p, _)| sim.is_torch_burned_out(*p))
        .map(|(p, _)| [p.0, p.1, p.2])
        .collect()
}

fn new_sim(world: &World) -> eyre::Result<Simulator> {
    Ok(Simulator::from_settled_with_limits_and_trace(
        world, 4096, 2_000_000, 0,
    )?)
}

fn main() -> eyre::Result<()> {
    let root = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or("target/alu-display/adder-8".into()),
    );
    let world = NBTRoot::from_nbt_bytes(&std::fs::read(root.join("adder.nbt"))?)?.to_world();
    let interface: Interface =
        serde_json::from_slice(&std::fs::read(root.join("interface.json"))?)?;
    let width = interface.inputs.len() / 2;
    let cases = original::static_cases(width);
    let started = Instant::now();
    for &(a, b) in &cases {
        let mut fresh = new_sim(&world)?;
        drive(&mut fresh, &interface, a, b)?;
        eyre::ensure!(
            observed(&fresh, &interface) == a + b && burned(&fresh).is_empty(),
            "static case ({a},{b}) failed"
        );
    }
    let mut sim = new_sim(&world)?;
    let mut previous = None;
    for (index, &(a, b)) in cases.iter().enumerate() {
        // Keep only the failing step's trace, with bounded event storage.
        sim.set_trace_limit(if index >= 45 { 20_000 } else { 0 });
        sim.clear_trace();
        drive(&mut sim, &interface, a, b)?;
        let actual = observed(&sim, &interface);
        let burnt = burned(&sim);
        if actual != a + b || !burnt.is_empty() {
            let mut fresh = new_sim(&world)?;
            drive(&mut fresh, &interface, a, b)?;
            let mut pair = new_sim(&world)?;
            if let Some((pa, pb)) = previous {
                drive(&mut pair, &interface, pa, pb)?;
            }
            drive(&mut pair, &interface, a, b)?;
            let differences=sim.world().iter_block().into_iter().filter(|(p,block)| block.kind.is_powered()!=fresh.world()[*p].kind.is_powered()).map(|(p,block)| json!({"pos":[p.0,p.1,p.2],"bad":format!("{:?}",block),"fresh":format!("{:?}",fresh.world()[p]),"burned":sim.is_torch_burned_out(p),"power_sources":sim.diagnostic_power_sources(p)})).collect::<Vec<_>>();
            let report = json!({"case":index+1,"previous":previous,"inputs":[a,b],"expected":a+b,"actual":actual,"fresh_actual":observed(&fresh,&interface),"pair_actual":observed(&pair,&interface),"burned":burnt,"pair_burned":burned(&pair),"fresh_burned":burned(&fresh),"powered_differences":differences,"trace":sim.trace(),"elapsed_ms":started.elapsed().as_millis()});
            std::fs::write(
                root.join("replay-failure-trace.json"),
                serde_json::to_vec_pretty(&report)?,
            )?;
            // A long idle followed by a real settle distinguishes retained
            // burnout/queue history from a stable geometric feedback loop.
            sim.set_trace_limit(0);
            sim.advance_idle_cycles(1000)?;
            sim.run_with_max_cycles(4096)?;
            let after_settle = observed(&sim, &interface);
            drive(&mut sim, &interface, a, b)?;
            let after_repeat = observed(&sim, &interface);
            let summary = json!({"case":index+1,"previous":previous,"inputs":[a,b],"expected":a+b,"actual":actual,"fresh_actual":observed(&fresh,&interface),"pair_actual":observed(&pair,&interface),"burned":burnt,"pair_burned":burned(&pair),"after_1000_idle_and_settle":after_settle,"after_same_input_again":after_repeat,"remaining_burned":burned(&sim),"powered_difference_count":differences.len(),"elapsed_ms":started.elapsed().as_millis()});
            std::fs::write(
                root.join("replay-summary.json"),
                serde_json::to_vec_pretty(&summary)?,
            )?;
            println!("{}", serde_json::to_string_pretty(&summary)?);
            return Ok(());
        }
        previous = Some((a, b));
    }
    let report = json!({"static_passed":cases.len(), "continuous_passed":cases.len(), "nbt_recompiled":false,"elapsed_ms":started.elapsed().as_millis()});
    std::fs::write(
        root.join("replay-verification.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
