//! Replay an existing decoder NBT without compilation or placement.
//! cargo run --release --example decoder_replay -- DIR [x,y,z ...]
use std::path::PathBuf;

use redstone_compiler::nbt::NBTRoot;
use redstone_compiler::output::OutputEndpoint;
use redstone_compiler::world::position::Position;
use redstone_compiler::world::simulator::Simulator;
use redstone_compiler::world::World;
use serde::Deserialize;
use serde_json::json;
#[path = "../test/alu_display/oracle.rs"]
mod oracle;
#[derive(Deserialize)]
struct Interface {
    inputs: Vec<OutputEndpoint>,
    outputs: Vec<OutputEndpoint>,
}
fn bit(name: &str) -> usize {
    if name == "blank" {
        4
    } else {
        name.rsplit_once('_')
            .expect("bus pin")
            .1
            .parse()
            .expect("pin index")
    }
}
fn fresh(world: &World) -> eyre::Result<Simulator> {
    Ok(Simulator::from_settled_with_limits_and_trace(
        world, 4096, 2_000_000, 0,
    )?)
}
fn drive(sim: &mut Simulator, io: &Interface, value: usize) -> eyre::Result<()> {
    sim.advance_idle_cycles(61)?;
    sim.drive_inputs_with_limits(
        io.inputs
            .iter()
            .map(|p| (p.position(), value & (1 << bit(&p.name)) != 0))
            .collect(),
        4096,
        2_000_000,
    )
}
fn actual(sim: &Simulator, io: &Interface) -> usize {
    io.outputs.iter().fold(0, |m, p| {
        m | (usize::from(sim.world()[p.position()].kind.is_powered()) << bit(&p.name))
    })
}
fn inspect(sim: &Simulator, p: Position) -> serde_json::Value {
    json!({"position":[p.0,p.1,p.2],"block":format!("{:?}",sim.world()[p]),"powered":sim.world()[p].kind.is_powered(),"burned":sim.is_torch_burned_out(p),"sources":sim.diagnostic_power_sources(p)})
}
fn main() -> eyre::Result<()> {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(
        args.next()
            .expect("directory containing decoder.nbt and interface.json"),
    );
    let queries = args
        .map(|s| {
            let xyz = s
                .split(',')
                .map(str::parse::<usize>)
                .collect::<Result<Vec<_>, _>>()?;
            eyre::ensure!(xyz.len() == 3, "query must be x,y,z");
            Ok(Position(xyz[0], xyz[1], xyz[2]))
        })
        .collect::<eyre::Result<Vec<_>>>()?;
    let world = NBTRoot::from_nbt_bytes(&std::fs::read(root.join("decoder.nbt"))?)?.to_world();
    let io: Interface = serde_json::from_slice(&std::fs::read(root.join("interface.json"))?)?;
    eyre::ensure!(
        [4, 5].contains(&io.inputs.len()) && io.outputs.len() == 7,
        "unexpected interface"
    );
    for &p in &queries {
        eyre::ensure!(world.size.bound_on(p), "query outside world");
    }
    let count = 1 << io.inputs.len();
    let mut rows = vec![];
    let mut walking = fresh(&world)?;
    let mut first_failure = false;
    for value in 0..count {
        let expected = oracle::segment_mask(value & 15, value & 16 != 0);
        let mut sim = fresh(&world)?;
        sim.set_trace_limit(30_000);
        drive(&mut sim, &io, value)?;
        let observed = actual(&sim, &io);
        let mut from_zero = fresh(&world)?;
        drive(&mut from_zero, &io, 0)?;
        drive(&mut from_zero, &io, value)?;
        drive(&mut walking, &io, value)?;
        let row = json!({"input":value,"expected":expected,"fresh":observed,"from_zero":actual(&from_zero,&io),"walk":actual(&walking,&io),"queries":queries.iter().map(|&p|inspect(&sim,p)).collect::<Vec<_>>()});
        println!("{row}");
        rows.push(row);
        if observed != expected && !first_failure {
            first_failure = true;
            let states = sim
                .world()
                .iter_block()
                .iter()
                .map(|(p, _)| inspect(&sim, *p))
                .collect::<Vec<_>>();
            std::fs::write(
                root.join("decoder-replay-first-failure.json"),
                serde_json::to_vec_pretty(
                    &json!({"input":value,"expected":expected,"actual":observed,"blocks":states,"trace":sim.trace()}),
                )?,
            )?;
        }
    }
    let mut transition_failures = vec![];
    let mut continuous_failures = vec![];
    let mut continuous = fresh(&world)?;
    for from in 0..count {
        for to in 0..count {
            for value in [from, to] {
                drive(&mut continuous, &io, value)?;
                let expected = oracle::segment_mask(value & 15, value & 16 != 0);
                let observed = actual(&continuous, &io);
                if observed != expected {
                    continuous_failures.push(json!({"from":from,"to":to,"input":value,"actual":observed,"expected":expected}));
                }
            }
            let mut sim = fresh(&world)?;
            drive(&mut sim, &io, from)?;
            drive(&mut sim, &io, to)?;
            let observed = actual(&sim, &io);
            let expected = oracle::segment_mask(to & 15, to & 16 != 0);
            if observed != expected {
                if transition_failures.is_empty() {
                    let mut traced = fresh(&world)?;
                    drive(&mut traced, &io, from)?;
                    traced.set_trace_limit(30_000);
                    drive(&mut traced, &io, to)?;
                    let mut good = fresh(&world)?;
                    drive(&mut good, &io, to)?;
                    let differences = traced
                        .world()
                        .iter_block()
                        .iter()
                        .filter(|(p, b)| b.kind.is_powered() != good.world()[*p].kind.is_powered())
                        .map(|(p, _)| json!({"bad":inspect(&traced,*p),"fresh":inspect(&good,*p)}))
                        .collect::<Vec<_>>();
                    let burned = traced
                        .world()
                        .iter_block()
                        .iter()
                        .filter(|(p, _)| traced.is_torch_burned_out(*p))
                        .map(|(p, _)| inspect(&traced, *p))
                        .collect::<Vec<_>>();
                    let inputs = io
                        .inputs
                        .iter()
                        .map(|p| json!({"name":p.name,"state":inspect(&traced,p.position())}))
                        .collect::<Vec<_>>();
                    let trace = traced.trace().to_vec();
                    let states = traced
                        .world()
                        .iter_block()
                        .iter()
                        .map(|(p, _)| inspect(&traced, *p))
                        .collect::<Vec<_>>();
                    traced.advance_idle_cycles(1000)?;
                    traced.run_with_max_cycles(4096)?;
                    let after_idle = actual(&traced, &io);
                    drive(&mut traced, &io, to)?;
                    std::fs::write(
                        root.join("decoder-replay-transition-failure.json"),
                        serde_json::to_vec_pretty(
                            &json!({"from":from,"to":to,"expected":expected,"actual":observed,"after_idle":after_idle,"after_same_input":actual(&traced,&io),"inputs":inputs,"burned":burned,"differences":differences,"blocks":states,"trace":trace}),
                        )?,
                    )?;
                }
                transition_failures
                    .push(json!({"from":from,"to":to,"actual":observed,"expected":expected}));
            }
        }
    }
    let static_failures = rows
        .iter()
        .filter(|row| {
            row["fresh"] != row["expected"]
                || row["from_zero"] != row["expected"]
                || row["walk"] != row["expected"]
        })
        .count();
    let report = json!({"static":rows,"static_failures":static_failures,"transitions_checked":count*count,"transition_failures":transition_failures,"continuous_failures":continuous_failures,"nbt_recompiled":false});
    std::fs::write(
        root.join("decoder-replay-report.json"),
        serde_json::to_vec_pretty(&report)?,
    )?;
    println!(
        "transition_failures={}/{}",
        transition_failures.len(),
        count * count
    );
    println!(
        "continuous_failures={}/{}",
        continuous_failures.len(),
        2 * count * count
    );
    eyre::ensure!(
        static_failures == 0 && transition_failures.is_empty() && continuous_failures.is_empty(),
        "saved decoder NBT replay failed; see decoder-replay-report.json"
    );
    Ok(())
}
