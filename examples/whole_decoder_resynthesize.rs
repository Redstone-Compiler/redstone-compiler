//! Synthesize each RTL decoder output with bounded frontier/2-input NOR/OR,
//! then share only the four primary inputs between output cones.
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;

use eyre::ensure;
use redstone_compiler::ir::LogicalDesign;
use redstone_compiler::transform::place_and_route::local_placer::exact::{
    synthesize, Net, NetDriver, NorNetlist, SynthesisOptions, SynthesisOutcome,
};

fn main() -> eyre::Result<()> {
    let directory = PathBuf::from(
        std::env::args()
            .nth(1)
            .unwrap_or("target/alu-display/decoder-resynthesis".into()),
    );
    std::fs::create_dir_all(&directory)?;
    let source = format!(
        "{}\n{}",
        include_str!("../test/alu_display/bcd_to_7seg_minimized.v"),
        include_str!("../test/alu_display/bcd_digit_to_7seg.v")
    );
    let graph = LogicalDesign::from_verilog_source(&source)?.lower_flat_logic_graph()?;
    let truth = graph.truth_table()?;
    let inputs = truth
        .input_names
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    let mut combined = NorNetlist {
        nets: inputs
            .iter()
            .enumerate()
            .map(|(i, name)| Net {
                name: name.to_string(),
                node_id: i,
                driver: NetDriver::Input(name.to_string()),
                gate_inputs: Vec::new(),
            })
            .collect(),
        outputs: Vec::new(),
        state: Vec::new(),
    };
    let mut reports = Vec::new();
    for (name, table) in &truth.output_tables {
        let expression = table
            .iter()
            .enumerate()
            .filter(|(_, on)| **on)
            .map(|(case, _)| {
                format!(
                    "({})",
                    inputs
                        .iter()
                        .enumerate()
                        .map(|(i, n)| format!("{}{n}", if case & (1 << i) == 0 { "~" } else { "" }))
                        .collect::<Vec<_>>()
                        .join("&")
                )
            })
            .collect::<Vec<_>>()
            .join("|");
        let cache = directory.join(format!("{name}.nor"));
        let netlist = if cache.exists() {
            NorNetlist::from_text(&std::fs::read_to_string(&cache)?)?
        } else {
            let mut found = None;
            for steps in [5, 6, 7, 8, 10, 12, 14, 16, 18] {
                let options = SynthesisOptions {
                    steps,
                    max_live: 4,
                    max_depth: None,
                    time_limit: Duration::from_secs(15),
                };
                let started = std::time::Instant::now();
                let outcome = synthesize(&inputs, &[(name, &expression)], &options)?;
                eprintln!(
                    "{name}: steps={steps} seconds={:.3} outcome={:?}",
                    started.elapsed().as_secs_f64(),
                    outcome
                );
                if let SynthesisOutcome::Found(result) = outcome {
                    reports.push(serde_json::json!({"output":name,"steps":steps,"torches":result.torches,"proven":result.proven}));
                    std::fs::write(&cache, result.netlist.to_text())?;
                    found = Some(result.netlist);
                    break;
                }
            }
            found.ok_or_else(|| eyre::eyre!("no bounded synthesis for {name}"))?
        };
        let values = netlist.net_values();
        let target = netlist
            .outputs
            .iter()
            .find(|(n, _)| n == name)
            .ok_or_else(|| eyre::eyre!("missing output"))?
            .1;
        ensure!(
            netlist.input_names() == truth.input_names && values[target] == *table,
            "resynthesis oracle mismatch for {name}"
        );
        let mut remap = BTreeMap::new();
        for (id, net) in netlist.nets.iter().enumerate() {
            if let NetDriver::Input(input) = &net.driver {
                remap.insert(id, inputs.iter().position(|n| *n == input).unwrap());
            } else {
                let mut copied = net.clone();
                copied.name = if id == target {
                    name.clone()
                } else {
                    format!("{name}_{}", net.name)
                };
                copied.node_id = combined.nets.len();
                copied.gate_inputs = net.gate_inputs.iter().map(|input| remap[input]).collect();
                remap.insert(id, combined.nets.len());
                combined.nets.push(copied);
            }
        }
        combined.outputs.push((name.clone(), remap[&target]));
    }
    let values = combined.net_values();
    for (name, net) in &combined.outputs {
        ensure!(
            values[*net] == truth.output_tables[name],
            "combined oracle mismatch"
        );
    }
    let output = directory.join("combined.nor");
    std::fs::write(&output, combined.to_text())?;
    std::fs::write(
        directory.join("synthesis.json"),
        serde_json::to_vec_pretty(&reports)?,
    )?;
    println!(
        "{}: {} gates, {} nets",
        output.display(),
        combined.gates().count(),
        combined.nets.len()
    );
    Ok(())
}
