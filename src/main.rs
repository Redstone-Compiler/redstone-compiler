use std::path::PathBuf;

use mimalloc::MiMalloc;
use redstone_compiler::nbt::ToNBT;
use redstone_compiler::transform::place_and_route::local_placer::{
    InputPlacementStrategy, LocalPlacer, LocalPlacerConfig, NotRouteStrategy,
    PlacementSamplingPolicy, TorchPlacementStrategy,
};
use redstone_compiler::transform::place_and_route::sampling::SamplingPolicy;
use redstone_compiler::verilog;
use redstone_compiler::world::position::DimSize;
use structopt::StructOpt;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[derive(Debug, StructOpt)]
#[structopt(name = "example", about = "An example of StructOpt usage.")]
pub struct CompilerOption {
    #[structopt(parse(from_os_str))]
    pub input: PathBuf,

    #[structopt(parse(from_os_str))]
    pub output: Option<PathBuf>,
}

fn main() -> eyre::Result<()> {
    tracing_subscriber::fmt::init();
    let opt = CompilerOption::from_args();

    if opt.input.extension().and_then(|ext| ext.to_str()) != Some("v") {
        eyre::bail!("unsupported input file extension: {:?}", opt.input);
    }

    let graph = verilog::load_logic_graph(&opt.input)?.prepare_place()?;

    let Some(output) = opt.output else {
        println!(
            "loaded Verilog graph: nodes={} inputs={} outputs={}",
            graph.nodes.len(),
            graph.inputs().len(),
            graph.outputs().len()
        );
        return Ok(());
    };

    let config = LocalPlacerConfig {
        random_seed: 42,
        greedy_input_generation: true,
        input_placement_strategy: InputPlacementStrategy::Boundary,
        input_candidate_limit: None,
        step_sampling_policy: SamplingPolicy::Random(10000),
        placement_sampling_policy: PlacementSamplingPolicy::StepPolicy,
        leak_sampling: false,
        route_torch_directly: true,
        torch_placement_strategy: TorchPlacementStrategy::DirectOnly,
        not_route_strategy: NotRouteStrategy::DirectOnly,
        max_not_route_step: 3,
        not_route_step_sampling_policy: SamplingPolicy::Random(100),
        max_route_step: 3,
        route_step_sampling_policy: SamplingPolicy::Random(100),
    };
    let placer = LocalPlacer::new(graph, config)?;
    let worlds = placer.generate(DimSize(10, 10, 5), None);
    let Some(world) = worlds.into_iter().next() else {
        eyre::bail!("placement produced no worlds");
    };
    world.to_nbt().save(&output);

    println!("exported Verilog graph: path={}", output.display());

    Ok(())
}
