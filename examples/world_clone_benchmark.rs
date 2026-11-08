//! Optional storage diagnostic, not a timing test.
//! cargo run --release --example world_clone_benchmark -- 1000
use std::collections::BTreeMap;
use std::hint::black_box;
use std::sync::Arc;
use std::time::{Duration, Instant};

use redstone_compiler::world::block::{Block, BlockKind};
use redstone_compiler::world::position::{DimSize, Position, PositionIndex};
use redstone_compiler::world::{World, World3D};

// Previous conversion algorithm, using the same current COW storage so this
// comparison isolates sparse construction from the earlier clone optimization.
fn dense_lookup_constructor(source: &World) -> World3D {
    let mut blocks = BTreeMap::new();
    for (position, block) in &source.blocks {
        blocks.insert(position.index(&source.size), *block);
    }
    let map = (0..source.size.2)
        .map(|z| {
            Arc::new(
                (0..source.size.1)
                    .map(|y| {
                        Arc::new(
                            (0..source.size.0)
                                .map(|x| {
                                    blocks
                                        .get(&PositionIndex(
                                            x + y * source.size.0
                                                + z * source.size.0 * source.size.1,
                                        ))
                                        .copied()
                                        .unwrap_or_default()
                                })
                                .collect(),
                        )
                    })
                    .collect(),
            )
        })
        .collect();
    World3D {
        size: source.size,
        map,
    }
}

fn sparse_constructor(source: &World) -> World3D {
    World3D::from(source)
}

fn construction_run(
    source: &World,
    count: usize,
    constructor: fn(&World) -> World3D,
) -> (Duration, usize) {
    let start = Instant::now();
    let mut checksum = 0;
    for _ in 0..count {
        let world = constructor(black_box(source));
        black_box(&world);
        checksum += usize::from(world[Position(0, 0, 0)].kind.is_powered());
    }
    (start.elapsed(), checksum)
}

// The previous World3D representation, retained here only for the A/B baseline.
#[derive(Clone)]
struct DeepWorld {
    map: Vec<Vec<Vec<Block>>>,
}

fn mutations(iteration: usize) -> [(Position, Block); 2] {
    [
        (
            Position(iteration % 64, (iteration / 64) % 64, iteration % 8),
            Block {
                kind: BlockKind::Switch { is_on: true },
                ..Default::default()
            },
        ),
        (
            Position(
                (iteration + 7) % 64,
                (iteration / 64 + 1) % 64,
                (iteration + 1) % 8,
            ),
            Block {
                kind: BlockKind::RedstoneBlock,
                ..Default::default()
            },
        ),
    ]
}

fn deep_run(source: &DeepWorld, count: usize) -> (Duration, usize) {
    let start = Instant::now();
    let mut checksum = 0;
    for iteration in 0..count {
        let mut branch = black_box(source).clone();
        for (p, block) in mutations(iteration) {
            branch.map[p.2][p.1][p.0] = block;
        }
        black_box(&branch);
        for (p, _) in mutations(iteration) {
            checksum += usize::from(branch.map[p.2][p.1][p.0].kind.is_powered());
        }
    }
    (start.elapsed(), checksum)
}

// The immediately previous row-only COW representation.
#[derive(Clone)]
struct RowCowWorld {
    map: Vec<Vec<Arc<Vec<Block>>>>,
}

fn row_cow_run(source: &RowCowWorld, count: usize) -> (Duration, usize) {
    let start = Instant::now();
    let mut checksum = 0;
    for iteration in 0..count {
        let mut branch = black_box(source).clone();
        for (p, block) in mutations(iteration) {
            Arc::make_mut(&mut branch.map[p.2][p.1])[p.0] = block;
        }
        black_box(&branch);
        for (p, _) in mutations(iteration) {
            checksum += usize::from(branch.map[p.2][p.1][p.0].kind.is_powered());
        }
    }
    (start.elapsed(), checksum)
}

fn cow_run(source: &World3D, count: usize) -> (Duration, usize) {
    let start = Instant::now();
    let mut checksum = 0;
    for iteration in 0..count {
        let mut branch = black_box(source).clone();
        for (p, block) in mutations(iteration) {
            branch[p] = block;
        }
        black_box(&branch);
        for (p, _) in mutations(iteration) {
            checksum += usize::from(branch[p].kind.is_powered());
        }
    }
    (start.elapsed(), checksum)
}

fn main() -> eyre::Result<()> {
    let count = std::env::args()
        .nth(1)
        .map(|arg| arg.parse())
        .transpose()?
        .unwrap_or(1000);
    eyre::ensure!(count > 0, "iteration count must be positive");
    let mut source = World3D::new(DimSize(64, 64, 8));
    for z in 0..8 {
        for y in 0..64 {
            source[Position((y + z) % 64, y, z)].kind = BlockKind::RedstoneBlock;
        }
    }
    let deep = DeepWorld {
        map: source
            .map
            .iter()
            .map(|plane| plane.iter().map(|row| row.as_ref().clone()).collect())
            .collect(),
    };
    let mut old_times = Vec::new();
    let mut new_times = Vec::new();
    for sample in 0..5 {
        let (old, new) = if sample % 2 == 0 {
            (deep_run(&deep, count), cow_run(&source, count))
        } else {
            let new = cow_run(&source, count);
            (deep_run(&deep, count), new)
        };
        assert_eq!(old.1, count * 2);
        assert_eq!(old.1, new.1);
        old_times.push(old.0);
        new_times.push(new.0);
        println!(
            "sample={sample} deep_ms={:.3} cow_ms={:.3} checksum={}",
            old.0.as_secs_f64() * 1000.0,
            new.0.as_secs_f64() * 1000.0,
            new.1
        );
    }
    old_times.sort();
    new_times.sort();
    println!("world=64x64x8 clones={count} edits_per_clone=2 samples=5 median_deep_ms={:.3} median_cow_ms={:.3} speedup={:.2}x", old_times[2].as_secs_f64()*1000.0, new_times[2].as_secs_f64()*1000.0, old_times[2].as_secs_f64()/new_times[2].as_secs_f64());
    let row_source = RowCowWorld {
        map: source
            .map
            .iter()
            .map(|plane| plane.as_ref().clone())
            .collect(),
    };
    old_times.clear();
    new_times.clear();
    for sample in 0..5 {
        let (old, new) = if sample % 2 == 0 {
            (row_cow_run(&row_source, count), cow_run(&source, count))
        } else {
            let new = cow_run(&source, count);
            (row_cow_run(&row_source, count), new)
        };
        assert_eq!(old.1, count * 2);
        assert_eq!(old.1, new.1);
        old_times.push(old.0);
        new_times.push(new.0);
        println!(
            "plane_sample={sample} row_cow_ms={:.3} plane_row_cow_ms={:.3} checksum={}",
            old.0.as_secs_f64() * 1000.0,
            new.0.as_secs_f64() * 1000.0,
            new.1
        );
    }
    old_times.sort();
    new_times.sort();
    println!("world=64x64x8 clones={count} edits_per_clone=2 samples=5 median_row_cow_ms={:.3} median_plane_row_cow_ms={:.3} speedup={:.2}x",old_times[2].as_secs_f64()*1000.0,new_times[2].as_secs_f64()*1000.0,old_times[2].as_secs_f64()/new_times[2].as_secs_f64());
    let sparse = World::from(&source);
    assert_eq!(
        dense_lookup_constructor(&sparse).iter_block(),
        World3D::from(&sparse).iter_block()
    );
    old_times.clear();
    new_times.clear();
    for sample in 0..5 {
        let (old, new) = if sample % 2 == 0 {
            (
                construction_run(&sparse, count, dense_lookup_constructor),
                construction_run(&sparse, count, sparse_constructor),
            )
        } else {
            let new = construction_run(&sparse, count, sparse_constructor);
            (
                construction_run(&sparse, count, dense_lookup_constructor),
                new,
            )
        };
        assert_eq!(old.1, count);
        assert_eq!(old.1, new.1);
        old_times.push(old.0);
        new_times.push(new.0);
        println!(
            "construction_sample={sample} dense_lookup_ms={:.3} sparse_write_ms={:.3} checksum={}",
            old.0.as_secs_f64() * 1000.0,
            new.0.as_secs_f64() * 1000.0,
            new.1
        );
    }
    old_times.sort();
    new_times.sort();
    println!("world=64x64x8 blocks={} constructions={count} samples=5 median_dense_lookup_ms={:.3} median_sparse_write_ms={:.3} speedup={:.2}x",sparse.blocks.len(),old_times[2].as_secs_f64()*1000.0,new_times[2].as_secs_f64()*1000.0,old_times[2].as_secs_f64()/new_times[2].as_secs_f64());
    Ok(())
}
