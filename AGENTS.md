# Agent Notes

## Project Notes

- Local placer improvement context: `docs/local_placer_improvement.md`
- Hierarchical placer design and SCC/primitive layout strategy: `docs/hierarchical_placer_design.md`
- Sequential primitive and soft macro placement notes: `docs/sequential_primitives.md`

## Documentation

When asked to create or preserve project documentation, add an appropriate file under `docs/` and link it from this file when it is useful for future agents.

## Testing

Run local placer and place-and-route tests with `cargo test --release`; debug builds are too slow for these search-heavy tests.
