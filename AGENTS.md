# Agent Notes

## Documentation

- Verilog RTL interface design notes: `docs/verilog_rtl_interface_design.md`
- Free 3D global placement design: `docs/superpowers/specs/2026-07-11-free-3d-global-placement-design.md`
- Feedback-directed global PnR search design: `docs/superpowers/specs/2026-07-12-feedback-directed-global-pnr-search-design.md`

When asked to create or preserve project documentation, add an appropriate file under `docs/` and link it from this file when it is useful for future agents.

## Git

When committing changes, include the intent behind the change in the commit message body.

## Testing

Run local placer and place-and-route tests with `cargo test --release`; debug builds are too slow for these search-heavy tests.
