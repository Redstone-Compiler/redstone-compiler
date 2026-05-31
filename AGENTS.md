# Agent Notes

## Documentation

When asked to create or preserve project documentation, add an appropriate file under `docs/` and link it from this file when it is useful for future agents.

## Git

When committing changes, include the intent behind the change in the commit message body.

## Testing

Run local placer and place-and-route tests with `cargo test --release`; debug builds are too slow for these search-heavy tests.
