# Native build and baseline verification

Use native entry points below. Build tooling and test selection are distinct from product requirements. Target task-specific cases and nearby regressions, record selected/filtered counts and exit codes. Zero selected tests is not success.

Pinned Rust 1.98.1; dependencies already cached locally. On a fresh machine run cargo fetch --locked once before offline tests. C compiler must be /usr/bin/gcc.

```sh
cargo test --locked --offline -p globset -p ignore --lib
```
