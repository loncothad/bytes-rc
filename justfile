set shell := ["bash", "-eu", "-o", "pipefail", "-c"]

cargo := "cargo +beta"
nightly_cargo := "cargo +nightly"
miri_flags := "-Zmiri-strict-provenance"

default: check

# Format Rust and TOML sources.
fmt:
    {{ nightly_cargo }} fmt
    taplo fmt

# Check formatting without changing files.
fmt-check:
    {{ nightly_cargo }} fmt -- --check
    taplo fmt --check

# Type-check every target with every feature.
check:
    {{ cargo }} check --all-targets --all-features

# Build every target with every feature.
build:
    {{ cargo }} build --all-targets --all-features

# Build every target with optimizations and every feature.
build-release:
    {{ cargo }} build --release --all-targets --all-features

# Run Clippy with the crate's deny-by-default lint policy.
clippy:
    {{ cargo }} clippy --all-targets --all-features

# Run unit and integration tests in debug mode.
test:
    {{ cargo }} test --all-targets

# Run unit and integration tests with optimizations enabled.
test-release:
    {{ cargo }} test --release --all-targets

# Run documentation tests.
doctest:
    {{ cargo }} test --doc

# Build API documentation and treat warnings as errors.
doc:
    RUSTDOCFLAGS=-Dwarnings {{ cargo }} doc --no-deps

# Run the full Miri suite except the intentionally long stress tests.
miri:
    MIRIFLAGS={{ miri_flags }} {{ nightly_cargo }} miri test --lib -- \
        --skip bytes_mut_advance_remaining_capacity \
        --skip operation_sequence_matches_vec_model

# Run the deterministic long operation-sequence test under Miri.
miri-model:
    MIRIFLAGS={{ miri_flags }} {{ nightly_cargo }} miri test --lib \
        tests::operation_sequence_matches_vec_model -- --exact

# Inspect the dependency tree.
tree:
    {{ cargo }} tree

# Update dependencies using Cargo's resolver.
update:
    {{ cargo }} update

# Build and verify the distributable crate archive.
package:
    {{ cargo }} package --allow-dirty --locked

# Build and verify the crate archive without network access.
package-offline:
    {{ cargo }} package --allow-dirty --locked --offline

# Validate the package manifest as a publishable crate.
publish-dry-run:
    {{ cargo }} publish --dry-run --locked

# Run the complete local quality gate.
ci: fmt-check check clippy test doctest doc

# Remove Cargo build artifacts.
clean:
    {{ cargo }} clean
