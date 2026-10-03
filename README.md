# bytes-rc

`bytes-rc` provides single-threaded byte buffers with the familiar
`bytes::Buf` and `bytes::BufMut` interfaces:

- `Bytes` is an immutable, cheaply cloneable view backed by non-atomic
  reference counting.
- `BytesMut` is a growable mutable buffer. Splitting is zero-copy, while
  cloning performs a deep copy.

Use this crate when buffers stay on one thread and atomic reference counting is
unnecessary. The buffer types intentionally do not implement `Send` or `Sync`;
use the `bytes` crate directly when values must cross thread boundaries.

## Example

```rust
use bytes_rc::{buf::{Buf, BufMut}, BytesMut};

let mut buffer = BytesMut::with_capacity(32);
buffer.put_slice(b"hello world");

let mut bytes = buffer.freeze();
assert_eq!(bytes.chunk(), b"hello world");

bytes.advance(6);
assert_eq!(&bytes[..], b"world");

let clone = bytes.clone();
assert_eq!(clone, b"world".as_slice());
```

## Rust version and development

The minimum supported Rust version (MSRV) is 1.100, which stabilizes the
`Allocator` API used for buffer allocations. The intended baseline is the
Rust 1.100 stable release; this repository does not pin a beta toolchain.

```sh
cargo check --all-targets
cargo clippy --all-targets
cargo test
```

The `justfile` uses ordinary Cargo commands for builds, checks, tests, and
documentation. Formatting still uses nightly because `.rustfmt.toml` contains
nightly-only options; Miri also requires nightly. Install these tools with
`rustup toolchain install nightly --profile minimal --component rustfmt,miri`,
then use `just fmt-check` (which also requires Taplo) and `just miri`.

## License

Licensed under either the Apache License, Version 2.0, or the MIT license, at
your option. Unless explicitly stated otherwise, contributions are dual
licensed under the same terms.
