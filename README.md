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

## Custom allocators

`Bytes<A = Global>` and `BytesMut<A = Global>` accept any stabilized
`Allocator`, including borrowed allocators. The existing constructors still use
`Global`. Use `new_in`, `with_capacity_in` (mutable buffers),
`copy_from_slice_in`, or `from_owner_in` (immutable buffers) to select an
allocator; `allocator()` returns the original instance.

Unique mutable buffers own their allocator directly. Splitting, freezing, and
owner-backed buffers put it in a non-atomic shared holder. Byte storage, the
holder, and shared/owner control blocks are allocated through that same allocator.
Owner-provided byte storage remains the owner's responsibility. Empty derived
views retain their allocator and can therefore keep the original storage alive.

Sharing and growth do not require `A: Clone` or clone the allocator. Deep mutable
clones, copying conversions from shared/owner-backed buffers, and infallible
`Vec<u8, A>` / `Box<[u8], A>` conversions require `A: Clone` only for fresh
allocations. A cloned allocator need not be equivalent to the original.

`Vec<u8, A>` and `Box<[u8], A>` inputs preserve the exact owning allocator,
including empty allocations. `try_into_vec()` supports non-Clone allocators and
transfers both storage and allocator without copying when exclusively held
(offset views may move their bytes back to the allocation base). It returns the
buffer on failure: sibling handles or detached allocations can still share the
allocator. Dropping those siblings allows the original allocator to be recovered.

```rust
use std::{
    alloc::{AllocError, Allocator, Global, Layout},
    cell::Cell,
    ptr::NonNull,
};
use bytes_rc::{BytesMut, buf::BufMut};

struct Counting<'a>(&'a Cell<usize>);

// SAFETY: allocation and deallocation both delegate to Global with the
// unchanged layouts and pointers.
unsafe impl Allocator for Counting<'_> {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.0.set(self.0.get() + 1);
        Global.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the caller supplies the original allocation and a fitting layout.
        unsafe { Global.deallocate(ptr, layout) }
    }
}

let allocations = Cell::new(0);
let mut buffer = BytesMut::with_capacity_in(8, Counting(&allocations));
buffer.put_slice(b"hello");
let bytes = buffer.freeze();
assert!(allocations.get() > 0);
assert!(std::ptr::eq(bytes.allocator().0, &allocations));
let vec = bytes.try_into_vec().unwrap();
assert_eq!(vec.as_slice(), b"hello");
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
