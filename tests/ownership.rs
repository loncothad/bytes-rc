//! Ownership and aliasing regressions for non-atomic byte buffers.

use std::{
    cell::Cell,
    rc::Rc,
};

use bytes_rc::{
    Bytes,
    BytesMut,
    buf::{
        Buf,
        BufMut,
    },
};

#[test]
fn frozen_prefix_survives_tail_growth_and_reuse() {
    let mut prefix = BytesMut::with_capacity(64);
    prefix.extend_from_slice(b"abcdefgh");
    let mut tail = prefix.split_off(4);
    let prefix = prefix.freeze();
    let snapshot = prefix.clone();
    tail.reserve(128);
    tail.extend_from_slice(&[9; 128]);
    assert_eq!(prefix.as_ref(), b"abcd");
    assert_eq!(snapshot.as_ref(), b"abcd");
    assert_eq!(&tail[.. 4], b"efgh");
    let mut tail = tail.freeze().try_into_mut().unwrap();
    tail[0] = 42;
    assert_eq!(prefix.as_ref(), b"abcd");
    assert_eq!(tail[0], 42);
}

#[test]
fn unique_frozen_tail_can_grow_and_transfer_to_vec() {
    let mut bytes = BytesMut::with_capacity(16);
    bytes.extend_from_slice(b"0123456789abcdef");
    let prefix = bytes.split_to(13);
    drop(prefix);
    let mut bytes = bytes.freeze().try_into_mut().unwrap();
    bytes.reserve(100);
    bytes.extend_from_slice(&[7; 100]);
    let bytes = Vec::<u8>::from(bytes);
    assert_eq!(&bytes[.. 3], b"def");
    assert_eq!(&bytes[3 ..], &[7; 100]);
}

#[test]
fn initialized_spare_capacity_supports_buf_mut_contract() {
    let mut bytes = BytesMut::with_capacity(32);
    bytes.put_slice(b"prefix");
    bytes.chunk_mut()[.. 4].copy_from_slice(b"tail");
    // SAFETY: the four bytes in the spare tail were initialized above.
    unsafe {
        bytes.advance_mut(4);
    }
    assert_eq!(bytes.as_ref(), b"prefixtail");
}

#[test]
fn owner_views_drop_exactly_once_in_each_drop_order() {
    struct Owner {
        bytes: [u8; 8],
        drops: Rc<Cell<usize>>,
    }
    impl AsRef<[u8]> for Owner {
        fn as_ref(&self) -> &[u8] {
            &self.bytes
        }
    }
    impl Drop for Owner {
        fn drop(&mut self) {
            self.drops.set(self.drops.get() + 1);
        }
    }
    for order in [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]] {
        let drops = Rc::new(Cell::new(0));
        let bytes = Bytes::from_owner(Owner {
            bytes: *b"abcdefgh",
            drops: Rc::clone(&drops),
        });
        let mut views = [Some(bytes.clone()), Some(bytes.slice(2 .. 6)), Some(bytes)];
        for (position, index) in order.into_iter().enumerate() {
            drop(views[index].take());
            assert_eq!(drops.get(), usize::from(position == 2));
            for view in views.iter().flatten() {
                assert!(!view.is_empty());
            }
        }
    }
}

#[test]
fn live_views_match_vec_models() {
    // Keep several mutable splits and immutable snapshots alive together.
    // Native runs cover 16,384 operations; Miri runs cover 1,024 operations.
    let seeds = if cfg!(miri) {
        4_u64
    } else {
        16
    };
    let steps = if cfg!(miri) {
        256
    } else {
        1024
    };
    for initial in 1 ..= seeds {
        let mut seed = initial.wrapping_mul(0x9E37_79B9_7F4A_7C15_u64);
        let mut buffers = vec![(BytesMut::from(&b"abcdefgh"[..]), b"abcdefgh".to_vec())];
        let mut snapshots: Vec<(Bytes, Vec<u8>)> = Vec::new();
        for _ in 0 .. steps {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let index = (seed as usize) % buffers.len();
            match (seed >> 32) % 10 {
                | 0 => {
                    let data = seed.to_le_bytes();
                    let count = (seed as usize) % 9;
                    buffers[index].0.extend_from_slice(&data[.. count]);
                    buffers[index].1.extend_from_slice(&data[.. count]);
                },
                | 1 => {
                    let new_len = (seed as usize) % 49;
                    buffers[index].0.resize(new_len, seed as u8);
                    buffers[index].1.resize(new_len, seed as u8);
                },
                | 2 => {
                    let count = (seed as usize) % (buffers[index].1.len() + 1);
                    buffers[index].0.advance(count);
                    buffers[index].1.drain(.. count);
                },
                | 3 if buffers.len() < 8 => {
                    let at = (seed as usize) % (buffers[index].1.len() + 1);
                    let tail = buffers[index].0.split_off(at);
                    let model = buffers[index].1.split_off(at);
                    buffers.push((tail, model));
                },
                | 4 if buffers.len() > 1 => {
                    let other = (index + 1) % buffers.len();
                    let (tail, model) = buffers.remove(other);
                    let index = index - usize::from(other < index);
                    buffers[index].0.unsplit(tail);
                    buffers[index].1.extend(model);
                },
                | 5 => {
                    let bytes = std::mem::take(&mut buffers[index].0).freeze();
                    snapshots.push((bytes.clone(), buffers[index].1.clone()));
                    buffers[index].0 = BytesMut::from(bytes);
                    if snapshots.len() > 4 {
                        snapshots.remove(0);
                    }
                },
                | 6 => {
                    let count = (seed as usize) % 97;
                    buffers[index].0.reserve(count);
                    assert!(buffers[index].0.capacity() - buffers[index].0.len() >= count);
                },
                | 7 => {
                    let count = (seed as usize) % 97;
                    if buffers[index].0.try_reclaim(count) {
                        assert!(buffers[index].0.capacity() - buffers[index].0.len() >= count);
                    }
                },
                | 8 => {
                    let bytes = Vec::<u8>::from(std::mem::take(&mut buffers[index].0));
                    buffers[index].0 = BytesMut::from(bytes);
                },
                | 9 if buffers.len() < 8 => {
                    let bytes = buffers[index].0.clone();
                    let model = buffers[index].1.clone();
                    buffers.push((bytes, model));
                },
                | _ => {},
            }
            for (bytes, model) in &buffers {
                assert_eq!(bytes.as_ref(), model.as_slice());
                assert!(bytes.len() <= bytes.capacity());
            }
            for (bytes, model) in &snapshots {
                assert_eq!(bytes.as_ref(), model.as_slice());
            }
        }
    }
}
