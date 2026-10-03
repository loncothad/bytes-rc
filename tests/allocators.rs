//! Public allocator ownership, layout, and panic-safety regressions.

use std::{
    alloc::{
        AllocError,
        Allocator,
        Global,
        Layout,
    },
    cell::{
        Cell,
        RefCell,
    },
    collections::BTreeMap,
    panic::{
        AssertUnwindSafe,
        catch_unwind,
    },
    ptr::NonNull,
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

#[derive(Default)]
struct State {
    next_id:        Cell<usize>,
    allocations:    Cell<usize>,
    grows:          Cell<usize>,
    deallocations:  Cell<usize>,
    clones:         Cell<usize>,
    panic_clone:    Cell<bool>,
    panic_allocate: Cell<Option<usize>>,
    live:           RefCell<BTreeMap<usize, (usize, Layout, Layout)>>,
}

struct Tracked {
    id:    usize,
    state: Rc<State>,
}

impl State {
    fn allocator(self: &Rc<Self>) -> Tracked {
        let id = self.next_id.get();
        self.next_id.set(id + 1);
        Tracked {
            id,
            state: self.clone(),
        }
    }

    fn assert_empty(&self) {
        assert!(self.live.borrow().is_empty());
        assert_eq!(self.allocations.get(), self.deallocations.get());
    }
}

impl Clone for Tracked {
    fn clone(&self) -> Self {
        assert!(!self.state.panic_clone.get(), "allocator clone panicked");
        self.state.clones.set(self.state.clones.get() + 1);
        self.state.allocator()
    }
}

fn excess_layout(layout: Layout) -> Layout {
    Layout::from_size_align(layout.size() + 16, layout.align()).unwrap()
}

// SAFETY: each allocation uses Global storage of at least the requested size
// and alignment, and the map preserves its actual layout across grow and free.
unsafe impl Allocator for Tracked {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let next = self.state.allocations.get() + 1;
        assert_ne!(
            self.state.panic_allocate.get(),
            Some(next),
            "allocator allocate panicked"
        );
        let actual = excess_layout(layout);
        let ptr = Global.allocate(actual)?;
        let old = self
            .state
            .live
            .borrow_mut()
            .insert(ptr.cast::<u8>().as_ptr() as usize, (self.id, layout, actual));
        assert!(old.is_none());
        self.state.allocations.set(next);
        Ok(ptr)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        let (id, requested, actual) = self
            .state
            .live
            .borrow_mut()
            .remove(&(ptr.as_ptr() as usize))
            .expect("unknown allocation");
        assert_eq!(id, self.id, "wrong owning allocator instance");
        assert_eq!(layout, requested, "wrong deallocation layout");
        self.state.deallocations.set(self.state.deallocations.get() + 1);
        // SAFETY: the recorded allocation came from Global with this layout.
        unsafe {
            Global.deallocate(ptr, actual);
        }
    }

    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old_layout: Layout,
        new_layout: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let (id, requested, actual) = self.state.live.borrow()[&(ptr.as_ptr() as usize)];
        assert_eq!(id, self.id, "wrong growing allocator instance");
        assert_eq!(old_layout, requested, "wrong growth layout");
        let new_actual = excess_layout(new_layout);
        // SAFETY: the original Global layout is recorded; the new layout grows
        // its size and preserves its alignment.
        let new_ptr = unsafe { Global.grow(ptr, actual, new_actual)? };
        let mut live = self.state.live.borrow_mut();
        live.remove(&(ptr.as_ptr() as usize));
        assert!(
            live.insert(
                new_ptr.cast::<u8>().as_ptr() as usize,
                (self.id, new_layout, new_actual)
            )
            .is_none()
        );
        self.state.grows.set(self.state.grows.get() + 1);
        Ok(new_ptr)
    }
}

struct NonClone(Tracked);

// SAFETY: every operation delegates to the same original Tracked allocator.
unsafe impl Allocator for NonClone {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        self.0.allocate(layout)
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        // SAFETY: the caller guarantees this allocation's ownership and layout.
        unsafe {
            self.0.deallocate(ptr, layout);
        }
    }

    unsafe fn grow(&self, ptr: NonNull<u8>, old: Layout, new: Layout) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the caller guarantees the allocation and compatible layouts.
        unsafe { self.0.grow(ptr, old, new) }
    }
}

#[test]
fn unique_vec_growth_preserves_exact_allocator_and_requested_capacity() {
    let state = Rc::new(State::default());
    let allocator = NonClone(state.allocator());
    let id = allocator.0.id;
    let mut vec = Vec::with_capacity_in(8, allocator);
    vec.extend_from_slice(b"abc");
    let mut bytes = BytesMut::from(vec);
    assert_eq!(bytes.capacity(), 8);
    bytes.reserve(100);
    bytes.put_slice(b"def");
    let vec = bytes.try_into_vec().unwrap();
    assert_eq!(vec.allocator().0.id, id);
    assert_eq!(vec.as_slice(), b"abcdef");
    assert_eq!(state.grows.get(), 1);
    drop(vec);
    state.assert_empty();
}

#[test]
fn unique_offset_growth_and_freeze_keep_original_allocation_owner() {
    let state = Rc::new(State::default());
    let mut bytes = BytesMut::copy_from_slice_in(b"abcdefgh", NonClone(state.allocator()));
    let id = bytes.allocator().0.id;
    let prefix = bytes.split_to(3);
    drop(prefix);
    bytes.reserve(100);
    let mut bytes = bytes.freeze().try_into_mut().unwrap();
    bytes.put_slice(b"ij");
    let vec = bytes.try_into_vec().unwrap();
    assert_eq!(vec.allocator().0.id, id);
    assert_eq!(vec.as_slice(), b"defghij");
    assert_eq!(state.grows.get(), 1);
    drop(vec);
    state.assert_empty();
}

#[test]
fn shared_detach_uses_original_allocator_without_clone_bound() {
    let state = Rc::new(State::default());
    let mut bytes = BytesMut::copy_from_slice_in(b"abcdefgh", NonClone(state.allocator()));
    let prefix = bytes.split_to(3).freeze();
    let id = bytes.allocator().0.id;
    bytes.reserve(100);
    bytes.put_slice(b"ij");
    assert_eq!(prefix.as_slice(), b"abc");
    assert_eq!(bytes.allocator().0.id, id);
    let bytes = bytes.try_into_vec().unwrap_err();
    drop(prefix);
    let vec = bytes.try_into_vec().unwrap();
    assert_eq!(vec.allocator().0.id, id);
    assert_eq!(vec.as_slice(), b"defghij");
    assert_eq!(state.clones.get(), 0);
    drop(vec);
    state.assert_empty();
}

#[test]
fn immutable_clones_and_empty_views_never_clone_allocator() {
    let state = Rc::new(State::default());
    state.panic_clone.set(true);
    let mut bytes = Bytes::copy_from_slice_in(b"abcdefgh", state.allocator());
    let id = bytes.allocator().id;
    let clone = bytes.clone();
    let slice = bytes.slice(2 .. 5);
    let empty = bytes.slice(3 .. 3);
    let empty_ref = bytes.slice_ref(b"");
    let end = bytes.len();
    let split_off = bytes.split_off(end);
    let split_to = bytes.split_to(0);
    assert_eq!(empty.allocator().id, id);
    assert_eq!(empty_ref.allocator().id, id);
    assert_eq!(split_off.allocator().id, id);
    assert_eq!(split_to.allocator().id, id);
    assert_eq!(slice.as_slice(), b"cde");
    drop((clone, slice, empty, empty_ref, split_off, split_to));
    let vec: Vec<u8, Tracked> = bytes.into();
    assert_eq!(vec.allocator().id, id);
    assert_eq!(state.clones.get(), 0);
    drop(vec);
    state.assert_empty();
}

#[test]
fn empty_vec_roundtrip_preserves_allocator_and_capacity() {
    let state = Rc::new(State::default());
    for capacity in [0, 8] {
        let allocator = NonClone(state.allocator());
        let id = allocator.0.id;
        let vec = Vec::with_capacity_in(capacity, allocator);
        let bytes = Bytes::from(vec);
        assert!(bytes.is_empty());
        let vec = bytes.try_into_vec().unwrap();
        assert_eq!(vec.allocator().0.id, id);
        assert_eq!(vec.capacity(), capacity);
        drop(vec);
        state.assert_empty();
    }
    let bytes = Bytes::new_in(NonClone(state.allocator()));
    let id = bytes.allocator().0.id;
    let mut bytes = bytes.try_into_mut().unwrap();
    bytes.put_slice(b"hello");
    let vec = bytes.try_into_vec().unwrap();
    assert_eq!(vec.allocator().0.id, id);
    drop(vec);
    state.assert_empty();
}

#[test]
fn shared_vec_conversion_and_deep_clone_allocate_with_distinct_clone() {
    let state = Rc::new(State::default());
    let bytes = Bytes::copy_from_slice_in(b"abc", state.allocator());
    let id = bytes.allocator().id;
    let sibling = bytes.clone();
    let vec: Vec<u8, Tracked> = bytes.into();
    assert_ne!(vec.allocator().id, id);
    assert_eq!(vec.as_slice(), b"abc");
    assert_eq!(sibling.allocator().id, id);
    drop((vec, sibling));
    state.assert_empty();

    let mut original = BytesMut::copy_from_slice_in(b"def", state.allocator());
    let mut clone = original.clone();
    assert_ne!(clone.allocator().id, original.allocator().id);
    original.put_u8(b'g');
    clone.put_u8(b'h');
    assert_eq!(original.as_ref(), b"defg");
    assert_eq!(clone.as_ref(), b"defh");
    drop((original, clone));
    assert_eq!(state.clones.get(), 2);
    state.assert_empty();
}

#[test]
fn box_roundtrips_preserve_exact_allocator() {
    let state = Rc::new(State::default());
    for immutable in [false, true] {
        let mut vec = Vec::with_capacity_in(4, state.allocator());
        let id = vec.allocator().id;
        vec.extend_from_slice(b"abcd");
        let boxed = vec.into_boxed_slice();
        let boxed: Box<[u8], Tracked> = if immutable {
            Bytes::from(boxed).into()
        } else {
            BytesMut::from(boxed).into()
        };
        assert_eq!(Box::allocator(&boxed).id, id);
        assert_eq!(boxed.as_ref(), b"abcd");
        drop(boxed);
        state.assert_empty();
    }
    assert_eq!(state.clones.get(), 0);
}

#[test]
fn borrowed_nonclone_allocator_propagates_through_traits() {
    let state = Rc::new(State::default());
    let allocator = NonClone(state.allocator());
    let mut bytes = BytesMut::new_in(&allocator);
    bytes.extend(*b"ab");
    bytes.extend(b"c".iter());
    bytes.put_u8(b'd');
    assert!(std::ptr::eq(*bytes.allocator(), &allocator));
    assert_eq!(format!("{bytes:?}"), "b\"abcd\"");
    let mut bytes = bytes.freeze();
    bytes.advance(1);
    assert_eq!(bytes.into_iter().collect::<Vec<_>>(), b"bcd");
    state.assert_empty();
}

#[test]
fn panicking_clone_preserves_shared_and_unique_buffers() {
    let state = Rc::new(State::default());
    let bytes = Bytes::copy_from_slice_in(b"abc", state.allocator());
    let sibling = bytes.clone();
    state.panic_clone.set(true);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _: Vec<u8, Tracked> = bytes.into();
        }))
        .is_err()
    );
    assert_eq!(sibling.as_slice(), b"abc");
    let bytes = sibling.try_into_mut().unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| bytes.clone())).is_err());
    assert_eq!(bytes.as_ref(), b"abc");
    drop(bytes);
    state.assert_empty();

    let bytes = Bytes::copy_from_slice_in(b"xyz", state.allocator());
    let sibling = bytes.clone();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _: BytesMut<Tracked> = bytes.into();
        }))
        .is_err()
    );
    assert_eq!(sibling.as_slice(), b"xyz");
    drop(sibling);
    state.assert_empty();
}

#[test]
fn promotion_and_detachment_are_unwind_safe() {
    let state = Rc::new(State::default());
    let mut bytes = BytesMut::copy_from_slice_in(b"abc", NonClone(state.allocator()));
    state.panic_allocate.set(Some(state.allocations.get() + 2));
    assert!(catch_unwind(AssertUnwindSafe(|| bytes.split_to(1))).is_err());
    assert_eq!(bytes.as_ref(), b"abc");
    assert_eq!(state.live.borrow().len(), 1);
    state.panic_allocate.set(None);
    let prefix = bytes.split_to(1).freeze();
    let live = state.live.borrow().len();
    state.panic_allocate.set(Some(state.allocations.get() + 2));
    assert!(catch_unwind(AssertUnwindSafe(|| bytes.reserve(100))).is_err());
    assert_eq!(bytes.as_ref(), b"bc");
    assert_eq!(prefix.as_slice(), b"a");
    assert_eq!(state.live.borrow().len(), live);
    state.panic_allocate.set(None);
    bytes.reserve(100);
    drop((prefix, bytes));
    state.assert_empty();
}

#[test]
fn owner_control_allocations_use_original_allocator_and_drop_once() {
    struct Owner {
        bytes:   [u8; 3],
        drops:   Rc<Cell<usize>>,
        address: Cell<*const u8>,
    }
    impl AsRef<[u8]> for Owner {
        fn as_ref(&self) -> &[u8] {
            self.address.set(self.bytes.as_ptr());
            &self.bytes
        }
    }
    impl Drop for Owner {
        fn drop(&mut self) {
            assert_eq!(self.address.get(), self.bytes.as_ptr(), "owner moved before drop");
            self.drops.set(self.drops.get() + 1);
        }
    }
    let state = Rc::new(State::default());
    let drops = Rc::new(Cell::new(0));
    let owner = Owner {
        bytes:   *b"abc",
        drops:   drops.clone(),
        address: Cell::new(std::ptr::null()),
    };
    let bytes = Bytes::from_owner_in(owner, NonClone(state.allocator()));
    let id = bytes.allocator().0.id;
    let clone = bytes.clone();
    assert_eq!(bytes.as_slice(), b"abc");
    assert_eq!(clone.allocator().0.id, id);
    assert_eq!(state.allocations.get(), 2);
    drop(bytes);
    assert_eq!(drops.get(), 0);
    assert_eq!(clone.as_slice(), b"abc");
    drop(clone);
    assert_eq!(drops.get(), 1);
    state.assert_empty();
}

#[test]
fn panicking_owner_borrow_and_drop_release_control_allocations() {
    struct Owner(bool);
    impl AsRef<[u8]> for Owner {
        fn as_ref(&self) -> &[u8] {
            assert!(!self.0, "owner borrow panicked");
            b"abc"
        }
    }
    impl Drop for Owner {
        fn drop(&mut self) {
            assert!(self.0, "owner drop panicked");
        }
    }
    let state = Rc::new(State::default());
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            Bytes::from_owner_in(Owner(true), state.allocator())
        }))
        .is_err()
    );
    state.assert_empty();
    let bytes = Bytes::from_owner_in(Owner(false), state.allocator());
    assert!(catch_unwind(AssertUnwindSafe(|| drop(bytes))).is_err());
    state.assert_empty();
}
