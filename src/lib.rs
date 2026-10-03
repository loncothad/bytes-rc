//! Rc version of `bytes::Bytes` and `bytes::BytesMut`.
//!
//! This crate provides a non-atomic `Rc`-based alternative to the `Bytes` and
//! `BytesMut` types from the `bytes` crate. This is useful in single-threaded
//! contexts where the overhead of atomic reference counting (`Arc`) is
//! unnecessary.
//!
//! The types provided here implement [`bytes::Buf`] and [`bytes::BufMut`],
//! allowing them to be seamlessly integrated with ecosystems that expect buffer
//! types.
//!
//! `Bytes` and `BytesMut` intentionally do not implement `Send` or `Sync`. Use
//! the `bytes` crate when a buffer must cross thread boundaries.
//!
//! # Example
//!
//! ```
//! use bytes_rc::{
//!     BytesMut,
//!     buf::{
//!         Buf,
//!         BufMut,
//!     },
//! };
//!
//! let mut buffer = BytesMut::with_capacity(32);
//! buffer.put_slice(b"hello world");
//!
//! let mut bytes = buffer.freeze();
//! bytes.advance(6);
//! assert_eq!(&bytes[..], b"world");
//! ```
//!
//! The single-threaded guarantee is enforced by the type system:
//!
//! ```compile_fail
//! fn require_send<T: Send>() {}
//! require_send::<bytes_rc::Bytes>();
//! ```

//! # Custom allocators
//!
//! `Bytes<A = Global>` and `BytesMut<A = Global>` accept any stabilized
//! `Allocator`, including borrowed allocators. The existing constructors still
//! use `Global`. Use `new_in`, `with_capacity_in` (mutable buffers),
//! `copy_from_slice_in`, or `from_owner_in` (immutable buffers) to select an
//! allocator; `allocator()` returns the original instance.
//!
//! Unique mutable buffers own their allocator directly. Splitting, freezing,
//! and owner-backed buffers put it in a non-atomic shared holder. Byte storage,
//! the holder, and shared/owner control blocks are allocated through that same
//! allocator. Owner-provided byte storage remains the owner's responsibility.
//! Empty derived views retain their allocator and can therefore keep the
//! original storage alive.
//!
//! Sharing and growth do not require `A: Clone` or clone the allocator. Deep
//! mutable clones, copying conversions from shared/owner-backed buffers, and
//! infallible `Vec<u8, A>` / `Box<[u8], A>` conversions require `A: Clone` only
//! for fresh allocations. A cloned allocator need not be equivalent to the
//! original.
//!
//! `Vec<u8, A>` and `Box<[u8], A>` inputs preserve the exact owning allocator,
//! including empty allocations. `try_into_vec()` supports non-Clone allocators
//! and transfers both storage and allocator without copying when exclusively
//! held (offset views may move their bytes back to the allocation base). It
//! returns the buffer on failure: sibling handles or detached allocations can
//! still share the allocator. Dropping those siblings allows the original
//! allocator to be recovered.
//!
//! ```
//! use std::{
//!     alloc::{
//!         AllocError,
//!         Allocator,
//!         Global,
//!         Layout,
//!     },
//!     cell::Cell,
//!     ptr::NonNull,
//! };
//!
//! use bytes_rc::{
//!     BytesMut,
//!     buf::BufMut,
//! };
//!
//! struct Counting<'a>(&'a Cell<usize>);
//!
//! // SAFETY: allocation and deallocation both delegate to Global with the
//! // unchanged layouts and pointers.
//! unsafe impl Allocator for Counting<'_> {
//!     fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
//!         self.0.set(self.0.get() + 1);
//!         Global.allocate(layout)
//!     }
//!
//!     unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
//!         // SAFETY: the caller supplies the original allocation and a fitting layout.
//!         unsafe { Global.deallocate(ptr, layout) }
//!     }
//! }
//!
//! let allocations = Cell::new(0);
//! let mut buffer = BytesMut::with_capacity_in(8, Counting(&allocations));
//! buffer.put_slice(b"hello");
//! let bytes = buffer.freeze();
//! assert!(allocations.get() > 0);
//! assert!(std::ptr::eq(bytes.allocator().0, &allocations));
//! let vec = bytes.try_into_vec().unwrap();
//! assert_eq!(vec.as_slice(), b"hello");
//! ```
//!
//! Custom allocator parameters do not make these buffers thread-safe:
//!
//! ```compile_fail
//! fn require_sync<T: Sync>() {}
//! require_sync::<bytes_rc::Bytes<&std::alloc::Global>>();
//! ```
//!
//! ```compile_fail
//! fn require_send<T: Send>() {}
//! require_send::<bytes_rc::BytesMut<&std::alloc::Global>>();
//! ```

#![deny(missing_docs)]

use core::{
    alloc::{
        Allocator,
        Layout,
    },
    ascii,
    borrow::{
        Borrow,
        BorrowMut,
    },
    cell::Cell,
    cmp::Ordering,
    fmt,
    hash::{
        Hash,
        Hasher,
    },
    mem,
    ops::{
        Bound,
        Deref,
        DerefMut,
        RangeBounds,
    },
    ptr::{
        self,
        NonNull,
    },
    slice,
};
use std::alloc::{
    Global,
    handle_alloc_error,
};

#[doc(no_inline)]
pub use bytes::buf;
use bytes::{
    Buf,
    BufMut,
};

/// A cheap, cloneable, non-atomic byte array.
pub struct Bytes<A: Allocator = Global> {
    ptr:    *const u8,
    len:    usize,
    data:   *mut SharedData<A>,
    vtable: &'static Vtable,
}

/// A unique reference to a contiguous slice of memory.
pub struct BytesMut<A: Allocator = Global> {
    ptr:     NonNull<u8>,
    len:     usize,
    cap:     usize,
    backing: Backing<A>,
}

enum Backing<A: Allocator> {
    Unique(A),
    Shared(NonNull<SharedData<A>>),
}

struct AllocatorData<A> {
    ref_cnt:   Cell<usize>,
    allocator: A,
}

struct AllocatorRef<A: Allocator> {
    ptr: NonNull<AllocatorData<A>>,
}

impl<A: Allocator> AllocatorRef<A> {
    fn new(allocator: A) -> Self {
        let ptr = allocate_value_ptr::<AllocatorData<A>, A>(&allocator);
        // SAFETY: the allocation has the exact size and alignment of this
        // value.
        unsafe {
            ptr.write(AllocatorData {
                ref_cnt: Cell::new(1),
                allocator,
            });
        }
        Self {
            ptr: NonNull::new(ptr).unwrap(),
        }
    }

    fn allocator(&self) -> &A {
        // SAFETY: each handle retains one reference to the initialized holder.
        unsafe { &self.ptr.as_ref().allocator }
    }

    fn is_unique(&self) -> bool {
        // SAFETY: the holder remains live through this handle.
        unsafe { self.ptr.as_ref().ref_cnt.get() == 1 }
    }

    fn try_unwrap(self) -> Result<A, Self> {
        if !self.is_unique() {
            return Err(self);
        }
        let this = mem::ManuallyDrop::new(self);
        // SAFETY: uniqueness permits moving A out; deallocation uses that exact
        // instance only after it has moved off the allocation being released.
        unsafe {
            let data = this.ptr.as_ptr().read();
            data.allocator
                .deallocate(this.ptr.cast(), Layout::new::<AllocatorData<A>>());
            Ok(data.allocator)
        }
    }
}

impl<A: Allocator> Clone for AllocatorRef<A> {
    fn clone(&self) -> Self {
        // SAFETY: this handle keeps the holder live while incrementing its
        // count.
        unsafe {
            increment_ref_count(&self.ptr.as_ref().ref_cnt);
        }
        Self {
            ptr: self.ptr
        }
    }
}

impl<A: Allocator> Drop for AllocatorRef<A> {
    fn drop(&mut self) {
        // SAFETY: this handle owns one reference. The last reference moves A
        // out before using it to release its own holder allocation.
        unsafe {
            let count = self.ptr.as_ref().ref_cnt.get();
            if count == 1 {
                let data = self.ptr.as_ptr().read();
                data.allocator
                    .deallocate(self.ptr.cast(), Layout::new::<AllocatorData<A>>());
            } else {
                self.ptr.as_ref().ref_cnt.set(count - 1);
            }
        }
    }
}

struct SharedData<A: Allocator> {
    ref_cnt:   Cell<usize>,
    alloc_ptr: NonNull<u8>,
    alloc_cap: usize,
    allocator: AllocatorRef<A>,
}

impl<A: Allocator> SharedData<A> {
    const VTABLE: Vtable = Vtable {
        clone:     clone_shared::<A>,
        drop:      drop_shared::<A>,
        allocator: shared_allocator::<A>,
        shared:    true,
    };

    fn new(alloc_ptr: NonNull<u8>, alloc_cap: usize, allocator: AllocatorRef<A>) -> *mut Self {
        let ptr = allocate_value_ptr::<Self, A>(allocator.allocator());
        // SAFETY: the allocation has the exact layout of the initialized value.
        unsafe {
            ptr.write(Self {
                ref_cnt: Cell::new(1),
                alloc_ptr,
                alloc_cap,
                allocator,
            });
        }
        ptr
    }

    /// # Safety
    /// The caller must uniquely own the initialized control block at ptr.
    unsafe fn into_parts(ptr: *mut Self) -> (NonNull<u8>, usize, AllocatorRef<A>) {
        // SAFETY: the caller uniquely owns the initialized control block.
        unsafe {
            let data = ptr.read();
            data.allocator
                .allocator()
                .deallocate(NonNull::new_unchecked(ptr).cast(), Layout::new::<Self>());
            (data.alloc_ptr, data.alloc_cap, data.allocator)
        }
    }
}

struct Vtable {
    clone:     fn(*mut ()),
    drop:      fn(*mut ()),
    allocator: fn(*mut ()) -> *const (),
    shared:    bool,
}

#[inline]
fn allocation_layout(capacity: usize) -> Layout {
    Layout::array::<u8>(capacity).expect("capacity overflow")
}

fn allocate_value_ptr<T, A: Allocator>(allocator: &A) -> *mut T {
    let layout = Layout::new::<T>();
    allocator
        .allocate(layout)
        .unwrap_or_else(|_| handle_alloc_error(layout))
        .cast::<T>()
        .as_ptr()
}

#[inline]
fn allocate<A: Allocator>(capacity: usize, allocator: &A) -> NonNull<u8> {
    if capacity == 0 {
        return NonNull::dangling();
    }
    let layout = allocation_layout(capacity);
    // The requested size fits even when an allocator returns excess storage.
    // Retaining it keeps the layout valid for Vec and subsequent deallocation.
    allocator
        .allocate(layout)
        .unwrap_or_else(|_| handle_alloc_error(layout))
        .cast()
}

fn clone_shared<A: Allocator>(data: *mut ()) {
    // SAFETY: this vtable is installed only with SharedData<A> storage.
    unsafe {
        increment_ref_count(&(*(data.cast::<SharedData<A>>())).ref_cnt);
    }
}

fn drop_shared<A: Allocator>(data: *mut ()) {
    if data.is_null() {
        return;
    }
    // SAFETY: this vtable owns one reference to a live SharedData<A>.
    unsafe {
        let ptr = data.cast::<SharedData<A>>();
        let shared = &*ptr;
        let count = shared.ref_cnt.get();
        if count == 1 {
            let (alloc_ptr, alloc_cap, allocator) = SharedData::into_parts(ptr);
            if alloc_cap > 0 {
                allocator
                    .allocator()
                    .deallocate(alloc_ptr, allocation_layout(alloc_cap));
            }
        } else {
            shared.ref_cnt.set(count - 1);
        }
    }
}

fn shared_allocator<A: Allocator>(data: *mut ()) -> *const () {
    // SAFETY: the shared vtable guarantees the live control-block type.
    unsafe { (&*data.cast::<SharedData<A>>()).allocator.allocator() as *const A as *const () }
}

static STATIC_VTABLE: Vtable = Vtable {
    clone:     |_| {},
    drop:      |_| {},
    allocator: |_| (&Global as *const Global).cast(),
    shared:    false,
};

struct OwnerData<T, A: Allocator = Global> {
    ref_cnt:   Cell<usize>,
    owner:     T,
    allocator: AllocatorRef<A>,
}

impl<T, A: Allocator> OwnerData<T, A> {
    const VTABLE: Vtable = Vtable {
        clone:     clone_owner::<T, A>,
        drop:      drop_owner::<T, A>,
        allocator: owner_allocator::<T, A>,
        shared:    false,
    };
}

#[inline]
fn increment_ref_count(ref_cnt: &Cell<usize>) {
    let count = ref_cnt.get();
    assert!(count > 0 && count <= usize::MAX >> 1, "invalid reference count");
    ref_cnt.set(count + 1);
}

fn drop_owner<T, A: Allocator>(data: *mut ()) {
    if data.is_null() {
        return;
    }
    // SAFETY: the owner vtable guarantees this exact live OwnerData<T, A>.
    unsafe {
        let ptr = data.cast::<OwnerData<T, A>>();
        let count = (*ptr).ref_cnt.get();
        if count == 1 {
            let allocator = ptr::read(&raw const (*ptr).allocator);
            let guard = Box::from_raw_in(ptr.cast::<mem::MaybeUninit<OwnerData<T, A>>>(), allocator.allocator());
            ptr::drop_in_place(&raw mut (*ptr).owner);
            drop(guard);
        } else {
            (*ptr).ref_cnt.set(count - 1);
        }
    }
}

fn clone_owner<T, A: Allocator>(data: *mut ()) {
    // SAFETY: the owner vtable guarantees this exact live control-block type.
    unsafe {
        increment_ref_count(&(*data.cast::<OwnerData<T, A>>()).ref_cnt);
    }
}

fn owner_allocator<T, A: Allocator>(data: *mut ()) -> *const () {
    // SAFETY: the owner vtable guarantees this exact live control-block type.
    unsafe { (&*data.cast::<OwnerData<T, A>>()).allocator.allocator() as *const A as *const () }
}

// --- Bytes ---

impl Bytes {
    /// Creates a new empty `Bytes` instance.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ptr:    NonNull::dangling().as_ptr(),
            len:    0,
            data:   ptr::null_mut(),
            vtable: &STATIC_VTABLE,
        }
    }

    /// Creates a new `Bytes` from a static slice.
    #[inline]
    #[must_use]
    pub const fn from_static(bytes: &'static [u8]) -> Self {
        Self {
            ptr:    bytes.as_ptr(),
            len:    bytes.len(),
            data:   ptr::null_mut(),
            vtable: &STATIC_VTABLE,
        }
    }

    /// Creates a new `Bytes` by copying from a slice.
    #[inline]
    #[must_use]
    pub fn copy_from_slice(data: &[u8]) -> Self {
        if data.is_empty() {
            Self::new()
        } else {
            let mut b = BytesMut::with_capacity(data.len());
            b.put_slice(data);
            b.freeze()
        }
    }

    /// Creates a new `Bytes` from an arbitrary owner type that implements
    /// `AsRef<[u8]>`.
    ///
    /// The owner is kept at a stable address and dropped after the final clone
    /// or slice derived from this value. Converting owner-backed bytes into
    /// [`BytesMut`] always copies the visible bytes.
    #[must_use]
    pub fn from_owner<T>(owner: T) -> Self
    where
        T: AsRef<[u8]> + 'static,
    {
        Self::from_owner_in(owner, Global)
    }
}

impl<A: Allocator> Bytes<A> {
    /// Creates an empty buffer using the supplied allocator.
    #[must_use]
    pub fn new_in(allocator: A) -> Self {
        BytesMut::new_in(allocator).freeze()
    }

    /// Copies a slice into a buffer using the supplied allocator.
    #[must_use]
    pub fn copy_from_slice_in(data: &[u8], allocator: A) -> Self {
        let mut bytes = BytesMut::with_capacity_in(data.len(), allocator);
        bytes.extend_from_slice(data);
        bytes.freeze()
    }

    /// Keeps an owner at a stable address, allocating its control blocks with
    /// A.
    ///
    /// The owner's own storage is not reallocated. Its allocator, if any,
    /// remains the owner's responsibility.
    #[must_use]
    pub fn from_owner_in<T: AsRef<[u8]> + 'static>(owner: T, allocator: A) -> Self {
        let allocator = AllocatorRef::new(allocator);
        let data = allocate_value_ptr::<OwnerData<T, A>, A>(allocator.allocator());
        // SAFETY: the allocated control block has the exact required layout.
        unsafe {
            data.write(OwnerData {
                ref_cnt: Cell::new(1),
                owner,
                allocator,
            });
        }
        let mut bytes = Self {
            ptr:    NonNull::dangling().as_ptr(),
            len:    0,
            data:   data.cast(),
            vtable: &OwnerData::<T, A>::VTABLE,
        };
        // SAFETY: bytes guards the initialized, pinned owner during unwinding.
        let slice = unsafe { (*data).owner.as_ref() };
        bytes.ptr = slice.as_ptr();
        bytes.len = slice.len();
        bytes
    }

    /// Returns the original allocator shared by this buffer's handles.
    #[must_use]
    pub fn allocator(&self) -> &A {
        // SAFETY: the constructors install a vtable matching A and keep the
        // returned allocator live throughout this borrow.
        unsafe { &*(self.vtable.allocator)(self.data.cast()).cast::<A>() }
    }

    /// Transfers the allocation and original allocator to a Vec without
    /// copying.
    ///
    /// Returns this buffer unchanged if its storage or allocator is shared, or
    /// if it is static or owner-backed. No allocator Clone bound is required.
    pub fn try_into_vec(self) -> Result<Vec<u8, A>, Self> {
        match self.try_into_mut() {
            | Ok(bytes) => bytes.try_into_vec().map_err(BytesMut::freeze),
            | Err(bytes) => Err(bytes),
        }
    }

    /// Returns the number of bytes contained in this `Bytes`.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the `Bytes` has a length of 0.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns a pointer to the start of the data.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const u8 {
        self.ptr
    }

    /// Returns a slice of self for the provided range without copying.
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds or its start exceeds its end.
    #[must_use]
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        let len = self.len();
        let start = match range.start_bound() {
            | Bound::Included(&n) => n,
            | Bound::Excluded(&n) => n.checked_add(1).expect("range start overflow"),
            | Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            | Bound::Included(&n) => n.checked_add(1).expect("range end overflow"),
            | Bound::Excluded(&n) => n,
            | Bound::Unbounded => len,
        };

        assert!(start <= end, "range start must not be greater than end");
        assert!(end <= len, "range end out of bounds");

        let mut ret = self.clone();
        // SAFETY: the validated start is within this slice's allocation.
        ret.ptr = unsafe { ret.ptr.add(start) };
        ret.len = end - start;
        ret
    }

    /// Creates a new `Bytes` from a subslice of this `Bytes`.
    ///
    /// # Panics
    ///
    /// Panics if a non-empty `subset` is not contained in this byte view.
    #[must_use]
    pub fn slice_ref(&self, subset: &[u8]) -> Self {
        if subset.is_empty() {
            return self.slice(0 .. 0);
        }

        let offset = (subset.as_ptr() as usize)
            .checked_sub(self.as_ptr() as usize)
            .filter(|&offset| offset <= self.len() && subset.len() <= self.len() - offset)
            .expect("subset is out of bounds or not part of this allocation");

        self.slice(offset .. offset + subset.len())
    }

    /// Splits the buffer into two at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `at` exceeds the buffer length.
    #[must_use = "consider Bytes::truncate if you don't need the other half"]
    pub fn split_off(&mut self, at: usize) -> Self {
        assert!(at <= self.len(), "split_off out of bounds: {} <= {}", at, self.len());
        if at == self.len() {
            return self.slice(0 .. 0);
        }
        if at == 0 {
            let ret = self.clone();
            self.clear();
            return ret;
        }

        let mut ret = self.clone();
        // SAFETY: `at` was validated against the current slice length.
        ret.ptr = unsafe { ret.ptr.add(at) };
        ret.len = self.len - at;
        self.len = at;
        ret
    }

    /// Splits the buffer into two at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `at` exceeds the buffer length.
    #[must_use = "consider Bytes::advance if you don't need the other half"]
    pub fn split_to(&mut self, at: usize) -> Self {
        assert!(at <= self.len(), "split_to out of bounds: {} <= {}", at, self.len());
        if at == 0 {
            return self.slice(0 .. 0);
        }
        if at == self.len() {
            let ret = self.clone();
            self.clear();
            return ret;
        }

        let mut ret = self.clone();
        ret.len = at;
        // SAFETY: `at` was validated against the current slice length.
        self.ptr = unsafe { self.ptr.add(at) };
        self.len -= at;
        ret
    }

    /// Shortens the buffer, keeping the first `len` bytes and dropping the
    /// rest.
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        if len < self.len {
            self.len = len;
        }
    }

    /// Clears the buffer, removing all data.
    #[inline]
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Returns a slice to the underlying data.
    #[inline]
    #[must_use]
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: every constructor maintains a live pointer to at least `len`
        // initialized bytes; the owning handle keeps that storage alive.
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Returns true if this is the only reference to the underlying data.
    #[inline]
    #[must_use]
    pub fn is_unique(&self) -> bool {
        if self.vtable.shared && !self.data.is_null() {
            // SAFETY: the vtable discriminant proves `data` is `SharedData`.
            unsafe { (*self.data).ref_cnt.get() == 1 }
        } else {
            false
        }
    }

    /// Tries to convert this `Bytes` to a `BytesMut` without copying.
    ///
    /// This succeeds only for uniquely owned allocation-backed bytes.
    pub fn try_into_mut(self) -> Result<BytesMut<A>, Bytes<A>> {
        if self.vtable.shared && !self.data.is_null() {
            // SAFETY: the vtable discriminant proves the control-block type.
            // Uniqueness permits transferring its mutable allocation, and the
            // visible slice is always contained in that allocation.
            unsafe {
                if (*self.data).ref_cnt.get() == 1 {
                    let data = self.data;
                    let alloc_ptr = (*data).alloc_ptr;
                    let alloc_cap = (*data).alloc_cap;
                    let ptr = self.ptr;
                    let offset = ptr as usize - alloc_ptr.as_ptr() as usize;
                    let cap = alloc_cap - offset;

                    let mut s = self;
                    s.data = ptr::null_mut(); // prevent drop

                    return Ok(BytesMut {
                        ptr: NonNull::new_unchecked(ptr as *mut u8),
                        len: s.len,
                        cap,
                        backing: Backing::Shared(NonNull::new_unchecked(data)),
                    });
                }
            }
        }
        Err(self)
    }
}

impl<A: Allocator> Clone for Bytes<A> {
    #[inline]
    fn clone(&self) -> Self {
        (self.vtable.clone)(self.data.cast());
        Self {
            ptr:    self.ptr,
            len:    self.len,
            data:   self.data,
            vtable: self.vtable,
        }
    }
}

impl<A: Allocator> Drop for Bytes<A> {
    #[inline]
    fn drop(&mut self) {
        (self.vtable.drop)(self.data.cast())
    }
}

// --- BytesMut ---

impl BytesMut {
    /// Creates a buffer with the specified capacity using Global.
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        Self::with_capacity_in(capacity, Global)
    }

    /// Creates an empty buffer using Global.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            ptr:     NonNull::dangling(),
            len:     0,
            cap:     0,
            backing: Backing::Unique(Global),
        }
    }
}

impl<A: Allocator> BytesMut<A> {
    /// Creates an empty buffer owning the supplied allocator.
    #[must_use]
    pub const fn new_in(allocator: A) -> Self {
        Self {
            ptr:     NonNull::dangling(),
            len:     0,
            cap:     0,
            backing: Backing::Unique(allocator),
        }
    }

    /// Creates a buffer with the specified capacity using the supplied
    /// allocator.
    ///
    /// # Panics
    ///
    /// Panics if capacity exceeds the maximum allocation size.
    #[must_use]
    pub fn with_capacity_in(capacity: usize, allocator: A) -> Self {
        let ptr = allocate(capacity, &allocator);
        Self {
            ptr,
            len: 0,
            cap: capacity,
            backing: Backing::Unique(allocator),
        }
    }

    /// Copies a slice using the supplied allocator.
    #[must_use]
    pub fn copy_from_slice_in(data: &[u8], allocator: A) -> Self {
        let mut bytes = Self::with_capacity_in(data.len(), allocator);
        bytes.extend_from_slice(data);
        bytes
    }

    /// Returns the original allocator responsible for this allocation.
    #[must_use]
    pub fn allocator(&self) -> &A {
        match &self.backing {
            | Backing::Unique(allocator) => allocator,
            // SAFETY: the shared backing retains one live control-block reference.
            | Backing::Shared(data) => unsafe { data.as_ref().allocator.allocator() },
        }
    }

    fn shared_ptr(&self) -> *mut SharedData<A> {
        match self.backing {
            | Backing::Shared(data) => data.as_ptr(),
            | _ => ptr::null_mut(),
        }
    }

    /// Transfers the allocation and original allocator to a Vec without
    /// copying.
    ///
    /// Returns the buffer unchanged while its allocation or allocator is
    /// shared.
    pub fn try_into_vec(self) -> Result<Vec<u8, A>, Self> {
        let data = self.shared_ptr();
        if !data.is_null() {
            // SAFETY: data is live while this handle retains its reference.
            let shared = unsafe { &*data };
            if shared.ref_cnt.get() != 1 || !shared.allocator.is_unique() {
                return Err(self);
            }
        }
        let this = mem::ManuallyDrop::new(self);
        // SAFETY: ManuallyDrop suppresses the old handle's drop after this
        // move.
        let backing = unsafe { ptr::read(&this.backing) };
        let (base, cap, allocator) = match backing {
            | Backing::Unique(allocator) => (this.ptr, this.cap, allocator),
            | Backing::Shared(data) => {
                // SAFETY: both control block and allocator holder are unique.
                let (base, cap, allocator) = unsafe { SharedData::into_parts(data.as_ptr()) };
                let allocator = match allocator.try_unwrap() {
                    | Ok(a) => a,
                    | Err(_) => unreachable!(),
                };
                (base, cap, allocator)
            },
        };
        // SAFETY: the original allocation and allocator are transferred
        // together. copy handles overlap when an offset view must move
        // back to its base.
        unsafe {
            ptr::copy(this.ptr.as_ptr(), base.as_ptr(), this.len);
            Ok(Vec::from_raw_parts_in(base.as_ptr(), this.len, cap, allocator))
        }
    }

    /// Returns the number of bytes contained in this `BytesMut`.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns true if the `BytesMut` has a length of 0.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns the total capacity of the buffer from its current start.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.cap
    }

    /// Promotes the unique allocation to a shared state if needed.
    #[inline]
    fn promote(&mut self) {
        if !self.shared_ptr().is_null() {
            return;
        }
        // These Boxes guard both allocations until all fallible work finishes,
        // while the original allocator and byte allocation remain in self.
        let holder = Box::new_in(mem::MaybeUninit::<AllocatorData<A>>::uninit(), self.allocator());
        let shared = Box::new_in(mem::MaybeUninit::<SharedData<A>>::uninit(), self.allocator());
        let (holder, _) = Box::into_raw_with_allocator(holder);
        let (shared, _) = Box::into_raw_with_allocator(shared);
        // SAFETY: all fallible allocations are complete; the moved backing is
        // replaced below before this handle can be observed or dropped.
        let Backing::Unique(allocator) = (unsafe { ptr::read(&self.backing) }) else {
            unreachable!()
        };
        // SAFETY: both raw allocations have their exact layouts, and no
        // fallible operations remain before installing the owning
        // shared reference.
        unsafe {
            let holder = holder.cast::<AllocatorData<A>>();
            holder.write(AllocatorData {
                ref_cnt: Cell::new(1),
                allocator,
            });
            let shared = shared.cast::<SharedData<A>>();
            shared.write(SharedData {
                ref_cnt:   Cell::new(1),
                alloc_ptr: self.ptr,
                alloc_cap: self.cap,
                allocator: AllocatorRef {
                    ptr: NonNull::new_unchecked(holder),
                },
            });
            ptr::write(&mut self.backing, Backing::Shared(NonNull::new_unchecked(shared)));
        }
    }

    /// Reserves capacity for at least `additional` more bytes to be inserted.
    ///
    /// # Panics
    ///
    /// Panics if the resulting capacity overflows or exceeds the maximum
    /// allocation size.
    pub fn reserve(&mut self, additional: usize) {
        let remaining = self.cap - self.len;
        if additional <= remaining {
            return;
        }

        let required = self.len.checked_add(additional).expect("capacity overflow");

        let alloc_cap = if self.shared_ptr().is_null() {
            self.cap
        } else {
            // SAFETY: non-null `data` always points to a live `SharedData`.
            unsafe { (*self.shared_ptr()).alloc_cap }
        };

        let doubled = alloc_cap.checked_mul(2).unwrap_or(required);
        let new_cap = doubled.max(required).max(64);

        if self.shared_ptr().is_null() {
            if self.cap == 0 {
                self.ptr = allocate(new_cap, self.allocator());
                self.cap = new_cap;
                return;
            }

            // SAFETY: an unshared handle with non-zero capacity owns the block
            // described by `self.ptr` and `self.cap`.
            let new_ptr = unsafe {
                let layout = allocation_layout(self.cap);
                let new_layout = allocation_layout(new_cap);
                self.allocator()
                    .grow(self.ptr, layout, new_layout)
                    .unwrap_or_else(|_| handle_alloc_error(new_layout))
                    .cast()
            };
            self.ptr = new_ptr;
            self.cap = new_cap;
        } else {
            // SAFETY: this branch is selected only for a live control block.
            let shared = unsafe { &*self.shared_ptr() };
            if shared.ref_cnt.get() == 1 {
                let offset = self.ptr.as_ptr() as usize - shared.alloc_ptr.as_ptr() as usize;

                if shared.alloc_cap == 0 {
                    let new_alloc_ptr = allocate(new_cap, self.allocator());
                    self.ptr = new_alloc_ptr;
                    self.cap = new_cap;
                    // SAFETY: uniqueness gives exclusive access to the control
                    // block, which did not previously own an allocation.
                    unsafe {
                        (*self.shared_ptr()).alloc_ptr = new_alloc_ptr;
                        (*self.shared_ptr()).alloc_cap = new_cap;
                    }
                    return;
                }

                if shared.alloc_cap - offset >= required {
                    self.cap = shared.alloc_cap - offset;
                    return;
                }

                if offset > 0 && shared.alloc_cap >= required {
                    // SAFETY: source and destination are within the same live
                    // allocation; `copy` permits overlap.
                    unsafe {
                        ptr::copy(self.ptr.as_ptr(), shared.alloc_ptr.as_ptr(), self.len);
                    }
                    self.ptr = shared.alloc_ptr;
                    self.cap = shared.alloc_cap;
                    return;
                }

                let required_alloc_cap = offset.checked_add(required).expect("capacity overflow");
                let new_alloc_cap = new_cap.max(required_alloc_cap);
                // SAFETY: uniqueness gives ownership of the allocation, whose
                // base and layout are recorded in `shared`.
                let new_alloc_ptr = unsafe {
                    let layout = allocation_layout(shared.alloc_cap);
                    let new_layout = allocation_layout(new_alloc_cap);
                    self.allocator()
                        .grow(shared.alloc_ptr, layout, new_layout)
                        .unwrap_or_else(|_| handle_alloc_error(new_layout))
                        .cast::<u8>()
                };

                // SAFETY: `new_alloc_cap` includes `offset`, so this pointer is
                // in bounds and cannot be null.
                self.ptr = unsafe { NonNull::new_unchecked(new_alloc_ptr.as_ptr().add(offset)) };
                self.cap = new_alloc_cap - offset;

                // SAFETY: uniqueness gives exclusive access to the live
                // control block.
                unsafe {
                    (*self.shared_ptr()).alloc_ptr = new_alloc_ptr;
                    (*self.shared_ptr()).alloc_cap = new_alloc_cap;
                }
            } else {
                let allocator = shared.allocator.clone();
                let new_shared = SharedData::new(NonNull::dangling(), 0, allocator);
                // Guard the new control block if allocating bytes panics.
                let mut detached = BytesMut {
                    ptr:     NonNull::dangling(),
                    len:     0,
                    cap:     0,
                    // SAFETY: new_shared is a freshly allocated control block.
                    backing: Backing::Shared(unsafe { NonNull::new_unchecked(new_shared) }),
                };
                let new_alloc_ptr = allocate(new_cap, detached.allocator());
                // SAFETY: detached exclusively owns this initialized control
                // block.
                unsafe {
                    (*new_shared).alloc_ptr = new_alloc_ptr;
                    (*new_shared).alloc_cap = new_cap;
                }
                detached.ptr = new_alloc_ptr;
                detached.cap = new_cap;

                // SAFETY: both regions are valid for `self.len` bytes and are
                // in distinct allocations.
                unsafe {
                    ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_alloc_ptr.as_ptr(), self.len);
                }

                let cnt = shared.ref_cnt.get();
                debug_assert!(cnt > 1, "shared allocation must have another owner");
                shared.ref_cnt.set(cnt - 1);

                detached.len = self.len;
                mem::forget(mem::replace(self, detached));
            }
        }
    }

    /// Tries to reclaim capacity without reallocating.
    ///
    /// Returns `true` when at least `additional` bytes of spare capacity are
    /// available afterward.
    #[must_use = "consider BytesMut::reserve if allocation is acceptable"]
    pub fn try_reclaim(&mut self, additional: usize) -> bool {
        if additional <= self.cap - self.len {
            return true;
        }

        let Some(required) = self.len.checked_add(additional) else {
            return false;
        };

        if self.shared_ptr().is_null() {
            // Cannot easily reclaim without `SharedData` because we don't know
            // original ptr
            false
        } else {
            // SAFETY: non-null `data` points to the live shared allocation.
            // Uniqueness is checked before moving bytes within it.
            unsafe {
                let shared = &*self.shared_ptr();
                if shared.ref_cnt.get() == 1 {
                    let offset = self.ptr.as_ptr() as usize - shared.alloc_ptr.as_ptr() as usize;
                    if shared.alloc_cap - offset >= required {
                        self.cap = shared.alloc_cap - offset;
                        true
                    } else if offset > 0 && shared.alloc_cap >= required {
                        ptr::copy(self.ptr.as_ptr(), shared.alloc_ptr.as_ptr(), self.len);
                        self.ptr = shared.alloc_ptr;
                        self.cap = shared.alloc_cap;
                        true
                    } else {
                        false
                    }
                } else {
                    false
                }
            }
        }
    }

    /// Extends the buffer with the given slice.
    #[inline]
    pub fn extend_from_slice(&mut self, extend: &[u8]) {
        self.reserve(extend.len());
        // SAFETY: reservation guarantees writable tail space. A safe caller
        // cannot alias `extend` with the mutably borrowed `self`.
        unsafe {
            ptr::copy_nonoverlapping(extend.as_ptr(), self.ptr.as_ptr().add(self.len), extend.len());
        }
        self.len += extend.len();
    }

    /// Extends the buffer from within itself.
    ///
    /// # Panics
    ///
    /// Panics if the range is out of bounds or capacity overflows.
    pub fn extend_from_within<R>(&mut self, range: R)
    where
        R: RangeBounds<usize>,
    {
        let len = self.len;
        let start = match range.start_bound() {
            | Bound::Included(&n) => n,
            | Bound::Excluded(&n) => n.checked_add(1).expect("range start overflow"),
            | Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            | Bound::Included(&n) => n.checked_add(1).expect("range end overflow"),
            | Bound::Excluded(&n) => n,
            | Bound::Unbounded => len,
        };
        assert!(start <= end && end <= len, "range out of bounds");

        let cnt = end - start;
        self.reserve(cnt);
        // SAFETY: the source range was validated against initialized bytes and
        // the reserved destination immediately follows them, so it is disjoint.
        unsafe {
            ptr::copy_nonoverlapping(self.ptr.as_ptr().add(start), self.ptr.as_ptr().add(self.len), cnt);
        }
        self.len += cnt;
    }

    /// Puts a single byte at the end of the buffer.
    #[inline]
    pub fn put_u8(&mut self, n: u8) {
        self.reserve(1);
        // SAFETY: reservation guarantees one writable byte at `len`.
        unsafe {
            self.ptr.as_ptr().add(self.len).write(n);
        }
        self.len += 1;
    }

    /// Splits the buffer into two at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `at` exceeds the buffer length.
    #[must_use = "consider BytesMut::truncate if you don't need the other half"]
    pub fn split_off(&mut self, at: usize) -> BytesMut<A> {
        assert!(at <= self.len, "split_off out of bounds: {} <= {}", at, self.len);

        self.promote();
        // SAFETY: promotion installs a live shared control block.
        unsafe {
            increment_ref_count(&(*self.shared_ptr()).ref_cnt);
        }

        // SAFETY: `at <= len <= cap`, including the permitted one-past pointer.
        let new_ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(at)) };
        let new_len = self.len - at;
        let new_cap = self.cap - at;

        self.len = at;
        self.cap = at;

        BytesMut {
            ptr:     new_ptr,
            len:     new_len,
            cap:     new_cap,
            // SAFETY: promotion retained a live shared control block.
            backing: Backing::Shared(unsafe { NonNull::new_unchecked(self.shared_ptr()) }),
        }
    }

    /// Splits the buffer into two at the given index.
    ///
    /// # Panics
    ///
    /// Panics if `at` exceeds the buffer length.
    #[must_use = "consider Buf::advance if you don't need the other half"]
    pub fn split_to(&mut self, at: usize) -> BytesMut<A> {
        assert!(at <= self.len, "split_to out of bounds: {} <= {}", at, self.len);

        self.promote();
        // SAFETY: promotion installs a live shared control block.
        unsafe {
            increment_ref_count(&(*self.shared_ptr()).ref_cnt);
        }

        let new_ptr = self.ptr;
        let new_len = at;
        let new_cap = at;

        // SAFETY: `at <= len <= cap`, including the permitted one-past pointer.
        self.ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(at)) };
        self.len -= at;
        self.cap -= at;

        BytesMut {
            ptr:     new_ptr,
            len:     new_len,
            cap:     new_cap,
            // SAFETY: promotion retained a live shared control block.
            backing: Backing::Shared(unsafe { NonNull::new_unchecked(self.shared_ptr()) }),
        }
    }

    /// Splits the buffer into two at the current length.
    #[must_use]
    pub fn split(&mut self) -> BytesMut<A> {
        let len = self.len;
        self.split_to(len)
    }

    /// Unsplits the buffer.
    pub fn unsplit(&mut self, mut other: BytesMut<A>) {
        if self.is_empty() {
            *self = other;
            return;
        }
        if other.is_empty() {
            return;
        }

        // SAFETY: `len <= cap`, so computing the end of the initialized region
        // stays within the allocation.
        let contiguous = unsafe { self.ptr.as_ptr().add(self.len) == other.ptr.as_ptr() };
        let same_alloc = !self.shared_ptr().is_null()
            && !other.shared_ptr().is_null()
            && ptr::eq(self.shared_ptr(), other.shared_ptr());

        if contiguous && same_alloc {
            self.len += other.len;
            self.cap += other.cap;
            other.len = 0; // Prevent other from having data
            return;
        }

        self.extend_from_slice(&other);
    }

    /// Shortens the buffer, keeping the first `len` bytes and dropping the
    /// rest.
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        if len <= self.len {
            self.len = len;
        }
    }

    /// Clears the buffer, removing all data.
    #[inline]
    pub fn clear(&mut self) {
        self.truncate(0);
    }

    /// Sets the length of the buffer.
    ///
    /// # Safety
    /// The caller must ensure that the requested length is less than or equal
    /// to the capacity and that all bytes up to `len` are initialized.
    #[inline]
    pub unsafe fn set_len(&mut self, len: usize) {
        debug_assert!(len <= self.cap);
        self.len = len;
    }

    /// Freezes the `BytesMut` into a `Bytes`.
    #[must_use]
    pub fn freeze(mut self) -> Bytes<A> {
        self.promote();
        let data = self.shared_ptr();
        let ptr = self.ptr.as_ptr();
        let len = self.len;

        // Forget self to prevent Drop
        mem::forget(self);

        Bytes {
            ptr,
            len,
            data,
            vtable: &SharedData::<A>::VTABLE,
        }
    }

    /// Resizes the buffer.
    pub fn resize(&mut self, new_len: usize, value: u8) {
        if new_len > self.len {
            let additional = new_len - self.len;
            self.reserve(additional);
            // SAFETY: reservation made the requested tail writable.
            unsafe {
                ptr::write_bytes(self.ptr.as_ptr().add(self.len), value, additional);
            }
        }
        self.len = new_len;
    }
}

impl<A: Allocator + Clone> Clone for BytesMut<A> {
    fn clone(&self) -> Self {
        Self::copy_from_slice_in(self, self.allocator().clone())
    }
}

impl<A: Allocator> Drop for BytesMut<A> {
    fn drop(&mut self) {
        match &self.backing {
            | Backing::Shared(data) => drop_shared::<A>(data.as_ptr().cast()),
            | Backing::Unique(allocator) if self.cap > 0 => {
                // SAFETY: a unique handle owns its byte allocation and exact A.
                unsafe {
                    allocator.deallocate(self.ptr, allocation_layout(self.cap));
                }
            },
            | _ => {},
        }
    }
}

// --- Trait Implementations ---

impl<A: Allocator> Buf for Bytes<A> {
    #[inline]
    fn remaining(&self) -> usize {
        self.len()
    }

    #[inline]
    fn chunk(&self) -> &[u8] {
        self.as_slice()
    }

    #[inline]
    fn advance(&mut self, cnt: usize) {
        assert!(cnt <= self.len(), "advance out of bounds: {} <= {}", cnt, self.len());
        // SAFETY: `cnt` was validated against the visible slice length.
        self.ptr = unsafe { self.ptr.add(cnt) };
        self.len -= cnt;
    }
}

impl<A: Allocator> Buf for BytesMut<A> {
    #[inline]
    fn remaining(&self) -> usize {
        self.len
    }

    #[inline]
    fn chunk(&self) -> &[u8] {
        self.as_ref()
    }

    #[inline]
    fn advance(&mut self, cnt: usize) {
        assert!(cnt <= self.len, "advance out of bounds: {} <= {}", cnt, self.len);
        if cnt == 0 {
            return;
        }

        self.promote();

        // SAFETY: `cnt <= len <= cap`, including a one-past pointer.
        self.ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(cnt)) };
        self.len -= cnt;
        self.cap -= cnt;
    }
}

// SAFETY: `chunk_mut` exposes only spare capacity, `advance_mut` validates its
// bound, and all initialized bytes remain live for the duration of the handle.
unsafe impl<A: Allocator> BufMut for BytesMut<A> {
    #[inline]
    fn remaining_mut(&self) -> usize {
        usize::MAX - self.len
    }

    #[inline]
    unsafe fn advance_mut(&mut self, cnt: usize) {
        assert!(cnt <= self.cap - self.len, "advance out of bounds");
        self.len += cnt;
    }

    #[inline]
    fn chunk_mut(&mut self) -> &mut bytes::buf::UninitSlice {
        if self.cap == self.len {
            self.reserve(64);
        }
        // SAFETY: the returned range is exactly the allocation's spare tail.
        unsafe {
            let ptr = self.ptr.as_ptr().add(self.len);
            let len = self.cap - self.len;
            bytes::buf::UninitSlice::from_raw_parts_mut(ptr, len)
        }
    }

    #[inline]
    fn put_slice(&mut self, src: &[u8]) {
        self.extend_from_slice(src);
    }
}

impl<A: Allocator> Deref for Bytes<A> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl<A: Allocator> AsRef<[u8]> for Bytes<A> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl<A: Allocator> Borrow<[u8]> for Bytes<A> {
    #[inline]
    fn borrow(&self) -> &[u8] {
        self.as_slice()
    }
}

impl<A: Allocator> Deref for BytesMut<A> {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        // SAFETY: `len` initialized bytes starting at `ptr` stay live while
        // this handle owns its allocation reference.
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<A: Allocator> DerefMut for BytesMut<A> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        // SAFETY: mutable handles either own the allocation uniquely or cover
        // a region disjoint from every other handle sharing it.
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl<A: Allocator> AsRef<[u8]> for BytesMut<A> {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl<A: Allocator> AsMut<[u8]> for BytesMut<A> {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self
    }
}

impl<A: Allocator> Borrow<[u8]> for BytesMut<A> {
    #[inline]
    fn borrow(&self) -> &[u8] {
        self
    }
}

impl<A: Allocator> BorrowMut<[u8]> for BytesMut<A> {
    #[inline]
    fn borrow_mut(&mut self) -> &mut [u8] {
        self
    }
}

impl Default for Bytes {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl Default for BytesMut {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl From<&'static [u8]> for Bytes {
    #[inline]
    fn from(slice: &'static [u8]) -> Self {
        Self::from_static(slice)
    }
}

impl From<&'static str> for Bytes {
    #[inline]
    fn from(s: &'static str) -> Self {
        Self::from_static(s.as_bytes())
    }
}

impl<A: Allocator> From<Vec<u8, A>> for Bytes<A> {
    fn from(vec: Vec<u8, A>) -> Self {
        BytesMut::from(vec).freeze()
    }
}

impl<A: Allocator> From<Vec<u8, A>> for BytesMut<A> {
    fn from(vec: Vec<u8, A>) -> Self {
        let (ptr, len, cap, allocator) = vec.into_raw_parts_with_allocator();
        Self {
            // SAFETY: Vec supplies a non-null pointer even for zero capacity.
            ptr: unsafe { NonNull::new_unchecked(ptr) },
            len,
            cap,
            backing: Backing::Unique(allocator),
        }
    }
}

impl<A: Allocator> From<Box<[u8], A>> for Bytes<A> {
    fn from(b: Box<[u8], A>) -> Self {
        Self::from(b.into_vec())
    }
}
impl<A: Allocator> From<Box<[u8], A>> for BytesMut<A> {
    fn from(b: Box<[u8], A>) -> Self {
        Self::from(b.into_vec())
    }
}

impl From<String> for Bytes {
    fn from(s: String) -> Self {
        s.into_bytes().into()
    }
}
impl From<String> for BytesMut {
    fn from(s: String) -> Self {
        s.into_bytes().into()
    }
}
impl From<&[u8]> for BytesMut {
    fn from(s: &[u8]) -> Self {
        Self::copy_from_slice_in(s, Global)
    }
}
impl From<&str> for BytesMut {
    fn from(s: &str) -> Self {
        Self::from(s.as_bytes())
    }
}

impl<A: Allocator + Clone> From<Bytes<A>> for BytesMut<A> {
    fn from(bytes: Bytes<A>) -> Self {
        match bytes.try_into_mut() {
            | Ok(bytes) => bytes,
            | Err(bytes) => Self::copy_from_slice_in(&bytes, bytes.allocator().clone()),
        }
    }
}
impl<A: Allocator> From<BytesMut<A>> for Bytes<A> {
    fn from(bytes: BytesMut<A>) -> Self {
        bytes.freeze()
    }
}
impl<A: Allocator + Clone> From<Bytes<A>> for Vec<u8, A> {
    fn from(bytes: Bytes<A>) -> Self {
        match bytes.try_into_vec() {
            | Ok(vec) => vec,
            | Err(bytes) => {
                let mut vec = Vec::with_capacity_in(bytes.len(), bytes.allocator().clone());
                vec.extend_from_slice(&bytes);
                vec
            },
        }
    }
}
impl<A: Allocator + Clone> From<BytesMut<A>> for Vec<u8, A> {
    fn from(bytes: BytesMut<A>) -> Self {
        match bytes.try_into_vec() {
            | Ok(vec) => vec,
            | Err(bytes) => {
                let mut vec = Vec::with_capacity_in(bytes.len(), bytes.allocator().clone());
                vec.extend_from_slice(&bytes);
                vec
            },
        }
    }
}
impl<A: Allocator + Clone> From<Bytes<A>> for Box<[u8], A> {
    fn from(bytes: Bytes<A>) -> Self {
        Vec::from(bytes).into_boxed_slice()
    }
}
impl<A: Allocator + Clone> From<BytesMut<A>> for Box<[u8], A> {
    fn from(bytes: BytesMut<A>) -> Self {
        Vec::from(bytes).into_boxed_slice()
    }
}

impl<A: Allocator> Extend<u8> for BytesMut<A> {
    #[inline]
    fn extend<T: IntoIterator<Item = u8>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for b in iter {
            self.put_u8(b);
        }
    }
}

impl<'a, A: Allocator> Extend<&'a u8> for BytesMut<A> {
    #[inline]
    fn extend<T: IntoIterator<Item = &'a u8>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for &b in iter {
            self.put_u8(b);
        }
    }
}

impl<A: Allocator + Default> FromIterator<u8> for BytesMut<A> {
    fn from_iter<T: IntoIterator<Item = u8>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut b = Self::with_capacity_in(lower, A::default());
        b.extend(iter);
        b
    }
}

impl<A: Allocator + Default> FromIterator<u8> for Bytes<A> {
    fn from_iter<T: IntoIterator<Item = u8>>(iter: T) -> Self {
        BytesMut::<A>::from_iter(iter).freeze()
    }
}

impl<'a, A: Allocator + Default> FromIterator<&'a u8> for Bytes<A> {
    fn from_iter<I: IntoIterator<Item = &'a u8>>(iter: I) -> Self {
        iter.into_iter().copied().collect::<BytesMut<A>>().freeze()
    }
}

impl<'a, A: Allocator + Default> FromIterator<&'a u8> for BytesMut<A> {
    fn from_iter<I: IntoIterator<Item = &'a u8>>(iter: I) -> Self {
        iter.into_iter().copied().collect()
    }
}

impl<A: Allocator> IntoIterator for Bytes<A> {
    type IntoIter = bytes::buf::IntoIter<Bytes<A>>;
    type Item = u8;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        bytes::buf::IntoIter::new(self)
    }
}

impl<'a, A: Allocator> IntoIterator for &'a Bytes<A> {
    type IntoIter = slice::Iter<'a, u8>;
    type Item = &'a u8;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl<A: Allocator> IntoIterator for BytesMut<A> {
    type IntoIter = bytes::buf::IntoIter<BytesMut<A>>;
    type Item = u8;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        bytes::buf::IntoIter::new(self)
    }
}

impl<'a, A: Allocator> IntoIterator for &'a BytesMut<A> {
    type IntoIter = slice::Iter<'a, u8>;
    type Item = &'a u8;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_ref().iter()
    }
}

impl<A: Allocator> fmt::Debug for Bytes<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "b\"")?;
        for &b in self.as_slice() {
            for c in ascii::escape_default(b) {
                write!(f, "{}", c as char)?;
            }
        }
        write!(f, "\"")
    }
}

impl<A: Allocator> fmt::Debug for BytesMut<A> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "b\"")?;
        for &b in self.as_ref() {
            for c in ascii::escape_default(b) {
                write!(f, "{}", c as char)?;
            }
        }
        write!(f, "\"")
    }
}

// Equality and comparison

impl<A: Allocator> PartialEq for Bytes<A> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<A: Allocator> Eq for Bytes<A> {}

impl<A: Allocator> PartialOrd for Bytes<A> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<A: Allocator> Ord for Bytes<A> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl<A: Allocator> Hash for Bytes<A> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl<A: Allocator> PartialEq<[u8]> for Bytes<A> {
    #[inline]
    fn eq(&self, other: &[u8]) -> bool {
        self.as_slice() == other
    }
}

impl<A: Allocator> PartialEq<&[u8]> for Bytes<A> {
    #[inline]
    fn eq(&self, other: &&[u8]) -> bool {
        self.as_slice() == *other
    }
}

impl<A: Allocator> PartialEq<Vec<u8>> for Bytes<A> {
    #[inline]
    fn eq(&self, other: &Vec<u8>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<A: Allocator> PartialEq<Bytes<A>> for [u8] {
    #[inline]
    fn eq(&self, other: &Bytes<A>) -> bool {
        self == other.as_slice()
    }
}

impl<A: Allocator> PartialEq<Bytes<A>> for &[u8] {
    #[inline]
    fn eq(&self, other: &Bytes<A>) -> bool {
        *self == other.as_slice()
    }
}

impl<A: Allocator> PartialEq<Bytes<A>> for Vec<u8> {
    #[inline]
    fn eq(&self, other: &Bytes<A>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl<A: Allocator> PartialEq for BytesMut<A> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl<A: Allocator> Eq for BytesMut<A> {}

impl<A: Allocator> PartialOrd for BytesMut<A> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl<A: Allocator> Ord for BytesMut<A> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_ref().cmp(other.as_ref())
    }
}

impl<A: Allocator> Hash for BytesMut<A> {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_ref().hash(state);
    }
}

impl<A: Allocator> PartialEq<Bytes<A>> for BytesMut<A> {
    #[inline]
    fn eq(&self, other: &Bytes<A>) -> bool {
        self.as_ref() == other.as_slice()
    }
}

impl<A: Allocator> PartialEq<BytesMut<A>> for Bytes<A> {
    #[inline]
    fn eq(&self, other: &BytesMut<A>) -> bool {
        self.as_slice() == other.as_ref()
    }
}

impl<A: Allocator> PartialEq<[u8]> for BytesMut<A> {
    #[inline]
    fn eq(&self, other: &[u8]) -> bool {
        self.as_ref() == other
    }
}

impl<A: Allocator> PartialEq<&[u8]> for BytesMut<A> {
    #[inline]
    fn eq(&self, other: &&[u8]) -> bool {
        self.as_ref() == *other
    }
}

impl<A: Allocator> PartialEq<Vec<u8>> for BytesMut<A> {
    #[inline]
    fn eq(&self, other: &Vec<u8>) -> bool {
        self.as_ref() == other.as_slice()
    }
}

impl<A: Allocator> PartialEq<BytesMut<A>> for [u8] {
    #[inline]
    fn eq(&self, other: &BytesMut<A>) -> bool {
        self == other.as_ref()
    }
}

impl<A: Allocator> PartialEq<BytesMut<A>> for &[u8] {
    #[inline]
    fn eq(&self, other: &BytesMut<A>) -> bool {
        *self == other.as_ref()
    }
}

impl<A: Allocator> PartialEq<BytesMut<A>> for Vec<u8> {
    #[inline]
    fn eq(&self, other: &BytesMut<A>) -> bool {
        self.as_slice() == other.as_ref()
    }
}

impl<A: Allocator> fmt::Write for BytesMut<A> {
    #[inline]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.put_slice(s.as_bytes());
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout() {
        use mem;
        assert_eq!(
            mem::size_of::<Bytes>(),
            mem::size_of::<usize>() * 4,
            "Bytes size should be 4 words",
        );
        assert_eq!(
            mem::size_of::<BytesMut>(),
            mem::size_of::<usize>() * 4,
            "BytesMut should be 4 words",
        );
    }

    #[test]
    fn bytes_mut_advance_remaining_capacity() {
        let max_capacity = 256;
        for capacity in 0 ..= max_capacity {
            for len in 0 ..= capacity {
                for advance in 0 ..= len {
                    let mut buf = BytesMut::with_capacity(capacity);
                    buf.resize(len, 42);
                    assert_eq!(buf.len(), len);
                    assert_eq!(buf.remaining(), len);
                    buf.advance(advance);
                    assert_eq!(buf.remaining(), len - advance);
                    assert_eq!(buf.capacity(), capacity - advance);
                }
            }
        }
    }

    #[test]
    fn bytes_into_vec() {
        let content = b"helloworld";
        let mut bytes = BytesMut::new();
        bytes.put_slice(content);
        let vec: Vec<u8> = bytes.into();
        assert_eq!(&vec, content);
    }

    #[test]
    fn freeze_clone_shared() {
        let s = &b"abcdefgh"[..];
        let b = BytesMut::from(s).split().freeze();
        assert_eq!(b, s);
        let c = b.clone();
        assert_eq!(c, s);
    }

    #[test]
    fn split_to_2() {
        let mut a = Bytes::from(b"mary had a little lamb, little lamb, little lamb".to_vec());
        let b = a.split_to(1);
        assert_eq!(b"ary had a little lamb, little lamb, little lamb"[..], a);
        drop(b);
    }

    #[test]
    fn bytesmut_from_bytes_promotable_even_arc_1() {
        let vec = vec![33u8; 1024];
        let b1 = Bytes::from(vec.clone());
        drop(b1.clone());
        let b1m = BytesMut::from(b1);
        assert_eq!(b1m, vec);
    }

    #[test]
    fn bytes_mut_unsplit_basic() {
        let mut buf = BytesMut::with_capacity(64);
        buf.extend_from_slice(b"aaabbbcccddd");
        let splitted = buf.split_off(6);
        assert_eq!(b"aaabbb", &buf[..]);
        assert_eq!(b"cccddd", &splitted[..]);
        buf.unsplit(splitted);
        assert_eq!(b"aaabbbcccddd", &buf[..]);
    }

    #[test]
    fn try_reclaim_vec() {
        let mut buf = BytesMut::with_capacity(6);
        buf.put_slice(b"abc");
        assert!(!buf.try_reclaim(usize::MAX));
        assert!(!buf.try_reclaim(6));
        buf.advance(2);
        assert_eq!(4, buf.capacity());
        assert!(!buf.try_reclaim(6));
        assert!(buf.try_reclaim(5));
        buf.advance(1);
        assert!(buf.try_reclaim(6));
        assert_eq!(6, buf.capacity());
    }

    #[test]
    fn try_into_mut_restores_capacity() {
        let mut bytes = BytesMut::with_capacity(100);
        bytes.put_slice(b"hello world");
        let frozen = bytes.freeze();

        let unfrozen = frozen.try_into_mut().unwrap();
        assert_eq!(unfrozen.capacity(), 100);
    }

    #[test]
    fn slice_ref() {
        let b = Bytes::from_static(b"hello world");
        let sub_slice = &b[6 ..];
        let sub_bytes = b.slice_ref(sub_slice);
        assert_eq!(sub_bytes.as_slice(), b"world");
    }

    #[derive(Clone)]
    struct SharedAtomicCounter(Rc<Cell<usize>>);

    use std::rc::Rc;

    impl SharedAtomicCounter {
        pub fn new() -> Self {
            SharedAtomicCounter(Rc::new(Cell::new(0)))
        }

        pub fn increment(&self) {
            self.0.set(self.0.get() + 1);
        }

        pub fn get(&self) -> usize {
            self.0.get()
        }
    }

    struct OwnedTester<const L: usize> {
        buf:        [u8; L],
        drop_count: SharedAtomicCounter,
    }

    impl<const L: usize> OwnedTester<L> {
        fn new(buf: [u8; L], drop_count: SharedAtomicCounter) -> Self {
            Self {
                buf,
                drop_count,
            }
        }
    }

    impl<const L: usize> AsRef<[u8]> for OwnedTester<L> {
        fn as_ref(&self) -> &[u8] {
            self.buf.as_slice()
        }
    }

    impl<const L: usize> Drop for OwnedTester<L> {
        fn drop(&mut self) {
            self.drop_count.increment();
        }
    }

    #[test]
    fn owned_dropped_exactly_once() {
        let buf: [u8; 5] = [1, 2, 3, 4, 5];
        let drop_counter = SharedAtomicCounter::new();
        let owner = OwnedTester::new(buf, drop_counter.clone());
        let b1 = Bytes::from_owner(owner);
        let b2 = b1.clone();
        assert_eq!(drop_counter.get(), 0);
        drop(b1);
        assert_eq!(drop_counter.get(), 0);
        let b3 = b2.slice(1 .. b2.len() - 1);
        drop(b2);
        assert_eq!(drop_counter.get(), 0);
        drop(b3);
        assert_eq!(drop_counter.get(), 1);
    }

    #[test]
    fn bytes_new_empty() {
        let b = Bytes::new();
        assert!(b.is_empty());
        assert_eq!(b.len(), 0);
        assert_eq!(b.as_slice(), b"");
    }

    #[test]
    fn bytes_from_static() {
        let b = Bytes::from_static(b"hello");
        assert_eq!(b.len(), 5);
        assert_eq!(b.as_slice(), b"hello");
        assert!(!b.is_unique());
    }

    #[test]
    fn bytes_copy_from_slice() {
        let original = b"world";
        let b = Bytes::copy_from_slice(original);
        assert_eq!(b.as_slice(), original);
        assert!(b.is_unique());
    }

    #[test]
    fn bytes_slice_various() {
        let b = Bytes::from_static(b"hello world");

        let sub1 = b.slice(0 .. 5);
        assert_eq!(sub1.as_slice(), b"hello");

        let sub2 = b.slice(6 ..);
        assert_eq!(sub2.as_slice(), b"world");

        let sub3 = b.slice(..);
        assert_eq!(sub3.as_slice(), b"hello world");

        let sub4 = b.slice(3 .. 3);
        assert!(sub4.is_empty());
    }

    #[test]
    #[should_panic(expected = "range start must not be greater than end")]
    fn bytes_slice_invalid_range() {
        let b = Bytes::from_static(b"hello");
        let start = 3;
        let end = 2;
        let _ = b.slice(start .. end);
    }

    #[test]
    #[should_panic(expected = "range end out of bounds")]
    fn bytes_slice_out_of_bounds() {
        let b = Bytes::from_static(b"hello");
        let _ = b.slice(0 .. 6);
    }

    #[test]
    fn bytes_split_off() {
        let mut b = Bytes::from_static(b"helloworld");
        let other = b.split_off(5);
        assert_eq!(b.as_slice(), b"hello");
        assert_eq!(other.as_slice(), b"world");
    }

    #[test]
    fn bytes_split_to() {
        let mut b = Bytes::from_static(b"helloworld");
        let other = b.split_to(5);
        assert_eq!(other.as_slice(), b"hello");
        assert_eq!(b.as_slice(), b"world");
    }

    #[test]
    fn bytes_truncate_clear() {
        let mut b = Bytes::from_static(b"hello");
        b.truncate(3);
        assert_eq!(b.as_slice(), b"hel");
        b.clear();
        assert!(b.is_empty());
    }

    #[test]
    fn bytes_mut_with_capacity_zero() {
        let b = BytesMut::with_capacity(0);
        assert_eq!(b.capacity(), 0);
        assert!(b.is_empty());
    }

    #[test]
    fn bytes_mut_reserve_and_put() {
        let mut b = BytesMut::new();
        assert_eq!(b.capacity(), 0);

        b.reserve(10);
        assert!(b.capacity() >= 10);

        b.put_u8(b'a');
        b.extend_from_slice(b"bc");
        assert_eq!(b.as_ref(), b"abc");
        assert_eq!(b.len(), 3);
    }

    #[test]
    fn bytes_mut_reserve_shared() {
        let mut b1 = BytesMut::with_capacity(10);
        b1.put_slice(b"hello");
        let b2 = b1.clone(); // triggers promotion/sharing

        b1.reserve(20); // should reallocate independently since it is shared
        b1.put_slice(b" world");

        assert_eq!(b1.as_ref(), b"hello world");
        assert_eq!(b2.as_ref(), b"hello");
    }

    #[test]
    fn bytes_mut_extend_from_within() {
        let mut b = BytesMut::with_capacity(20);
        b.put_slice(b"hello");
        b.extend_from_within(1 .. 4);
        assert_eq!(b.as_ref(), b"helloell");
    }

    #[test]
    fn buf_trait_impl() {
        let mut b = Bytes::copy_from_slice(b"abcdef");
        assert_eq!(b.remaining(), 6);
        assert_eq!(b.chunk(), b"abcdef");

        b.advance(2);
        assert_eq!(b.remaining(), 4);
        assert_eq!(b.chunk(), b"cdef");
    }

    #[test]
    fn buf_mut_trait_impl() {
        let mut b = BytesMut::with_capacity(10);
        assert_eq!(b.remaining_mut(), usize::MAX);

        b.put_slice(b"abc");
        assert_eq!(b.as_ref(), b"abc");

        // SAFETY: the test is deliberately marking two spare bytes initialized;
        // their values are not subsequently read.
        unsafe {
            b.advance_mut(2); // directly advance length
        }
        assert_eq!(b.len(), 5);
    }

    #[test]
    fn conversions_vec_and_box() {
        let original_vec = vec![1, 2, 3, 4, 5];

        // Vec -> Bytes -> Vec
        let b = Bytes::from(original_vec.clone());
        let roundtrip_vec: Vec<u8> = b.into();
        assert_eq!(roundtrip_vec, original_vec);

        // Box -> BytesMut -> Vec
        let original_box: Box<[u8]> = vec![6, 7, 8].into_boxed_slice();
        let bm = BytesMut::from(original_box);
        let roundtrip_vec2: Vec<u8> = bm.into();
        assert_eq!(roundtrip_vec2, vec![6, 7, 8]);
    }

    #[test]
    fn string_conversion() {
        let s = String::from("hello string");
        let b = Bytes::from(s.clone());
        assert_eq!(b.as_slice(), s.as_bytes());

        let bm = BytesMut::from(s.clone());
        assert_eq!(bm.as_ref(), s.as_bytes());
    }

    #[test]
    fn comparisons() {
        let b = Bytes::from_static(b"abc");
        let bm = BytesMut::from(b"abc" as &[u8]);

        assert_eq!(b, bm);
        assert_eq!(bm, b);

        assert_eq!(b, b"abc"[..]);
        assert_eq!(bm, b"abc"[..]);

        let vec = vec![97, 98, 99];
        assert_eq!(bm, vec);
    }

    #[test]
    fn fmt_write() {
        use std::fmt::Write;
        let mut b = BytesMut::with_capacity(20);
        write!(b, "hello {}", 42).unwrap();
        assert_eq!(b.as_ref(), b"hello 42");
    }

    #[test]
    fn from_iter_behavior() {
        let data = vec![1, 2, 3];
        let b: Bytes = data.clone().into_iter().collect();
        assert_eq!(b.as_slice(), &data[..]);

        let bm: BytesMut = data.clone().into_iter().collect();
        assert_eq!(bm.as_ref(), &data[..]);

        let from_refs: BytesMut = data.iter().collect();
        assert_eq!(from_refs.as_ref(), data.as_slice());
    }

    #[test]
    fn bytes_mut_order_hash_and_iteration_follow_slice() {
        use std::collections::hash_map::DefaultHasher;

        let first = BytesMut::from(&b"abc"[..]);
        let second = BytesMut::from(&b"abd"[..]);
        assert!(first < second);

        let mut buffer_hasher = DefaultHasher::new();
        first.hash(&mut buffer_hasher);
        let mut slice_hasher = DefaultHasher::new();
        b"abc".as_slice().hash(&mut slice_hasher);
        assert_eq!(buffer_hasher.finish(), slice_hasher.finish());

        assert_eq!((&first).into_iter().copied().collect::<Vec<_>>(), b"abc");
        assert_eq!(first.into_iter().collect::<Vec<_>>(), b"abc");
    }

    #[test]
    fn test_unsplit_empty_variants() {
        // Both empty
        let mut b1 = BytesMut::new();
        let b2 = BytesMut::new();
        b1.unsplit(b2);
        assert!(b1.is_empty());

        // Self empty, other non-empty
        let mut b1 = BytesMut::new();
        let mut b2 = BytesMut::with_capacity(10);
        b2.put_slice(b"hello");
        b1.unsplit(b2);
        assert_eq!(b1.as_ref(), b"hello");

        // Self non-empty, other empty
        let mut b1 = BytesMut::with_capacity(10);
        b1.put_slice(b"world");
        let b2 = BytesMut::new();
        b1.unsplit(b2);
        assert_eq!(b1.as_ref(), b"world");
    }

    #[test]
    fn test_split_boundary_indices() {
        let mut b = BytesMut::from(&b"hello"[..]);

        // split_to at 0
        let s0 = b.split_to(0);
        assert!(s0.is_empty());
        assert_eq!(b.as_ref(), b"hello");

        // split_to at len
        let s_len = b.split_to(b.len());
        assert_eq!(s_len.as_ref(), b"hello");
        assert!(b.is_empty());

        // reset
        let mut b = BytesMut::from(&b"world"[..]);

        // split_off at len
        let s_off_len = b.split_off(b.len());
        assert!(s_off_len.is_empty());
        assert_eq!(b.as_ref(), b"world");

        // split_off at 0
        let s_off_0 = b.split_off(0);
        assert_eq!(s_off_0.as_ref(), b"world");
        assert!(b.is_empty());
    }

    #[test]
    fn test_slice_ref_overlapping_and_empty() {
        let b = Bytes::from_static(b"hello world");

        // slice_ref with empty slice
        let empty_slice = &b[0 .. 0];
        let empty_bytes = b.slice_ref(empty_slice);
        assert!(empty_bytes.is_empty());

        // slice_ref on full slice
        let full_slice = &b[..];
        let full_bytes = b.slice_ref(full_slice);
        assert_eq!(full_bytes.as_slice(), b"hello world");
    }

    #[test]
    #[should_panic(expected = "subset is out of bounds or not part of this allocation")]
    fn test_slice_ref_non_overlapping_panic() {
        let b = Bytes::from_static(b"hello");
        let alien_slice = b"world" as &[u8];
        let _ = b.slice_ref(alien_slice);
    }

    #[test]
    fn test_bytes_mut_cow_behavior() {
        let mut b1 = BytesMut::with_capacity(10);
        b1.put_slice(b"abc");

        let mut b2 = b1.clone(); // both point to same shared data internally

        // Write to b1, triggering copy-on-write (detaching from b2)
        b1.put_slice(b"def");

        assert_eq!(b1.as_ref(), b"abcdef");
        assert_eq!(b2.as_ref(), b"abc");

        // Ensure further mutations on b2 are clean
        b2.put_slice(b"xyz");
        assert_eq!(b2.as_ref(), b"abcxyz");
    }

    #[test]
    fn test_from_vec_excess_capacity() {
        let mut vec = Vec::with_capacity(100);
        vec.extend_from_slice(b"short");

        let b = Bytes::from(vec);
        assert_eq!(b.as_slice(), b"short");

        let bm = BytesMut::from(b);
        assert_eq!(bm.as_ref(), b"short");
        assert_eq!(bm.capacity(), 100);
    }

    #[test]
    fn test_range_overflow_handling() {
        let b = Bytes::from_static(b"abc");

        // Bounded range checks using maximum bounds
        let s = b.slice(..);
        assert_eq!(s.as_slice(), b"abc");

        let s_from_max = b.slice(0 .. 3);
        assert_eq!(s_from_max.as_slice(), b"abc");

        assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| b.slice(..= usize::MAX))).is_err());
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                b.slice((Bound::Excluded(usize::MAX), Bound::Unbounded))
            }))
            .is_err()
        );

        let mut mutable = BytesMut::from(&b"abc"[..]);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                mutable.extend_from_within(..= usize::MAX);
            }))
            .is_err()
        );
        assert_eq!(mutable.as_ref(), b"abc");
    }

    #[test]
    fn test_zero_reserve_no_op() {
        let mut b = BytesMut::with_capacity(10);
        b.put_slice(b"abc");
        let initial_cap = b.capacity();

        b.reserve(0);
        assert_eq!(b.capacity(), initial_cap);
    }

    #[test]
    #[should_panic(expected = "advance out of bounds")]
    fn test_buf_mut_advance_past_capacity_panic() {
        let mut b = BytesMut::with_capacity(5);
        // SAFETY: this intentionally violates the precondition to verify the
        // runtime bounds assertion before any length change occurs.
        unsafe {
            b.advance_mut(10);
        }
    }

    #[test]
    fn from_owner_is_pinned_before_as_ref() {
        let drop_counter = SharedAtomicCounter::new();
        let bytes = Bytes::from_owner(OwnedTester::new([1, 2, 3, 4, 5], drop_counter));
        // SAFETY: `bytes` was constructed with this exact owner type, and the
        // test holds the owning handle for the duration of the borrow.
        let owner = unsafe { &*(bytes.data as *const OwnerData<OwnedTester<5>>) };

        assert_eq!(bytes.as_ptr(), owner.owner.buf.as_ptr());
        assert_eq!(bytes.as_slice(), &[1, 2, 3, 4, 5]);
    }

    #[test]
    fn from_owner_reuses_type_vtable() {
        let first = Bytes::from_owner(vec![1, 2, 3]);
        let second = Bytes::from_owner(vec![4, 5, 6]);

        assert!(ptr::eq(first.vtable, second.vtable));
    }

    #[test]
    fn from_owner_drops_owner_when_as_ref_panics() {
        struct PanickingOwner(SharedAtomicCounter);

        impl AsRef<[u8]> for PanickingOwner {
            fn as_ref(&self) -> &[u8] {
                panic!("AsRef panic");
            }
        }

        impl Drop for PanickingOwner {
            fn drop(&mut self) {
                self.0.increment();
            }
        }

        let drop_counter = SharedAtomicCounter::new();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe({
            let drop_counter = drop_counter.clone();
            move || {
                let _ = Bytes::from_owner(PanickingOwner(drop_counter));
            }
        }));

        assert!(result.is_err());
        assert_eq!(drop_counter.get(), 1);
    }

    #[test]
    fn shared_bytes_to_bytes_mut_makes_an_independent_copy() {
        let bytes = Bytes::from(vec![1, 2, 3, 4]);
        let clone = bytes.clone();
        let mut mutable = BytesMut::from(clone);

        mutable[0] = 9;

        assert_eq!(bytes.as_slice(), &[1, 2, 3, 4]);
        assert_eq!(mutable.as_ref(), &[9, 2, 3, 4]);
    }

    #[test]
    fn cloning_cannot_overflow_the_reference_count() {
        let bytes = BytesMut::from(&b"data"[..]).freeze();
        // SAFETY: this white-box test has a live shared control block and
        // restores its count before the owning handle is dropped.
        unsafe {
            (*bytes.data).ref_cnt.set((usize::MAX >> 1) + 1);
        }

        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| bytes.clone()));
        assert!(result.is_err());

        // SAFETY: see above; restoring the sole real reference reestablishes
        // the normal drop invariant.
        unsafe {
            (*bytes.data).ref_cnt.set(1);
        }
    }

    #[test]
    fn reserve_after_advance_accounts_for_allocation_offset() {
        let mut bytes = BytesMut::with_capacity(64);
        bytes.resize(64, 7);
        bytes.advance(60);

        bytes.reserve(1_000);
        assert!(bytes.capacity() - bytes.len() >= 1_000);
        bytes.extend_from_slice(&vec![8; 1_000]);

        assert_eq!(bytes.len(), 1_004);
        assert_eq!(&bytes[.. 4], &[7; 4]);
        assert!(bytes[4 ..].iter().all(|&byte| byte == 8));
    }

    #[test]
    fn reserving_a_promoted_empty_buffer_allocates_normally() {
        let frozen = BytesMut::new().freeze();
        let mut bytes = frozen.try_into_mut().expect("the empty allocation is unique");

        bytes.reserve(8);
        bytes.extend_from_slice(b"contents");

        assert_eq!(bytes.as_ref(), b"contents");
    }

    #[test]
    fn allocator_vec_round_trip_preserves_requested_capacity() {
        for capacity in [0, 1, 7, 64, 129] {
            let mut bytes = BytesMut::with_capacity(capacity);
            assert_eq!(bytes.capacity(), capacity);
            bytes.resize(capacity, 42);
            let original_ptr = bytes.as_ptr();

            let vec = Vec::from(bytes);
            assert_eq!(vec.capacity(), capacity);
            assert_eq!(vec.as_ptr(), original_ptr);
            assert_eq!(vec.as_slice(), vec![42; capacity]);

            let frozen = BytesMut::from(vec).freeze();
            let clone = frozen.clone();
            drop(frozen);
            let vec = Vec::from(clone);
            assert_eq!(vec.capacity(), capacity);
            assert_eq!(vec.as_ptr(), original_ptr);
            assert_eq!(vec.as_slice(), vec![42; capacity]);
        }
    }

    #[test]
    fn allocator_growth_accepts_vec_allocations_and_returns_them() {
        let mut vec = Vec::with_capacity(7);
        vec.extend_from_slice(b"initial");
        let original_ptr = vec.as_ptr();
        let mut bytes = BytesMut::from(vec);
        assert_eq!(bytes.as_ptr(), original_ptr);

        bytes.reserve(100);
        bytes.extend_from_slice(&[42; 100]);
        let grown_ptr = bytes.as_ptr();
        let grown_capacity = bytes.capacity();
        let mut vec = Vec::from(bytes.freeze());
        assert_eq!(vec.as_ptr(), grown_ptr);
        assert_eq!(vec.capacity(), grown_capacity);
        assert_eq!(&vec[.. 7], b"initial");
        assert_eq!(&vec[7 ..], &[42; 100]);

        vec.reserve(grown_capacity);
        vec.extend_from_slice(b"returned");
        let bytes = BytesMut::from(vec);
        assert_eq!(&bytes[.. 7], b"initial");
        assert_eq!(&bytes[7 .. 107], &[42; 100]);
        assert_eq!(&bytes[107 ..], b"returned");
    }

    #[test]
    fn allocator_growth_of_unique_offset_split_round_trips_to_vec() {
        let mut bytes = BytesMut::with_capacity(16);
        bytes.extend_from_slice(b"abcdefghijklmnop");
        drop(bytes.split_to(12));

        bytes.reserve(100);
        assert!(bytes.capacity() >= 104);
        bytes.extend_from_slice(&[42; 100]);

        let mut vec = Vec::from(bytes.freeze());
        assert_eq!(&vec[.. 4], b"mnop");
        assert_eq!(&vec[4 ..], &[42; 100]);
        vec.reserve(vec.capacity());
        vec.extend_from_slice(b"end");
        assert_eq!(&vec[104 ..], b"end");
    }

    #[test]
    fn allocator_growth_detaches_from_frozen_split() {
        let mut bytes = BytesMut::with_capacity(16);
        bytes.extend_from_slice(b"abcdefghijklmnop");
        let prefix = bytes.split_to(12).freeze();
        let prefix_clone = prefix.clone();

        bytes.reserve(100);
        bytes.extend_from_slice(&[42; 100]);
        assert_eq!(prefix.as_slice(), b"abcdefghijkl");
        drop(prefix);
        assert_eq!(prefix_clone.as_slice(), b"abcdefghijkl");
        drop(prefix_clone);

        let detached_ptr = bytes.as_ptr();
        let vec = Vec::from(bytes);
        assert_eq!(vec.as_ptr(), detached_ptr);
        assert_eq!(&vec[.. 4], b"mnop");
        assert_eq!(&vec[4 ..], &[42; 100]);
    }

    #[test]
    fn capacity_overflow_panics_without_modifying_buffer() {
        let mut bytes = BytesMut::from(&b"x"[..]);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            bytes.reserve(usize::MAX);
        }));

        assert!(result.is_err());
        assert_eq!(bytes.as_ref(), b"x");
        assert!(!bytes.try_reclaim(usize::MAX));
    }

    #[test]
    fn try_reclaim_succeeds_when_spare_capacity_is_already_available() {
        let mut bytes = BytesMut::with_capacity(16);
        bytes.extend_from_slice(b"data");

        assert!(bytes.try_reclaim(12));
        assert_eq!(bytes.as_ref(), b"data");
        assert_eq!(bytes.capacity(), 16);
    }

    #[test]
    fn try_reclaim_recovers_dropped_split_capacity() {
        let mut prefix = BytesMut::with_capacity(64);
        prefix.extend_from_slice(b"abcdefgh");
        let suffix = prefix.split_off(4);
        drop(suffix);

        assert_eq!(prefix.capacity(), 4);
        assert!(prefix.try_reclaim(60));
        assert_eq!(prefix.capacity(), 64);
        assert_eq!(prefix.as_ref(), b"abcd");
    }

    #[test]
    fn try_reclaim_moves_an_advanced_unique_split_to_the_allocation_base() {
        let mut prefix = BytesMut::with_capacity(64);
        prefix.extend_from_slice(b"abcdefgh");
        let mut suffix = prefix.split_off(4);
        drop(prefix);

        assert!(suffix.try_reclaim(60));
        assert_eq!(suffix.capacity(), 64);
        assert_eq!(suffix.as_ref(), b"efgh");
    }

    #[test]
    fn operation_sequence_matches_vec_model() {
        let mut seed = 0x4D59_5DF4_D0F3_3173_u64;
        let mut bytes = BytesMut::new();
        let mut model = Vec::new();

        for _ in 0 .. 10_000 {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            match (seed >> 32) % 8 {
                | 0 => {
                    let byte = seed as u8;
                    bytes.put_u8(byte);
                    model.push(byte);
                },
                | 1 => {
                    let data = seed.to_le_bytes();
                    let count = (seed as usize) % (data.len() + 1);
                    bytes.extend_from_slice(&data[.. count]);
                    model.extend_from_slice(&data[.. count]);
                },
                | 2 if !model.is_empty() => {
                    let new_len = (seed as usize) % (model.len() + 1);
                    bytes.truncate(new_len);
                    model.truncate(new_len);
                },
                | 3 if !model.is_empty() => {
                    let count = (seed as usize) % (model.len() + 1);
                    bytes.advance(count);
                    model.drain(.. count);
                },
                | 4 => {
                    let additional = (seed as usize) % 128;
                    bytes.reserve(additional);
                    assert!(bytes.capacity() - bytes.len() >= additional);
                },
                | 5 if !model.is_empty() => {
                    let start = (seed as usize) % model.len();
                    let end = start + ((seed >> 16) as usize) % (model.len() - start + 1);
                    bytes.extend_from_within(start .. end);
                    model.extend_from_within(start .. end);
                },
                | 6 => {
                    let at = if model.is_empty() {
                        0
                    } else {
                        (seed as usize) % (model.len() + 1)
                    };
                    let mut suffix = bytes.split_off(at);
                    assert_eq!(bytes.as_ref(), &model[.. at]);
                    assert_eq!(suffix.as_ref(), &model[at ..]);
                    bytes.unsplit(mem::take(&mut suffix));
                },
                | 7 => {
                    let frozen = mem::take(&mut bytes).freeze();
                    let snapshot = frozen.clone();
                    bytes = BytesMut::from(frozen);
                    assert_eq!(snapshot.as_slice(), model.as_slice());
                },
                | _ => {},
            }

            assert_eq!(bytes.as_ref(), model.as_slice());
        }

        let round_trip: Vec<u8> = bytes.into();
        assert_eq!(round_trip, model);
    }
}
