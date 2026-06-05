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

#![deny(missing_docs)]
#![allow(clippy::all)]

use core::{
    alloc::Layout,
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

#[doc(no_inline)]
pub use bytes::buf;
use bytes::{
    Buf,
    BufMut,
};

/// A cheap, cloneable, non-atomic byte array.
pub struct Bytes {
    ptr:    *const u8,
    len:    usize,
    data:   *mut SharedData,
    vtable: &'static Vtable,
}

/// A unique reference to a contiguous slice of memory.
pub struct BytesMut {
    ptr:  NonNull<u8>,
    len:  usize,
    cap:  usize,
    data: *mut SharedData,
}

struct SharedData {
    ref_cnt:   Cell<usize>,
    alloc_ptr: NonNull<u8>,
    alloc_cap: usize,
}

struct Vtable {
    clone: fn(&Bytes) -> Bytes,
    drop:  fn(&mut Bytes),
}

static SHARED_VTABLE: Vtable = Vtable {
    clone: |b| {
        if !b.data.is_null() {
            unsafe {
                let cnt = (*b.data).ref_cnt.get();
                (*b.data).ref_cnt.set(cnt + 1);
            }
        }
        Bytes {
            ptr:    b.ptr,
            len:    b.len,
            data:   b.data,
            vtable: b.vtable,
        }
    },
    drop:  |b| {
        if !b.data.is_null() {
            unsafe {
                let shared = b.data;
                let cnt = (*shared).ref_cnt.get();
                if cnt == 1 {
                    if (*shared).alloc_cap > 0 {
                        let layout = Layout::from_size_align_unchecked((*shared).alloc_cap, 1);
                        std::alloc::dealloc((*shared).alloc_ptr.as_ptr(), layout);
                    }
                    drop(Box::from_raw(shared));
                } else {
                    (*shared).ref_cnt.set(cnt - 1);
                }
            }
        }
    },
};

static STATIC_VTABLE: Vtable = Vtable {
    clone: |b| {
        Bytes {
            ptr:    b.ptr,
            len:    b.len,
            data:   ptr::null_mut(),
            vtable: &STATIC_VTABLE,
        }
    },
    drop:  |_| {},
};

struct OwnerData<T> {
    ref_cnt: Cell<usize>,
    #[allow(unused)]
    owner:   T,
}

fn drop_owner<T>(b: &mut Bytes) {
    if !b.data.is_null() {
        unsafe {
            let shared = b.data as *mut OwnerData<T>;
            let cnt = (*shared).ref_cnt.get();
            if cnt == 1 {
                drop(Box::from_raw(shared));
            } else {
                (*shared).ref_cnt.set(cnt - 1);
            }
        }
    }
}

fn clone_owner<T>(b: &Bytes) -> Bytes {
    if !b.data.is_null() {
        unsafe {
            let shared = b.data as *mut OwnerData<T>;
            let cnt = (*shared).ref_cnt.get();
            (*shared).ref_cnt.set(cnt + 1);
        }
    }
    Bytes {
        ptr:    b.ptr,
        len:    b.len,
        data:   b.data,
        vtable: b.vtable,
    }
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
    #[must_use]
    pub fn from_owner<T>(owner: T) -> Self
    where
        T: AsRef<[u8]> + 'static,
    {
        let slice = owner.as_ref();
        let ptr = slice.as_ptr();
        let len = slice.len();

        let data = Box::new(OwnerData {
            ref_cnt: Cell::new(1),
            owner:   owner,
        });
        let vtable = Box::leak(Box::new(Vtable {
            clone: clone_owner::<T>,
            drop:  drop_owner::<T>,
        }));

        Self {
            ptr,
            len,
            data: Box::into_raw(data) as *mut SharedData,
            vtable,
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

    /// Returns a slice of self for the provided range.
    #[must_use]
    pub fn slice(&self, range: impl RangeBounds<usize>) -> Self {
        let len = self.len();
        let start = match range.start_bound() {
            | Bound::Included(&n) => n,
            | Bound::Excluded(&n) => n.saturating_add(1),
            | Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            | Bound::Included(&n) => n.saturating_add(1),
            | Bound::Excluded(&n) => n,
            | Bound::Unbounded => len,
        };

        assert!(start <= end, "range start must not be greater than end");
        assert!(end <= len, "range end out of bounds");

        if start == end {
            return Self::new();
        }

        let mut ret = self.clone();
        ret.ptr = unsafe { ret.ptr.add(start) };
        ret.len = end - start;
        ret
    }

    /// Creates a new `Bytes` from a subslice of this `Bytes`.
    #[must_use]
    pub fn slice_ref(&self, subset: &[u8]) -> Self {
        if subset.is_empty() {
            return Self::new();
        }

        let self_ptr = self.as_ptr() as usize;
        let self_len = self.len();
        let subset_ptr = subset.as_ptr() as usize;
        let subset_len = subset.len();

        assert!(
            subset_ptr >= self_ptr && subset_ptr + subset_len <= self_ptr + self_len,
            "subset is out of bounds or not part of this allocation"
        );

        let offset = subset_ptr - self_ptr;
        self.slice(offset .. (offset + subset_len))
    }

    /// Splits the buffer into two at the given index.
    #[must_use = "consider Bytes::truncate if you don't need the other half"]
    pub fn split_off(&mut self, at: usize) -> Self {
        assert!(at <= self.len(), "split_off out of bounds: {} <= {}", at, self.len());
        if at == self.len() {
            return Self::new();
        }
        if at == 0 {
            let ret = self.clone();
            self.clear();
            return ret;
        }

        let mut ret = self.clone();
        ret.ptr = unsafe { ret.ptr.add(at) };
        ret.len = self.len - at;
        self.len = at;
        ret
    }

    /// Splits the buffer into two at the given index.
    #[must_use = "consider Bytes::advance if you don't need the other half"]
    pub fn split_to(&mut self, at: usize) -> Self {
        assert!(at <= self.len(), "split_to out of bounds: {} <= {}", at, self.len());
        if at == 0 {
            return Self::new();
        }
        if at == self.len() {
            let ret = self.clone();
            self.clear();
            return ret;
        }

        let mut ret = self.clone();
        ret.len = at;
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
        unsafe { slice::from_raw_parts(self.ptr, self.len) }
    }

    /// Returns true if this is the only reference to the underlying data.
    #[inline]
    #[must_use]
    pub fn is_unique(&self) -> bool {
        if ptr::eq(self.vtable, &SHARED_VTABLE) && !self.data.is_null() {
            unsafe { (*self.data).ref_cnt.get() == 1 }
        } else {
            false
        }
    }

    /// Tries to convert this `Bytes` to a `BytesMut`.
    pub fn try_into_mut(self) -> Result<BytesMut, Bytes> {
        if ptr::eq(self.vtable, &SHARED_VTABLE) && !self.data.is_null() {
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
                        data,
                    });
                }
            }
        }
        Err(self)
    }
}

impl Clone for Bytes {
    #[inline]
    fn clone(&self) -> Self {
        (self.vtable.clone)(self)
    }
}

impl Drop for Bytes {
    #[inline]
    fn drop(&mut self) {
        (self.vtable.drop)(self)
    }
}

// --- BytesMut ---

impl BytesMut {
    /// Creates a new `BytesMut` with the specified capacity.
    #[inline]
    #[must_use]
    pub fn with_capacity(capacity: usize) -> Self {
        if capacity == 0 {
            return Self::new();
        }
        let layout = Layout::from_size_align(capacity, 1).unwrap();
        let ptr = unsafe { std::alloc::alloc(layout) };
        let ptr = NonNull::new(ptr).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
        BytesMut {
            ptr,
            len: 0,
            cap: capacity,
            data: ptr::null_mut(),
        }
    }

    /// Creates a new empty `BytesMut` instance.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        BytesMut {
            ptr:  NonNull::dangling(),
            len:  0,
            cap:  0,
            data: ptr::null_mut(),
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

    /// Returns the remaining capacity in the buffer.
    #[inline]
    #[must_use]
    pub const fn capacity(&self) -> usize {
        self.cap
    }

    /// Promotes the unique allocation to a shared state if needed.
    #[inline]
    fn promote(&mut self) {
        if self.data.is_null() {
            let shared = Box::new(SharedData {
                ref_cnt:   Cell::new(1),
                alloc_ptr: self.ptr,
                alloc_cap: self.cap,
            });
            self.data = Box::into_raw(shared);
        }
    }

    /// Reserves capacity for at least `additional` more bytes to be inserted.
    pub fn reserve(&mut self, additional: usize) {
        if self.cap >= self.len + additional {
            return;
        }

        let alloc_cap = if self.data.is_null() {
            self.cap
        } else {
            unsafe { (*self.data).alloc_cap }
        };

        let new_cap = (alloc_cap * 2).max(self.len + additional).max(64);

        if self.data.is_null() {
            if self.cap == 0 {
                let layout = Layout::from_size_align(new_cap, 1).unwrap();
                let new_ptr = unsafe { std::alloc::alloc(layout) };
                self.ptr = NonNull::new(new_ptr).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));
                self.cap = new_cap;
                return;
            }

            let new_ptr = unsafe {
                let layout = Layout::from_size_align_unchecked(self.cap, 1);
                let ptr = std::alloc::realloc(self.ptr.as_ptr(), layout, new_cap);
                NonNull::new(ptr)
                    .unwrap_or_else(|| std::alloc::handle_alloc_error(Layout::from_size_align_unchecked(new_cap, 1)))
            };
            self.ptr = new_ptr;
            self.cap = new_cap;
        } else {
            let shared = unsafe { &*self.data };
            if shared.ref_cnt.get() == 1 {
                let offset = self.ptr.as_ptr() as usize - shared.alloc_ptr.as_ptr() as usize;

                if offset > 0 && shared.alloc_cap >= self.len + additional {
                    unsafe {
                        ptr::copy(self.ptr.as_ptr(), shared.alloc_ptr.as_ptr(), self.len);
                    }
                    self.ptr = shared.alloc_ptr;
                    self.cap = shared.alloc_cap;
                    return;
                }

                let new_alloc_ptr = unsafe {
                    let layout = Layout::from_size_align_unchecked(shared.alloc_cap, 1);
                    let ptr = std::alloc::realloc(shared.alloc_ptr.as_ptr(), layout, new_cap);
                    NonNull::new(ptr).unwrap_or_else(|| {
                        std::alloc::handle_alloc_error(Layout::from_size_align_unchecked(new_cap, 1))
                    })
                };

                self.ptr = unsafe { NonNull::new_unchecked(new_alloc_ptr.as_ptr().add(offset)) };
                self.cap = new_cap - offset;

                unsafe {
                    (*self.data).alloc_ptr = new_alloc_ptr;
                    (*self.data).alloc_cap = new_cap;
                }
            } else {
                let layout = Layout::from_size_align(new_cap, 1).unwrap();
                let new_alloc_ptr = unsafe { std::alloc::alloc(layout) };
                let new_alloc_ptr =
                    NonNull::new(new_alloc_ptr).unwrap_or_else(|| std::alloc::handle_alloc_error(layout));

                unsafe {
                    ptr::copy_nonoverlapping(self.ptr.as_ptr(), new_alloc_ptr.as_ptr(), self.len);
                }

                let cnt = shared.ref_cnt.get();
                shared.ref_cnt.set(cnt - 1);

                self.ptr = new_alloc_ptr;
                self.cap = new_cap;
                self.data = ptr::null_mut();
            }
        }
    }

    /// Try to reclaim capacity without reallocating
    pub fn try_reclaim(&mut self, additional: usize) -> bool {
        if self.data.is_null() {
            // Cannot easily reclaim without `SharedData` because we don't know original ptr
            false
        } else {
            unsafe {
                let shared = &*self.data;
                if shared.ref_cnt.get() == 1 {
                    let offset = self.ptr.as_ptr() as usize - shared.alloc_ptr.as_ptr() as usize;
                    if offset > 0 && shared.alloc_cap >= self.len + additional {
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
        unsafe {
            ptr::copy_nonoverlapping(extend.as_ptr(), self.ptr.as_ptr().add(self.len), extend.len());
        }
        self.len += extend.len();
    }

    /// Extends the buffer from within itself.
    pub fn extend_from_within<R>(&mut self, range: R)
    where
        R: RangeBounds<usize>,
    {
        let len = self.len;
        let start = match range.start_bound() {
            | Bound::Included(&n) => n,
            | Bound::Excluded(&n) => n + 1,
            | Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            | Bound::Included(&n) => n + 1,
            | Bound::Excluded(&n) => n,
            | Bound::Unbounded => len,
        };
        assert!(start <= end && end <= len, "range out of bounds");

        let cnt = end - start;
        self.reserve(cnt);
        unsafe {
            ptr::copy_nonoverlapping(self.ptr.as_ptr().add(start), self.ptr.as_ptr().add(self.len), cnt);
        }
        self.len += cnt;
    }

    /// Puts a single byte at the end of the buffer.
    #[inline]
    pub fn put_u8(&mut self, n: u8) {
        self.reserve(1);
        unsafe {
            self.ptr.as_ptr().add(self.len).write(n);
        }
        self.len += 1;
    }

    /// Splits the buffer into two at the given index.
    pub fn split_off(&mut self, at: usize) -> BytesMut {
        assert!(at <= self.len, "split_off out of bounds: {} <= {}", at, self.len);

        self.promote();
        unsafe {
            let cnt = (*self.data).ref_cnt.get();
            (*self.data).ref_cnt.set(cnt + 1);
        }

        let new_ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(at)) };
        let new_len = self.len - at;
        let new_cap = self.cap - at;

        self.len = at;
        self.cap = at;

        BytesMut {
            ptr:  new_ptr,
            len:  new_len,
            cap:  new_cap,
            data: self.data,
        }
    }

    /// Splits the buffer into two at the given index.
    pub fn split_to(&mut self, at: usize) -> BytesMut {
        assert!(at <= self.len, "split_to out of bounds: {} <= {}", at, self.len);

        self.promote();
        unsafe {
            let cnt = (*self.data).ref_cnt.get();
            (*self.data).ref_cnt.set(cnt + 1);
        }

        let new_ptr = self.ptr;
        let new_len = at;
        let new_cap = at;

        self.ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(at)) };
        self.len -= at;
        self.cap -= at;

        BytesMut {
            ptr:  new_ptr,
            len:  new_len,
            cap:  new_cap,
            data: self.data,
        }
    }

    /// Splits the buffer into two at the current length.
    pub fn split(&mut self) -> BytesMut {
        let len = self.len;
        self.split_to(len)
    }

    /// Unsplits the buffer.
    pub fn unsplit(&mut self, mut other: BytesMut) {
        if self.is_empty() {
            *self = other;
            return;
        }
        if other.is_empty() {
            return;
        }

        let contiguous = unsafe { self.ptr.as_ptr().add(self.len) == other.ptr.as_ptr() };
        let same_alloc = !self.data.is_null() && !other.data.is_null() && ptr::eq(self.data, other.data);

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
    pub fn freeze(mut self) -> Bytes {
        self.promote();
        let data = self.data;
        let ptr = self.ptr.as_ptr();
        let len = self.len;

        // Forget self to prevent Drop
        mem::forget(self);

        Bytes {
            ptr,
            len,
            data,
            vtable: &SHARED_VTABLE,
        }
    }

    /// Resizes the buffer.
    pub fn resize(&mut self, new_len: usize, value: u8) {
        if new_len > self.len {
            let additional = new_len - self.len;
            self.reserve(additional);
            unsafe {
                ptr::write_bytes(self.ptr.as_ptr().add(self.len), value, additional);
            }
        }
        self.len = new_len;
    }
}

impl Clone for BytesMut {
    fn clone(&self) -> Self {
        let mut b = BytesMut::with_capacity(self.len);
        b.extend_from_slice(self);
        b
    }
}

impl Drop for BytesMut {
    fn drop(&mut self) {
        if !self.data.is_null() {
            unsafe {
                let shared = self.data;
                let cnt = (*shared).ref_cnt.get();
                if cnt == 1 {
                    if (*shared).alloc_cap > 0 {
                        let layout = Layout::from_size_align_unchecked((*shared).alloc_cap, 1);
                        std::alloc::dealloc((*shared).alloc_ptr.as_ptr(), layout);
                    }
                    drop(Box::from_raw(shared));
                } else {
                    (*shared).ref_cnt.set(cnt - 1);
                }
            }
        } else if self.cap > 0 {
            unsafe {
                let layout = Layout::from_size_align_unchecked(self.cap, 1);
                std::alloc::dealloc(self.ptr.as_ptr(), layout);
            }
        }
    }
}

// --- Trait Implementations ---

impl Buf for Bytes {
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
        self.ptr = unsafe { self.ptr.add(cnt) };
        self.len -= cnt;
    }
}

impl Buf for BytesMut {
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

        if self.data.is_null() {
            let shared = Box::new(SharedData {
                ref_cnt:   Cell::new(1),
                alloc_ptr: self.ptr,
                alloc_cap: self.cap,
            });
            self.data = Box::into_raw(shared);
        }

        self.ptr = unsafe { NonNull::new_unchecked(self.ptr.as_ptr().add(cnt)) };
        self.len -= cnt;
        self.cap -= cnt;
    }
}

unsafe impl BufMut for BytesMut {
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

impl Deref for Bytes {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl AsRef<[u8]> for Bytes {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self.as_slice()
    }
}

impl Borrow<[u8]> for Bytes {
    #[inline]
    fn borrow(&self) -> &[u8] {
        self.as_slice()
    }
}

impl Deref for BytesMut {
    type Target = [u8];

    #[inline]
    fn deref(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl DerefMut for BytesMut {
    #[inline]
    fn deref_mut(&mut self) -> &mut [u8] {
        unsafe { slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl AsRef<[u8]> for BytesMut {
    #[inline]
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl AsMut<[u8]> for BytesMut {
    #[inline]
    fn as_mut(&mut self) -> &mut [u8] {
        self
    }
}

impl Borrow<[u8]> for BytesMut {
    #[inline]
    fn borrow(&self) -> &[u8] {
        self
    }
}

impl BorrowMut<[u8]> for BytesMut {
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

impl From<Vec<u8>> for Bytes {
    #[inline]
    fn from(mut vec: Vec<u8>) -> Self {
        if vec.is_empty() {
            return Self::new();
        }
        let ptr = vec.as_mut_ptr();
        let len = vec.len();
        let cap = vec.capacity();
        let alloc_ptr = NonNull::new(ptr).unwrap();
        mem::forget(vec);

        let shared = Box::new(SharedData {
            ref_cnt: Cell::new(1),
            alloc_ptr,
            alloc_cap: cap,
        });

        Bytes {
            ptr,
            len,
            data: Box::into_raw(shared),
            vtable: &SHARED_VTABLE,
        }
    }
}

impl From<Box<[u8]>> for Bytes {
    #[inline]
    fn from(b: Box<[u8]>) -> Self {
        Self::from(b.into_vec())
    }
}

impl From<String> for Bytes {
    #[inline]
    fn from(s: String) -> Self {
        s.into_bytes().into()
    }
}

impl From<Vec<u8>> for BytesMut {
    #[inline]
    fn from(mut vec: Vec<u8>) -> Self {
        if vec.capacity() == 0 {
            return Self::new();
        }
        let ptr = vec.as_mut_ptr();
        let len = vec.len();
        let cap = vec.capacity();
        let alloc_ptr = NonNull::new(ptr).unwrap();
        mem::forget(vec);

        BytesMut {
            ptr: alloc_ptr,
            len,
            cap,
            data: ptr::null_mut(),
        }
    }
}

impl From<Box<[u8]>> for BytesMut {
    #[inline]
    fn from(b: Box<[u8]>) -> Self {
        Self::from(b.into_vec())
    }
}

impl From<String> for BytesMut {
    #[inline]
    fn from(s: String) -> Self {
        Self::from(s.into_bytes())
    }
}

impl From<&[u8]> for BytesMut {
    #[inline]
    fn from(s: &[u8]) -> Self {
        let mut b = BytesMut::with_capacity(s.len());
        b.put_slice(s);
        b
    }
}

impl From<&str> for BytesMut {
    #[inline]
    fn from(s: &str) -> Self {
        Self::from(s.as_bytes())
    }
}

impl From<Bytes> for BytesMut {
    fn from(bytes: Bytes) -> Self {
        if ptr::eq(bytes.vtable, &SHARED_VTABLE) && !bytes.data.is_null() {
            let data = bytes.data;
            let ptr = unsafe { NonNull::new_unchecked(bytes.ptr as *mut u8) };
            let len = bytes.len;
            mem::forget(bytes);
            BytesMut {
                ptr,
                len,
                cap: len,
                data,
            }
        } else {
            let mut b = BytesMut::with_capacity(bytes.len());
            b.put_slice(&bytes);
            b
        }
    }
}

impl From<BytesMut> for Bytes {
    #[inline]
    fn from(bm: BytesMut) -> Self {
        bm.freeze()
    }
}

impl From<Bytes> for Vec<u8> {
    fn from(mut bytes: Bytes) -> Self {
        let len = bytes.len;
        let ptr = bytes.ptr;

        if ptr::eq(bytes.vtable, &SHARED_VTABLE) && !bytes.data.is_null() {
            unsafe {
                if (*bytes.data).ref_cnt.get() == 1 {
                    let shared = Box::from_raw(bytes.data);
                    let alloc_ptr = shared.alloc_ptr.as_ptr();
                    let alloc_cap = shared.alloc_cap;

                    bytes.data = ptr::null_mut(); // prevent drop from doing anything

                    let offset = ptr as usize - alloc_ptr as usize;
                    if offset > 0 && len > 0 {
                        ptr::copy(ptr, alloc_ptr, len);
                    }
                    return Vec::from_raw_parts(alloc_ptr, len, alloc_cap);
                }
            }
        }

        let mut vec = Vec::with_capacity(len);
        unsafe {
            ptr::copy_nonoverlapping(ptr, vec.as_mut_ptr(), len);
            vec.set_len(len);
        }
        vec
    }
}

impl From<BytesMut> for Vec<u8> {
    fn from(mut bytes: BytesMut) -> Self {
        let len = bytes.len;
        let ptr = bytes.ptr.as_ptr();

        if bytes.data.is_null() {
            let cap = bytes.cap;
            bytes.cap = 0; // prevent drop
            unsafe {
                return Vec::from_raw_parts(ptr, len, cap);
            }
        } else {
            unsafe {
                let shared = bytes.data;
                if (*shared).ref_cnt.get() == 1 {
                    let shared = Box::from_raw(shared);
                    let alloc_ptr = shared.alloc_ptr.as_ptr();
                    let alloc_cap = shared.alloc_cap;

                    bytes.data = ptr::null_mut(); // prevent drop
                    bytes.cap = 0;

                    let offset = ptr as usize - alloc_ptr as usize;
                    if offset > 0 && len > 0 {
                        ptr::copy(ptr, alloc_ptr, len);
                    }
                    return Vec::from_raw_parts(alloc_ptr, len, alloc_cap);
                }
            }
        }

        let mut vec = Vec::with_capacity(len);
        unsafe {
            ptr::copy_nonoverlapping(ptr, vec.as_mut_ptr(), len);
            vec.set_len(len);
        }
        vec
    }
}

impl Extend<u8> for BytesMut {
    #[inline]
    fn extend<T: IntoIterator<Item = u8>>(&mut self, iter: T) {
        for b in iter {
            self.put_u8(b);
        }
    }
}

impl<'a> Extend<&'a u8> for BytesMut {
    #[inline]
    fn extend<T: IntoIterator<Item = &'a u8>>(&mut self, iter: T) {
        for &b in iter {
            self.put_u8(b);
        }
    }
}

impl FromIterator<u8> for BytesMut {
    fn from_iter<T: IntoIterator<Item = u8>>(iter: T) -> Self {
        let iter = iter.into_iter();
        let (lower, _) = iter.size_hint();
        let mut b = BytesMut::with_capacity(lower);
        b.extend(iter);
        b
    }
}

impl FromIterator<u8> for Bytes {
    fn from_iter<T: IntoIterator<Item = u8>>(iter: T) -> Self {
        BytesMut::from_iter(iter).freeze()
    }
}

impl<'a> FromIterator<&'a u8> for Bytes {
    fn from_iter<I: IntoIterator<Item = &'a u8>>(iter: I) -> Self {
        iter.into_iter().copied().collect::<BytesMut>().freeze()
    }
}

impl IntoIterator for Bytes {
    type IntoIter = bytes::buf::IntoIter<Bytes>;
    type Item = u8;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        bytes::buf::IntoIter::new(self)
    }
}

impl<'a> IntoIterator for &'a Bytes {
    type IntoIter = slice::Iter<'a, u8>;
    type Item = &'a u8;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

impl fmt::Debug for Bytes {
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

impl fmt::Debug for BytesMut {
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

impl PartialEq for Bytes {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl Eq for Bytes {}

impl PartialOrd for Bytes {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Bytes {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.as_slice().cmp(other.as_slice())
    }
}

impl Hash for Bytes {
    #[inline]
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.as_slice().hash(state);
    }
}

impl PartialEq<[u8]> for Bytes {
    #[inline]
    fn eq(&self, other: &[u8]) -> bool {
        self.as_slice() == other
    }
}

impl PartialEq<&[u8]> for Bytes {
    #[inline]
    fn eq(&self, other: &&[u8]) -> bool {
        self.as_slice() == *other
    }
}

impl PartialEq<Vec<u8>> for Bytes {
    #[inline]
    fn eq(&self, other: &Vec<u8>) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl PartialEq<Bytes> for [u8] {
    #[inline]
    fn eq(&self, other: &Bytes) -> bool {
        self == other.as_slice()
    }
}

impl PartialEq<Bytes> for &[u8] {
    #[inline]
    fn eq(&self, other: &Bytes) -> bool {
        *self == other.as_slice()
    }
}

impl PartialEq<Bytes> for Vec<u8> {
    #[inline]
    fn eq(&self, other: &Bytes) -> bool {
        self.as_slice() == other.as_slice()
    }
}

impl PartialEq for BytesMut {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.as_ref() == other.as_ref()
    }
}

impl PartialEq<Bytes> for BytesMut {
    #[inline]
    fn eq(&self, other: &Bytes) -> bool {
        self.as_ref() == other.as_slice()
    }
}

impl PartialEq<BytesMut> for Bytes {
    #[inline]
    fn eq(&self, other: &BytesMut) -> bool {
        self.as_slice() == other.as_ref()
    }
}

impl PartialEq<[u8]> for BytesMut {
    #[inline]
    fn eq(&self, other: &[u8]) -> bool {
        self.as_ref() == other
    }
}

impl PartialEq<&[u8]> for BytesMut {
    #[inline]
    fn eq(&self, other: &&[u8]) -> bool {
        self.as_ref() == *other
    }
}

impl PartialEq<Vec<u8>> for BytesMut {
    #[inline]
    fn eq(&self, other: &Vec<u8>) -> bool {
        self.as_ref() == other.as_slice()
    }
}

impl PartialEq<BytesMut> for [u8] {
    #[inline]
    fn eq(&self, other: &BytesMut) -> bool {
        self == other.as_ref()
    }
}

impl PartialEq<BytesMut> for &[u8] {
    #[inline]
    fn eq(&self, other: &BytesMut) -> bool {
        *self == other.as_ref()
    }
}

impl PartialEq<BytesMut> for Vec<u8> {
    #[inline]
    fn eq(&self, other: &BytesMut) -> bool {
        self.as_slice() == other.as_ref()
    }
}

impl fmt::Write for BytesMut {
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
        assert_eq!(false, buf.try_reclaim(usize::MAX));
        assert_eq!(false, buf.try_reclaim(6));
        buf.advance(2);
        assert_eq!(4, buf.capacity());
        assert_eq!(false, buf.try_reclaim(6));
        assert_eq!(true, buf.try_reclaim(5));
        buf.advance(1);
        assert_eq!(true, buf.try_reclaim(6));
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
        let _ = b.slice(3 .. 2);
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
        // Capacity of reconstructed unfrozen mut bytes starts as the length of the
        // slice
        assert_eq!(bm.capacity(), 5);
    }

    #[test]
    fn test_range_overflow_handling() {
        let b = Bytes::from_static(b"abc");

        // Bounded range checks using maximum bounds
        let s = b.slice(..);
        assert_eq!(s.as_slice(), b"abc");

        let s_from_max = b.slice(0 .. 3);
        assert_eq!(s_from_max.as_slice(), b"abc");
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
        unsafe {
            b.advance_mut(10);
        }
    }
}
