//! Rc version of `bytes::Bytes`.
//!
//! This crate provides a non-atomic `Rc`-based alternative to the `Bytes` type
//! from the `bytes` crate. This is useful in single-threaded contexts where
//! the overhead of atomic reference counting (`Arc`) is unnecessary.
//!
//! The [`Bytes`] type provided here implements [`bytes::Buf`], allowing it to
//! be seamlessly integrated with ecosystems that expect buffer types.

#![deny(missing_docs)]

use std::{
    borrow::Borrow,
    cmp::Ordering,
    fmt,
    hash::{
        Hash,
        Hasher,
    },
    ops::{
        Bound,
        Deref,
        RangeBounds,
    },
    rc::Rc,
};

use bytes::Buf;
#[doc(no_inline)]
pub use bytes::buf;

/// A cheap, cloneable, non-atomic byte array.
#[derive(Clone)]
pub struct Bytes {
    inner: Inner,
    start: usize,
    end:   usize,
}

#[derive(Clone)]
enum Inner {
    Static(&'static [u8]),
    Rc(Rc<[u8]>),
}

impl Bytes {
    /// Creates a new empty `Bytes` instance.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            inner: Inner::Static(b""),
            start: 0,
            end:   0,
        }
    }

    /// Creates a new `Bytes` from a static slice.
    #[inline]
    #[must_use]
    pub const fn from_static(bytes: &'static [u8]) -> Self {
        Self {
            inner: Inner::Static(bytes),
            start: 0,
            end:   bytes.len(),
        }
    }

    /// Creates a new `Bytes` by copying from a slice.
    #[inline]
    #[must_use]
    pub fn copy_from_slice(data: &[u8]) -> Self {
        if data.is_empty() {
            Self::new()
        } else {
            let rc: Rc<[u8]> = Rc::from(data);
            Self {
                inner: Inner::Rc(rc),
                start: 0,
                end:   data.len(),
            }
        }
    }

    /// Returns the number of bytes contained in this `Bytes`.
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.end - self.start
    }

    /// Returns true if the `Bytes` has a length of 0.
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns a pointer to the start of the data.
    #[inline]
    #[must_use]
    pub fn as_ptr(&self) -> *const u8 {
        self.as_slice().as_ptr()
    }

    /// Returns true if this is the only reference to the underlying data.
    ///
    /// If this `Bytes` is backed by a static slice, this method will always
    /// return `false`.
    #[inline]
    #[must_use]
    pub fn is_unique(&self) -> bool {
        match &self.inner {
            | Inner::Static(_) => false,
            | Inner::Rc(rc) => Rc::strong_count(rc) == 1 && Rc::weak_count(rc) == 0,
        }
    }

    /// Returns a slice of self for the provided range.
    ///
    /// # Panics
    ///
    /// Panics if the start of the range is greater than the end, or if the
    /// end of the range is out of bounds.
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

        assert!(
            start <= end,
            "range start must not be greater than end: {} <= {}",
            start,
            end
        );
        assert!(end <= len, "range end out of bounds: {} <= {}", end, len);

        if start == end {
            return Self::new();
        }

        Self {
            inner: self.inner.clone(),
            start: self.start + start,
            end:   self.start + end,
        }
    }

    /// Creates a new `Bytes` from a subslice of this `Bytes`.
    ///
    /// # Panics
    ///
    /// Panics if the provided slice is not a subslice of this `Bytes` (e.g.,
    /// it points to a different allocation or is out of bounds).
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
    ///
    /// Afterwards `self` contains elements `[0, at)`, and the returned `Bytes`
    /// contains elements `[at, len)`.
    ///
    /// # Panics
    ///
    /// Panics if `at > len`.
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

        let ret = Self {
            inner: self.inner.clone(),
            start: self.start + at,
            end:   self.end,
        };
        self.end = self.start + at;
        ret
    }

    /// Splits the buffer into two at the given index.
    ///
    /// Afterwards `self` contains elements `[at, len)`, and the returned
    /// `Bytes` contains elements `[0, at)`.
    ///
    /// # Panics
    ///
    /// Panics if `at > len`.
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

        let ret = Self {
            inner: self.inner.clone(),
            start: self.start,
            end:   self.start + at,
        };
        self.start += at;
        ret
    }

    /// Shortens the buffer, keeping the first `len` bytes and dropping the
    /// rest.
    #[inline]
    pub fn truncate(&mut self, len: usize) {
        if len < self.len() {
            self.end = self.start + len;
        }
    }

    /// Clears the buffer, removing all data.
    #[inline]
    pub fn clear(&mut self) {
        self.start = self.end;
    }

    /// Returns a slice to the underlying data.
    #[inline]
    #[must_use]
    #[allow(clippy::indexing_slicing)]
    pub fn as_slice(&self) -> &[u8] {
        match &self.inner {
            | Inner::Static(s) => &s[self.start .. self.end],
            | Inner::Rc(rc) => &rc[self.start .. self.end],
        }
    }
}

impl Buf for Bytes {
    #[inline]
    fn remaining(&self) -> usize {
        self.len()
    }

    #[inline]
    fn chunk(&self) -> &[u8] {
        self.as_slice()
    }

    /// # Panics
    ///
    /// Panics if `cnt > len`.
    #[inline]
    fn advance(&mut self, cnt: usize) {
        assert!(cnt <= self.len(), "advance out of bounds: {} <= {}", cnt, self.len());
        self.start += cnt;
    }
}

impl Default for Bytes {
    #[inline]
    fn default() -> Self {
        Self::new()
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
    fn from(vec: Vec<u8>) -> Self {
        let len = vec.len();
        if len == 0 {
            Self::new()
        } else {
            let rc: Rc<[u8]> = Rc::from(vec);
            Self {
                inner: Inner::Rc(rc),
                start: 0,
                end:   len,
            }
        }
    }
}

impl From<Box<[u8]>> for Bytes {
    #[inline]
    fn from(b: Box<[u8]>) -> Self {
        let len = b.len();
        if len == 0 {
            Self::new()
        } else {
            let rc: Rc<[u8]> = Rc::from(b);
            Self {
                inner: Inner::Rc(rc),
                start: 0,
                end:   len,
            }
        }
    }
}

impl From<String> for Bytes {
    #[inline]
    fn from(s: String) -> Self {
        s.into_bytes().into()
    }
}

impl fmt::Debug for Bytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "b\"")?;
        for &b in self.as_slice() {
            for c in std::ascii::escape_default(b) {
                write!(f, "{}", c as char)?;
            }
        }
        write!(f, "\"")
    }
}

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

impl FromIterator<u8> for Bytes {
    fn from_iter<I: IntoIterator<Item = u8>>(iter: I) -> Self {
        Vec::from_iter(iter).into()
    }
}

impl<'a> FromIterator<&'a u8> for Bytes {
    fn from_iter<I: IntoIterator<Item = &'a u8>>(iter: I) -> Self {
        iter.into_iter().copied().collect::<Vec<u8>>().into()
    }
}

impl IntoIterator for Bytes {
    type IntoIter = IntoIter;
    type Item = u8;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        IntoIter {
            bytes: self
        }
    }
}

impl<'a> IntoIterator for &'a Bytes {
    type IntoIter = std::slice::Iter<'a, u8>;
    type Item = &'a u8;

    #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.as_slice().iter()
    }
}

/// Iterator over the bytes of a `Bytes` object.
#[derive(Debug, Clone)]
pub struct IntoIter {
    bytes: Bytes,
}

impl Iterator for IntoIter {
    type Item = u8;

    #[inline]
    #[allow(clippy::indexing_slicing)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.bytes.is_empty() {
            None
        } else {
            let ret = self.bytes.as_slice()[0];
            self.bytes.advance(1);
            Some(ret)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let len = self.bytes.len();
        (len, Some(len))
    }
}

impl ExactSizeIterator for IntoIter {
    #[inline]
    fn len(&self) -> usize {
        self.bytes.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new() {
        let b = Bytes::new();
        assert!(b.is_empty());
        assert_eq!(b.len(), 0);
        assert_eq!(b.as_slice(), b"");
        assert!(!b.is_unique());
    }

    #[test]
    fn from_static() {
        let b = Bytes::from_static(b"hello");
        assert_eq!(b.len(), 5);
        assert_eq!(b.as_slice(), b"hello");
        assert!(!b.is_unique());
    }

    #[test]
    fn from_vec() {
        let v = vec![1, 2, 3];
        let b = Bytes::from(v);
        assert_eq!(b.len(), 3);
        assert_eq!(b.as_slice(), &[1, 2, 3]);
        assert!(b.is_unique());
    }

    #[test]
    fn from_box() {
        let b: Box<[u8]> = vec![1, 2, 3].into_boxed_slice();
        let bytes = Bytes::from(b);
        assert_eq!(bytes.len(), 3);
        assert_eq!(bytes.as_slice(), &[1, 2, 3]);
        assert!(bytes.is_unique());
    }

    #[test]
    fn copy_from_slice() {
        let arr = [1, 2, 3];
        let b = Bytes::copy_from_slice(&arr);
        assert_eq!(b.as_slice(), &[1, 2, 3]);
        assert!(b.is_unique());

        let c = b.clone();
        assert!(!b.is_unique());
        assert!(!c.is_unique());
    }

    #[test]
    fn slice() {
        let b = Bytes::from_static(b"hello world");

        let sub1 = b.slice(0 .. 5);
        assert_eq!(sub1.as_slice(), b"hello");

        let sub2 = b.slice(6 ..);
        assert_eq!(sub2.as_slice(), b"world");

        let sub3 = b.slice(..);
        assert_eq!(sub3.as_slice(), b"hello world");
    }

    #[test]
    fn slice_ref() {
        let b = Bytes::from_static(b"hello world");
        let sub_slice = &b[6 ..];
        let sub_bytes = b.slice_ref(sub_slice);
        assert_eq!(sub_bytes.as_slice(), b"world");

        let empty = &b[0 .. 0];
        assert_eq!(b.slice_ref(empty).len(), 0);
    }

    #[test]
    fn split_off() {
        let mut b = Bytes::from_static(b"hello world");
        let off = b.split_off(5);
        assert_eq!(b.as_slice(), b"hello");
        assert_eq!(off.as_slice(), b" world");
    }

    #[test]
    fn split_to() {
        let mut b = Bytes::from_static(b"hello world");
        let to = b.split_to(6);
        assert_eq!(to.as_slice(), b"hello ");
        assert_eq!(b.as_slice(), b"world");
    }

    #[test]
    fn truncate() {
        let mut b = Bytes::from_static(b"hello world");
        b.truncate(5);
        assert_eq!(b.as_slice(), b"hello");
    }

    #[test]
    fn clear() {
        let mut b = Bytes::from_static(b"hello");
        b.clear();
        assert!(b.is_empty());
    }

    #[test]
    fn buf_advance() {
        let mut b = Bytes::from_static(b"hello");
        assert_eq!(b.remaining(), 5);
        b.advance(2);
        assert_eq!(b.remaining(), 3);
        assert_eq!(b.chunk(), b"llo");
    }

    #[test]
    fn eq() {
        let b1 = Bytes::from_static(b"hello");
        let b2 = Bytes::from(b"hello".to_vec());
        assert_eq!(b1, b2);
        assert_eq!(b1, b"hello"[..]);
    }

    #[test]
    fn from_iter() {
        let b: Bytes = vec![1, 2, 3].into_iter().collect();
        assert_eq!(b.as_slice(), &[1, 2, 3]);

        let arr = [4, 5, 6];
        let b2: Bytes = arr.iter().collect();
        assert_eq!(b2.as_slice(), &[4, 5, 6]);
    }

    #[test]
    fn debug() {
        let b = Bytes::from_static(b"hello\n\t\\");
        assert_eq!(format!("{:?}", b), r#"b"hello\n\t\\""#);
    }
}
