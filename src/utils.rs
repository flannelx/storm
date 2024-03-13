use std::{collections::{BTreeSet, HashSet, HashMap}, hash::Hash, sync::Arc};

use crate::dtype::NumType;

pub fn getenv<T: std::fmt::Display + Default + std::str::FromStr>(s: &str, default: T) -> T {
    let s = s.to_uppercase();
    std::env::var(s)
        .unwrap_or(format!("{}", default))
        .parse::<T>()
        .unwrap_or(T::default())
}

pub fn roundup<N: NumType>(num: N, amt: N) -> N {
    (num + amt - N::one()) / amt * amt
}

pub fn prod<'a, N: NumType + std::iter::Product<&'a N>>(v: &'a [N]) -> N {
    v.iter().product::<N>()
}

pub fn sum<'a, N: NumType + std::iter::Sum<&'a N>>(v: &'a [N]) -> N {
    v.iter().sum::<N>()
}

pub fn all(v: &[bool]) -> bool {
    v.iter().all(|a| *a)
}

pub fn any(v: &[bool]) -> bool {
    v.iter().any(|a| *a)
}

/// This retains the order of the original vec
pub fn dedup<T: Hash + Eq + Clone>(v: Vec<T>) -> Vec<T> {
    let set:HashSet<&T> = HashSet::from_iter(v.iter());
    let mut inserted:HashSet<&T> = HashSet::new();
    let mut ret = vec![];
    for x in v.iter() {
        if set.contains(&x) && !inserted.contains(x){
            ret.push((*x).clone());
            inserted.insert(x);
        }
    }
    ret
}

#[derive(Clone, Debug)]
pub struct DefaultDict<K: Hash + Eq + Clone, V: Default> {
    val: Arc<HashMap<K, V>>,
    ptr: Option<*mut HashMap<K, V>>,
}

impl<K: Hash + Eq + Clone, V: Default> Default for DefaultDict<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + Clone, V: Default> DefaultDict<K, V> {
    pub fn new() -> Self {
        Self{
            val: Arc::new(HashMap::new()),
            ptr: None,
        }
    }

    pub fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + Clone, V: Default> std::ops::Deref for DefaultDict<K, V> {
    type Target = HashMap<K, V>;

    fn deref(&self) -> &Self::Target {
        &self.val
    }
}

impl<K: Hash + Eq + Clone, V: Default> std::ops::DerefMut for DefaultDict<K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe {
            self.ptr = Some(Arc::as_ptr(&self.val) as *mut _);
            &mut **self.ptr.as_mut().unwrap()
        }
    }
}

impl<K: Hash + Eq + Clone, V: Default> std::ops::Index<K> for DefaultDict<K, V> {
    type Output = V;

    fn index(&self, index: K) -> &Self::Output {
        if !self.val.contains_key(&index) {
            unsafe {
                let mut val_clone = self.val.clone();
                let val_mut = Arc::get_mut_unchecked(&mut val_clone);
                val_mut.insert(index.clone(), V::default());
            }
        }
        &self.val[&index]
    }
}

impl<K: Hash + Eq + Clone, V: Default> std::ops::IndexMut<K> for DefaultDict<K, V> {
    fn index_mut(&mut self, index: K) -> &mut Self::Output {
        &self[index.clone()];
        self.get_mut(&index).unwrap()
    }
}

impl<K: Hash + Eq + Clone, V: Default> std::ops::Index<&K> for DefaultDict<K, V> {
    type Output = V;

    fn index(&self, index: &K) -> &Self::Output {
        if !self.val.contains_key(&index) {
            unsafe {
                let mut val_clone = self.val.clone();
                let val_mut = Arc::get_mut_unchecked(&mut val_clone);
                val_mut.insert(index.clone(), V::default());
            }
        }
        &self.val[&index]
    }
}

impl<K: Hash + Eq + Clone, V: Default> std::ops::IndexMut<&K> for DefaultDict<K, V> {
    fn index_mut(&mut self, index: &K) -> &mut Self::Output {
        &self[index.clone()];
        self.get_mut(&index).unwrap()
    }
}
