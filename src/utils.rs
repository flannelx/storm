use crate::dtype::NumType;

pub fn getenv<T: std::fmt::Debug + Default + std::str::FromStr>(s: &str, default: T) -> T {
    let s = s.to_uppercase();
    std::env::var(s)
        .unwrap_or(format!("{:?}", default))
        .parse::<T>()
        .unwrap_or(T::default())
}

pub fn roundup<N: NumType>(num: N, amt: N) -> N {
    (num + amt - N::one()) / amt * amt
}

pub fn prod<'a, N: NumType + std::iter::Product<&'a N>>(v: &'a [N]) -> N {
    v.iter().product::<N>()
}
pub fn all(v: &[bool]) -> bool {
    v.iter().all(|a| *a)
}

pub fn any(v: &[bool]) -> bool {
    v.iter().any(|a| *a)
}
