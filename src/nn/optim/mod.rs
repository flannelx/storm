use crate::prelude::*;

pub trait Optimizer {
    fn zero_grad(&mut self);
    fn realize(&mut self);
    fn step(&mut self);
}

pub fn adam<'a>(params: &[*mut Tensor], lr: f32) -> LAMP {
    LAMP::new(params.to_vec(), lr, 0.9, 0.999, 1e-8, 0.0, true)
}

pub fn adam_with<'a>(params: &[*mut Tensor], p: &[f32]) -> LAMP {
    assert!(p.len() > 0, "need lr");
    let mut default = [0.001, 0.9, 0.999, 1e-8, 0.0];
    default.iter_mut().zip(p).for_each(|(d, p)| *d = *p);
    LAMP::new(
        params.to_vec(),
        default[0],
        default[1],
        default[2],
        default[3],
        default[4],
        true,
    )
}
//def __init__(self, params: List[Tensor], lr=0.001, b1=0.9, b2=0.999, eps=1e-6, wd=0.0, adam=False):
#[derive(Debug)]
pub struct LAMP {
    pub(crate) params: Vec<*mut Tensor>,
    pub(crate) buffers: Vec<*mut Tensor>,
    pub(crate) lr: Tensor,
    pub(crate) b1: Tensor,
    pub(crate) b2: Tensor,
    pub(crate) eps: f32,
    pub(crate) wd: f32,
    pub(crate) adam: bool,
    pub(crate) t: Tensor,
    pub(crate) m: Vec<Tensor>,
    pub(crate) v: Vec<Tensor>,
}

impl LAMP {
    pub fn new(
        mut _params: Vec<*mut Tensor>,
        lr: f32,
        b1: f32,
        b2: f32,
        eps: f32,
        wd: f32,
        adam: bool,
    ) -> Self {
        unsafe {
            let mut params = Vec::new();
            while !_params.is_empty() {
                let t = _params.pop().unwrap();
                // if (*t).require_grad {
                //     params.push(t);
                // } else {
                //     buffers.push(t);
                // }
                (*t).require_grad = true;
                params.push(t);
            }
            let lr = Tensor::from([lr]);
            params.dedup_by_key(|t| (*(*t)).id);
            //buffers.dedup_by_key(|t| (*(*t)).id);
            let m = params
                .iter()
                .map(|t| Tensor::zeros((**t).shape()))
                .collect();
            let v = params
                .iter()
                .map(|t| Tensor::zeros((**t).shape()))
                .collect();
            Self {
                params,
                buffers: vec![],
                lr,
                b1: Tensor::from([b1]),
                b2: Tensor::from([b2]),
                eps,
                wd,
                adam,
                t: Tensor::from([0.]),
                m,
                v,
            }
        }
    }
}

impl Optimizer for LAMP {
    fn zero_grad(&mut self) {
        for p in self.params.iter_mut() {
            unsafe {
                let p = &(*(*p));
                *p.grad.lock().unwrap() = None;
            }
        }
    }

    fn realize(&mut self) {
        let mut lst = vec![];
        for b in self.m.iter() {
            b.realize();
        }
        for b in self.v.iter() {
            b.realize();
        }
        for b in self.buffers.iter() {
            unsafe {
                lst.push((**b).clone());
            }
        }
        for b in self.params.iter() {
            unsafe {
                lst.push((**b).clone());
            }
        }
        Tensor::corealize(lst);
    }

    fn step(&mut self) {
        self.t.assign((&self.t + 1.).realize());
        unsafe {
            for (i, t) in self.params.iter_mut().enumerate() {
                let t = &mut (**t);
                assert!(t.grad.lock().unwrap().is_some());
                let g = (*t.grad.lock().unwrap()).clone();
                let mut g = g.unwrap().realize();

                // self.m[i].assign(self.m[i] * self.b1 + g * (1.0 - self.b1)).realize()
                // self.v[i].assign(self.v[i] * self.b2 + (g * g) * (1.0 - self.b2)).realize()
                let mi = (&self.m[i] * &self.b1 + &g * &(1.0 - &self.b1)).realize();
                let vi = (&self.v[i] * &self.b2 + (&g * &g) * (1.0 - &self.b2)).realize();
                self.m[i].assign(mi);
                self.v[i].assign(vi);
                // m_hat = self.m[i] / (1.0 - self.b1**self.t)
                let m_hat = (&self.m[i] / &(1.0 - self.b1.pow(self.t.clone(), false)));
                // v_hat = self.v[i] / (1.0 - self.b2**self.t)
                let v_hat = &self.v[i] / &(1.0 - self.b2.pow(self.t.clone(), false));
                // up = (m_hat / (v_hat.sqrt() + self.eps)) + self.wd * t.detach()
                let up = (m_hat / (v_hat.sqrt() + self.eps)) + t.detach() * self.wd;
                let r = if !self.adam { todo!() } else { 1.0 };
                //println!("{:?}", tmp.to_vec());
                t.assign(t.detach() - &(&self.lr * r * up)).realize();

                // just in case _ctx is attach in Function{}.apply() but shouldnt matter, but
                // why not
            }
        }
        self.realize()
    }
}
