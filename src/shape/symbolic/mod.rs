pub mod core_ops;

use std::hash::Hash;
use std::{cmp::PartialEq, collections::HashMap, sync::Arc};

#[derive(Clone, Eq, Debug)]
pub struct ArcNode(Arc<dyn Node>);

impl PartialEq for ArcNode {
    fn eq(&self, other: &Self) -> bool {
        if self.is_num() && other.is_num() {
            return self.num_val().unwrap() == other.num_val().unwrap();
        }
        self.key() == other.key()
    }
}

impl Hash for ArcNode {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.key().hash(state)
    }
}

impl core::ops::Deref for ArcNode {
    type Target = Arc<dyn Node>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

fn lt(lhs: ArcNode, b: ArcNode) -> ArcNode {
    create_node(LtNode::new(lhs, b))
}

#[allow(unused_variables)]
fn div(lhs: ArcNode, rhs: ArcNode, factoring_allowed: Option<bool>) -> ArcNode {
    if lhs.key() == rhs.key() {
        return num(1);
    }
    if (&rhs - &lhs).min().unwrap() > 0 && lhs.min().unwrap() >= 0 {
        return num(0);
    }
    let b = rhs.num_val().unwrap();
    assert!(b != 0);
    if b < 0 {
        return div(lhs, num(-b), None) * -1;
    }
    if b == 1 {
        return lhs;
    }
    let min = lhs.min().unwrap();
    if min < 0 {
        let offset = min.div_euclid(b);
        // println!("{min}/{b}={offset}");
        return div(&lhs + -offset * b, rhs, Some(false)) + offset;
    }
    create_node(DivNode::new(lhs, rhs))
}

fn _mod(lhs: ArcNode, rhs: ArcNode) -> ArcNode {
    if lhs.key() == rhs.key() {
        return num(0);
    }
    if (&rhs - &lhs).min().unwrap() > 0 && lhs.min().unwrap() >= 0 {
        return lhs.to_arc();
    }

    let b = rhs.num_val().unwrap();
    let min = lhs.min().unwrap();
    let max = lhs.max().unwrap();
    if b == 1 {
        return num(0);
    }
    if min >= 0 && max < b {
        return lhs.to_arc();
    }
    if min < 0 {
        return &lhs - (min.div_euclid(b) * b).rem_euclid(b);
    }
    create_node(ModNode::new(lhs.to_arc(), rhs))
}

pub trait Node: core::fmt::Debug {
    // Node variants possible fields
    fn a(&self) -> Option<ArcNode> {
        None
    }

    fn b(&self) -> Option<ArcNode> {
        None
    }

    fn num_val(&self) -> Option<isize> {
        None
    }

    fn expr(&self) -> Option<&str> {
        None
    }

    fn key(&self) -> String {
        //NOTE:: This is the default NodeOp impl.
        self.render(Arc::new(CStyle), None, false)
    }

    fn min(&self) -> Option<isize> {
        None
    }

    fn max(&self) -> Option<isize> {
        None
    }

    fn _add(&self, rhs: ArcNode) -> ArcNode {
        sum(&[self.to_arc(), rhs])
    }

    fn _sub(&self, rhs: ArcNode) -> ArcNode {
        self._add(rhs.neg())
    }

    fn _mul(&self, rhs: ArcNode) -> ArcNode {
        if let Some(b) = rhs.num_val() {
            if b == 0 {
                return num(0);
            }
            if b == 1 {
                return self.to_arc();
            }
        }
        if self.is_num() {
            if rhs.is_num() {
                return num(self.num_val().unwrap() * rhs.num_val().unwrap());
            } else {
                return rhs._mul(self.b().unwrap().to_arc());
            }
        }
        create_node(MulNode::new(self.to_arc(), rhs))
    }

    fn _div(&self, rhs: ArcNode, factoring_allowed: Option<bool>) -> ArcNode {
        div(self.to_arc(), rhs, factoring_allowed)
    }

    fn _mod(&self, rhs: ArcNode) -> ArcNode {
        _mod(self.to_arc(), rhs)
    }

    fn neg(&self) -> ArcNode {
        self._mul(num(-1))
    }

    fn lt(&self, b: ArcNode) -> ArcNode {
        lt(self.to_arc(), b)
    }

    fn le(&self, b: ArcNode) -> ArcNode {
        // self < (b+1)
        self.lt(b + 1)
    }

    fn gt(&self, b: ArcNode) -> ArcNode {
        // (-self) < (-b)
        self.neg().lt(-b)
    }

    fn ge(&self, b: ArcNode) -> ArcNode {
        //(-self) < (-b+1)
        self.neg().lt(-b + 1)
    }

    fn is_add(&self) -> bool {
        false
    }

    fn is_sub(&self) -> bool {
        false
    }

    fn is_mul(&self) -> bool {
        false
    }

    fn is_div(&self) -> bool {
        false
    }

    fn is_mod(&self) -> bool {
        false
    }

    fn is_and(&self) -> bool {
        false
    }

    fn is_num(&self) -> bool {
        false
    }

    fn is_sum(&self) -> bool {
        false
    }

    fn is_var(&self) -> bool {
        false
    }

    fn render(&self, ops: Arc<dyn NodeOp>, ctx: Option<&str>, strip_paren: bool) -> String;

    fn vars(&self) -> Vec<ArcNode> {
        vec![]
    }

    fn nodes(&self) -> Vec<ArcNode> {
        vec![]
    }

    fn flat_components(&self) -> Vec<ArcNode> {
        vec![self.to_arc()]
    }

    fn to_arc(&self) -> ArcNode;

    fn get_bounds(&self) -> Option<(isize, isize)> {
        None
    }
}

pub fn create_node(ret: ArcNode) -> ArcNode {
    assert!(
        ret.min().unwrap() <= ret.max().unwrap(),
        "min greater than max! {} {} when creating {ret:?}",
        ret.min().unwrap(),
        ret.max().unwrap()
    );
    if ret.min().unwrap() == ret.max().unwrap() {
        return num(ret.min().unwrap());
    }
    ret
}

pub fn gcd(mut n: isize, mut m: isize) -> isize {
    assert!(n != 0 && m != 0);
    while m != 0 {
        if m < n {
            std::mem::swap(&mut m, &mut n);
        }
        m %= n;
    }
    n
}

pub fn var(expr: &str, min: isize, max: isize) -> ArcNode {
    create_node(Variable::new(expr, min, max))
}

pub fn num(v: isize) -> ArcNode {
    NumNode::new(v)
}

pub fn sum(_nodes: &[ArcNode]) -> ArcNode {
    let mut nodes = vec![];
    for n in _nodes {
        if n.min().is_some() || n.max().is_some() {
            nodes.push((*n).to_arc())
        }
    }
    if nodes.len() == 0 {
        return num(0);
    }

    if nodes.len() == 1 {
        return nodes[0].to_arc();
    }

    let mut new_nodes = vec![];
    let mut num_node_sum = 0;
    for n in (SumNode { nodes }).flat_components() {
        if n.is_num() {
            num_node_sum += n.num_val().unwrap();
            continue;
        }
        new_nodes.push(n)
    }
    let mut unique_mul_a: Vec<ArcNode> = new_nodes
        .iter()
        .map(|x| {
            if x.is_mul() {
                x.a().unwrap()
            } else {
                x.clone()
            }
        })
        .collect();
    unique_mul_a.dedup_by(|a, b| a.key() == b.key());
    if new_nodes.len() > 1 && unique_mul_a.len() < new_nodes.len() {
        new_nodes = factorize(new_nodes);
    }
    if num_node_sum != 0 {
        new_nodes.push(num(num_node_sum));
    }

    // return create_rednode(SumNode, new_nodes) if len(new_nodes) > 1 else new_nodes[0] if len(new_nodes) == 1 else NumNode(0)
    if new_nodes.len() == 0 {
        return num(0);
    }
    if new_nodes.len() == 1 {
        return new_nodes[0].to_arc();
    }
    ArcNode(Arc::new(SumNode { nodes: new_nodes }))
}

pub fn ands(nodes: &[ArcNode]) -> ArcNode {
    if nodes.is_empty() {
        return num(1);
    }
    if nodes.len() == 1 {
        return nodes[0].clone();
    }
    // def __bool__(self): return not (self.max == self.min == 0)
    // if any(not x for x in nodes): return NumNode(0)
    if nodes.iter().any(|x| {
        x.min().is_some()
            && x.max().is_some()
            && x.max().unwrap() == x.min().unwrap()
            && x.max().unwrap() == 0
    }) {
        return num(0);
    }
    let nodes: Vec<ArcNode> = nodes
        .iter()
        .filter(|n| n.min().unwrap() != n.max().unwrap())
        .map(|n| n.clone())
        .collect();
    if nodes.len() > 1 {
        return AndNode::new(&nodes);
    }
    if nodes.len() == 1 {
        return nodes[0].to_owned();
    }
    num(1)
}

pub fn factorize(nodes: Vec<ArcNode>) -> Vec<ArcNode> {
    let mut mul_groups: HashMap<ArcNode, isize> = HashMap::new();
    for x in nodes {
        let (a, b) = if x.is_mul() {
            (x.a().unwrap().to_arc(), x.b().unwrap().to_arc())
        } else {
            (x, num(1))
        };
        *mul_groups.entry(a).or_default() += b.num_val().unwrap();
    }
    let mut ret = vec![];
    for (a, b_sum) in mul_groups.keys().zip(mul_groups.values()) {
        if *b_sum == 0 {
            continue;
        }

        if *b_sum != 1 {
            ret.push(MulNode::new(a.to_arc(), num(*b_sum)));
        } else {
            ret.push(a.to_arc())
        }
    }
    ret
}

#[derive(Clone, Debug)]
pub struct Variable {
    expr: String,
    min: isize,
    max: isize,
}

impl Variable {
    fn new(expr: &str, min: isize, max: isize) -> ArcNode {
        ArcNode(Arc::new(Self {
            expr: expr.to_string(),
            min,
            max,
        }))
    }
}

impl Node for Variable {
    fn is_var(&self) -> bool {
        true
    }

    fn to_arc(&self) -> ArcNode {
        ArcNode(Arc::new(Self {
            expr: self.expr.clone(),
            min: self.min,
            max: self.max,
        }))
    }

    fn expr(&self) -> Option<&str> {
        Some(&self.expr)
    }

    fn min(&self) -> Option<isize> {
        Some(self.min)
    }

    fn max(&self) -> Option<isize> {
        Some(self.max)
    }

    fn render(&self, ops: Arc<dyn NodeOp>, ctx: Option<&str>, strip_paren: bool) -> String {
        let mut ret = ops.variable(self.to_arc(), ctx);
        if strip_paren && ret.chars().nth(0).unwrap() == '(' {
            ret = ret.replace("(", "").replace(")", "");
        }
        ret
    }

    fn vars(&self) -> Vec<ArcNode> {
        vec![self.to_arc()]
    }
}

#[derive(Debug)]
pub struct SumNode {
    nodes: Vec<ArcNode>,
}

impl Node for SumNode {
    // Not sure about this, in sum() there is a check for min or max. but why. what node doesnt
    // have min max and is it safe to do this?
    fn min(&self) -> Option<isize> {
        Some(self.nodes.iter().map(|x| x.min().unwrap()).sum())
    }

    fn max(&self) -> Option<isize> {
        Some(self.nodes.iter().map(|x| x.max().unwrap()).sum())
    }

    fn _mul(&self, rhs: ArcNode) -> ArcNode {
        let a: Vec<ArcNode> = self.nodes.iter().map(|n| n._mul(rhs.to_arc())).collect();
        sum(&a)
    }

    fn is_sum(&self) -> bool {
        true
    }

    fn to_arc(&self) -> ArcNode {
        ArcNode(Arc::new(Self {
            nodes: self.nodes.iter().map(|x| (*x).to_arc()).collect(),
        }))
    }

    fn nodes(&self) -> Vec<ArcNode> {
        self.nodes.clone()
    }

    fn vars(&self) -> Vec<ArcNode> {
        let mut ret = vec![];
        for n in self.nodes.iter() {
            ret.extend(n.vars());
        }
        ret
    }

    fn render(&self, ops: Arc<dyn NodeOp>, ctx: Option<&str>, strip_paren: bool) -> String {
        assert!(
            self.min().unwrap() != self.max().unwrap(),
            "min:{} eq max:{}",
            self.min().unwrap(),
            self.max().unwrap()
        );
        let mut ret = ops.sum(self.to_arc(), ctx);
        if strip_paren && ret.chars().nth(0).unwrap() == '(' {
            ret = ret.replace("(", "").replace(")", "");
        }
        ret
    }

    fn flat_components(&self) -> Vec<ArcNode> {
        self.nodes
            .iter()
            .map(|n| n.flat_components())
            .collect::<Vec<Vec<ArcNode>>>()
            .concat()
    }

    fn lt(&self, mut b: ArcNode) -> ArcNode {
        if !b.is_num() {
            return lt(self.to_arc(), b);
        }
        let mut new_sum = vec![];
        for x in self.nodes.iter() {
            if x.is_num() {
                b = b - x.b().unwrap();
            } else {
                new_sum.push(x.clone());
            }
        }
        let mut lhs = sum(&new_sum);
        if !lhs.is_sum() {
            return lt(lhs, b);
        }
        let mut muls = vec![];
        let mut others = vec![];
        for x in lhs.nodes() {
            if x.is_mul()
                && x.b().unwrap().num_val().unwrap() > 0
                && x.max().unwrap() >= b.num_val().unwrap()
            {
                muls.push(x.clone());
            } else {
                others.push(x.clone());
            }
        }
        if muls.is_empty() {
            return create_node(LtNode::new(lhs, b.clone()));
        }

        let mut mul_gcd = muls[0].b().unwrap().num_val().unwrap();
        for x in muls[1..].iter() {
            mul_gcd = gcd(mul_gcd, x.b().unwrap().num_val().unwrap());
        }
        if b.num_val().unwrap().rem_euclid(mul_gcd) == 0 {
            let all_others = sum(&others);
            if all_others.min().unwrap() >= 0 && all_others.max().unwrap() < mul_gcd {
                lhs = sum(&muls);
            }
        }
        lt(lhs, b)
    }

    fn _mod(&self, rhs: ArcNode) -> ArcNode {
        if rhs.is_sum() {
            let nu_num = sum(&self
                .flat_components()
                .into_iter()
                .filter(|n| n.is_num())
                .collect::<Vec<ArcNode>>());
            let de_num = sum(&rhs
                .flat_components()
                .into_iter()
                .filter(|n| n.is_num())
                .collect::<Vec<ArcNode>>());
            // we filter only num and sum of these nodes is a num node
            if nu_num.num_val().unwrap() > 0
                && de_num.is_num()
                && nu_num.num_val().unwrap() / de_num.num_val().unwrap() > 0
            {
                return (self.to_arc()
                    - rhs.clone() * (nu_num.num_val().unwrap() / de_num.num_val().unwrap()))
                    % rhs;
            }
        }
        if (&rhs - self.to_arc()).min().unwrap() > 0 {
            return self.to_arc();
        }
        let mut new_nodes = vec![];
        for x in self.nodes.iter() {
            if x.is_num() {
                new_nodes.push(num(x.num_val().unwrap().rem_euclid(rhs.num_val().unwrap())));
            } else if x.is_mul() {
                new_nodes.push(x.a().unwrap() * (x.b().unwrap() % rhs.clone()));
            } else {
                new_nodes.push(x.clone())
            }
        }
        _mod(sum(&new_nodes), rhs)
    }

    fn _div(&self, rhs: ArcNode, factoring_allowed: Option<bool>) -> ArcNode {
        if rhs.is_sum() {
            let nu_num = sum(&self
                .flat_components()
                .into_iter()
                .filter(|n| n.is_num())
                .collect::<Vec<ArcNode>>());
            let de_num = sum(&rhs
                .flat_components()
                .into_iter()
                .filter(|n| n.is_num())
                .collect::<Vec<ArcNode>>());
            if nu_num.num_val().unwrap() > 0
                && de_num.is_num()
                && nu_num.num_val().unwrap() / de_num.num_val().unwrap() > 0
            {
                let d = nu_num.num_val().unwrap() / de_num.num_val().unwrap();
                return num(d) + (self.to_arc() - &rhs * d) / rhs;
            }
        }
        let mut fully_divided = vec![];
        let mut rest = vec![];
        if !rhs.is_num() {
            for x in self.flat_components() {
                if &x % &rhs == num(0) {
                    fully_divided.push(&x / &rhs);
                } else {
                    rest.push(x.clone());
                }
            }
            let sum_fully_divied = sum(&fully_divided);
            if sum_fully_divied != num(0) {
                return sum_fully_divied + sum(&rest) / rhs;
            }
            return div(self.to_arc(), rhs, Some(false));
        }
        if rhs.num_val().unwrap() == 1 {
            return self.to_arc();
        }
        if factoring_allowed.is_some_and(|x| x == false) {
            return div(self.to_arc(), rhs, Some(false));
        }
        let b = rhs.num_val().unwrap();
        let mut fully_divided = vec![];
        let mut rest = vec![];
        let mut _gcd = rhs.num_val().unwrap();
        let mut divisor = 1;
        for x in self.flat_components() {
            if x.is_num() || x.is_mul() {
                if x.b().unwrap().num_val().unwrap().rem_euclid(b) == 0 {
                    fully_divided.push(x / b)
                } else {
                    rest.push(x.clone());
                    _gcd = gcd(_gcd, x.b().unwrap().num_val().unwrap());
                    if x.is_mul() && divisor == 1 && &rhs % &x.b().unwrap() == num(0) {
                        divisor = x.b().unwrap().num_val().unwrap();
                    }
                }
            } else {
                rest.push(x.clone());
                _gcd = 1;
            }
        }
        if _gcd > 1 {
            return sum(&fully_divided) + sum(&rest) / _gcd / (rhs / _gcd);
        }
        if divisor > 1 {
            return sum(&fully_divided) + sum(&rest) / divisor / (rhs / divisor);
        }
        sum(&fully_divided) + div(sum(&rest), rhs, None)
    }
}

#[derive(Debug)]
pub struct MulNode {
    a: ArcNode,
    b: ArcNode,
    min: isize,
    max: isize,
}

impl MulNode {
    fn new(a: ArcNode, b: ArcNode) -> ArcNode {
        let mut ret = Self {
            a,
            b,
            min: 0,
            max: 0,
        };
        (ret.min, ret.max) = ret
            .get_bounds()
            .expect("MulNode should have impl get_bounds()");
        ArcNode(Arc::new(ret))
    }
}

impl Node for MulNode {
    fn vars(&self) -> Vec<ArcNode> {
        vec![self.a.vars(), self.b.vars()].concat()
    }
    fn get_bounds(&self) -> Option<(isize, isize)> {
        let b = if self.b.is_num() {
            self.b.num_val().unwrap()
        } else {
            self.b.max().unwrap() // FIXME: when self.b is not a int, min max becomes 'Node' instead of int in tinygrad
        };
        if b >= 0 {
            return Some((self.a.min().unwrap() * b, self.a.max().unwrap() * b));
        }
        Some((self.a.max().unwrap() * b, self.a.min().unwrap() * b))
    }

    fn _mul(&self, rhs: ArcNode) -> ArcNode {
        self.a._mul(self.b._mul(rhs))
    }

    fn _div(&self, rhs: ArcNode, factoring_allowed: Option<bool>) -> ArcNode {
        assert!(self.b.is_num() && rhs.is_num(), "Tinygrad requires both self.b and rhs to be num implicitly, will crash in Node.__mod__ otherwise");
        let b = self.b.num_val().unwrap();
        let r = rhs.num_val().unwrap();
        if &b % &r == 0 {
            return &self.a * (b / r);
        }
        if &r % &b == 0 && b > 1 {
            return &self.a / (r / b);
        }
        div(self.to_arc(), rhs, factoring_allowed)
    }

    fn is_mul(&self) -> bool {
        true
    }
    fn to_arc(&self) -> ArcNode {
        ArcNode(Arc::new(Self {
            a: self.a.to_arc(),
            b: self.b.to_arc(),
            min: self.min,
            max: self.max,
        }))
    }

    fn a(&self) -> Option<ArcNode> {
        Some(self.a.clone())
    }

    fn b(&self) -> Option<ArcNode> {
        Some(self.b.clone())
    }

    fn min(&self) -> Option<isize> {
        Some(self.min)
    }
    fn max(&self) -> Option<isize> {
        Some(self.max)
    }

    fn render(&self, ops: Arc<dyn NodeOp>, ctx: Option<&str>, strip_paren: bool) -> String {
        assert!(
            self.min().unwrap() != self.max().unwrap(),
            "min:{} eq max:{}",
            self.min().unwrap(),
            self.max().unwrap()
        );
        let mut ret = ops.mul(self.to_arc(), ctx);
        if strip_paren && ret.chars().nth(0).unwrap() == '(' {
            ret = ret.replace("(", "").replace(")", "");
        }
        ret
    }

    fn _mod(&self, rhs: ArcNode) -> ArcNode {
        let a = &self.a * &(&self.b % &rhs);
        _mod(a, rhs)
    }

    fn nodes(&self) -> Vec<ArcNode> {
        vec![self.a.clone(), self.b.clone()]
    }
}

#[derive(Debug)]
pub struct NumNode {
    b: isize,
}

impl NumNode {
    pub fn new(b: isize) -> ArcNode {
        ArcNode(Arc::new(Self { b }))
    }
}

impl Node for NumNode {
    fn num_val(&self) -> Option<isize> {
        Some(self.b)
    }

    fn b(&self) -> Option<ArcNode> {
        Some(self.to_arc())
    }

    fn min(&self) -> Option<isize> {
        Some(self.b)
    }

    fn max(&self) -> Option<isize> {
        Some(self.b)
    }

    fn is_num(&self) -> bool {
        true
    }

    #[allow(unused_variables)]
    fn render(&self, ops: Arc<dyn NodeOp>, ctx: Option<&str>, strip_paren: bool) -> String {
        self.b.to_string()
    }

    fn to_arc(&self) -> ArcNode {
        ArcNode(Arc::new(Self { b: self.b }))
    }
}

#[derive(Debug)]
pub struct DivNode {
    a: ArcNode,
    b: ArcNode,
    min: isize,
    max: isize,
}

impl DivNode {
    fn new(a: ArcNode, b: ArcNode) -> ArcNode {
        let mut ret = Self {
            a,
            b,
            min: 0,
            max: 0,
        };
        let (min, max) = ret
            .get_bounds()
            .expect("OpNode Div should have impl get_bounds()");
        ret.min = min;
        ret.max = max;
        ArcNode(Arc::new(ret))
    }
}

impl Node for DivNode {
    #[allow(unused_variables)]
    fn _div(&self, rhs: ArcNode, factoring_allowed: Option<bool>) -> ArcNode {
        self.a._div(self.b._mul(rhs), None)
    }

    fn get_bounds(&self) -> Option<(isize, isize)> {
        let b = if self.b.is_num() {
            self.b.num_val().unwrap()
        } else {
            self.b.max().unwrap() // FIXME: when self.b is not a int, min max becomes 'Node' instead of int in tinygrad
        };
        Some((self.a.min().unwrap() / b, self.a.max().unwrap() / b))
    }

    fn is_div(&self) -> bool {
        true
    }

    fn vars(&self) -> Vec<ArcNode> {
        vec![self.a.vars(), self.b.vars()].concat()
    }
    fn to_arc(&self) -> ArcNode {
        ArcNode(Arc::new(Self {
            a: self.a.to_arc(),
            b: self.b.to_arc(),
            min: self.min,
            max: self.max,
        }))
    }

    fn a(&self) -> Option<ArcNode> {
        Some(self.a.clone())
    }

    fn b(&self) -> Option<ArcNode> {
        Some(self.b.clone())
    }

    fn min(&self) -> Option<isize> {
        Some(self.min)
    }

    fn max(&self) -> Option<isize> {
        Some(self.max)
    }

    fn render(&self, ops: Arc<dyn NodeOp>, ctx: Option<&str>, strip_paren: bool) -> String {
        assert!(
            self.min().unwrap() != self.max().unwrap(),
            "min:{} eq max:{}",
            self.min().unwrap(),
            self.max().unwrap()
        );
        let mut ret = ops.div(self.to_arc(), ctx);
        if strip_paren && ret.chars().nth(0).unwrap() == '(' {
            ret = ret.replace("(", "").replace(")", "");
        }
        ret
    }
}

#[derive(Debug)]
pub struct LtNode {
    a: ArcNode,
    b: ArcNode,
    min: isize,
    max: isize,
}

impl LtNode {
    fn new(a: ArcNode, b: ArcNode) -> ArcNode {
        let mut ret = Self {
            a,
            b,
            min: 0,
            max: 0,
        };

        (ret.min, ret.max) = ret
            .get_bounds()
            .expect("LtNode should have impl get_bounds()");
        ArcNode(Arc::new(ret))
    }
}

#[allow(unused_variables)]
impl Node for LtNode {
    fn _mul(&self, rhs: ArcNode) -> ArcNode {
        (self.a.clone() * rhs.clone()).lt(self.b.clone() * rhs.clone())
    }

    fn _div(&self, rhs: ArcNode, factoring_allowed: Option<bool>) -> ArcNode {
        (self.a.clone() / rhs.clone()).lt(self.b.clone() / rhs.clone())
    }

    fn a(&self) -> Option<ArcNode> {
        Some(self.a.clone())
    }

    fn b(&self) -> Option<ArcNode> {
        Some(self.b.clone())
    }

    fn min(&self) -> Option<isize> {
        Some(self.min)
    }

    fn max(&self) -> Option<isize> {
        Some(self.max)
    }

    fn render(&self, ops: Arc<dyn NodeOp>, ctx: Option<&str>, strip_paren: bool) -> String {
        assert!(
            self.min().unwrap() != self.max().unwrap(),
            "min:{} eq max:{}",
            self.min().unwrap(),
            self.max().unwrap()
        );
        let mut ret = ops.lt(self.to_arc(), ctx);
        if strip_paren && ret.chars().nth(0).unwrap() == '(' {
            ret = ret.replace("(", "").replace(")", "");
        }
        ret
    }

    fn to_arc(&self) -> ArcNode {
        ArcNode(Arc::new(Self {
            a: self.a.clone(),
            b: self.b.clone(),
            min: self.min,
            max: self.max,
        }))
    }

    fn get_bounds(&self) -> Option<(isize, isize)> {
        if self.b.is_num() {
            let (a_min, a_max) = (self.a.min().unwrap(), self.a.max().unwrap());
            let b = self.b.num_val().unwrap();

            return Some((if a_max < b { 1 } else { 0 }, if a_min < b { 1 } else { 0 }));
        }
        if self.a.max().unwrap() < self.b.min().unwrap() {
            Some((1, 1))
        } else if self.a.min().unwrap() > self.b.max().unwrap() {
            Some((0, 0))
        } else {
            Some((0, 1))
        }
    }

    fn vars(&self) -> Vec<ArcNode> {
        vec![self.a.vars(), self.b.vars()].concat()
    }
}

#[derive(Debug)]
pub struct ModNode {
    a: ArcNode,
    b: ArcNode,
    min: isize,
    max: isize,
}

impl ModNode {
    fn new(a: ArcNode, b: ArcNode) -> ArcNode {
        let mut ret = Self {
            a,
            b,
            min: 0,
            max: 0,
        };
        (ret.min, ret.max) = ret.get_bounds().unwrap();
        ArcNode(Arc::new(ret))
    }
}

impl Node for ModNode {
    fn vars(&self) -> Vec<ArcNode> {
        vec![self.a.vars(), self.b.vars()].concat()
    }

    fn a(&self) -> Option<ArcNode> {
        Some(self.a.clone())
    }

    fn b(&self) -> Option<ArcNode> {
        Some(self.b.clone())
    }

    fn min(&self) -> Option<isize> {
        Some(self.min)
    }

    fn max(&self) -> Option<isize> {
        Some(self.max)
    }

    fn nodes(&self) -> Vec<ArcNode> {
        vec![self.a.clone(), self.b.clone()]
    }

    fn get_bounds(&self) -> Option<(isize, isize)> {
        assert!(self.a.min().unwrap() >= 0 && self.b.is_num());
        let (a_min, a_max) = (self.a.min().unwrap(), self.a.max().unwrap());
        let b = if self.b.is_num() {
            self.b.num_val().unwrap()
        } else {
            self.b.max().unwrap() // FIXME: when self.b is not a int, min max becomes 'Node' instead of int in tinygrad
        };
        // if self.a.max - self.a.min >= self.b or (self.a.min != self.a.max and self.a.min%self.b >= self.a.max%self.b):
        //    return (0, self.b-1)
        // else
        //    return (self.a.min%self.b, self.a.max%self.b)
        if a_max - a_min >= b || (a_min != a_max && a_min % b >= a_max % b) {
            Some((0, b - 1))
        } else {
            Some((a_min % b, a_max % b))
        }
    }

    fn render(&self, ops: Arc<dyn NodeOp>, ctx: Option<&str>, strip_paren: bool) -> String {
        assert!(
            self.min().unwrap() != self.max().unwrap(),
            "min:{} eq max:{}",
            self.min().unwrap(),
            self.max().unwrap()
        );
        let mut ret = ops._mod(self.to_arc(), ctx);
        if strip_paren && ret.chars().nth(0).unwrap() == '(' {
            ret = ret.replace("(", "").replace(")", "");
        }
        ret
    }

    fn to_arc(&self) -> ArcNode {
        ArcNode(Arc::new(Self {
            a: self.a.clone(),
            b: self.b.clone(),
            min: self.min,
            max: self.max,
        }))
    }

    fn is_mod(&self) -> bool {
        true
    }

    fn _div(&self, rhs: ArcNode, factoring_allowed: Option<bool>) -> ArcNode {
        if &self.b % &rhs == num(0) {
            return (&self.a / &rhs) % (&self.b / &rhs);
        }
        div(self.to_arc(), rhs, factoring_allowed)
    }
}

#[derive(Debug)]
pub struct AndNode {
    nodes: Vec<ArcNode>,
}

impl AndNode {
    fn new(b: &[ArcNode]) -> ArcNode {
        ArcNode(Arc::new(Self { nodes: b.to_vec() }))
    }
}

#[allow(unused_variables)]
impl Node for AndNode {
    fn min(&self) -> Option<isize> {
        self.nodes.iter().map(|n| n.min().unwrap()).min()
    }

    fn vars(&self) -> Vec<ArcNode> {
        let mut ret = vec![];
        for n in self.nodes.iter() {
            ret.extend(n.vars());
        }
        ret
    }

    fn max(&self) -> Option<isize> {
        self.nodes.iter().map(|n| n.max().unwrap()).max()
    }

    fn _mul(&self, rhs: ArcNode) -> ArcNode {
        ands(
            &self
                .nodes
                .iter()
                .map(|n| n * &rhs)
                .collect::<Vec<ArcNode>>(),
        )
    }

    fn _div(&self, rhs: ArcNode, factoring_allowed: Option<bool>) -> ArcNode {
        ands(
            &self
                .nodes
                .iter()
                .map(|n| n / &rhs)
                .collect::<Vec<ArcNode>>(),
        )
    }

    fn render(&self, ops: Arc<dyn NodeOp>, ctx: Option<&str>, strip_paren: bool) -> String {
        let mut ret = ops.and(self.to_arc(), ctx);
        if strip_paren && ret.chars().nth(0).unwrap() == '(' {
            ret = ret.replace("(", "").replace(")", "");
        }
        ret
    }

    fn to_arc(&self) -> ArcNode {
        ArcNode(Arc::new(Self {
            nodes: self.nodes.clone(),
        }))
    }

    fn nodes(&self) -> Vec<ArcNode> {
        self.nodes.clone()
    }
}

#[allow(unused_variables)]
pub trait NodeOp {
    fn variable(&self, s: ArcNode, ctx: Option<&str>) -> String {
        // Variable: lambda self,ops,ctx: f"{self.expr}[{self.min}-{self.max}]" if ctx == "DEBUG" else f"{self.expr}",
        if ctx.is_some_and(|f| f == "DEBUG") {
            return format!(
                "{}[{}-{}]",
                s.expr().unwrap(),
                s.min().unwrap(),
                s.max().unwrap()
            );
        }
        s.expr().unwrap().to_string()
    }

    fn num(&self, s: ArcNode, ctx: Option<&str>) -> String {
        // NumNode: lambda self,ops,ctx: f"{self.b}",
        s.b().unwrap().to_string()
    }

    fn mul(&self, s: ArcNode, ctx: Option<&str>) -> String {
        // MulNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}*{sym_render(self.b,ops,ctx)})",
        format!(
            "({}*{})",
            s.a().unwrap().render(self.to_arc(), ctx, false),
            s.b().unwrap().render(self.to_arc(), ctx, false), // <-- Everything should be a Node here,
                                                              // so no need to "sym_render()"
        )
    }

    fn div(&self, s: ArcNode, ctx: Option<&str>) -> String {
        // DivNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}/{self.b})",
        format!(
            "({}/{})",
            s.a().unwrap().render(self.to_arc(), ctx, false),
            s.b().unwrap()
        )
    }

    fn _mod(&self, s: ArcNode, ctx: Option<&str>) -> String {
        // ModNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}%{self.b})",
        format!(
            "({}%{})",
            s.a().unwrap().render(self.to_arc(), ctx, false),
            s.b().unwrap()
        )
    }

    fn lt(&self, s: ArcNode, ctx: Option<&str>) -> String {
        //LtNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}<{sym_render(self.b,ops,ctx)})",
        format!(
            "({}<{})",
            s.a().unwrap().render(self.to_arc(), ctx, false),
            s.b().unwrap().render(self.to_arc(), ctx, false),
        )
    }

    fn sum(&self, s: ArcNode, ctx: Option<&str>) -> String {
        let mut renders = vec![];
        for n in s.nodes() {
            renders.push(n.render(self.to_arc(), ctx, false));
        }
        renders.sort();
        format!("({})", renders.join("+"))
    }

    fn and(&self, s: ArcNode, ctx: Option<&str>) -> String {
        let mut renders = vec![];
        for n in s.nodes() {
            renders.push(n.render(self.to_arc(), ctx, false));
        }
        renders.sort();
        format!("({})", renders.join("&&"))
    }

    fn to_arc(&self) -> Arc<dyn NodeOp>;
}

pub struct CStyle;
#[allow(dead_code)]
impl CStyle {
    fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}
impl NodeOp for CStyle {
    fn to_arc(&self) -> Arc<dyn NodeOp> {
        Arc::new(CStyle)
    }
}

pub struct Python;
#[allow(dead_code)]
impl Python {
    fn new() -> Arc<Self> {
        Arc::new(Self)
    }
}
impl NodeOp for Python {
    fn div(&self, s: ArcNode, ctx: Option<&str>) -> String {
        format!(
            "({}//{})",
            s.a().unwrap().render(Arc::new(Self), ctx, false),
            s.b().unwrap()
        )
    }
    fn and(&self, s: ArcNode, ctx: Option<&str>) -> String {
        let mut renders = vec![];
        for n in s.nodes() {
            renders.push(n.render(Arc::new(Self), ctx, false));
        }
        renders.sort();
        format!("({})", renders.join(" and "))
    }

    fn to_arc(&self) -> Arc<dyn NodeOp> {
        Arc::new(Python)
    }
}
