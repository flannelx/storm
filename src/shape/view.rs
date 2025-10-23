use super::list::ShapeVec;

#[derive(Clone, Debug)]
pub struct View {
    pub shape: Vec<ShapeVec>,
    pub stride: Vec<ShapeVec>,
    pub mask: Option<Vec<(isize, isize)>>,
}
