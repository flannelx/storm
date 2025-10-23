pub mod list;
pub mod view;

use view::View;

#[derive(Clone, Debug)]
pub struct Shape {
    views: Vec<View>,
}
