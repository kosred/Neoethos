mod layout;
mod tabs;
mod viewer;

pub use layout::WorkspaceState;
pub use tabs::{WorkspaceGroup, WorkspaceTab};
pub use viewer::{WorkspaceViewer, render_workspace};
