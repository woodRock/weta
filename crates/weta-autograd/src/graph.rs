//! The recorded graph and the backward pass.
//!
//! Each op on a tensor that requires grad creates one [`Node`]: the op
//! (holding whatever its backward needs) plus an [`Edge`] per input. Edges
//! only point backwards, from an output to the nodes that made its inputs,
//! so the graph is a DAG with no reference cycles. The output tensor holds
//! an `Arc` to its node, and the graph is freed when the last tensor that
//! refers to it is dropped (or its saved tensors are released after
//! `backward`), with no global tape to clear.
//!
//! `backward` sorts the nodes reachable from the loss topologically *once*,
//! then visits them in reverse. A node used by several later ops has all of
//! their gradient contributions summed before it is visited, so its own
//! backward runs exactly once (fixing the C++ version, whose recursive
//! `parent->backward()` passed gradients down twice through skip
//! connections).

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use smallvec::SmallVec;
use weta_core::{DType, Device};

use crate::{Inner, Tensor};

/// A differentiable operation's backward pass.
///
/// Implement this to add a new op without touching the core: compute the
/// forward result on raw tensors, then call [`Tensor::from_op`] with the op
/// and its inputs. `backward` is written with ordinary [`Tensor`] ops, so it
/// runs on every backend, and is itself differentiable.
pub trait Op: Send + Sync {
    fn name(&self) -> &'static str;
    /// Gradients for each input, given the output's gradient. Return `None`
    /// for an input that needs no gradient (or has none, like an index).
    fn backward(&self, ctx: &BackwardCtx, grad_out: &Tensor) -> Vec<Option<Tensor>>;
}

/// Shape, dtype and device of an op's input, kept so backward can produce
/// gradients of the right shape without holding on to the input itself.
#[derive(Clone, Debug)]
pub struct InputMeta {
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub device: Device,
    pub requires_grad: bool,
}

/// What a backward pass may need to know besides the output gradient.
pub struct BackwardCtx<'a> {
    pub inputs: &'a [InputMeta],
}

impl BackwardCtx<'_> {
    pub fn input_shape(&self, i: usize) -> &[usize] {
        &self.inputs[i].shape
    }
    pub fn needs_grad(&self, i: usize) -> bool {
        self.inputs[i].requires_grad
    }
}

/// Where an input's gradient goes.
pub(crate) enum Edge {
    /// The input doesn't require grad.
    None,
    /// The input was made by another op: pass the gradient on to it.
    Node(Arc<Node>),
    /// The input is a leaf (e.g. a parameter): accumulate into its `.grad()`.
    Leaf(Arc<Inner>),
}

pub struct Node {
    /// Taken (and so freed, with its saved tensors) after backward unless
    /// the graph is retained.
    op: Mutex<Option<Box<dyn Op>>>,
    name: &'static str,
    pub(crate) inputs: SmallVec<[Edge; 2]>,
    meta: SmallVec<[InputMeta; 2]>,
    /// Creation order, used to break ties deterministically.
    seq: u64,
}

static SEQ: AtomicU64 = AtomicU64::new(0);

impl Node {
    pub(crate) fn new(op: Box<dyn Op>, inputs: SmallVec<[Edge; 2]>, meta: SmallVec<[InputMeta; 2]>) -> Node {
        let name = op.name();
        Node { op: Mutex::new(Some(op)), name, inputs, meta, seq: SEQ.fetch_add(1, Ordering::Relaxed) }
    }
    pub fn name(&self) -> &'static str {
        self.name
    }
}

// Dropping a long chain of nodes recursively (node → its input's node → ...)
// would overflow the stack on deep graphs, so nodes are released through a
// thread-local queue instead: each drop only pushes its children.
thread_local! {
    static DROP_QUEUE: RefCell<Vec<Arc<Node>>> = const { RefCell::new(Vec::new()) };
    static DRAINING: Cell<bool> = const { Cell::new(false) };
}

pub(crate) fn defer_drop(node: Arc<Node>) {
    let queued = DROP_QUEUE.try_with(|q| q.borrow_mut().push(node.clone())).is_ok();
    drop(node);
    if !queued || DRAINING.with(|d| d.replace(true)) {
        return;
    }
    while let Some(n) = DROP_QUEUE.with(|q| q.borrow_mut().pop()) {
        drop(n);
    }
    DRAINING.with(|d| d.set(false));
}

impl Drop for Node {
    fn drop(&mut self) {
        for e in self.inputs.drain(..) {
            if let Edge::Node(n) = e {
                defer_drop(n);
            }
        }
        // The op (and the tensors it saved) drops after this, and those
        // tensors' own nodes go through the queue too (see `Inner::drop`).
    }
}

// ---------------------------------------------------------------------------
// Grad mode
// ---------------------------------------------------------------------------

thread_local! {
    static GRAD_ENABLED: Cell<bool> = const { Cell::new(true) };
}

pub fn is_grad_enabled() -> bool {
    GRAD_ENABLED.with(|g| g.get())
}

/// Restores the previous grad mode when dropped.
#[must_use = "recording resumes as soon as the guard is dropped; bind it with `let _guard = no_grad();`"]
pub struct GradModeGuard {
    prev: bool,
}

impl Drop for GradModeGuard {
    fn drop(&mut self) {
        GRAD_ENABLED.with(|g| g.set(self.prev));
    }
}

pub fn set_grad_enabled(enabled: bool) -> GradModeGuard {
    let prev = GRAD_ENABLED.with(|g| g.replace(enabled));
    GradModeGuard { prev }
}

/// Stops recording until the guard is dropped, like `with torch.no_grad():`.
///
/// ```
/// # use weta_autograd::{no_grad, Tensor};
/// let w = Tensor::ones(&[2]).requires_grad_(true);
/// {
///     let _guard = no_grad();
///     assert!(!(&w * 2.0).requires_grad());
/// }
/// assert!((&w * 2.0).requires_grad());
/// ```
pub fn no_grad() -> GradModeGuard {
    set_grad_enabled(false)
}

pub fn enable_grad() -> GradModeGuard {
    set_grad_enabled(true)
}

// ---------------------------------------------------------------------------
// Backward
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, Debug, Default)]
pub struct BackwardOptions {
    /// Keep saved tensors so backward can run through this graph again.
    pub retain_graph: bool,
    /// Record the backward pass itself, for higher-order gradients.
    pub create_graph: bool,
}

/// Nodes reachable from `root`, in topological order (inputs first).
/// Iterative, so deep graphs can't overflow the stack.
fn topo_order(root: &Arc<Node>) -> Vec<Arc<Node>> {
    let mut order = Vec::new();
    let mut visited: HashMap<*const Node, ()> = HashMap::new();
    let mut stack: Vec<(Arc<Node>, usize)> = vec![(root.clone(), 0)];
    visited.insert(Arc::as_ptr(root), ());
    while let Some((node, child)) = stack.pop() {
        // Find the next unvisited child that is itself a node.
        let mut next = child;
        let mut pushed = false;
        while next < node.inputs.len() {
            if let Edge::Node(c) = &node.inputs[next] {
                if visited.insert(Arc::as_ptr(c), ()).is_none() {
                    let c = c.clone();
                    stack.push((node.clone(), next + 1));
                    stack.push((c, 0));
                    pushed = true;
                    break;
                }
            }
            next += 1;
        }
        if !pushed {
            order.push(node);
        }
    }
    order
}

fn accumulate(slot: &mut Option<Tensor>, g: Tensor) {
    *slot = Some(match slot.take() {
        Some(prev) => &prev + &g,
        None => g,
    });
}

pub(crate) fn run_backward(root: &Tensor, grad: Tensor, opts: BackwardOptions) {
    let _mode = set_grad_enabled(opts.create_graph);
    let Some(root_node) = root.grad_fn() else {
        // A leaf: its gradient is the seed itself.
        if root.requires_grad() {
            root.accumulate_grad(grad);
            return;
        }
        panic!("element 0 of tensors does not require grad and does not have a grad_fn");
    };
    let order = topo_order(&root_node);
    let mut grads: HashMap<*const Node, Tensor> = HashMap::new();
    grads.insert(Arc::as_ptr(&root_node), grad);
    // Reverse topological order: every node's consumers run before it, so
    // its gradient is complete when we reach it.
    let mut order = order;
    order.sort_by_key(|n| std::cmp::Reverse(n.seq));
    for node in &order {
        let Some(g) = grads.remove(&Arc::as_ptr(node)) else { continue };
        let op_guard = node.op.lock().unwrap_or_else(|e| e.into_inner());
        let op = op_guard.as_ref().unwrap_or_else(|| {
            panic!(
                "{}: trying to backward through the graph a second time; \
                 use backward_with(BackwardOptions {{ retain_graph: true, .. }}) the first time",
                node.name
            )
        });
        let ctx = BackwardCtx { inputs: &node.meta };
        let input_grads = op.backward(&ctx, &g);
        drop(op_guard);
        if !opts.retain_graph {
            node.op.lock().unwrap_or_else(|e| e.into_inner()).take();
        }
        assert_eq!(input_grads.len(), node.inputs.len(), "{}: backward returned the wrong number of gradients", node.name);
        for (i, (edge, gi)) in node.inputs.iter().zip(input_grads).enumerate() {
            let Some(gi) = gi else { continue };
            let want = &node.meta[i].shape;
            assert_eq!(gi.shape(), &want[..], "{}: gradient for input {i} has shape {:?}, expected {want:?}", node.name, gi.shape());
            match edge {
                Edge::None => {}
                Edge::Node(n) => {
                    let mut slot = grads.remove(&Arc::as_ptr(n));
                    accumulate(&mut slot, gi);
                    grads.insert(Arc::as_ptr(n), slot.unwrap());
                }
                Edge::Leaf(inner) => Tensor::from_inner(inner.clone()).accumulate_grad(gi),
            }
        }
    }
}
