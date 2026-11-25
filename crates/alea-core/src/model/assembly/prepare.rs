//! Shared pure preparation for scalar factors and typed native block calls.
use super::{Op, PrimitiveMetadata};
use std::{
    any::{Any, TypeId},
    collections::HashMap,
};

#[derive(Clone, Copy)]
pub(super) struct Rule {
    pub(super) argument: usize,
    pub(super) prepare: fn(f64) -> f64,
}
struct Entry {
    prepare: fn(f64) -> f64,
    bits: Option<u64>,
    value: f64,
    constant: bool,
    pending: bool,
}
pub(super) struct Plan {
    callbacks: Box<[(usize, usize)]>,
    entries: Vec<Entry>,
    pub(super) blocks: BlockEntries,
}
impl Plan {
    pub(super) fn new(
        nodes: &[Op],
        primitives: &[(usize, PrimitiveMetadata)],
        callbacks: usize,
        active: Option<&[bool]>,
    ) -> Option<Box<Self>> {
        #[derive(Hash, PartialEq, Eq)]
        enum Input {
            Constant(u64),
            Node(usize),
        }
        type Key = (TypeId, Input, usize);
        let mut groups: HashMap<Key, Vec<(usize, Rule, bool)>> = HashMap::new();
        for (node, primitive) in primitives {
            let Some(rule) = primitive.preparation else {
                continue;
            };
            let Op::Factor {
                inputs, callback, ..
            } = &nodes[*node]
            else {
                unreachable!("registered primitive")
            };
            let source = inputs[rule.argument];
            let input = match nodes[source] {
                Op::Constant(v) => Input::Constant(v.to_bits()),
                _ => Input::Node(source),
            };
            // Process-local identity only, not a serialized address or replay
            // contract. Preparation is pure; identical-code folding is harmless.
            groups
                .entry((
                    primitive.descriptor.as_ref().type_id(),
                    input,
                    rule.prepare as usize,
                ))
                .or_default()
                .push((*callback, rule, matches!(nodes[source], Op::Constant(_))));
        }
        let blocks = BlockEntries::new(nodes, active);
        let shared_scalar = groups.values().any(|group| group.len() > 1);
        if !shared_scalar && blocks.slots.is_empty() {
            return None;
        }
        let mut mapping = vec![(usize::MAX, 0); if shared_scalar { callbacks } else { 0 }];
        let mut entries = Vec::new();
        // Group iteration order has no effect on evaluation/reduction order.
        for group in groups.values().filter(|group| group.len() > 1) {
            let index = entries.len();
            entries.push(Entry {
                prepare: group[0].1.prepare,
                bits: None,
                value: 0.,
                constant: group[0].2,
                pending: true,
            });
            for &(callback, rule, _) in group {
                mapping[callback] = (index, rule.argument);
            }
        }
        Some(Box::new(Self {
            callbacks: mapping.into_boxed_slice(),
            blocks,
            entries,
        }))
    }
    pub(super) fn begin_evaluation(&mut self) {
        self.blocks.begin_evaluation();
        for entry in &mut self.entries {
            if !entry.constant {
                entry.pending = true;
            }
        }
    }
    pub(super) fn value(&mut self, callback: usize, inputs: &[f64]) -> Option<f64> {
        let &(index, argument) = self.callbacks.get(callback)?;
        if index == usize::MAX {
            return None;
        }
        let input = inputs[argument];
        let entry = &mut self.entries[index];
        if entry.pending {
            if entry.bits != Some(input.to_bits()) {
                entry.value = (entry.prepare)(input);
                entry.bits = Some(input.to_bits());
            }
            entry.pending = false;
        }
        Some(entry.value)
    }
}

// The recipe sees only its declared scalar, never the descriptor or workspace.
pub(super) struct BlockRule {
    pub(super) argument: usize,
    identity: (TypeId, TypeId),
    slot: Box<dyn Fn() -> Box<dyn BlockSlot> + Send + Sync>,
}
impl BlockRule {
    pub(super) fn new<P, F>(argument: usize, prepare: F) -> Self
    where
        P: Copy + Send + Sync + 'static,
        F: Fn(f64) -> P + Copy + Send + Sync + 'static,
    {
        debug_assert_eq!(size_of::<F>(), 0);
        Self {
            argument,
            identity: (TypeId::of::<P>(), TypeId::of::<F>()),
            slot: Box::new(move || {
                Box::new(TypedSlot {
                    prepare,
                    value: None,
                })
            }),
        }
    }
}
trait BlockSlot: Send {
    fn clear(&mut self);
    fn value(&mut self, input: f64) -> &(dyn Any + Send + Sync);
}
struct TypedSlot<P, F> {
    prepare: F,
    value: Option<P>,
}
impl<P, F> BlockSlot for TypedSlot<P, F>
where
    P: Copy + Send + Sync + 'static,
    F: Fn(f64) -> P + Send,
{
    fn clear(&mut self) {
        self.value = None;
    }
    fn value(&mut self, input: f64) -> &(dyn Any + Send + Sync) {
        self.value.get_or_insert_with(|| (self.prepare)(input))
    }
}
#[derive(Default)]
pub(super) struct BlockEntries {
    mapping: Vec<usize>,
    slots: Vec<Box<dyn BlockSlot>>,
}
impl BlockEntries {
    fn new(nodes: &[Op], active: Option<&[bool]>) -> Self {
        let mut entries = Self::default();
        let mut groups = HashMap::new();
        for (i, node) in nodes.iter().enumerate() {
            if active.is_some_and(|active| !active[i]) {
                continue;
            }
            let Op::Region(call) = node else { continue };
            let Some(rule) = &call.preparation else {
                continue;
            };
            let key = (call.inputs[rule.argument], rule.identity);
            let index = *groups.entry(key).or_insert_with(|| {
                let index = entries.slots.len();
                entries.slots.push((rule.slot)());
                index
            });
            entries.mapping.resize(call.slot + 1, usize::MAX);
            entries.mapping[call.slot] = index;
        }
        entries
    }
    fn begin_evaluation(&mut self) {
        for slot in &mut self.slots {
            slot.clear();
        }
    }
    pub(super) fn value(&mut self, call: usize, input: f64) -> Option<&(dyn Any + Send + Sync)> {
        let index = *self.mapping.get(call)?;
        Some(self.slots.get_mut(index)?.value(input))
    }
}
