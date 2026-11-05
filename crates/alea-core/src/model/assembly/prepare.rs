//! Shared pure scalar preparation; never reorder factor admission or reverse edges.
use super::{Op, PrimitiveMetadata};
use std::{any::TypeId, collections::HashMap};

#[derive(Clone, Copy)]
pub(super) struct Rule {
    pub argument: usize,
    pub prepare: fn(f64) -> f64,
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
}
impl Plan {
    pub(super) fn new(
        nodes: &[Op],
        primitives: &[(usize, PrimitiveMetadata)],
        callbacks: usize,
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
        if !groups.values().any(|group| group.len() > 1) {
            return None;
        }
        let mut mapping = vec![(usize::MAX, 0); callbacks];
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
            entries,
        }))
    }
    pub(super) fn begin_evaluation(&mut self) {
        for entry in &mut self.entries {
            if !entry.constant {
                entry.pending = true;
            }
        }
    }
    pub(super) fn value(&mut self, callback: usize, inputs: &[f64]) -> Option<f64> {
        let (index, argument) = self.callbacks[callback];
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
