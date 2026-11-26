//! Directional composition over the native controller's admitted frames.
use super::*;

pub(in crate::model::tensor::execute) enum Control<'a> {
    Cond([Program<'a>; 2]),
    Scan {
        first: Program<'a>,
        steady: Option<Program<'a>>,
    },
}

impl<'a> Control<'a> {
    pub(super) fn compile(
        rule: Rule<'a>,
        ranges: &[Option<Range<usize>>],
        curvature: bool,
    ) -> Result<Option<Self>, E> {
        if !matches!(rule, Rule::Cond(_)) && !matches!(rule, Rule::Scan(s) if s.steps != 0) {
            return Ok(None);
        }
        let mut mask: Vec<_> = ranges.iter().map(Option::is_some).collect();
        Ok(match rule {
            Rule::Cond(c) => Some(Self::Cond([
                Program::compile_mode(&c.yes.plan, &mask, curvature)?,
                Program::compile_mode(&c.no.plan, &mask, curvature)?,
            ])),
            Rule::Scan(s) if s.steps != 0 => {
                let first = Program::compile_mode(&s.body.plan, &mask, curvature)?;
                // The region has one whole carry activity bit. Its transfer
                // depends on the old bit and the same row/capture mask at every
                // step, so after the first transition the mask is stationary.
                let carry = first.output.is_some();
                let steady = if s.steps > 1 && mask[0] != carry {
                    mask[0] = carry;
                    Some(Program::compile_mode(&s.body.plan, &mask, curvature)?)
                } else {
                    None
                };
                Some(Self::Scan { first, steady })
            }
            _ => None,
        })
    }

    pub(super) fn scratch_elements(&self) -> usize {
        match self {
            Self::Cond([yes, no]) => yes.tangent_elements().max(no.tangent_elements()),
            Self::Scan { first, steady } => first
                .tangent_elements()
                .max(steady.as_ref().map_or(0, Program::tangent_elements)),
        }
    }

    pub(in crate::model::tensor::execute) fn frame_program(&self, index: usize) -> &Program<'a> {
        match self {
            Self::Cond(programs) => &programs[index],
            Self::Scan { first, steady } => {
                if index == 0 {
                    first
                } else {
                    steady.as_ref().unwrap_or(first)
                }
            }
        }
    }

    pub(super) fn cond<'d>(
        &self,
        plan: &Plan,
        c: &Cond,
        region: &mut RegionWork,
        directions: impl Fn(usize) -> Option<&'d [f64]>,
        (output, scratch): (&mut [f64], &mut [f64]),
        curvature: Option<&mut curvature::Region>,
    ) -> Result<(), E> {
        let Self::Cond(programs) = self else {
            unreachable!("compiled condition")
        };
        let choice = region.branch.ok_or(E::Shape)?;
        let program = programs.get(choice).ok_or(E::Shape)?;
        let binding = if choice == 0 { &c.yes } else { &c.no };
        let guard = plan
            .facts
            .guard(region.site, FrameInstance::Branch(choice))?;
        let frame = region.residual(guard, choice)?;
        let arguments = (0..c.args.len()).map(directions);
        program.child_direction(
            (binding, frame),
            (0..).zip(arguments),
            output,
            scratch,
            curvature.map(|tape| &mut tape.frames[choice]),
        )?;
        Ok(())
    }

    pub(super) fn scan<'d>(
        &self,
        plan: &Plan,
        s: &Scan,
        (parent, region): (&[f64], &mut RegionWork),
        directions: impl Fn(usize) -> Option<&'d [f64]>,
        (output, scratch): (&mut [f64], &mut [f64]),
        mut curvature: Option<&mut curvature::Region>,
    ) -> Result<(), E> {
        let interval = region.checkpoint.as_ref().map_or(0, |c| c.plan.width());
        let replay = if let Some(checkpoint) = &region.checkpoint {
            let carry = output.len();
            // Full forward admission already saved the initial boundary. Run
            // the existing checkpoint replay in forward order, consuming each
            // version before another iteration can overwrite its frame.
            region.args[..carry].copy_from_slice(&checkpoint.carries[..carry]);
            Some(
                plan.facts
                    .lowered
                    .call(region.site.node)
                    .expect("checked replay")
                    .native_primals(parent)
                    .inputs,
            )
        } else {
            None
        };
        for step in 0..s.steps {
            if let Some(inputs) = replay {
                if s.body.plan.execution.is_some() {
                    scan_step::<true>(s, inputs, region, step)?;
                } else {
                    scan_step::<false>(s, inputs, region, step)?;
                }
            }
            if let Some(tape) = &mut curvature
                && interval != 0
                && step % interval == 0
            {
                let start = step / interval * output.len();
                tape.boundaries[start..start + output.len()].copy_from_slice(output);
            }
            self.scan_direction(
                (plan, s),
                region,
                step,
                &directions,
                (output, scratch),
                curvature
                    .as_deref_mut()
                    .map(|tape| &mut tape.frames[scan_frame(interval, step)]),
            )?;
        }
        Ok(())
    }

    pub(in crate::model::tensor::execute) fn scan_direction<'d>(
        &self,
        (plan, s): (&Plan, &Scan),
        region: &mut RegionWork,
        step: usize,
        directions: impl Fn(usize) -> Option<&'d [f64]>,
        (carry, scratch): (&mut [f64], &mut [f64]),
        mut curvature: Option<&mut curvature::Frame>,
    ) -> Result<(), E> {
        let interval = region.checkpoint.as_ref().map_or(0, |c| c.plan.width());
        let index = scan_frame(interval, step);
        let guard = plan
            .facts
            .guard(region.site, FrameInstance::Iteration(step))?;
        let frame = region.residual(guard, index)?;
        let row = directions(1).map(|x| &x[step * s.row..(step + 1) * s.row]);
        let program = self.frame_program(index);
        let arguments =
            std::iter::once(row).chain((0..s.captures.len()).map(|i| directions(i + 2)));
        if let Some(tape) = &mut curvature {
            program.load_direction(0, Some(carry), &mut tape.tangents);
        } else {
            program.load_direction(0, Some(carry), scratch);
        }
        program.child_direction(
            (&s.body, frame),
            (1..).zip(arguments),
            carry,
            scratch,
            curvature,
        )
    }
}

impl Program<'_> {
    fn child_direction<'d>(
        &self,
        (binding, workspace): (&BoundProgram, &mut TensorWorkspace),
        arguments: impl Iterator<Item = (usize, Option<&'d [f64]>)>,
        output: &mut [f64],
        scratch: &mut [f64],
        curvature: Option<&mut curvature::Frame>,
    ) -> Result<(), E> {
        if let Some(frame) = curvature {
            for (i, direction) in arguments {
                self.load_direction(i, direction, &mut frame.tangents);
            }
            self.directional_frame(binding, workspace, frame)?;
            self.copy_result(output, &frame.tangents);
        } else {
            for (i, direction) in arguments {
                self.load_direction(i, direction, scratch);
            }
            self.run_prepared(binding, workspace, scratch)?;
            self.copy_result(output, scratch);
        }
        Ok(())
    }
}
