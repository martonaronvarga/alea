//! Binding-preparation proof retained for the later preparation cutover.
use super::*;
#[test]
fn binding_rules_reject_dynamic_dependencies_bad_arguments_and_extent_before_graph_mutation() {
    let mut b = RegionBuilder::new();
    let x = b.input("x", Shape::scalar()).unwrap();
    let data = b.data("data", Shape::scalar()).unwrap();
    let derived_data = b.unary(UnaryOp::Neg, data).unwrap();
    for (input, argument, elements) in [
        (x, 0, 1),
        (derived_data, 0, 1),
        (data, 1, 1),
        (data, 0, usize::MAX),
    ] {
        let nodes = b.nodes.len();
        let result = b.register(
            Signature {
                inputs: vec![BlockType::of(b.node(input).unwrap())],
                output: BlockType {
                    dtype: ElementType::Real,
                    shape: Shape::scalar(),
                },
                scratch: 0,
            },
            &[input],
            || {
                Ok(Kernel {
                    forward: Box::new(|_, out, _| {
                        out[0] = 0.;
                        Ok(())
                    }),
                    vjp: None,
                    binding: Some(BindingRule {
                        arguments: vec![argument].into_boxed_slice(),
                        elements,
                        prepare: Box::new(|_, _| panic!("invalid binding preparation executed")),
                    }),
                })
            },
        );
        assert!(matches!(result, Err(E::Shape)));
        assert_eq!(b.nodes.len(), nodes);
    }
    // A direct immutable constant is permitted, and its preparation may
    // itself be empty; neither requires a parameter or derivative buffer.
    let constant = b.constant(Shape::scalar(), vec![2.]).unwrap();
    let output = b
        .register(
            Signature {
                inputs: vec![BlockType::of(b.node(constant).unwrap())],
                output: BlockType {
                    dtype: ElementType::Real,
                    shape: Shape::scalar(),
                },
                scratch: 0,
            },
            &[constant],
            || {
                Ok(Kernel {
                    binding: Some(BindingRule {
                        arguments: vec![0].into_boxed_slice(),
                        elements: 0,
                        prepare: Box::new(|args, out| {
                            assert!(out.is_empty());
                            assert!(matches!(args.get(0)?, BindingValue::Real([2.])));
                            Ok(())
                        }),
                    }),
                    forward: Box::new(|args, out, _| {
                        assert!(args.prepared.is_empty());
                        out[0] = args.real(0)?[0];
                        Ok(())
                    }),
                    vjp: None,
                })
            },
        )
        .unwrap();
    let e = b.compile(output).unwrap().bind(&[("data", &[1.])]).unwrap();
    let (mut out, mut g) = ([17.], [19.]);
    e.value_and_vjp(&[1.], &[1.], &mut out, &mut g, &mut e.workspace())
        .unwrap();
    assert_eq!((out, g), ([2.], [0.])); // Unused x stays inactive.
}

#[test]
fn binding_preparation_reads_discrete_leaves_without_float_conversion() {
    let mut b = RegionBuilder::new();
    let flag = b
        .typed_data(Shape::scalar(), Data::Bool(vec![true].into()))
        .unwrap();
    let count = b
        .typed_data(
            Shape::scalar(),
            Data::Count(vec![9_007_199_254_740_993].into()),
        )
        .unwrap();
    let index = b
        .typed_data(Shape::scalar(), Data::Index(vec![usize::MAX].into()))
        .unwrap();
    let inputs = [flag, count, index];
    let signature = Signature {
        inputs: inputs
            .iter()
            .map(|&i| BlockType::of(b.node(i).unwrap()))
            .collect(),
        output: BlockType {
            dtype: ElementType::Real,
            shape: Shape::scalar(),
        },
        scratch: 0,
    };
    let out = b
        .register(signature, &inputs, || {
            Ok(Kernel {
                binding: Some(BindingRule {
                    arguments: vec![0, 1, 2].into(),
                    elements: 1,
                    prepare: Box::new(|args, out| {
                        assert!(matches!(args.get(0)?, BindingValue::Bool([true])));
                        assert!(matches!(
                            args.get(1)?,
                            BindingValue::Count([9_007_199_254_740_993])
                        ));
                        assert!(matches!(args.get(2)?, BindingValue::Index([usize::MAX])));
                        out[0] = 1.;
                        Ok(())
                    }),
                }),
                forward: Box::new(|args, out, _| {
                    out[0] = args.prepared[0];
                    Ok(())
                }),
                vjp: None,
            })
        })
        .unwrap();
    let bound = b.compile(out).unwrap().bind(&[]).unwrap();
    let mut value = [19.];
    bound
        .value_and_vjp(&[], &[1.], &mut value, &mut [], &mut bound.workspace())
        .unwrap();
    assert_eq!(value, [1.]);
}
