// Appended only in the isolated pinned upstream checkout. No algorithm is replaced.
impl LowRankMassMatrixStrategy {
    pub fn emit_alea_updates() {
        use crate::math::{CpuMath, test_logps::NormalLogp};
        use crate::transform::Transformation;
        type Backend = CpuMath<NormalLogp>;
        fn vector(values: impl Iterator<Item=f64>) -> String {
            values.map(|v|format!("{v:.17e}")).collect::<Vec<_>>().join(";")
        }
        println!("# nuts-rs a762aae513bf7bb5ccb27ffc9a938a923fc837ec actual update_estimators/switch/update/transform actions");
        println!("event,dim,good,switched,current,background,q,score,scales,ridge,threshold,rank,inverse_mass,logdet_mass,version");
        for dim in [2,8] {
            let settings=LowRankSettings { gamma:0.01,eigval_cutoff:1.1,..Default::default() };
            let mut math=CpuMath::new(NormalLogp::new(dim,0.0));
            let mut strategy=Self::new(dim,settings);
            let mut metric=LowRankMassMatrix::new(&mut math,settings);
            for event in 1..=16 {
                let q=Col::from_fn(dim,|j| (((event+1)*(j+3)+event*event)%17) as f64/4.0-2.0);
                let g=Col::from_fn(dim,|j| -0.7*q[j]+0.3*q[(j+1)%dim]+((event*(j+1)+3)%7) as f64/10.0);
                let good=event%5!=0;
                let collector=DrawGradCollector {draw:q,grad:g,is_good:good};
                <Self as MassMatrixAdaptStrategy<Backend>>::update_estimators(&mut strategy,&mut math,&collector);
                let switched=event==6 || event==12;
                if switched { <Self as MassMatrixAdaptStrategy<Backend>>::switch(&mut strategy,&mut math); }
                let n=strategy.draws.len();
                if n<3 {continue;}
                // Exercise the actual updater and the installed transformation.
                strategy.update(&mut math,&mut metric);
                let draws=Mat::from_fn(dim,n,|i,j|strategy.draws[j][i]);
                let grads=Mat::from_fn(dim,n,|i,j|strategy.grads[j][i]);
                let (scales,_,values,_,_)=strategy.compute_update(draws.clone(),grads.clone()).unwrap();
                let mut base=Col::zeros(dim);
                let mut grad=Col::zeros(dim);
                let mut tg=Col::zeros(dim);
                let (_,logdet)=metric.init_from_transformed_position(&mut math,&mut base,&mut grad,&Col::zeros(dim),&mut tg).unwrap();
                let mut factor=Mat::zeros(dim,dim);
                for j in 0..dim {
                    let z=Col::from_fn(dim,|i|f64::from(i==j));
                    let mut out=Col::zeros(dim);
                    metric.init_from_transformed_position(&mut math,&mut out,&mut grad,&z,&mut tg).unwrap();
                    for i in 0..dim {factor[(i,j)]=out[i]-base[i];}
                }
                let imm=&factor*factor.transpose();
                println!("{event},{dim},{good},{switched},{},{},{},{},{},0.01,1.1,{},{},{:.17e},{}",
                    <Self as MassMatrixAdaptStrategy<Backend>>::current_count(&strategy),
                    <Self as MassMatrixAdaptStrategy<Backend>>::background_count(&strategy),
                    vector((0..n*dim).map(|k|draws[(k%dim,k/dim)])),
                    vector((0..n*dim).map(|k|grads[(k%dim,k/dim)])),vector(scales.iter().copied()),values.nrows(),
                    vector((0..dim*dim).map(|k|imm[(k/dim,k%dim)])),2.0*logdet,metric.transformation_id(&mut math));
            }
        }
    }
}
