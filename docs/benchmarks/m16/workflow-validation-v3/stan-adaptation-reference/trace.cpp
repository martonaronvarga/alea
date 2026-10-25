// Original harness executing the pinned Stan 2.38 public adaptation classes.
// No Alea headers, binaries or outputs are used.
#include <stan/mcmc/stepsize_adaptation.hpp>
#include <stan/mcmc/var_adaptation.hpp>
#include <stan/mcmc/covar_adaptation.hpp>
#include <iomanip>
#include <iostream>
#include <string>
int main(int argc, char** argv) {
  if (argc != 2) return 2;
  std::cout << std::setprecision(17);
  stan::callbacks::logger logger;
  if (std::string(argv[1]) == "schedule") {
    std::cout << "length,index,slow,end\n";
    for (unsigned n : {0,1,19,20,21,149,150,151,200,1000}) {
      stan::mcmc::var_adaptation variance(2);
      variance.set_window_params(n,75,50,25,logger);
      Eigen::VectorXd metric = Eigen::VectorXd::Ones(2);
      Eigen::VectorXd q = Eigen::VectorXd::Zero(2);
      for (unsigned i = 0; i < n; ++i) {
        std::cout << n << ',' << i << ',' << variance.adaptation_window()
                  << ',' << variance.end_adaptation_window() << '\n';
        variance.learn_variance(metric,q);
      }
    }
  } else if (std::string(argv[1]) == "trace") {
    std::cout << "kind,index,x,y,acceptance,step,c00,c01,c11\n";
    for (bool diagonal : {true,false}) {
      stan::mcmc::var_adaptation variance(2);
      stan::mcmc::covar_adaptation covariance(2);
      variance.set_window_params(200,75,50,25,logger);
      covariance.set_window_params(200,75,50,25,logger);
      stan::mcmc::stepsize_adaptation step;
      step.set_mu(std::log(10.*.3));step.set_delta(.8);
      double epsilon=.3;
      Eigen::VectorXd metric=Eigen::VectorXd::Ones(2);
      Eigen::MatrixXd matrix=Eigen::MatrixXd::Identity(2,2);
      for (int i=0;i<200;++i) {
        Eigen::VectorXd q(2);q << i%13-6,(i*7)%17-8;
        const double observations[]={.1,.5,.8,0.,1.,.95,.75};
        const double acceptance=observations[i%7];
        step.learn_stepsize(epsilon,acceptance);
        const bool update=diagonal ? variance.learn_variance(metric,q)
                                   : covariance.learn_covariance(matrix,q);
        if (update) {
          // Isolate the same recurrence/reset inputs as the BlackJAX fixture.
          // Full Stan NUTS instead runs a new metric-dependent step search here.
          step.complete_adaptation(epsilon);
          step.set_mu(std::log(10.*epsilon));step.restart();
        }
        std::cout << (diagonal ? "diagonal" : "dense") << ',' << i << ','
                  << q[0] << ',' << q[1] << ',' << acceptance << ',' << epsilon << ','
                  << (diagonal ? metric[0] : matrix(0,0)) << ','
                  << (diagonal ? 0. : matrix(0,1)) << ','
                  << (diagonal ? metric[1] : matrix(1,1)) << '\n';
      }
    }
  } else return 2;
}
