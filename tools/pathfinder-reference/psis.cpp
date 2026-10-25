// Execute Stan 2.38 PSIS without copying its implementation.
#include <stan/services/pathfinder/psis.hpp>
#include <iostream>
#include <iomanip>
int main() {
  std::cout << std::setprecision(17);
  stan::callbacks::logger logger;
  for (int mode=0;mode<3;++mode) {
    const int n=mode==0 ? 64 : 256;
    Eigen::ArrayXd logs(n);
    for(int i=0;i<n;++i) {
      const double p=(i+0.5)/n;
      logs[i]=mode==0 ? 2.0*p : (mode==1 ? -0.75*std::log1p(-p) : 0.3*std::log(p)-1.2*std::log1p(-p));
    }
    const int tail=std::min(0.2*n,3.0*std::sqrt(n));
    auto weights=stan::services::psis::psis_weights(logs,tail,logger);
    auto sorted=stan::services::psis::internal::largest_n_elements((logs-logs.maxCoeff()).eval(),tail+1);
    auto fit=stan::services::psis::internal::psis_smooth_tail(sorted.first.tail(tail),sorted.first[0]);
    std::cout << "case " << mode << ' ' << n << ' ' << tail << ' ' << fit.second << '\n';
    for(int i=0;i<n;++i) std::cout << logs[i] << ' ' << weights[i] << '\n';
  }
}
