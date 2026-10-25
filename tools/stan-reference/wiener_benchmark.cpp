// Operation-matched Stan worker. Build/source provenance is sealed by the driver.
#include <stan/math/rev.hpp>
#include <stan/math/prim/prob/wiener_full_lpdf.hpp>
#include <array>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc != 5 && argc != 12) return 2;
  const int family = std::stoi(argv[1]);
  const std::string operation = argv[2];
  const int repetitions = std::stoi(argv[3]);
  const int size = std::stoi(argv[4]);
  const bool batch = operation.find("batch") == 0;
  const bool gradient = operation != "logp";
  const int count = batch ? size : 1;
  const double precision = family == 7 ? 1e-4 : 1e-12;
  std::array<double,7> parameters{1.6,0.2,0.45,0.7,0.1,0.05,0.1};
  if (argc==12) for (int j=0;j<7;++j) parameters[j]=std::stod(argv[j+5]);
  double value = 0.0;
  std::array<double, 7> scores{};
  std::vector<double> upper, lower;
  if (batch) for (int i=0;i<count;++i) {
    const double rt=0.35+1.5*((i*37)%1000)/1000.0;
    (i%3==0 ? lower : upper).push_back(rt);
  }
  auto evaluate = [&]() {
    value = 0.0; scores.fill(0.0);
    if (operation=="batch_logp") {
      const auto& p=parameters;
      for (int i=0;i<count;++i) {
        const double rt=0.35+1.5*((i*37)%1000)/1000.0;
        const double w=i%3==0 ? 1.0-p[2] : p[2];
        const double v=i%3==0 ? -p[3] : p[3];
        if (family==4) value+=stan::math::wiener_lpdf(rt,p[0],p[1],w,v);
        else if (family==5) value+=stan::math::wiener_lpdf(rt,p[0],p[1],w,v,p[4],precision);
        else value+=stan::math::wiener_lpdf(rt,p[0],p[1],w,v,p[4],p[5],p[6],precision);
      }
      asm volatile("" : : "g"(value) : "memory");
      return;
    }
    if (batch) {
      // Share parameters and perform one reverse sweep for either batch API.
      const auto& p=parameters;
      std::array<stan::math::var,7> q{p[0],p[1],p[2],p[3],p[4],p[5],p[6]};
      stan::math::var reflected_w=1.0-q[2], reflected_v=-q[3], result;
      if (operation!="batch_vectorized") {
        result=0.0;
        for (int i=0;i<count;++i) {
          const double rt=0.35+1.5*((i*37)%1000)/1000.0;
          const auto w=i%3==0 ? reflected_w : q[2];
          const auto v=i%3==0 ? reflected_v : q[3];
          if (family==4) result+=stan::math::wiener_lpdf(rt,q[0],q[1],w,v);
          else if (family==5) result+=stan::math::wiener_lpdf(rt,q[0],q[1],w,v,q[4],precision);
          else result+=stan::math::wiener_lpdf(rt,q[0],q[1],w,v,q[4],q[5],q[6],precision);
        }
      } else if (family==4) result=stan::math::wiener_lpdf(upper,q[0],q[1],q[2],q[3])
        +stan::math::wiener_lpdf(lower,q[0],q[1],reflected_w,reflected_v);
      else if (family==5) result=stan::math::wiener_lpdf(upper,q[0],q[1],q[2],q[3],q[4],precision)
        +stan::math::wiener_lpdf(lower,q[0],q[1],reflected_w,reflected_v,q[4],precision);
      else result=stan::math::wiener_lpdf(upper,q[0],q[1],q[2],q[3],q[4],q[5],q[6],precision)
        +stan::math::wiener_lpdf(lower,q[0],q[1],reflected_w,reflected_v,q[4],q[5],q[6],precision);
      result.grad(); value=result.val();
      for (int j=0;j<family;++j) scores[j]=q[j].adj();
      stan::math::recover_memory();
      asm volatile("" : : "g"(value) : "memory");
      return;
    }
    for (int i = 0; i < count; ++i) {
      const double rt = batch ? 0.35 + 1.5 * ((i * 37) % 1000) / 1000.0 : 1.2;
      const bool lower = batch && i % 3 == 0;
      const double sign = lower ? -1.0 : 1.0;
      const std::array<double, 7> p{1.6, 0.2, lower ? 0.55 : 0.45, sign * 0.7, 0.1, 0.05, 0.1};
      if (!gradient) {
        if (family == 4) value += stan::math::wiener_lpdf(rt,p[0],p[1],p[2],p[3]);
        else if (family == 5) value += stan::math::wiener_lpdf(rt,p[0],p[1],p[2],p[3],p[4],precision);
        else value += stan::math::wiener_lpdf(rt,p[0],p[1],p[2],p[3],p[4],p[5],p[6],precision);
      } else {
        std::array<stan::math::var,7> q{p[0],p[1],p[2],p[3],p[4],p[5],p[6]};
        stan::math::var result;
        if (family == 4) result = stan::math::wiener_lpdf(rt,q[0],q[1],q[2],q[3]);
        else if (family == 5) result = stan::math::wiener_lpdf(rt,q[0],q[1],q[2],q[3],q[4],precision);
        else result = stan::math::wiener_lpdf(rt,q[0],q[1],q[2],q[3],q[4],q[5],q[6],precision);
        result.grad(); value += result.val();
        for (int j = 0; j < family; ++j) scores[j] += q[j].adj() * ((j == 2 || j == 3) ? sign : 1.0);
        stan::math::recover_memory();
      }
    }
    asm volatile("" : : "g"(value) : "memory");
  };
  for (int i = 0; i < 6; ++i) evaluate();
  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repetitions; ++i) evaluate();
  const auto elapsed = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now()-start).count();
  std::cout << std::setprecision(17) << "{\"family\":" << family << ",\"operation\":\"" << operation
    << "\",\"repetitions\":" << repetitions << ",\"observations\":" << count << ",\"elapsed_ns\":" << elapsed
    << ",\"logp\":" << value << ",\"scores\":[";
  for (int j=0;j<7;++j) std::cout << (j ? "," : "") << scores[j];
  std::cout << "]}\n";
}
