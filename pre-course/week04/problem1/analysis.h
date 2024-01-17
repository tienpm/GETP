#ifndef ANALYSIS_H
#define ANALYSIS_H
#include <vector>
#include <cstdint>
#include <string>

typedef int64_t (*func)(int32_t, int32_t);
typedef void (*func1)(const std::string&);

class Analysis {
  // Equivalence Testing
 private:
  std::vector<double> exp_res;
  std::vector<double> samples;
  int32_t n_sample, max_sample, sample_cnt;
  double epsilon;
  bool record_exp;
 protected:
  int32_t has_converged();
  void set_sample_element(double val);
 public:
  Analysis(int32_t n_sample, int32_t max_sample, double epsilon, bool record_exp);
  std::vector<double> get_experiment_results();
  double get_execution_cpu_cycles(func f, int32_t param1, int32_t param2);
  double get_execution_cpu_cycles(func1 f, const std::string& param1);
};

#endif  // ANALYSIS_H
