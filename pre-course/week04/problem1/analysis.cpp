#include "analysis.h"
#include "clock.h"

Analysis::Analysis(int32_t n_sample, int32_t max_sample, double epsilon, bool record_exp) {
  this->n_sample = n_sample;
  samples.assign(n_sample, 0);
  this->max_sample = max_sample;
  this->epsilon = epsilon;
  this->record_exp = record_exp;
  if (record_exp)
    exp_res.assign(max_sample, 0.0);
  sample_cnt = 0;
}

std::vector<double> Analysis::get_experiment_results() {
  return exp_res;
}

void Analysis::set_sample_element(double val) {
  int32_t p = 0;
  if (sample_cnt < n_sample) {
    p = sample_cnt;
    samples[p] = val;
  }
  else if (val < samples[n_sample-1]) {
    p = n_sample - 1;
    samples[p] = val;
  }
  if (record_exp)
    exp_res[sample_cnt] = val;
  sample_cnt += 1;
  // Insertion sort - O(n)
  while (p > 0 and samples[p-1] > samples[p]) {
    std::swap(samples[p], samples[p-1]);
    p -= 1;
  }
}

int32_t Analysis::has_converged() {
  // The experiment converge if the current experiment result not change compare with the previous 
  if (sample_cnt >= n_sample and (1 + epsilon)*samples[0] >= samples[n_sample-1])
    return sample_cnt;
  
  if (sample_cnt >= max_sample)
    return -1;

  return 0;
}

double Analysis::get_execution_cpu_cycles(func f, int32_t param1, int32_t param2) {
  double cycles;
  do {
    double c;
    start_counter();
    f(param1, param2);
    c = get_counter();
    set_sample_element(c);
  }
  while (!has_converged() && sample_cnt < max_sample);
  cycles = samples.front();

  return cycles;
}


double Analysis::get_execution_cpu_cycles(func1 f, const std::string& param1) {
  double cycles;
  do {
    double c;
    start_counter();
    f(param1);
    c = get_counter();
    set_sample_element(c);
  }
  while (!has_converged() && sample_cnt < max_sample);
  cycles = samples.front();

  return cycles;
}
