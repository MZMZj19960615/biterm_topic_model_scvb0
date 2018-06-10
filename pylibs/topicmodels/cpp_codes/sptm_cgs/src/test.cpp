#include <iostream>
#include <vector>
#include "core.h"
#include "macros.h"
#include <cmath>
#include <cinttypes>
#include <iomanip>

int main() {
  std::default_random_engine urng(123);

  int n_topics = 10;
  int n_word_types = 1000;
  int n_pseudo_docs = 3;
  topicmodels::docs_t docs = {
    {0, 2, 4, 1},
    {1, 4, 0},
    {0, 4, 0, 10, 20, 5, 1},
    {10, 20, 5, 1},
    {1, 0},
    {10, 20, 33, 3, 4, 5},
  };
  std::uint32_t seed = 123;
  topicmodels::SPTM<std::default_random_engine> model(n_topics, n_word_types, n_pseudo_docs, docs, seed);
  FOR(i, 10) {
    std::cout << model.log_marginalized_joint() << std::endl;
    model.update();
  }

  return 0;
}
