#ifndef MY_MACROS__H


#define FOR(i, n) for (int i = 0, loop_N_macro_##i = (n); i < loop_N_macro_##i; ++i)
#define SUM(i, n, expression) ([&] {double macro_s_##i = 0.0; FOR(i, n) macro_s_##i += (expression); return macro_s_##i;})()

#define MIN(i, n, expression) ([&] {\
  double macro_current_min_##i = std::numeric_limits<double>::infinity();\
  FOR(i, n) {\
    const auto macro_s_1##i = (expression);\
    if (macro_current_min_##i > macro_s_1##i) macro_current_min_##i = macro_s_1##i;\
  }\
  return macro_current_min_##i;\
})()

#define MAX(i, n, expression) ([&] {\
  double macro_current_max_##i = -std::numeric_limits<double>::infinity();\
  FOR(i, n) {\
    const auto macro_s_1##i = (expression);\
    if (macro_current_max_##i < macro_s_1##i) macro_current_max_##i = macro_s_1##i;\
  }\
  return macro_current_max_##i;\
})()

#define LOGSUMEXP(i, n, expression) ([&] {\
  const auto macro_max_##i = MAX(i, n, (expression));\
  return macro_max_##i + log(SUM(i, n, exp((expression) - macro_max_##i)));\
})()

#define BE(it) std::begin(it), std::end(it)
#define LLL(x) ([&] {const auto hogetarou = (x); std::cout << #x << " = " << hogetarou << std::endl; return hogetarou;})()

#define MY_MACROS__H
#endif

