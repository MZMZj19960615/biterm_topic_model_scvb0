#include <iostream>
#include <vector>
#include <tuple>
#include <utility>
#include <random>
#include <functional>
#include <cmath>
#include <cinttypes>
#include <Eigen/Eigen>
#include <cassert>
#include <numeric>
#include <random>

#include "macros.h"


namespace topicmodels {
  // math functions
  using std::exp;
  using std::log;
  using std::lgamma;

  // array types
  using ArrayXd = Eigen::ArrayXd;
  using ArrayXXd = Eigen::ArrayXXd;
  using ArrayXi = Eigen::Array<int, Eigen::Dynamic, 1>;
  using ArrayXXi = Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic>;

  // docs type
  using docs_t = std::vector<std::vector<int>>;


  // helper functions
  template<typename URNG, typename RealType>
  RealType draw_from_beta_distribution(RealType alpha, RealType beta, URNG &urng) {
    const auto a = std::gamma_distribution<RealType>(alpha, 1.0)(urng);
    const auto b = std::gamma_distribution<RealType>(beta, 1.0)(urng);
    return a / (a + b);
  }

  // Sparcified SPTM class
  template<typename URNG>
  class SPTM {
    private:
      // constants
      const docs_t wdn;
      const int K, V, D, P;
      const std::vector<int> Nd;


      // variables of the model
      docs_t zdn;

      double alpha, alphabar;
      double beta;
      double gamma0, gamma1;
      double lambda;

      ArrayXXi bpk; // topic selector for pseudo document p (binary)
      ArrayXi ld; // pseudo document index for a document d

      // deterministic variables
      ArrayXXi Npk; // Npk(p, k) := SUM(d, D, SUM(n, Nd[d], [ld(d) == p][zdn(d, n) == k]))    depends on: zdn, ld
      ArrayXi Np; // Np(p) := SUM(k, K, Npk(p, k))                                            depends on: ld
      ArrayXXi Nkv; // Nkv(k, v) := SUM(d, D, SUM(n, Nd[d], [wdn(d, n)== v][zdn(d, n) == k])) depends on: zdn
      ArrayXi Nk; // Nk(k) := SUM(v, V, Nkv(k, v))                                            depends on: zdn
      ArrayXi Dp; // Dp(p) := SUM(d, D, [ld(d) == p])                                         depends on: ld
      ArrayXi bp; // bp(p) := SUM(k, K, bpk(p, k))                                            depends on: bpk


      // temporary array
      std::vector<double> tmp_array;
      std::vector<int> tmp_arrayi_zeros;
      std::vector<int> tmp_arrayi2;


      // indices for sampling
      std::vector<std::tuple<int, int, int>> indices;

      // random number generator
      URNG urng;

    public:
      SPTM(int n_topics, int n_word_types, int n_pseudo_docs, const docs_t &docs, std::uint32_t seed) :
        wdn(docs),
        K(n_topics), V(n_word_types), D(static_cast<int>(wdn.size())), P(n_pseudo_docs),
        Nd(([&] {
          std::vector<int> Nd_tmp;
          FOR(d, D) Nd_tmp.push_back(static_cast<int>(wdn[d].size()));
          return Nd_tmp;
        })()),

        zdn(),

        alpha(0.1), alphabar(1e-12),
        beta(0.01),
        gamma0(0.1), gamma1(0.1),
        lambda(0.1),

        bpk(ArrayXXi::Zero(P, K)),
        ld(ArrayXi::Zero(D)),

        Npk(ArrayXXi::Zero(P, K)),
        Np(ArrayXi::Zero(P)),
        Nkv(ArrayXXi::Zero(K, V)),
        Nk(ArrayXi::Zero(K)),
        Dp(ArrayXi::Zero(P)),
        bp(ArrayXi::Zero(P)),
        tmp_array(std::max(K, P)),
        tmp_arrayi_zeros(K),
        tmp_arrayi2(K),

        urng(seed)
      {
        init_zdn_randomly();
        init_bpk_randomly();
        init_ld_randomly();
        update_deterministic_variables();
        FOR(k, K) tmp_arrayi_zeros[k] = 0;


        FOR(d, D) FOR(n, Nd[d]) indices.emplace_back(0, d, n);
        FOR(p, P) FOR(k, K)     indices.emplace_back(1, p, k);
        FOR(d, D)               indices.emplace_back(2, d, 0);
      }

      void update() {
        std::shuffle(BE(indices), urng);
        for (const auto &index : indices) {
          const auto variable_type = std::get<0>(index);
          switch (variable_type) {
            case 0: {
              const auto d = std::get<1>(index);
              const auto n = std::get<2>(index);
              sample_zdn(d, n);
            } break;

            case 1: {
              const auto p = std::get<1>(index);
              const auto k = std::get<2>(index);
              sample_bpk(p, k);
            } break;

            case 2: {
              const auto d = std::get<1>(index);
              sample_ld(d);
            } break;
          }
        }
      }

      void init_zdn_randomly() { // randomly initialize zdn
        FOR(d, D) {
          zdn.emplace_back();
          FOR(n, Nd[d]) {
            zdn[d].push_back(std::uniform_int_distribution<int>(0, K - 1)(urng));
          }
        }
      }

      void init_bpk_randomly() { // randomly initialize bpk
        FOR(p, P) FOR(k, K) bpk(p, k) = std::bernoulli_distribution(1.0 / K)(urng) ? 1 : 0;
      }

      void init_ld_randomly() { // randomly initialize ld
        FOR(d, D) ld(d) = std::uniform_int_distribution<int>(0, P - 1)(urng);
      }

      void sample_zdn(const int d, const int n) {
        auto &weights = tmp_array;
        const auto p = ld(d);
        const auto v = wdn[d][n];
        const auto k_old = zdn[d][n];
        --Npk(p, k_old);
        --Nk(k_old);
        --Nkv(k_old, v);

        FOR(k, K) weights[k] = (alpha*bpk(p, k) + alphabar + Npk(p, k)) / (V*beta + Nk(k)) * (beta + Nkv(k, v));

        const auto k_new = std::discrete_distribution<int>(weights.begin(), weights.begin() + K)(urng);
        ++Npk(p, k_new);
        ++Nk(k_new);
        ++Nkv(k_new, v);
        zdn[d][n] = k_new;
      }


      void sample_ld(const int d) {
        const auto p_old = ld(d);

        FOR(n, Nd[d]) --Npk(p_old, zdn[d][n]);
        Np(p_old) -= Nd[d];
        --Dp(p_old);

        auto &topic_counts = tmp_arrayi_zeros;
        auto &weights = tmp_array;
        auto &found_topics = tmp_arrayi2;

#ifndef NDEBUG
        FOR(k, K) {
          assert(topic_counts[k] == 0);
        }
#endif
        // calculate weights for sampling
        int n_topic_types = 0;

        FOR(p, P) weights[p] = 1.0;
        FOR(p, P) weights[p] *= lambda + Dp(p);
        FOR(n, Nd[d]) {
          const auto k = zdn[d][n];
          if (topic_counts[k] == 0) found_topics[n_topic_types++] = k;
          const int i = topic_counts[k]++;
          FOR(p, P) weights[p] *=
            static_cast<double>(bpk(p, k)*alpha + K*alphabar + Npk(p, k) + i)
            / (bp(p)*alpha + K*alphabar + Np(p) + n);
        }
        FOR(i, n_topic_types) topic_counts[found_topics[i]] = 0;

        const auto p_new = std::discrete_distribution<int>(tmp_array.begin(), tmp_array.begin() + P)(urng);

        FOR(n, Nd[d]) ++Npk(p_new, zdn[d][n]);
        Np(p_new) += Nd[d];
        ++Dp(p_new);

        ld(d) = p_new;
      }

     void sample_bpk_using_pip(const int p) {
       const auto pip = draw_from_beta_distribution(gamma0 + bp(p), gamma1 + K - bp(p), urng);

       auto &weights = tmp_array;
       FOR(k, K) {
         const auto s_old = bpk(p, k);
         bp(p) -= s_old;

         FOR(s, 2) weights[s] =
           + lgamma((bp(p) + s)*alpha + K*alphabar) - lgamma((bp(p) + s)*alpha + K*alphabar + Np(p))
           + lgamma(s*alpha + alphabar + Npk(p, k)) - lgamma(s*alpha + alphabar);

         const auto log_normalizer = LOGSUMEXP(s, 2, weights[s]);
         FOR(s, 2) weights[s] = exp(weights[s] - log_normalizer);

         weights[0] *= 1.0 - pip;
         weights[1] *= pip;


         const auto s_new = std::discrete_distribution<int>(weights.begin(), weights.begin() + 2)(urng);

         bp(p) += s_new;
         bpk(p, k) = s_new;
       }
     }
     void sample_bpk(const int p, const int k) {
       auto &weights = tmp_array;

       const auto s_old = bpk(p, k);
       bp(p) -= s_old;

       FOR(s, 2) weights[s] =
         + lgamma((bp(p) + s)*alpha + K*alphabar) - lgamma((bp(p) + s)*alpha + K*alphabar + Np(p))
         + lgamma(s*alpha + alphabar + Npk(p, k)) - lgamma(s*alpha + alphabar)
         + lgamma(gamma0 + bp(p) + s) + lgamma(gamma1 + K - bp(p) - s);
       const auto log_normalizer = LOGSUMEXP(s, 2, weights[s]);
       FOR(s, 2) weights[s] = exp(weights[s] - log_normalizer);


       const auto s_new = std::discrete_distribution<int>(weights.begin(), weights.begin() + 2)(urng);
       bp(p) += s_new;
       bpk(p, k) = s_new;
     }

 //     void sample_bpk_slow(const int p, const int k) {
 //       LLL("slow");
 //       auto &weights = tmp_array;

 //       FOR(s, 2) {
 //         bpk(p, k) = s;
 //         update_deterministic_variables();
 //         weights[s] = log_marginalized_joint();
 //       }
 //       const auto log_normalizer = LOGSUMEXP(s, 2, weights[s]);
 //       FOR(s, 2) weights[s] = exp(weights[s] - log_normalizer);

 //       LLL(weights[0]);
 //       LLL(weights[1]);
 //     }


      void update_deterministic_variables() {
        //Npk
        FOR(p, P) FOR(k, K) Npk(p, k) = 0;
        FOR(d, D) FOR(n, Nd[d]) ++Npk(ld(d), zdn[d][n]);

        // Np
        FOR(p, P) Np(p) = SUM(k, K, Npk(p, k));

        // Nkv
        FOR(k, K) FOR(v, V) Nkv(k, v) = 0;
        FOR(d, D) FOR(n, Nd[d]) ++Nkv(zdn[d][n], wdn[d][n]);

        // Nk
        FOR(k, K) Nk(k) = SUM(v, V, Nkv(k, v));

        // Dp
        FOR(p, P) Dp(p) = 0;
        FOR(d, D) ++Dp(ld(d));

        // bp
        FOR(p, P) bp(p) = SUM(k, K, bpk(p, k));
      }

      double log_marginalized_joint() const { // returns current log P(w, z, b, l|alpha, alphabar, beta, lambda)
        return

        + SUM(p, P,
          + lgamma(bp(p)*alpha + K*alphabar)
          - lgamma(bp(p)*alpha + K*alphabar + Np(p))
          + SUM(k, K,
              + lgamma(bpk(p, k)*alpha + alphabar + Npk(p, k))
              - lgamma(bpk(p, k)*alpha + alphabar)
            )
        )

        + SUM(k, K,
          + lgamma(V*beta) - lgamma(V*beta + Nk(k))
          + SUM(v, V, lgamma(Nkv(k, v) + beta) - lgamma(beta))
        )

        + lgamma(P*lambda) - lgamma(P*lambda + D)
        + SUM(p, P, lgamma(lambda + Dp(p)) - lgamma(lambda))

        + SUM(p, P,
          + lgamma(gamma0 + gamma1) - lgamma(gamma0 + gamma1 + K)
          + lgamma(gamma0 + bp(p))     - lgamma(gamma0)
          + lgamma(gamma1 + K - bp(p)) - lgamma(gamma1)
        )

        ;
      }

      Eigen::ArrayXXd get_phikv() const {
        Eigen::ArrayXXd phikv = Eigen::ArrayXXd::Zero(K, V);

        FOR(k, K) {
          FOR(v, V) phikv(k, v) = Nkv(k, v) + beta;
          const auto s = SUM(v, V, phikv(k, v));
          FOR(v, V) phikv(k, v) /= s;
        }

        return phikv;
      }

      Eigen::ArrayXd get_thetak() const {
        Eigen::ArrayXd thetak = Eigen::ArrayXd::Zero(K);
        Eigen::ArrayXd thetadk = Eigen::ArrayXd::Zero(K);

        FOR(d, D) {
          FOR(k, K) thetadk(k) = 0.0;
          FOR(n, Nd[d]) thetadk(zdn[d][n]) += 1.0;
          FOR(k, K) thetadk(k) /= Nd[d];
          FOR(k, K) thetak(k) += thetadk(k);
        }

        const double sum = SUM(k, K, thetak(k));
        FOR(k, K) thetak(k) /= sum;

        return thetak;
      }

      Eigen::ArrayXXd get_thetadk() const {
        Eigen::ArrayXXd thetadk = Eigen::ArrayXXd::Zero(D, K);

        FOR(d, D) {
          FOR(k, K) thetadk(d, k) = Npk(ld(d), k) + bpk(ld(d), k) * alpha + alphabar;
          const double s = SUM(k, K, thetadk(d, k));
          FOR(k, K) thetadk(d, k) /= s;
        }

        return thetadk;
      }

      Eigen::ArrayXi get_ld() const {
        return ld;
      }

  };
}

