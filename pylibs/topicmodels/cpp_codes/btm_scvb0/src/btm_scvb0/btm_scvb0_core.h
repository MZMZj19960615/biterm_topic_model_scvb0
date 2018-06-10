#include "lazymultarray.h"
#include <vector>
#include <tuple>
#include <utility>
#include <random>
#include <functional>
#include <cmath>
#include <cinttypes>
#include <Eigen/Eigen>

namespace topicmodel {
  class btm_scvb0_core {
    public:
      template <typename URNG>
      btm_scvb0_core(int num_vocabulary, int num_biterms, int num_topics, double alpha, double beta, double tau, double kappa, URNG &urng) :
          num_vocabulary_(num_vocabulary),
          num_biterms_(num_biterms),
          num_topics_(num_topics),
          alpha_(alpha),
          beta_(beta),
          tau_(tau),
          kappa_(kappa),
          Nkv_(Nkv_array_)
      {
        Nk_.resize(num_topics_);
        Nk_ = 0.0;
        Nkv_array_.resize(num_topics_, num_vocabulary_);
        Nkv_array_ = 0.0;
        weights_.resize(num_topics_);

        std::uniform_real_distribution<double> distribution {0.1, 1000.0};
        for (int k = 0; k < num_topics_; ++k) {
          Nk_(k) = distribution(urng);
        }
        Nk_ *= num_biterms / Nk_.sum();
        assert(std::round(Nk_.sum()) == num_biterms);

        for (int k = 0; k < num_topics_; ++k) {
          for (int v = 0; v < num_vocabulary_; ++v) {
            Nkv_array_(k, v) = distribution(urng);
          }
        }
        Nkv_array_ *= 2*num_biterms / Nkv_array_.sum();
        assert(std::round(Nkv_array_.sum()) == 2*num_biterms);
      }

      void update_with_biterm(const int w1, const int w2) {
        for (int k = 0; k < num_topics_; ++k) {
          const double topic_p = Nk_(k) + alpha_;
          const double w1_p = (Nkv_.get(k, w1) + beta_) / (num_vocabulary_ * beta_ + 2*Nk_(k) + 1);
          const double w2_p = (Nkv_.get(k, w2) + beta_) / (num_vocabulary_ * beta_ + 2*Nk_(k));
          weights_(k) = topic_p * w1_p * w2_p;
        }
        weights_ /= weights_.sum();

        const double rate = robbins_monro();

        Nk_ *= (1.0 - rate);
        for (int k = 0; k < num_topics_; ++k) {
          Nk_(k) += (num_biterms_ * weights_[k]) * rate;
        }

        Nkv_.lazy_mult(1.0 - rate);
        for (int k = 0; k < num_topics_; ++k) {
          Nkv_.set(k, w1, Nkv_.get(k, w1) + (num_biterms_ * weights_[k]) * rate);
          Nkv_.set(k, w2, Nkv_.get(k, w2) + (num_biterms_ * weights_[k]) * rate);
        }
      }

      const Eigen::ArrayXd &Nk() {
        return Nk_;
      }
      const Eigen::ArrayXXd &Nkv() {
        Nkv_.refresh();
        return Nkv_array_;
      }


    private:
      const int num_vocabulary_;
      const int num_biterms_;
      const int num_topics_;
      const double alpha_;
      const double beta_;
      const double tau_;
      const double kappa_;
      std::int64_t t_ {0};
      Eigen::ArrayXd Nk_;
      Eigen::ArrayXXd Nkv_array_;
      LazyMultArray<Eigen::ArrayXXd> Nkv_;
      Eigen::ArrayXd weights_;

      inline double robbins_monro() {
        const std::int64_t t = t_++;
        const double s = 1.0;
        return s / std::pow(t + tau_, kappa_);
      }
  };
}
