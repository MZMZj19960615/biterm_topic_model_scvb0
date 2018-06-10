#include <vector>
#include <utility>
#include <tuple>
#include <random>
#include <cmath>
#include <Eigen/Eigen>

namespace topicmodel {
  class btm_cgs_core {
    public:
      using biterms_t = std::vector<std::tuple<int, int>>;
      template <typename URNG>
      btm_cgs_core(const biterms_t &biterms, int num_vocabulary, int num_topics, double alpha, double beta, URNG &urng) :
          biterms_(biterms),
          num_vocabulary_(num_vocabulary),
          num_topics_(num_topics),
          alpha_(alpha),
          beta_(beta)
      {
        const int num_biterms = static_cast<int>(biterms_.size());

        Nk_.resize(num_topics_); Nk_ = 0;
        Nkv_.resize(num_topics_, num_vocabulary_); Nkv_ = 0;
        weights_.resize(num_topics_); weights_ = 1.0;
        z_.resize(num_biterms);

        for (int b = 0; b < num_biterms; ++b) {
          const auto &biterm = biterms_[b];
          const int w0 = std::get<0>(biterm);
          const int w1 = std::get<1>(biterm);
          const int k = std::uniform_int_distribution<int>(0, num_topics_ - 1)(urng);

          z_(b) = k;
          Nk_(k) += 1;
          Nkv_(k, w0) += 1;
          Nkv_(k, w1) += 1;
        }

        biterm_indices_.reserve(num_biterms);
        for (int b = 0; b < num_biterms; ++b) {
          biterm_indices_.push_back(b);
        }
      }

      template <typename URNG>
      void update(URNG &urng) {
        const int num_biterms = static_cast<int>(biterms_.size());

        std::shuffle(biterm_indices_.begin(), biterm_indices_.end(), urng);

        for (int i = 0; i < num_biterms; ++i) {
          const int b = biterm_indices_[i];
          const auto &biterm = biterms_[b];
          const int w0 = std::get<0>(biterm);
          const int w1 = std::get<1>(biterm);
          const int kold = z_(b);

          Nk_(kold) -= 1;
          Nkv_(kold, w0) -= 1;
          Nkv_(kold, w1) -= 1;

          for (int k = 0; k < num_topics_; ++k) {
            weights_[k] = Nk_(k) + alpha_;
            weights_[k] *= (Nkv_(k, w0) + beta_) / (2*Nk_(k) + num_vocabulary_*beta_ + 1);
            weights_[k] *= (Nkv_(k, w1) + beta_) / (2*Nk_(k) + num_vocabulary_*beta_    );
          }

          const int knew = std::discrete_distribution<int>(weights_.data(), weights_.data() + weights_.size())(urng);
          z_(b) = knew;

          Nk_(knew) += 1;
          Nkv_(knew, w0) += 1;
          Nkv_(knew, w1) += 1;
        }
      }

      Eigen::ArrayXi Nk() {
        return Nk_;
      }

      Eigen::ArrayXXi Nkv() {
        return Nkv_;
      }

    private:

      const biterms_t biterms_;
      const int num_vocabulary_;
      const int num_topics_;

      const double alpha_;
      const double beta_;

      Eigen::ArrayXi Nk_;
      Eigen::ArrayXXi Nkv_;
      Eigen::ArrayXd weights_;
      Eigen::ArrayXi z_;
      std::vector<int> biterm_indices_;
  };
}

