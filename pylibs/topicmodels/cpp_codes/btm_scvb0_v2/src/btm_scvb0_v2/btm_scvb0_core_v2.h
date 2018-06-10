#include "lazymultarray.h"
#include <vector>
#include <tuple>
#include <utility>
#include <random>
#include <functional>
#include <cmath>
#include <cinttypes>
#include <Eigen/Eigen>
#include <boost/math/special_functions/digamma.hpp>
#include <iostream>
using namespace std;

namespace topicmodel {
  class btm_scvb0_core_v2 {
    public:
      template <typename URNG>
      btm_scvb0_core_v2(int num_vocabulary, int num_biterms, int num_topics, double alpha, double beta, double tau, double kappa, URNG &urng) :
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

        //add
        alphak_.resize(num_topics_);
        alphak_ = 0.1;
        newbeta_ = 0.01;
        newalpha_ = 0.1;
      }

      void update_with_biterm(const int w1, const int w2) {
        for (int k = 0; k < num_topics_; ++k) {
          const double topic_p = Nk_(k) + alphak_(k);
          const double w1_p = (Nkv_.get(k, w1) + newbeta_) / (num_vocabulary_ * newbeta_ + 2*Nk_(k) + 1);
          const double w2_p = (Nkv_.get(k, w2) + newbeta_) / (num_vocabulary_ * newbeta_ + 2*Nk_(k));
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


      void update_parameter(){
          Nkv_.refresh();
          double alpha_sum = alphak_.sum();
          double numer = 0.0;
          double denom_sum1 = 0.0;

          double alpha_numer = 0.0;
          //symmetric alpha
          for(int k =0; k <num_topics_ ;k++){
            alpha_numer +=  boost::math::digamma( Nk_(k) + newalpha_) - boost::math::digamma(newalpha_);
          }

          newalpha_ = (1.0 / num_topics_)*( (alpha_numer) /
          (boost::math::digamma( num_biterms_ +  num_topics_*newalpha_ ) - boost::math::digamma(num_topics_*newalpha_ ) ) ) * newalpha_ ;
          cout<< "symmea_lpha"<<newalpha_ << endl;

/*
          for (int k = 0; k< num_topics_; k++){
            alphak_(k) = ( ( boost::math::digamma( Nk_(k) + alphak_(k)) - boost::math::digamma(alphak_(k))  ) * alphak_(k) ) /
            (boost::math::digamma( num_biterms_ +  alpha_sum ) - boost::math::digamma(alpha_sum ) ) ;
            //cout << "ALPHA" << alphak_(k) << endl;
          }
*/
/*//asymme
          for (int k = 0; k< num_topics_; k++){

            long double alpha_numer = ( boost::math::digamma( Nk_(k) + alphak_(k)) - boost::math::digamma(alphak_(k))  ) * alphak_(k) ;
            long double  alpha_denom = boost::math::digamma( num_biterms_ +  alpha_sum ) - boost::math::digamma(alpha_sum );

            alphak_(k) = ( alpha_numer)/(alpha_denom);
            //cout << "DENOM" << alpha_denom << endl;
            //cout << "NUMER" << alpha_numer << endl;
            cout << "ALPHA" << alphak_(k) << endl;
          }
          double numer = 0.0;
          */
          //cout << "--------------" << endl;

          /*
          for (int k = 0; k < Nkv_array_.rows(); ++k) {
              for (int v = 0; v < Nkv_array_.cols(); ++v) {
                numer = (boost::math::digamma( Nkv_array_(k,v) + newbeta_ ) - boost::math::digamma( newbeta_ ) ) * newbeta_  ;
                const double numer_sum  += numer;
              }
          }
          */
          for (int k = 0; k < Nkv_array_.rows(); ++k) {
            for (int v = 0; v < Nkv_array_.cols(); ++v) {
              numer += boost::math::digamma( Nkv_array_(k,v) + newbeta_ ) ;
            }
          }
          const double numer_sum = ( numer - num_topics_* num_vocabulary_* boost::math::digamma( newbeta_ ) ) *newbeta_ ;

          for (int k = 0; k < num_topics_; ++k ){
            denom_sum1 += boost::math::digamma( 2*Nk_(k) + num_vocabulary_ *newbeta_ );
          }
          const double denom_sum2 =  num_topics_ *(boost::math::digamma( num_vocabulary_ *newbeta_  ) );
          //cout << "betaorig" << newbeta_ << endl;
          newbeta_ = ( 1.0 / num_vocabulary_ ) * ( numer_sum /  (denom_sum1 - denom_sum2) )  ;
          //cout << "beta" << newbeta_ << endl;

      }

      const Eigen::ArrayXd &Nk() {
        return Nk_;
      }
      const Eigen::ArrayXXd &Nkv() {
        Nkv_.refresh();
        return Nkv_array_;
      }
      //add
      const Eigen::ArrayXd &alphak() {
        return alphak_;
      }
      const double &newbeta(){
          return newbeta_;
      }
      const double &newalpha(){
          return newalpha_;
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
      //add
      Eigen::ArrayXd alphak_;
      double newbeta_;
      double newalpha_;



      inline double robbins_monro() {
        const std::int64_t t = t_++;
        const double s = 1.0;
        return s / std::pow(t + tau_, kappa_);
      }
  };
}
