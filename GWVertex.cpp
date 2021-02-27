#include <poplar/Vertex.hpp>

#include <cmath>
#include <array>

static const float M_PI = 3.14159265359;

class GW : public poplar::Vertex {
public:

    static const int nParams = 2;

    poplar::Input<poplar::Vector<float>> data;
    poplar::Input<poplar::Vector<float>> uniformRand; // dim 2
    poplar::Input<poplar::Vector<int>> otherRand; // dim 1
    poplar::Input<poplar::Vector<float>> thisTheta;
    poplar::Vector<poplar::Input<poplar::Vector<float>>> otherThetas; // (nExchange, 2)

    poplar::Output<poplar::Vector<float>> out;

    float gaussian(float x, float mu, float sigma)
    {
        float ex = -(x - mu) * (x - mu) / (2 * sigma * sigma);
        float a = 1. / (sqrt(2. * M_PI) * sigma);

        return a * exp(ex);
    }

    template<typename T>
    float logLikelihood(T & theta, poplar::Vector<float> & data)
    {
        float mu = theta[0];
        float sigma = theta[1];

        float loglike = 0;
        for (auto d : data) loglike += log(gaussian(d, mu, sigma));

        return loglike;
    }
    template<typename T>
    float logPrior(T & theta)
    {
        if (theta[1] < 1E-8) return -std::numeric_limits<float>::infinity();
        return 0;
    }

    template<typename T>
    float logPosterior(T & theta, poplar::Vector<float> & data)
    {
        float prior = logPrior(theta);

        if (prior == -std::numeric_limits<float>::infinity()) return prior;

        float lh = logLikelihood(theta, data);

        return prior + lh;
    }

    float zDist(float a, float rand)
    {
        float z = (a - 1.0) * rand + 1.0;

        return (z * z) / a;
    }

    std::array<float, nParams> calcTestTheta(poplar::Vector<float> & thisTheta,
                                             poplar::Vector<float> & otherTheta,
                                             float z)
    {
        std::array<float, nParams> testTheta;

        for (int i = 0; i < nParams; i++) {
            testTheta[i] = thisTheta[i] - z * (thisTheta[i] - otherTheta[i]);
        }

        return testTheta;
    }

    bool compute()
    {
        float z = zDist(2.0, uniformRand[0]);

        for (auto otherTheta : otherThetas) {

            std::array<float, nParams> testTheta = calcTestTheta(thisTheta, otherTheta, z);

            float thisPost = logPosterior(thisTheta, data);
            float testPost = logPosterior(testTheta, data);

            float q = log(z) * (nParams - 1) + (testPost - thisPost);

            if (q > log(uniformRand[1])) {
                for (int i = 0; i < nParams; i++) out[i] = testTheta[i];
            } else {
                for (int i = 0; i < nParams; i++) out[i] = thisTheta[i];
            }
        }

        return true;
    }

};
