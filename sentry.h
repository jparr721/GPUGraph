#pragma once

#include <opencv2/opencv.hpp>
#include <utility>

class sentry {
  public:
    sentry(const cv::Mat& me) : me_(me) {};
    ~sentry();

    void scream();
    std::pair<int, cv::Mat> detect(cv::Mat& img);

    int watch();

    double frobenius_norm(cv::Mat me, cv::Mat them);
  private:
    const cv::Mat me_;
};
