/*
 *  Author: Qibin (Andrew) Hou
 *  Email:  andrewhoux@gmail.com
 *  This is a free software for computing all the metrics in salient object detection.
 */

#ifndef INCLUDE_SAL_METRIC_HPP
#define INCLUDE_SAL_METRIC_HPP

#include <vector>
#include <string>
#include <opencv2/core/core.hpp>

struct thread_param {
    float precision[256];
    float recall[256];
    float mae;
    int start_line;
    int end_line;
    std::vector<std::vector<std::string> > lines;
};

class SalMetric {
public:
    SalMetric() { }
    explicit SalMetric(const int num_threads) : num_threads_(num_threads) { }
    explicit SalMetric(const std::string list_file) {
        list_file_ = list_file;
        num_threads_ = 4; // in default
    }
    SalMetric(const std::string list_file, const int num_threads) : num_threads_(num_threads) {
        list_file_ = list_file;
    }
    static float compute_mae(const cv::Mat &sal, const cv::Mat &gt);
    static void compute_precision_and_recall(const cv::Mat &sal, const cv::Mat &gt, float* precision, float* recall);
    void load_list(const std::string &list_file);
    void load_list(std::vector<std::string> &sal_lst, std::vector<std::string> &gt_lst);
    void load_list();
    void do_evaluation();
    void do_evaluation_gpu();
    void set_num_thread(const int num_threads = 4) {
        num_threads_ = num_threads;
    }
private:
    //static void* compute_metrics(void *thread_args);

    std::string list_file_;
    std::vector<std::vector<std::string> > lines_;
    unsigned int num_threads_;
    static constexpr const int THREASHOLDS = 256;
    static constexpr const float EPSILON = 1e-4;
    static constexpr const float BETA = 0.3;

    // No copies
    SalMetric(const SalMetric&);
    SalMetric& operator=(const SalMetric&);
};


#endif //INCLUDE_SAL_METRIC_HPP
