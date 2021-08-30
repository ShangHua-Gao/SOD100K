/*
 *  Author: Qibin (Andrew) Hou
 *  Email:  andrewhoux@gmail.com
 *  This is a free software for computing all the metrics in salient object detection.
 */

#include "sal_metric.hpp"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <string>
#include <pthread.h>
#include <unistd.h>
#include <fstream>

using std::cout;
using std::endl;
using std::string;
using std::vector;

using cv::Mat;
//using namespace cv;

void* compute_metrics(void *thread_args) {
    thread_param* param = (thread_param *)thread_args;
    for (int i = param->start_line; i < param->end_line; ++i) {
        Mat sal = cv::imread(param->lines[i][0], 0);
        Mat gt  = cv::imread(param->lines[i][1], 0);
        sal.convertTo(sal, CV_32F);
        gt.convertTo(gt, CV_32F);
        if (sal.rows != gt.rows || sal.cols != gt.cols) {
            cout << "Saliency map should share the same size as ground truth, " << param->lines[i][0] << endl;
        }
        param->mae += SalMetric::compute_mae(sal, gt);
        SalMetric::compute_precision_and_recall(sal, gt, param->precision, param->recall);
    }
    return NULL;
}

/*
 * Read the list file that stores paris of prediction map and ground truth images.
 * It should have the following format:
 *      1_sal.png 1_gt.png
 *      2_sal.png 2_gt.png
 *      3_sal.png 3_gt.png
 *              ...
 */
void SalMetric::load_list(const string &list_file) {
    string sal_name;
    string gt_name;
    std::ifstream infile(list_file.c_str());
    while (infile >> sal_name >> gt_name) {
        vector<string> line;
        line.push_back(sal_name);
        line.push_back(gt_name);
        lines_.push_back(line);
    }
}

void SalMetric::load_list(vector<string> &sal_lst, vector<string> &gt_lst) {
    if (sal_lst.size() != gt_lst.size()) {
        cout << "The number of saliency maps should be equal to that of the ground truth maps." << endl;
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < sal_lst.size(); ++i) {
        vector<string> line;
        line.push_back(sal_lst[i]);
        line.push_back(gt_lst[i]);
        lines_.push_back(line);
    }
}

void SalMetric::load_list() {
    string sal_name;
    string gt_name;
    std::ifstream infile(list_file_.c_str());
    while (infile >> sal_name >> gt_name) {
        vector<string> line;
        line.push_back(sal_name);
        line.push_back(gt_name);
        lines_.push_back(line);
    }
}

float SalMetric::compute_mae(const Mat &sal, const Mat &gt) {
    int height = sal.rows;
    int width  = sal.cols;
    float mae = 0;
    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            mae += abs(sal.at<float>(h, w) - gt.at<float>(h, w)) / 255.;
        }
    }
    mae = mae / (height * width);
    return mae;
}

void SalMetric::compute_precision_and_recall(const Mat &sal, const Mat &gt, float* precision, float* recall) {
    for (int th = 0; th < THREASHOLDS; ++th) {
        float a_sum = 0;
        float b_sum = 0;
        float ab = 0;
        for (int h = 0; h < sal.rows; ++h) {
            for (int w = 0; w < sal.cols; ++w) {
                unsigned int a = (sal.at<float>(h, w) > th) ? 1 : 0;
                unsigned int b = (gt.at<float>(h, w) > THREASHOLDS / 2) ? 1 : 0;
                ab += (a & b);
                a_sum += a;
                b_sum += b;
            }
        }
        float pre = (ab + EPSILON) / (a_sum + EPSILON);
        float rec = (ab + EPSILON) / (b_sum + EPSILON);
        //precision[th] += a_sum == 0 ? 0 : pre;
        //recall[th]    += b_sum == 0 ? 0 : rec;
        precision[th] += pre;
        recall[th]    += rec;
    }
}

void SalMetric::do_evaluation() {
    cout << num_threads_ << " threads are being used for accelerating." << endl;
    int num_lines = lines_.size();
    int lines_per_thread = (num_lines + num_threads_ - 1) / num_threads_;
    pthread_t *pthread_id = new pthread_t[num_threads_];
    thread_param *param = new thread_param[num_threads_];
    for (size_t i = 0; i < num_threads_; ++i) {
        param[i].lines = lines_;
        param[i].mae = 0;
        memset(param[i].precision, 0, sizeof(float) * THREASHOLDS);
        memset(param[i].recall, 0, sizeof(float) * THREASHOLDS);
        param[i].start_line = lines_per_thread * i;
        param[i].end_line = lines_per_thread * (i + 1);
        if (param[i].end_line > num_lines)
            param[i].end_line = num_lines;
        pthread_create(&pthread_id[i], NULL, compute_metrics, (void*)&param[i]);
    }
    for (size_t i = 0; i < num_threads_; ++i) {
        pthread_join(pthread_id[i], NULL);
    }

    // post-processing
    float precision[THREASHOLDS];
    float recall[THREASHOLDS];
    float mae = 0;
    for (int th = 0; th < THREASHOLDS; ++th) {
        precision[th] = 0;
        recall[th] = 0;
    }
    int fmeasure_argmax = 0;
    float fmeasure_max = 0;
    float fmeasure_mean = 0;
    float precision_mean = 0;
    float recall_mean = 0;

    for (size_t i = 0; i < num_threads_; ++i) {
        mae += param[i].mae / num_lines;
        for (int th = 0; th < THREASHOLDS; ++th) {
            precision[th] += param[i].precision[th] / num_lines;
            recall[th] += param[i].recall[th] / num_lines;
        }
    }
    for (int th = 0; th < THREASHOLDS; ++th) {
        float fmeasure = ((1 + BETA) * precision[th] * recall[th]) / (BETA * precision[th] + recall[th]);
        fmeasure_mean += fmeasure;
        precision_mean += precision[th];
        recall_mean += recall[th];
        if (fmeasure > fmeasure_max) {
            fmeasure_max = fmeasure;
            fmeasure_argmax = th;
        }
        cout << "Threshold " << th << ":\tMAE: " << mae << "\tPrecision: " << precision[th];
        cout << "\tRecall: " << recall[th] << "\tFmeasure: " << fmeasure << endl;
    }
    fmeasure_mean /= THREASHOLDS;
    precision_mean /= THREASHOLDS;
    recall_mean /= THREASHOLDS;
    cout << "Max_F-measre:   " << fmeasure_max << endl;
    cout << "Mean_F-measre:  " << fmeasure_mean << endl;
    cout << "Precision:      " << precision[fmeasure_argmax] << endl;
    cout << "Recall:         " << recall[fmeasure_argmax] << endl;
    cout << "Mean_Precision: " << precision_mean << endl;
    cout << "Mean_Recall:    " << recall_mean << endl;
    cout << "MAE:            " << mae << endl;

    delete [] pthread_id;
    delete [] param;
}

void usage(int argc, char* argv[]) {
    cout << "Usage: " << argv[0] << " list_file [num_threads]" << endl;
    cout << "List file should have the following format:\n" << endl;
    cout << "\t1_sal.png 1_gt.png" << endl;
    cout << "\t2_sal.png 2_gt.png" << endl;
    cout << "\t3_sal.png 3_gt.png" << endl;
    cout << "\t\t..." << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "List file is required." << endl;
        usage(argc, argv);
        exit(EXIT_FAILURE);
    } else if (argc > 3) {
        cout << "Too many arguments provided." << endl;
        usage(argc, argv);
        exit(EXIT_FAILURE);
    }

    string list_file(argv[1]);
    if (argc == 2) {
        SalMetric metric(list_file);
        metric.load_list();
        metric.do_evaluation();
    } else if (argc == 3) {
        int num_threads = atoi(argv[2]);
        SalMetric metric(list_file, num_threads);
        metric.load_list();
        metric.do_evaluation();
    }

    return 0;
}
