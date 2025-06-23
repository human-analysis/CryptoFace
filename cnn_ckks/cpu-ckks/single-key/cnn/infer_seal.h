#pragma	once
#include "seal/seal.h"
#include "SEALcomp.h"
#include "MinicompFunc.h"
#include "func.h"
#include "PolyUpdate.h"
#include "program.h"
#include "Bootstrapper.h"
#include "cnn_seal.h"
#include <omp.h>
#include <NTL/RR.h>
#include <fstream>
#include <vector>
#include <chrono>

#include <iomanip>
#include <cstdlib>

// AutoFHE: Automated Adaption of CNNs for Efficient Evaluation over FHE. USENIX Security 2024

void resnet_autofhe(string &model, string &dataset, size_t fold, string &weight_dir, string &dataset_dir, string &output_dir, size_t start_image_id, size_t end_image_id);
void import_weights_resnet_autofhe(string &dataset, size_t fold, string &dir, vector<double> &linear_weight, vector<double> &linear_bias, vector<vector<double>> &conv_weight, vector<vector<double>> &bn_bias, vector<vector<double>> &bn_running_mean, vector<vector<double>> &bn_running_var, vector<vector<double>> &bn_weight, vector<double> &coef_threshold, vector<double> &in_range, vector<double> &out_range,  vector<int> &depth, vector<int> &boot_loc, size_t layer_num, size_t end_num);

// FHE-MP-CNN: Low-complexity deep convolutional neural networks on fully homomorphic encryption using multiplexed parallel convolutions. ICML 2022

void resnet_mpcnn(string &model, string &dataset, size_t fold, string &weight_dir, string &dataset_dir, string &output_dir, size_t start_image_id, size_t end_image_id);
void import_weights_resnet_mpcnn(string &dataset, size_t fold, string &dir, vector<double> &linear_weight, vector<double> &linear_bias, vector<vector<double>> &conv_weight, vector<vector<double>> &bn_bias, vector<vector<double>> &bn_running_mean, vector<vector<double>> &bn_running_var, vector<vector<double>> &bn_weight, vector<double> &coef_threshold, double &B, size_t layer_num, size_t end_num);

// CryptoFace: End-to-End Encrypted Face Recogntion. CVPR 2025.
void patchcnn(size_t input_size, string &dataset, size_t fold, string &weight_dir, string &dataset_dir, string &output_dir, size_t start_image_id, size_t end_image_id);
void import_weights_pcnn(size_t num_nets, string &dataset, size_t fold, string &dir, vector<vector<double>> &all_linear_weight, vector<double> &linear_bias, vector<double> &coef_threshold, vector<vector<double>> &all_conv_weight, vector<vector<vector<double>>> &all_layer_weight, vector<vector<vector<double>>> &all_shortcut_weight, vector<vector<vector<double>>> &all_a2, vector<vector<vector<double>>> &all_a1, vector<vector<vector<double>>> &all_a0, vector<vector<vector<double>>> &all_b1, vector<vector<vector<double>>> &all_b0,  vector<vector<vector<double>>> &all_shortcut_bn_weight, vector<vector<vector<double>>> &all_shortcut_bn_bias, vector<vector<vector<double>>> &all_shortcut_bn_running_mean, vector<vector<vector<double>>> &all_shortcut_bn_running_var, vector<vector<double>> &all_bn_weight, vector<vector<double>> &all_bn_bias, vector<vector<double>> &all_bn_running_mean, vector<vector<double>> &all_bn_running_var);
void import_weights_pcnn_net(string &dir, vector<double> &linear_weight, vector<double> &conv_weight, vector<vector<double>> &layer_weight, vector<vector<double>> &shortcut_weight, vector<vector<double>> &a2, vector<vector<double>> &a1, vector<vector<double>> &a0, vector<vector<double>> &b1, vector<vector<double>> &b0,  vector<vector<double>> &shortcut_bn_weight, vector<vector<double>> &shortcut_bn_bias, vector<vector<double>> &shortcut_bn_running_mean, vector<vector<double>> &shortcut_bn_running_var, vector<double> &bn_weight, vector<double> &bn_bias, vector<double> &bn_running_mean, vector<double> &bn_running_var);

// load dataset
void load_dataset(string &dataset, string &dataset_dir, int fold, vector<vector<double>> &images1, vector<vector<double>> &images2, vector<int> &labels);