#include <iostream>
#include "cnn_seal.h"
#include "infer_seal.h"
#include <algorithm>
#include <fstream>

using namespace std;

int main(int argc, char **argv) {
	// parse arguments
	string algo = argv[1];
	string model = argv[2];
	string dataset = argv[3];
	int fold = atoi(argv[4]);
	string weight_dir = argv[5];
	string dataset_dir = argv[6];
	string output_dir = argv[7];
	int start = atoi(argv[8]);
	int end = atoi(argv[9]);
	int num_pairs = 600;
	if (dataset == "cfp_fp") num_pairs = 700;
	if (start < 0 || start >= end || end > num_pairs) throw std::invalid_argument("start or end pair no. is wrong.");
	if (fold < 0 || fold >= 10) throw std::invalid_argument("fold no. is wrong.");

	cout << "--------------------------------------" << endl;
	cout << "=> Algorithm: " << algo << endl;
	cout << "=> Model: " << model << endl;
	cout << "=> Dataset: " << dataset << endl;
	cout << "=> Fold: " << to_string(fold) << endl;
	cout << "=> Start image id (include): " << start << endl;
	cout << "=> End image id (exclude): " << end << endl;

	if(algo == "autofhe") resnet_autofhe(model, dataset, fold, weight_dir, dataset_dir, output_dir, start, end);
	else if(algo == "mpcnn") resnet_mpcnn(model, dataset, fold, weight_dir, dataset_dir, output_dir, start, end);
	else if(algo == "cryptoface"){
		int input_size = atoi(model.c_str()); 
		patchcnn(input_size, dataset, fold, weight_dir, dataset_dir, output_dir, start, end);
	}
	else throw std::invalid_argument(algo + " is not known.");

	return 0;
}
