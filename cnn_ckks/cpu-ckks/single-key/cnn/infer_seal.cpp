#include "infer_seal.h"

void load_dataset(string &dataset, string &dataset_dir, int fold, vector<vector<double>> &images1, vector<vector<double>> &images2, vector<int> &labels)
{	
	int num_pairs = 600;
	if (dataset == "cfp_fp") num_pairs = 700;
	images1.resize(num_pairs);
	images2.resize(num_pairs);
	labels.resize(num_pairs);
	
	ifstream in_images1;
	in_images1.open(dataset_dir + "/" + dataset + "/fold" + to_string(fold) + "_images1.txt");
	if(!in_images1.is_open()) throw std::runtime_error(dataset_dir + "/" + dataset + "/fold" + to_string(fold) + "_images1.txt is not open");

	ifstream in_images2;
	in_images2.open(dataset_dir + "/" + dataset + "/fold" + to_string(fold) + "_images2.txt");
	if(!in_images2.is_open()) throw std::runtime_error(dataset_dir + "/" + dataset + "/fold" + to_string(fold) + "_images2.txt is not open");
	
	ifstream in_labels;
	in_labels.open(dataset_dir + "/" + dataset + "/fold" + to_string(fold) + "_labels.txt");
	if(!in_labels.is_open()) throw std::runtime_error(dataset_dir + "/" + dataset + "/fold" + to_string(fold) + "_labels.txt is not open");
	
	string line_img;
	string w0;
	int w1;
	for(size_t i=0; i<num_pairs; i++){
		std::getline(in_images1, line_img, '\n');	
		std::istringstream iss_img1(line_img);
		while (std::getline(iss_img1, w0, ',')) images1[i].emplace_back(atof(w0.c_str()));
		std::getline(in_images2, line_img, '\n');	
		std::istringstream iss_img2(line_img);
		while (std::getline(iss_img2, w0, ',')) images2[i].emplace_back(atof(w0.c_str()));
		in_labels >> w1;
		labels[i] = w1;
	}

	in_images1.close();
	in_images2.close();
	in_labels.close();

}

void import_weights_resnet_autofhe(string &dataset, size_t fold, string &dir, vector<double> &linear_weight, vector<double> &linear_bias, vector<vector<double>> &conv_weight, vector<vector<double>> &bn_bias, vector<vector<double>> &bn_running_mean, vector<vector<double>> &bn_running_var, vector<vector<double>> &bn_weight, vector<double> &coef_threshold, vector<double> &in_range, vector<double> &out_range,  vector<int> &depth, vector<int> &boot_loc, size_t layer_num, size_t end_num)
{
	std::cout << "load ResNet's weights from: " << dir << std::endl;

	ifstream in;
	double val;
	size_t num_c = 0, num_b = 0, num_m = 0, num_v = 0, num_w = 0;

	conv_weight.clear();
	conv_weight.resize(layer_num-1);
	bn_bias.clear();
	bn_bias.resize(layer_num-1);
	bn_running_mean.clear();
	bn_running_mean.resize(layer_num);
	bn_running_var.clear();
	bn_running_var.resize(layer_num);
	bn_weight.clear();
	bn_weight.resize(layer_num-1);
	in_range.clear();
	out_range.clear();
	depth.clear();
	boot_loc.clear();

	int fh = 3, fw = 3;
	int ci = 0, co = 0;

	// convolution parameters
	ci = 3, co = 16;
	in.open(dir + "/conv1_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<fh*fw*ci*co; i++) {in>>val; conv_weight[num_c].emplace_back(val);} in.close(); num_c++;

	// convolution parameters
	for(int j=1; j<=3; j++)
	{
		for(int k=0; k<=end_num; k++)
		{
			// co setting
			if(j==1) co=16;
			else if(j==2) co=32;
			else if(j==3) co=64;

			// ci setting
			if(j==1 || (j==2 && k==0)) ci=16;
			else if((j==2 && k!=0) || (j==3 && k==0)) ci=32;
			else ci=64;
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_conv1_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<fh*fw*ci*co; i++) {in>>val; conv_weight[num_c].emplace_back(val);} in.close(); num_c++;

			// ci setting
			if(j==1) ci = 16;
			else if(j==2) ci = 32;
			else if(j==3) ci = 64;
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_conv2_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<fh*fw*ci*co; i++) {in>>val; conv_weight[num_c].emplace_back(val);} in.close(); num_c++;
		}
	}	

	// batch_normalization parameters
	ci = 16;
	in.open(dir + "/bn1_bias.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; bn_bias[num_b].emplace_back(val);} in.close(); num_b++;
	in.open(dir + "/bn1_running_mean.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; bn_running_mean[num_m].emplace_back(val);} in.close(); num_m++;
	in.open(dir + "/bn1_running_var.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; bn_running_var[num_v].emplace_back(val);} in.close(); num_v++;
	in.open(dir + "/bn1_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; bn_weight[num_w].emplace_back(val);} in.close(); num_w++;

	// batch_normalization parameters
	for(int j=1; j<=3; j++)
	{
		int ci;
		if(j==1) ci=16;
		else if(j==2) ci=32;
		else if(j==3) ci=64;

		for(int k=0; k<=end_num; k++)
		{
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_bn1_bias.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<ci; i++) {in>>val; bn_bias[num_b].emplace_back(val);} in.close(); num_b++;
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_bn1_running_mean.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<ci; i++) {in>>val; bn_running_mean[num_m].emplace_back(val);} in.close(); num_m++;
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_bn1_running_var.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<ci; i++) {in>>val; bn_running_var[num_v].emplace_back(val);} in.close(); num_v++;
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_bn1_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<ci; i++) {in>>val; bn_weight[num_w].emplace_back(val);} in.close(); num_w++;
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_bn2_bias.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<ci; i++) {in>>val; bn_bias[num_b].emplace_back(val);} in.close(); num_b++;
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_bn2_running_mean.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<ci; i++) {in>>val; bn_running_mean[num_m].emplace_back(val);} in.close(); num_m++;
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_bn2_running_var.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<ci; i++) {in>>val; bn_running_var[num_v].emplace_back(val);} in.close(); num_v++;
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_bn2_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<ci; i++) {in>>val; bn_weight[num_w].emplace_back(val);} in.close(); num_w++;
		}
	}

	// output 1D batchnorm
	in.open(dir + "/norm_running_mean.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<256; i++) {in>>val; bn_running_mean[num_m].emplace_back(val);} in.close(); num_m++;
	in.open(dir + "/norm_running_var.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<256; i++) {in>>val; bn_running_var[num_v].emplace_back(val);} in.close(); num_v++;

	// FC
	in.open(dir + "/output_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<256*256; i++) {in>>val; linear_weight.emplace_back(val);} in.close();
	in.open(dir + "/output_bias.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<256; i++) {in>>val; linear_bias.emplace_back(val);} in.close();

	// in and out range of EvoReLU
	in.open(dir + "/evorelu_in_ranges.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<layer_num-1; i++) {in>>val; in_range.emplace_back(val);} in.close();
	in.open(dir + "/evorelu_out_ranges.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<layer_num-1; i++) {in>>val; out_range.emplace_back(val);} in.close();

	// depth
	int val2;
	in.open(dir + "/depth.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<layer_num-1; i++) {in>>val2; depth.emplace_back(val2);} in.close();

	// bootstrapping location 
	in.open(dir + "/boot_loc.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<(layer_num-2) * 2; i++) {in>>val2; boot_loc.emplace_back(val2);} in.close();

	// polynomial coefficients of 1/sqrt(*) and threshold
	in.open(dir + "/threshold_" + dataset + ".txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	string line;
	string w0;
	for(size_t i=0; i<10; i++){
		std::getline(in, line, '\n');
		if (i == fold){
			std::istringstream iss(line);
			while (std::getline(iss, w0, ',')) coef_threshold.emplace_back(atof(w0.c_str()));
		}	
	}
}

void import_weights_resnet_mpcnn(string &dataset, size_t fold, string &dir, vector<double> &linear_weight, vector<double> &linear_bias, vector<vector<double>> &conv_weight, vector<vector<double>> &bn_bias, vector<vector<double>> &bn_running_mean, vector<vector<double>> &bn_running_var, vector<vector<double>> &bn_weight, vector<double> &coef_threshold, double &B, size_t layer_num, size_t end_num)
{
	std::cout << "load ResNet's weights from: " << dir << std::endl;

	ifstream in;
	double val;
	size_t num_c = 0, num_b = 0, num_m = 0, num_v = 0, num_w = 0;

	conv_weight.clear();
	conv_weight.resize(layer_num-1);
	bn_bias.clear();
	bn_bias.resize(layer_num-1);
	bn_running_mean.clear();
	bn_running_mean.resize(layer_num);
	bn_running_var.clear();
	bn_running_var.resize(layer_num);
	bn_weight.clear();
	bn_weight.resize(layer_num-1);

	int fh = 3, fw = 3;
	int ci = 0, co = 0;

	// convolution parameters
	ci = 3, co = 16;
	in.open(dir + "/conv1_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<fh*fw*ci*co; i++) {in>>val; conv_weight[num_c].emplace_back(val);} in.close(); num_c++;

	// convolution parameters
	for(int j=1; j<=3; j++)
	{
		for(int k=0; k<=end_num; k++)
		{
			// co setting
			if(j==1) co=16;
			else if(j==2) co=32;
			else if(j==3) co=64;

			// ci setting
			if(j==1 || (j==2 && k==0)) ci=16;
			else if((j==2 && k!=0) || (j==3 && k==0)) ci=32;
			else ci=64;
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_conv1_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<fh*fw*ci*co; i++) {in>>val; conv_weight[num_c].emplace_back(val);} in.close(); num_c++;

			// ci setting
			if(j==1) ci = 16;
			else if(j==2) ci = 32;
			else if(j==3) ci = 64;
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_conv2_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<fh*fw*ci*co; i++) {in>>val; conv_weight[num_c].emplace_back(val);} in.close(); num_c++;
		}
	}	

	// batch_normalization parameters
	ci = 16;
	in.open(dir + "/bn1_bias.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; bn_bias[num_b].emplace_back(val);} in.close(); num_b++;
	in.open(dir + "/bn1_running_mean.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; bn_running_mean[num_m].emplace_back(val);} in.close(); num_m++;
	in.open(dir + "/bn1_running_var.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; bn_running_var[num_v].emplace_back(val);} in.close(); num_v++;
	in.open(dir + "/bn1_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; bn_weight[num_w].emplace_back(val);} in.close(); num_w++;

	// batch_normalization parameters
	for(int j=1; j<=3; j++)
	{
		int ci;
		if(j==1) ci=16;
		else if(j==2) ci=32;
		else if(j==3) ci=64;

		for(int k=0; k<=end_num; k++)
		{
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_bn1_bias.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<ci; i++) {in>>val; bn_bias[num_b].emplace_back(val);} in.close(); num_b++;
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_bn1_running_mean.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<ci; i++) {in>>val; bn_running_mean[num_m].emplace_back(val);} in.close(); num_m++;
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_bn1_running_var.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<ci; i++) {in>>val; bn_running_var[num_v].emplace_back(val);} in.close(); num_v++;
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_bn1_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<ci; i++) {in>>val; bn_weight[num_w].emplace_back(val);} in.close(); num_w++;
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_bn2_bias.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<ci; i++) {in>>val; bn_bias[num_b].emplace_back(val);} in.close(); num_b++;
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_bn2_running_mean.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<ci; i++) {in>>val; bn_running_mean[num_m].emplace_back(val);} in.close(); num_m++;
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_bn2_running_var.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<ci; i++) {in>>val; bn_running_var[num_v].emplace_back(val);} in.close(); num_v++;
			in.open(dir + "/layer" + to_string(j) + "_" + to_string(k) + "_bn2_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
			for(long i=0; i<ci; i++) {in>>val; bn_weight[num_w].emplace_back(val);} in.close(); num_w++;
		}
	}

	// output 1D batchnorm
	in.open(dir + "/norm_running_mean.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<256; i++) {in>>val; bn_running_mean[num_m].emplace_back(val);} in.close(); num_m++;
	in.open(dir + "/norm_running_var.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<256; i++) {in>>val; bn_running_var[num_v].emplace_back(val);} in.close(); num_v++;

	// FC
	in.open(dir + "/output_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<256*256; i++) {in>>val; linear_weight.emplace_back(val);} in.close();
	in.open(dir + "/output_bias.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<256; i++) {in>>val; linear_bias.emplace_back(val);} in.close();

	// B
	in.open(dir + "/B.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	in >> B; in.close();

	// polynomial coefficients of 1/sqrt(*) and threshold
	in.open(dir + "/threshold_" + dataset + ".txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	string line;
	string w0;
	for(size_t i=0; i<10; i++){
		std::getline(in, line, '\n');
		if (i == fold){
			std::istringstream iss(line);
			while (std::getline(iss, w0, ',')) coef_threshold.emplace_back(atof(w0.c_str()));
		}	
	}

	
}


void import_weights_pcnn(size_t num_nets, string &dataset, size_t fold, string &dir, vector<vector<double>> &all_linear_weight, vector<double> &linear_bias, vector<double> &coef_threshold, vector<vector<double>> &all_conv_weight, vector<vector<vector<double>>> &all_layer_weight, vector<vector<vector<double>>> &all_shortcut_weight, vector<vector<vector<double>>> &all_a2, vector<vector<vector<double>>> &all_a1, vector<vector<vector<double>>> &all_a0, vector<vector<vector<double>>> &all_b1, vector<vector<vector<double>>> &all_b0,  vector<vector<vector<double>>> &all_shortcut_bn_weight, vector<vector<vector<double>>> &all_shortcut_bn_bias, vector<vector<vector<double>>> &all_shortcut_bn_running_mean, vector<vector<vector<double>>> &all_shortcut_bn_running_var, vector<vector<double>> &all_bn_weight, vector<vector<double>> &all_bn_bias, vector<vector<double>> &all_bn_running_mean, vector<vector<double>> &all_bn_running_var)
{
	// subnet weights
	all_linear_weight.clear();
	all_linear_weight.resize(num_nets);
	all_conv_weight.clear();
	all_conv_weight.resize(num_nets);
	all_layer_weight.clear();
	all_layer_weight.resize(num_nets);
	all_shortcut_weight.clear();
	all_shortcut_weight.resize(num_nets);
	all_a2.clear();
	all_a2.resize(num_nets);
	all_a1.clear();
	all_a1.resize(num_nets);
	all_a0.clear();
	all_a0.resize(num_nets);
	all_b1.clear();
	all_b1.resize(num_nets);
	all_b0.clear();
	all_b0.resize(num_nets);
	all_shortcut_bn_weight.clear();
	all_shortcut_bn_weight.resize(num_nets);
	all_shortcut_bn_bias.clear();
	all_shortcut_bn_bias.resize(num_nets);
	all_shortcut_bn_running_mean.clear();
	all_shortcut_bn_running_mean.resize(num_nets);
	all_shortcut_bn_running_var.clear();
	all_shortcut_bn_running_var.resize(num_nets);
	all_bn_weight.clear();
	all_bn_weight.resize(num_nets);
	all_bn_bias.clear();
	all_bn_bias.resize(num_nets);
	all_bn_running_mean.clear();
	all_bn_running_mean.resize(num_nets);
	all_bn_running_var.clear();
	all_bn_running_var.resize(num_nets);
	for (size_t i = 0; i < num_nets; i++)
	{
		string net_dir = dir + "/net" + to_string(i);
		import_weights_pcnn_net(net_dir, all_linear_weight[i], all_conv_weight[i], all_layer_weight[i], all_shortcut_weight[i], all_a2[i], all_a1[i], all_a0[i], all_b1[i], all_b0[i],  all_shortcut_bn_weight[i], all_shortcut_bn_bias[i], all_shortcut_bn_running_mean[i], all_shortcut_bn_running_var[i], all_bn_weight[i], all_bn_bias[i], all_bn_running_mean[i], all_bn_running_var[i]);
	}

	ifstream in;
	double val;

	// FC
	in.open(dir + "/bias.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<256; i++) {in>>val; linear_bias.emplace_back(val);} in.close();

	// polynomial coefficients of 1/sqrt(*) and threshold
	in.open(dir + "/threshold_" + dataset + ".txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	string line;
	string w0;
	for(size_t i=0; i<10; i++){
		std::getline(in, line, '\n');
		if (i == fold){
			std::istringstream iss(line);
			while (std::getline(iss, w0, ',')) coef_threshold.emplace_back(atof(w0.c_str()));
		}	
	}
	
}

void import_weights_pcnn_net(string &dir, vector<double> &linear_weight, vector<double> &conv_weight, vector<vector<double>> &layer_weight, vector<vector<double>> &shortcut_weight, vector<vector<double>> &a2, vector<vector<double>> &a1, vector<vector<double>> &a0, vector<vector<double>> &b1, vector<vector<double>> &b0,  vector<vector<double>> &shortcut_bn_weight, vector<vector<double>> &shortcut_bn_bias, vector<vector<double>> &shortcut_bn_running_mean, vector<vector<double>> &shortcut_bn_running_var, vector<double> &bn_weight, vector<double> &bn_bias, vector<double> &bn_running_mean, vector<double> &bn_running_var)
{
	linear_weight.clear();
	conv_weight.clear();
	layer_weight.clear();
	layer_weight.resize(10);
	a2.clear();
	a2.resize(6);
	a1.clear();
	a1.resize(6);
	a0.clear();
	a0.resize(6);
	b1.clear();
	b1.resize(5);
	b0.clear();
	b0.resize(5);
	shortcut_weight.clear();
	shortcut_weight.resize(2);
	shortcut_bn_weight.clear();
	shortcut_bn_weight.resize(2);
	shortcut_bn_bias.clear();
	shortcut_bn_bias.resize(2);
	shortcut_bn_running_mean.clear();
	shortcut_bn_running_mean.resize(2);
	shortcut_bn_running_var.clear();
	shortcut_bn_running_var.resize(2);

	bn_weight.clear();
	bn_bias.clear();
	bn_running_mean.clear();
	bn_running_var.clear();
	

	std::cout << "load PCNN subnet weights from: " << dir << std::endl;
	ifstream in;
	double val;	
	int fh = 3, fw = 3;
	int ci = 0, co = 0;

	// Conv
	ci = 3, co = 16;
	in.open(dir + "/conv_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<fh*fw*ci*co; i++) {in>>val; conv_weight.emplace_back(val);} in.close();
	
	// HerPNConv(16, 16)
	ci = 16, co = 16;
	in.open(dir + "/layers_0_weight1.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<fh*fw*ci*co; i++) {in>>val; layer_weight[0].emplace_back(val);} in.close();
	
	in.open(dir + "/layers_0_a2.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; a2[0].emplace_back(val);} in.close();
	in.open(dir + "/layers_0_a1.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; a1[0].emplace_back(val);} in.close();
	in.open(dir + "/layers_0_a0.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; a0[0].emplace_back(val);} in.close();
	
	in.open(dir + "/layers_0_weight2.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<fh*fw*co*co; i++) {in>>val; layer_weight[1].emplace_back(val);} in.close();
	
	in.open(dir + "/layers_0_b1.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; b1[0].emplace_back(val);} in.close();
	in.open(dir + "/layers_0_b0.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; b0[0].emplace_back(val);} in.close();

	// HerPNConv(16, 32, 2)
	ci = 16, co = 32;
	in.open(dir + "/layers_1_weight1.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<fh*fw*ci*co; i++) {in>>val; layer_weight[2].emplace_back(val);} in.close();
	
	in.open(dir + "/layers_1_a2.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; a2[1].emplace_back(val);} in.close();
	in.open(dir + "/layers_1_a1.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; a1[1].emplace_back(val);} in.close();
	in.open(dir + "/layers_1_a0.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; a0[1].emplace_back(val);} in.close();
	
	in.open(dir + "/layers_1_weight2.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<fh*fw*co*co; i++) {in>>val; layer_weight[3].emplace_back(val);} in.close();
	
	in.open(dir + "/layers_1_b1.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; b1[1].emplace_back(val);} in.close();
	in.open(dir + "/layers_1_b0.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; b0[1].emplace_back(val);} in.close();
	
	in.open(dir + "/layers_1_shortcut_0_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci*co; i++) {in>>val; shortcut_weight[0].emplace_back(val);} in.close();
	in.open(dir + "/layers_1_shortcut_1_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; shortcut_bn_weight[0].emplace_back(val);} in.close();
	in.open(dir + "/layers_1_shortcut_1_bias.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; shortcut_bn_bias[0].emplace_back(val);} in.close();
	in.open(dir + "/layers_1_shortcut_1_running_mean.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; shortcut_bn_running_mean[0].emplace_back(val);} in.close();
	in.open(dir + "/layers_1_shortcut_1_running_var.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; shortcut_bn_running_var[0].emplace_back(val);} in.close();

	// HerPNConv(32, 32)
	ci = 32, co = 32;
	in.open(dir + "/layers_2_weight1.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<fh*fw*ci*co; i++) {in>>val; layer_weight[4].emplace_back(val);} in.close();
	
	in.open(dir + "/layers_2_a2.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; a2[2].emplace_back(val);} in.close();
	in.open(dir + "/layers_2_a1.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; a1[2].emplace_back(val);} in.close();
	in.open(dir + "/layers_2_a0.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; a0[2].emplace_back(val);} in.close();
	
	in.open(dir + "/layers_2_weight2.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<fh*fw*co*co; i++) {in>>val; layer_weight[5].emplace_back(val);} in.close();
	
	in.open(dir + "/layers_2_b1.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; b1[2].emplace_back(val);} in.close();
	in.open(dir + "/layers_2_b0.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; b0[2].emplace_back(val);} in.close();

	// HerPNConv(32, 64, 2)
	ci = 32, co = 64;
	in.open(dir + "/layers_3_weight1.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<fh*fw*ci*co; i++) {in>>val; layer_weight[6].emplace_back(val);} in.close();
	
	in.open(dir + "/layers_3_a2.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; a2[3].emplace_back(val);} in.close();
	in.open(dir + "/layers_3_a1.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; a1[3].emplace_back(val);} in.close();
	in.open(dir + "/layers_3_a0.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; a0[3].emplace_back(val);} in.close();
	
	in.open(dir + "/layers_3_weight2.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<fh*fw*co*co; i++) {in>>val; layer_weight[7].emplace_back(val);} in.close();
	
	in.open(dir + "/layers_3_b1.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; b1[3].emplace_back(val);} in.close();
	in.open(dir + "/layers_3_b0.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; b0[3].emplace_back(val);} in.close();
	
	in.open(dir + "/layers_3_shortcut_0_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci*co; i++) {in>>val; shortcut_weight[1].emplace_back(val);} in.close();
	in.open(dir + "/layers_3_shortcut_1_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; shortcut_bn_weight[1].emplace_back(val);} in.close();
	in.open(dir + "/layers_3_shortcut_1_bias.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; shortcut_bn_bias[1].emplace_back(val);} in.close();
	in.open(dir + "/layers_3_shortcut_1_running_mean.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; shortcut_bn_running_mean[1].emplace_back(val);} in.close();
	in.open(dir + "/layers_3_shortcut_1_running_var.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; shortcut_bn_running_var[1].emplace_back(val);} in.close();

	// HerPNConv(64, 64)
	ci = 64, co = 64;
	in.open(dir + "/layers_4_weight1.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<fh*fw*ci*co; i++) {in>>val; layer_weight[8].emplace_back(val);} in.close();
	
	in.open(dir + "/layers_4_a2.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; a2[4].emplace_back(val);} in.close();
	in.open(dir + "/layers_4_a1.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; a1[4].emplace_back(val);} in.close();
	in.open(dir + "/layers_4_a0.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<ci; i++) {in>>val; a0[4].emplace_back(val);} in.close();
	
	in.open(dir + "/layers_4_weight2.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<fh*fw*co*co; i++) {in>>val; layer_weight[9].emplace_back(val);} in.close();
	
	in.open(dir + "/layers_4_b1.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; b1[4].emplace_back(val);} in.close();
	in.open(dir + "/layers_4_b0.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; b0[4].emplace_back(val);} in.close();

	// HerPNPool
	in.open(dir + "/herpnpool_a2.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; a2[5].emplace_back(val);} in.close();
	in.open(dir + "/herpnpool_a1.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; a1[5].emplace_back(val);} in.close();
	in.open(dir + "/herpnpool_a0.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co; i++) {in>>val; a0[5].emplace_back(val);} in.close();

	// BatchNorm1d 
	in.open(dir + "/bn_weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co*2*2; i++) {in>>val; bn_weight.emplace_back(val);} in.close();
	in.open(dir + "/bn_bias.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co*2*2; i++) {in>>val; bn_bias.emplace_back(val);} in.close();
	in.open(dir + "/bn_running_mean.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co*2*2; i++) {in>>val; bn_running_mean.emplace_back(val);} in.close();
	in.open(dir + "/bn_running_var.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<co*2*2; i++) {in>>val; bn_running_var.emplace_back(val);} in.close();

	// FC
	in.open(dir + "/weight.txt"); if(!in.is_open()) throw std::runtime_error("file is not open");
	for(long i=0; i<256*256; i++) {in>>val; linear_weight.emplace_back(val);} in.close();
}


void resnet_autofhe(string &model, string &dataset, size_t fold, string &weight_dir, string &dataset_dir, string &output_dir, size_t start_image_id, size_t end_image_id)
{
	// SEAL and bootstrapping setting
	long boundary_K = 25;
	long boot_deg = 59;
    long scale_factor = 2;
    long inverse_deg = 1; 
	long logN = 16;
	long loge = 10; 
	long logn = 15;		// full slots
	long logn_1 = 14;	// sparse slots
	long logn_2 = 13;
	long logn_3 = 12;
	int logp = 46;
	int logq = 51;
	int log_special_prime = 51;
    int log_integer_part = logq - logp - loge + 5;
	int remaining_level = 16; // Calculation required
	int boot_level = 14; // 
	int total_level = remaining_level + boot_level;

	vector<int> coeff_bit_vec;
	coeff_bit_vec.push_back(logq);
	for (int i = 0; i < remaining_level; i++) coeff_bit_vec.push_back(logp);
	for (int i = 0; i < boot_level; i++) coeff_bit_vec.push_back(logq);
	coeff_bit_vec.push_back(log_special_prime);

	cout << "Setting Parameters" << endl;
	EncryptionParameters parms(scheme_type::ckks);
	size_t poly_modulus_degree = (size_t)(1 << logN);
	parms.set_poly_modulus_degree(poly_modulus_degree);
	parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec)); 

	// added
	size_t secret_key_hamming_weight = 192;
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
	// parms.set_sparse_slots(1 << logn_1);
	double scale = pow(2.0, logp);

	SEALContext context(parms);
	// KeyGenerator keygen(context, 192);
	KeyGenerator keygen(context);
    PublicKey public_key;
	keygen.create_public_key(public_key);
	auto secret_key = keygen.secret_key();
    RelinKeys relin_keys;
	keygen.create_relin_keys(relin_keys);
	GaloisKeys gal_keys;

	CKKSEncoder encoder(context);
	Encryptor encryptor(context, public_key);
	Evaluator evaluator(context, encoder);
	Decryptor decryptor(context, secret_key);
	// ScaleInvEvaluator scale_evaluator(context, encoder, relin_keys);

	Bootstrapper bootstrapper_1(loge, logn_1, logN - 1, total_level, scale, boundary_K, boot_deg, scale_factor, inverse_deg, context, keygen, encoder, encryptor, decryptor, evaluator, relin_keys, gal_keys);
	Bootstrapper bootstrapper_2(loge, logn_2, logN - 1, total_level, scale, boundary_K, boot_deg, scale_factor, inverse_deg, context, keygen, encoder, encryptor, decryptor, evaluator, relin_keys, gal_keys);
	Bootstrapper bootstrapper_3(loge, logn_3, logN - 1, total_level, scale, boundary_K, boot_deg, scale_factor, inverse_deg, context, keygen, encoder, encryptor, decryptor, evaluator, relin_keys, gal_keys);

//	additional rotation kinds for CNN
	vector<int> rotation_kinds = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33
		// ,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55
		,56
		// ,57,58,59,60,61
		,62,63,64,66,84,124,128,132,256,512,959,960,990,991,1008
		,1023,1024,1036,1064,1092,1952,1982,1983,2016,2044,2047,2048,2072,2078,2100,3007,3024,3040,3052,3070,3071,3072,3080,3108,4031
		,4032,4062,4063,4095,4096,5023,5024,5054,5055,5087,5118,5119,5120,6047,6078,6079,6111,6112,6142,6143,6144,7071,7102,7103,7135
		,7166,7167,7168,8095,8126,8127,8159,8190,8191,8192,9149,9183,9184,9213,9215,9216,10173,10207,10208,10237,10239,10240,11197,11231
		,11232,11261,11263,11264,12221,12255,12256,12285,12287,12288,13214,13216,13246,13278,13279,13280,13310,13311,13312,14238,14240
		,14270,14302,14303,14304,14334,14335,15262,15264,15294,15326,15327,15328,15358,15359,15360,16286,16288,16318,16350,16351,16352
		,16382,16383,16384,17311,17375,18335,18399,18432,19359,19423,20383,20447,20480,21405,21406,21437,21469,21470,21471,21501,21504
		,22429,22430,22461,22493,22494,22495,22525,22528,23453,23454,23485,23517,23518,23519,23549,24477,24478,24509,24541,24542,24543
		,24573,24576,25501,25565,25568,25600,26525,26589,26592,26624,27549,27613,27616,27648,28573,28637,28640,28672,29600,29632,29664
		,29696,30624,30656,30688,30720,31648,31680,31712,31743,31744,31774,32636,32640,32644,32672,32702,32704,32706,32735
		,32736,32737,32759,32760,32761,32762,32763,32764,32765,32766,32767
	};

	// bootstrapping preprocessing
	cout << "Generating Optimal Minimax Polynomials..." << endl;
	bootstrapper_1.prepare_mod_polynomial();
	bootstrapper_2.prepare_mod_polynomial();
	bootstrapper_3.prepare_mod_polynomial();

	cout << "Adding Bootstrapping Keys..." << endl;
	vector<int> gal_steps_vector;
	gal_steps_vector.push_back(0);
	for(int i=0; i<logN-1; i++) gal_steps_vector.push_back((1 << i));
	for(auto rot: rotation_kinds)
	{
		if(find(gal_steps_vector.begin(), gal_steps_vector.end(), rot) == gal_steps_vector.end()) gal_steps_vector.push_back(rot);
	} 
	bootstrapper_1.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);
	bootstrapper_2.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);
	bootstrapper_3.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);
	keygen.create_galois_keys(gal_steps_vector, gal_keys);

	bootstrapper_1.slot_vec.push_back(logn_1);
	bootstrapper_2.slot_vec.push_back(logn_2);
	bootstrapper_3.slot_vec.push_back(logn_3);

	cout << "Generating Linear Transformation Coefficients..." << endl;
	bootstrapper_1.generate_LT_coefficient_3();
	bootstrapper_2.generate_LT_coefficient_3();
	bootstrapper_3.generate_LT_coefficient_3();

	// end number
	size_t layer_num, end_num;
	if(model == "resnet20") layer_num = 20, end_num = 2;
	else if(model == "resnet32") layer_num = 32, end_num = 4;
	else if(model == "resnet44") layer_num = 44, end_num = 6;
	else if(model == "resnet56") layer_num = 56, end_num = 8;
	else if(model == "resnet110") layer_num = 110, end_num = 17;
	else throw std::invalid_argument(model + "is not known.");

	// load images and labels
	size_t img_s = 64;
	vector<vector<double>> image_share1;
	vector<vector<double>> image_share2;
	vector<int> label_share;
	load_dataset(dataset, dataset_dir, fold, image_share1, image_share2, label_share);

	// import network weights 
	vector<double> linear_weight_share, linear_bias_share, in_range_share, out_range_share;
	vector<vector<double>> conv_weight_share, bn_bias_share, bn_running_mean_share, bn_running_var_share, bn_weight_share;
	vector<int> depth_share;
	vector<int> boot_loc_share;
	vector<double> coef_threshold;
	weight_dir = weight_dir + "/" + "autofhe_" + model;
	import_weights_resnet_autofhe(dataset, fold, weight_dir, linear_weight_share, linear_bias_share, conv_weight_share, bn_bias_share, bn_running_mean_share, bn_running_var_share, bn_weight_share, coef_threshold, in_range_share, out_range_share, depth_share, boot_loc_share, layer_num, end_num);
	double coef_a = coef_threshold[0], coef_b = coef_threshold[1], coef_c = coef_threshold[2], threshold = coef_threshold[3];

	int nthred = end_image_id - start_image_id;
	vector<vector<int>> all_thread_results;
	all_thread_results.resize(nthred);

	// time setting
	chrono::high_resolution_clock::time_point total_time_start, total_time_end;
	chrono::microseconds total_time_diff;
	total_time_start = chrono::high_resolution_clock::now();
	
	// #pragma omp parallel for num_threads(nthred) // batch inference
	for(size_t image_id = start_image_id; image_id < end_image_id; image_id++)
	{
		// each thread output result file
		ofstream output;
		string output_path = output_dir + "/autofhe/" + model + "/" + dataset + "-fold" + to_string(fold) + "-" + to_string(image_id) + ".txt";
		output.open(output_path);
		if(!output.is_open()) throw std::runtime_error(output_path + "not open.");

		// ciphertext pool generation
		vector<Ciphertext> cipher_pool(14);

		// variables
		TensorCipher cnn, temp;
		Ciphertext probe, gallery; 

		// time setting
		chrono::high_resolution_clock::time_point offline_time_start, offline_time_end, online_time_start, online_time_end;
		chrono::microseconds offline_time_diff, online_time_diff;

		// face recognition
		// i = 0: offline, enroll
		// i = 1: online, verification
		int label = label_share[image_id];
		for(size_t i = 0; i < 2; i++){
			// deep learning parameters and import
			int co = 0, st = 0, fh = 3, fw = 3;
			long init_p = 2, n = 1<<logn;
			int stage = 0;
			double epsilon = 0.00001;
			vector<double> linear_weight(linear_weight_share.begin(), linear_weight_share.end()), linear_bias(linear_bias_share.begin(), linear_bias_share.end());
			vector<vector<double>> conv_weight(conv_weight_share.begin(), conv_weight_share.end()), bn_bias(bn_bias_share.begin(), bn_bias_share.end()); 
			vector<vector<double>> bn_running_mean(bn_running_mean_share.begin(), bn_running_mean_share.end()), bn_running_var(bn_running_var_share.begin(), bn_running_var_share.end()); 
			vector<vector<double>> bn_weight(bn_weight_share.begin(), bn_weight_share.end());
			vector<double> in_range(in_range_share.begin(), in_range_share.end()), out_range(out_range_share.begin(), out_range_share.end()); 
			vector<int> depth(depth_share.begin(), depth_share.end());
			vector<int> boot_loc(boot_loc_share.begin(), boot_loc_share.end());

			// pack images compactly
			vector<double> image;		
			if (i == 0){
				vector<double> image1(image_share1[image_id].begin(), image_share1[image_id].end());
				for(long i=img_s*img_s*3; i<1<<logn; i++) image1.emplace_back(0.);
				for(long i=n/init_p; i<n; i++) image1[i] = image1[i%(n/init_p)];
				image = image1;
				offline_time_start = chrono::high_resolution_clock::now();
				cout << "------------" << "offline" << "------------" << endl;
				output << "------------" << "offline" << "------------" << endl;
			}
			else{
				vector<double> image2(image_share2[image_id].begin(), image_share2[image_id].end());
				for(long i=img_s*img_s*3; i<1<<logn; i++) image2.emplace_back(0.);
				for(long i=n/init_p; i<n; i++) image2[i] = image2[i%(n/init_p)];
				image = image2;
				online_time_start = chrono::high_resolution_clock::now();
				cout << "------------" << "online" << "------------" << endl;
				output << "------------" << "online" << "------------" << endl;
			}

			// generate encrypted face image
			cnn = TensorCipher(logn, 1, img_s, img_s, 3, 3, init_p, image, encryptor, encoder, logq);
			cout << "remaining level : " << context.get_context_data(cnn.cipher().parms_id())->chain_index() << endl;
			cout << "scale: " << cnn.cipher().scale() << endl;

			// modulus down
			Ciphertext ctxt;
			ctxt = cnn.cipher();
			for(int i=0; i<boot_level-3; i++) evaluator.mod_switch_to_next_inplace(ctxt);
			cnn.set_ciphertext(ctxt);

			// AutoFHE
			double relu_in, relu_out, scale, res_scale;
			int relu_depth = depth[stage];
			size_t loc = 0;
			bool curr_boot, next_boot; 

			// layer 0
			cout << "layer 0" << endl;
			output << "layer 0" << endl;
			relu_in = in_range[stage], relu_out = out_range[stage];
			if (relu_depth > 2){
				for (auto &w : bn_weight[stage]) w /= relu_in;
				for (auto &w : bn_bias[stage])  w /= relu_in;
			}
			multiplexed_parallel_convolution_print(cnn, cnn, 16, 2, fh, fw, conv_weight[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, gal_keys, cipher_pool, output, decryptor, context, stage);

			// scaling factor ~2^51 -> 2^46s
			const auto &modulus = iter(context.first_context_data()->parms().coeff_modulus());
			ctxt = cnn.cipher();
			size_t cur_level = ctxt.coeff_modulus_size();
			Plaintext scaler;
			double scale_change = pow(2.0,46) * ((double)modulus[cur_level-1].value()) / ctxt.scale();
			encoder.encode(1, scale_change, scaler);
			evaluator.mod_switch_to_inplace(scaler, ctxt.parms_id());
			evaluator.multiply_plain_inplace(ctxt, scaler);
			evaluator.rescale_to_next_inplace(ctxt);
			ctxt.scale() = pow(2.0,46);
			cnn.set_ciphertext(ctxt);

			multiplexed_parallel_batch_norm_seal_print(cnn, cnn, bn_bias[stage], bn_running_mean[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, 1., output, decryptor, context, stage);
			if (relu_depth == 2) evorelu_seal_print(cnn, cnn, weight_dir, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, output, context, gal_keys, 1., 1., stage);
			if (relu_depth > 2) evorelu_seal_print(cnn, cnn, weight_dir, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, output, context, gal_keys, 1., relu_in, stage);
			
			scale = 1.;
			res_scale = 1.;
			for(int j=0; j<3; j++)		// layer 1_x, 2_x, 3_x
			{
				if(j==0) co = 16;
				else if(j==1) co = 32;
				else if(j==2) co = 64;

				for(int k=0; k<=end_num; k++)	// 0 ~ 2/4/6/8/17
				{
					stage = 2*((end_num+1)*j+k)+1;
					cout << "layer " << stage << endl;
					output << "layer " << stage << endl;
					temp = cnn;
					if(j>=1 && k==0) st = 2;
					else st = 1;

					res_scale = scale;
					curr_boot = boot_loc[loc]==1, next_boot = boot_loc[loc+1]==1, loc += 2;
					relu_depth = depth[stage], relu_in = in_range[stage], relu_out = out_range[stage];
					for (auto &w : conv_weight[stage]) w *= scale;

					// ConvBN -> [Bootstrapping] -> EvoReLU -> [Bootstrapping]
					// Case I: ConvBN -> Bootstrapping -> EvoReLU
					if (curr_boot){
						// ConvBN
						if (relu_in > 1.){
							for (auto &w : bn_weight[stage]) w /= relu_in;
							for (auto &w : bn_bias[stage])  w /= relu_in;
						}
						multiplexed_parallel_convolution_print(cnn, cnn, co, st, fh, fw, conv_weight[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, gal_keys, cipher_pool, output, decryptor, context, stage);
						multiplexed_parallel_batch_norm_seal_print(cnn, cnn, bn_bias[stage], bn_running_mean[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, 1., output, decryptor, context, stage);
						// Bootstrapping
						cur_level = context.get_context_data(cnn.cipher().parms_id())->chain_index();
						if (cur_level > 0){
							ctxt = cnn.cipher();
							for (size_t i = 0; i < cur_level; i++) evaluator.mod_switch_to_next_inplace(ctxt);
							cnn.set_ciphertext(ctxt);
						}
						if(j==0) bootstrap_print(cnn, cnn, bootstrapper_1, output, decryptor, encoder, context, stage);
						else if(j==1) bootstrap_print(cnn, cnn, bootstrapper_2, output, decryptor, encoder, context, stage);
						else if(j==2) bootstrap_print(cnn, cnn, bootstrapper_3, output, decryptor, encoder, context, stage);
						// EvoReLU
						if (relu_depth == 0) scale = relu_in > 1. ? relu_in : 1.;
						if (relu_depth == 2) evorelu_seal_print(cnn, cnn, weight_dir, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, output, context, gal_keys, relu_in > 1. ? relu_in : 1., 1., stage), scale =1.;
						else if (relu_depth > 2) evorelu_seal_print(cnn, cnn, weight_dir, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, output, context, gal_keys, 1., relu_in, stage), scale = 1.;
					}
					// Case II: ConvBN -> EvoReLU -> [Bootstrapping]
					else{
						if (relu_depth == 0)
						{
							if (next_boot){
								// ConvBN
								if (relu_in > 1.){
									for (auto &w : bn_weight[stage]) w /= relu_in;
									for (auto &w : bn_bias[stage])  w /= relu_in;
								}
								multiplexed_parallel_convolution_print(cnn, cnn, co, st, fh, fw, conv_weight[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, gal_keys, cipher_pool, output, decryptor, context, stage);
								multiplexed_parallel_batch_norm_seal_print(cnn, cnn, bn_bias[stage], bn_running_mean[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, 1., output, decryptor, context, stage);
								// Bootstrapping
								cur_level = context.get_context_data(cnn.cipher().parms_id())->chain_index();
								if (cur_level > 0){
									ctxt = cnn.cipher();
									for (size_t i = 0; i < cur_level; i++) evaluator.mod_switch_to_next_inplace(ctxt);
									cnn.set_ciphertext(ctxt);
								}
								if(j==0) bootstrap_print(cnn, cnn, bootstrapper_1, output, decryptor, encoder, context, stage);
								else if(j==1) bootstrap_print(cnn, cnn, bootstrapper_2, output, decryptor, encoder, context, stage);
								else if(j==2) bootstrap_print(cnn, cnn, bootstrapper_3, output, decryptor, encoder, context, stage);
								// Next Conv 
								scale = relu_in > 1. ? relu_in : 1.;
							}
							else{
								// ConvBN
								multiplexed_parallel_convolution_print(cnn, cnn, co, st, fh, fw, conv_weight[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, gal_keys, cipher_pool, output, decryptor, context, stage);
								multiplexed_parallel_batch_norm_seal_print(cnn, cnn, bn_bias[stage], bn_running_mean[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, 1., output, decryptor, context, stage);
								// Next Conv
								scale = 1.;
							}
						}
						else if (relu_depth == 2){
							// ConvBN
							multiplexed_parallel_convolution_print(cnn, cnn, co, st, fh, fw, conv_weight[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, gal_keys, cipher_pool, output, decryptor, context, stage);
							multiplexed_parallel_batch_norm_seal_print(cnn, cnn, bn_bias[stage], bn_running_mean[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, 1., output, decryptor, context, stage);
							if (next_boot){
								// EvoReLU
								evorelu_seal_print(cnn, cnn, weight_dir, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, output, context, gal_keys, 1., relu_out > 1. ? 1. / relu_out : 1., stage);
								// Bootstrapping
								cur_level = context.get_context_data(cnn.cipher().parms_id())->chain_index();
								if (cur_level > 0){
									ctxt = cnn.cipher();
									for (size_t i = 0; i < cur_level; i++) evaluator.mod_switch_to_next_inplace(ctxt);
									cnn.set_ciphertext(ctxt);
								}
								if(j==0) bootstrap_print(cnn, cnn, bootstrapper_1, output, decryptor, encoder, context, stage);
								else if(j==1) bootstrap_print(cnn, cnn, bootstrapper_2, output, decryptor, encoder, context, stage);
								else if(j==2) bootstrap_print(cnn, cnn, bootstrapper_3, output, decryptor, encoder, context, stage);
								// Next Conv
								scale = relu_out > 1. ? relu_out : 1.;
							}
							else{
								// EvoReLU
								evorelu_seal_print(cnn, cnn, weight_dir, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, output, context, gal_keys, 1., 1., stage);
								// Next Conv
								scale = 1.;
							}
						}
						else if (relu_depth > 2){
							// ConvBN
							for (auto &w : bn_weight[stage]) w /= relu_in;
							for (auto &w : bn_bias[stage])  w /= relu_in;
							multiplexed_parallel_convolution_print(cnn, cnn, co, st, fh, fw, conv_weight[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, gal_keys, cipher_pool, output, decryptor, context, stage);
							multiplexed_parallel_batch_norm_seal_print(cnn, cnn, bn_bias[stage], bn_running_mean[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, 1., output, decryptor, context, stage);
							if (next_boot){
								// EvoReLU
								evorelu_seal_print(cnn, cnn, weight_dir, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, output, context, gal_keys, 1., relu_out > 1.? relu_in / relu_out : relu_in, stage);
								// Bootstrapping
								cur_level = context.get_context_data(cnn.cipher().parms_id())->chain_index();
								if (cur_level > 0){
									ctxt = cnn.cipher();
									for (size_t i = 0; i < cur_level; i++) evaluator.mod_switch_to_next_inplace(ctxt);
									cnn.set_ciphertext(ctxt);
								}
								if(j==0) bootstrap_print(cnn, cnn, bootstrapper_1, output, decryptor, encoder, context, stage);
								else if(j==1) bootstrap_print(cnn, cnn, bootstrapper_2, output, decryptor, encoder, context, stage);
								else if(j==2) bootstrap_print(cnn, cnn, bootstrapper_3, output, decryptor, encoder, context, stage);
								// Next Conv
								scale = relu_out > 1. ? relu_out : 1.;
							}
							else{
								// EvoReLU
								evorelu_seal_print(cnn, cnn, weight_dir, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, output, context, gal_keys, 1., relu_in, stage);
								// Next Conv
								scale = 1.;
							}
						}
					}

					stage = 2*((end_num+1)*j+k)+2;
					cout << "layer " << stage << endl;
					output << "layer " << stage << endl;
					st = 1;
					curr_boot = boot_loc[loc]==1, next_boot = boot_loc[loc+1]==1, loc += 2;
					relu_depth = depth[stage], relu_in = in_range[stage], relu_out = out_range[stage];
					for (auto &w : conv_weight[stage]) w *= scale;
					if(j>=1 && k==0) multiplexed_parallel_downsampling_seal_print(temp, temp, evaluator, decryptor, encoder, context, gal_keys, output);


					// ConvBN -> [Bootstrapping] -> EvoReLU -> [Bootstrapping]
					// Case I: ConvBN -> Bootstrapping -> EvoReLU
					if (curr_boot){
						// ConvBN
						if (relu_in > 1.){
							for (auto &w : bn_weight[stage]) w /= relu_in;
							for (auto &w : bn_bias[stage])  w /= relu_in;
						}
						multiplexed_parallel_convolution_print(cnn, cnn, co, st, fh, fw, conv_weight[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, gal_keys, cipher_pool, output, decryptor, context, stage);
						multiplexed_parallel_batch_norm_seal_print(cnn, cnn, bn_bias[stage], bn_running_mean[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, 1., output, decryptor, context, stage);
						if (relu_in > 1.){
							ctxt = temp.cipher();
							evaluator.multiply_const_inplace(ctxt, res_scale / relu_in);
							evaluator.rescale_to_next_inplace(ctxt);
							temp.set_ciphertext(ctxt);
						}
						else if (res_scale != 1.){
							ctxt = temp.cipher();
							evaluator.multiply_const_inplace(ctxt, res_scale);
							evaluator.rescale_to_next_inplace(ctxt);
							temp.set_ciphertext(ctxt);
						}
						cipher_add_seal_print(temp, cnn, cnn, evaluator, output, decryptor, encoder, context);
						// Bootstrapping
						cur_level = context.get_context_data(cnn.cipher().parms_id())->chain_index();
						if (cur_level > 0){
							ctxt = cnn.cipher();
							for (size_t i = 0; i < cur_level; i++) evaluator.mod_switch_to_next_inplace(ctxt);
							cnn.set_ciphertext(ctxt);
						}
						if(j==0) bootstrap_print(cnn, cnn, bootstrapper_1, output, decryptor, encoder, context, stage);
						else if(j==1) bootstrap_print(cnn, cnn, bootstrapper_2, output, decryptor, encoder, context, stage);
						else if(j==2) bootstrap_print(cnn, cnn, bootstrapper_3, output, decryptor, encoder, context, stage);
						// EvoReLU
						if (relu_depth == 0) scale = relu_in > 1. ? relu_in : 1.;
						if (relu_depth == 2) evorelu_seal_print(cnn, cnn, weight_dir, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, output, context, gal_keys, relu_in > 1. ? relu_in : 1., 1., stage), scale =1.;
						else if (relu_depth > 2) evorelu_seal_print(cnn, cnn, weight_dir, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, output, context, gal_keys, 1., relu_in, stage), scale = 1.;
					}
					// Case II: ConvBN -> EvoReLU -> [Bootstrapping]
					else{
						if (relu_depth == 0)
						{
							if (next_boot){
								// ConvBN
								if (relu_in > 1.){
									for (auto &w : bn_weight[stage]) w /= relu_in;
									for (auto &w : bn_bias[stage])  w /= relu_in;
								}
								multiplexed_parallel_convolution_print(cnn, cnn, co, st, fh, fw, conv_weight[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, gal_keys, cipher_pool, output, decryptor, context, stage);
								multiplexed_parallel_batch_norm_seal_print(cnn, cnn, bn_bias[stage], bn_running_mean[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, 1., output, decryptor, context, stage);
								if (relu_in > 1.){
									ctxt = temp.cipher();
									evaluator.multiply_const_inplace(ctxt, res_scale / relu_in);
									evaluator.rescale_to_next_inplace(ctxt);
									temp.set_ciphertext(ctxt);
								}
								else if (res_scale != 1.){
									ctxt = temp.cipher();
									evaluator.multiply_const_inplace(ctxt, res_scale);
									evaluator.rescale_to_next_inplace(ctxt);
									temp.set_ciphertext(ctxt);
								}
								cipher_add_seal_print(temp, cnn, cnn, evaluator, output, decryptor, encoder, context);
								// Bootstrapping
								cur_level = context.get_context_data(cnn.cipher().parms_id())->chain_index();
								if (cur_level > 0){
									ctxt = cnn.cipher();
									for (size_t i = 0; i < cur_level; i++) evaluator.mod_switch_to_next_inplace(ctxt);
									cnn.set_ciphertext(ctxt);
								}
								if(j==0) bootstrap_print(cnn, cnn, bootstrapper_1, output, decryptor, encoder, context, stage);
								else if(j==1) bootstrap_print(cnn, cnn, bootstrapper_2, output, decryptor, encoder, context, stage);
								else if(j==2) bootstrap_print(cnn, cnn, bootstrapper_3, output, decryptor, encoder, context, stage);
								// Next Conv 
								scale = relu_in > 1. ? relu_in : 1.;
							}
							else{
								// ConvBN
								multiplexed_parallel_convolution_print(cnn, cnn, co, st, fh, fw, conv_weight[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, gal_keys, cipher_pool, output, decryptor, context, stage);
								multiplexed_parallel_batch_norm_seal_print(cnn, cnn, bn_bias[stage], bn_running_mean[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, 1., output, decryptor, context, stage);
								if (res_scale != 1.){
									ctxt = temp.cipher();
									evaluator.multiply_const_inplace(ctxt, res_scale);
									evaluator.rescale_to_next_inplace(ctxt);
									temp.set_ciphertext(ctxt);
								}
								cipher_add_seal_print(temp, cnn, cnn, evaluator, output, decryptor, encoder, context);
								// Next Conv
								scale = 1.;
							}
						}
						else if (relu_depth == 2){
							// ConvBN
							multiplexed_parallel_convolution_print(cnn, cnn, co, st, fh, fw, conv_weight[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, gal_keys, cipher_pool, output, decryptor, context, stage);
							multiplexed_parallel_batch_norm_seal_print(cnn, cnn, bn_bias[stage], bn_running_mean[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, 1., output, decryptor, context, stage);
							if (res_scale != 1.){
									ctxt = temp.cipher();
									evaluator.multiply_const_inplace(ctxt, res_scale);
									evaluator.rescale_to_next_inplace(ctxt);
									temp.set_ciphertext(ctxt);
								}
							cipher_add_seal_print(temp, cnn, cnn, evaluator, output, decryptor, encoder, context);
							if (next_boot){
								// EvoReLU
								evorelu_seal_print(cnn, cnn, weight_dir, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, output, context, gal_keys, 1., relu_out > 1. ? 1. / relu_out : 1., stage);
								// Bootstrapping
								cur_level = context.get_context_data(cnn.cipher().parms_id())->chain_index();
								if (cur_level > 0){
									ctxt = cnn.cipher();
									for (size_t i = 0; i < cur_level; i++) evaluator.mod_switch_to_next_inplace(ctxt);
									cnn.set_ciphertext(ctxt);
								}
								if(j==0) bootstrap_print(cnn, cnn, bootstrapper_1, output, decryptor, encoder, context, stage);
								else if(j==1) bootstrap_print(cnn, cnn, bootstrapper_2, output, decryptor, encoder, context, stage);
								else if(j==2) bootstrap_print(cnn, cnn, bootstrapper_3, output, decryptor, encoder, context, stage);
								// Next Conv
								scale = relu_out > 1. ? relu_out : 1.;
							}
							else{
								// EvoReLU
								evorelu_seal_print(cnn, cnn, weight_dir, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, output, context, gal_keys, 1., 1., stage);
								// Next Conv
								scale = 1.;
							}
						}
						else if (relu_depth > 2){
							// ConvBN
							for (auto &w : bn_weight[stage]) w /= relu_in;
							for (auto &w : bn_bias[stage])  w /= relu_in;
							multiplexed_parallel_convolution_print(cnn, cnn, co, st, fh, fw, conv_weight[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, gal_keys, cipher_pool, output, decryptor, context, stage);
							multiplexed_parallel_batch_norm_seal_print(cnn, cnn, bn_bias[stage], bn_running_mean[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, 1., output, decryptor, context, stage);
							ctxt = temp.cipher();
							evaluator.multiply_const_inplace(ctxt, res_scale / relu_in);
							evaluator.rescale_to_next_inplace(ctxt);
							temp.set_ciphertext(ctxt);
							cipher_add_seal_print(temp, cnn, cnn, evaluator, output, decryptor, encoder, context);
							if (next_boot){
								// EvoReLU
								evorelu_seal_print(cnn, cnn, weight_dir, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, output, context, gal_keys, 1., relu_out > 1.? relu_in / relu_out : relu_in, stage);
								// Bootstrapping
								cur_level = context.get_context_data(cnn.cipher().parms_id())->chain_index();
								if (cur_level > 0){
									ctxt = cnn.cipher();
									for (size_t i = 0; i < cur_level; i++) evaluator.mod_switch_to_next_inplace(ctxt);
									cnn.set_ciphertext(ctxt);
								}
								if(j==0) bootstrap_print(cnn, cnn, bootstrapper_1, output, decryptor, encoder, context, stage);
								else if(j==1) bootstrap_print(cnn, cnn, bootstrapper_2, output, decryptor, encoder, context, stage);
								else if(j==2) bootstrap_print(cnn, cnn, bootstrapper_3, output, decryptor, encoder, context, stage);
								// Next Conv
								scale = relu_out > 1. ? relu_out : 1.;
							}
							else{
								// EvoReLU
								evorelu_seal_print(cnn, cnn, weight_dir, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, output, context, gal_keys, 1., relu_in, stage);
								// Next Conv
								scale = 1.;
							}
						}
					}
				}
			}
			cout << "layer " << layer_num - 1 << endl;
			output << "layer " << layer_num - 1 << endl;
			cur_level = context.get_context_data(cnn.cipher().parms_id())->chain_index();
			if (cur_level < 8){
				if (cur_level > 0){
					ctxt = cnn.cipher();
					for (size_t i = 0; i < cur_level; i++) evaluator.mod_switch_to_next_inplace(ctxt);
					cnn.set_ciphertext(ctxt);}
				bootstrap_print(cnn, cnn, bootstrapper_3, output, decryptor, encoder, context, stage);
			}
			averagepooling_seal_scale_print(cnn, cnn, evaluator, gal_keys, scale, output, decryptor, encoder, context);
			fully_connected_seal_print(cnn, cnn, linear_weight, linear_bias, 256, 256, evaluator, gal_keys, output, decryptor, encoder, context);
			// BatchNorm 1D
			batchnorm1d_seal_print(cnn, cnn, evaluator, bn_running_mean.back(), bn_running_var.back(), epsilon, output, decryptor, encoder, context);
			// L2 Normalization
			l2norm_seal_print(cnn, cnn, evaluator, gal_keys, relin_keys, coef_a, coef_b, coef_c, output, decryptor, encoder, context);
			// enroll or verify
			if (i == 0){
				// enroll
				// probe = cnn.cipher();
				gallery = cnn.cipher();
				offline_time_end = chrono::high_resolution_clock::now();
				offline_time_diff = chrono::duration_cast<chrono::milliseconds>(offline_time_end - offline_time_start);
				cout << endl << "offline latency: " << offline_time_diff.count() / 1000 << " ms" << endl;
				output << endl << "offline latency: " << offline_time_diff.count() / 1000 << " ms" << endl << endl << endl;
			}
			else{
				// verify
				// gallery = cnn.cipher();
				probe = cnn.cipher();
			}
		}

		// face matching between probe and gallery (normalized features)

		// face matching time setting
		chrono::high_resolution_clock::time_point match_time_start, match_time_end;
		chrono::microseconds match_time_diff;
		match_time_start = chrono::high_resolution_clock::now();

		// (probe - gallery)
		evaluator.sub_inplace_reduced_error(probe, gallery);

		// (probe - gallery)^2
		evaluator.multiply_inplace_reduced_error(probe, probe, relin_keys);
		evaluator.rescale_to_next_inplace(probe);

		// sum(probe - gallery)^2
		for (size_t i = 0; i < log2_long(256); i++)
		{
			gallery = probe;
			memory_save_rotate(gallery, gallery, pow2(i), evaluator, gal_keys);
			evaluator.add_inplace_reduced_error(probe, gallery);		
		}

		//  sum(probe - gallery)^2 - threshold
		vector<double> b(1<<logn, 0.);
		fill(b.begin(), b.begin()+256, threshold);
		Plaintext plain;
		encoder.encode(b, probe.scale(), plain);
		evaluator.mod_switch_to_inplace(plain, probe.parms_id());
		evaluator.sub_plain_inplace(probe, plain);

		// /*test*/
		// decrypt_and_print(probe, decryptor, encoder, 1<<logn, 256, 2);

		// match latency
		match_time_end = chrono::high_resolution_clock::now();
		match_time_diff = chrono::duration_cast<chrono::milliseconds>(match_time_end - match_time_start);
		cout << "match," << match_time_diff.count() / 1000 << endl << endl;
		output << "match," << match_time_diff.count() / 1000 << endl << endl;

		// online latency
		online_time_end = chrono::high_resolution_clock::now();
		online_time_diff = chrono::duration_cast<chrono::milliseconds>(online_time_end - online_time_start);

		// result
		decryptor.decrypt(probe, plain);
		vector<complex<double>> rtn_vec;
		encoder.decode(plain, rtn_vec);
		double score = rtn_vec[0].real();
		size_t pred = 0;
		if(score < 0) pred = 1;
		cout << "online latency: " << online_time_diff.count() / 1000 << " ms" << endl;
		output << "online latency: " << online_time_diff.count() / 1000 << " ms" << endl;
		cout << endl << "label: " << label << endl;
		cout << "pred: " << pred << endl;
		output << endl << "label: " << label << endl;
		output << "pred: " << pred << endl;
		output.close();
		all_thread_results[image_id - start_image_id] = {image_id, label, pred, offline_time_diff.count() / 1000, online_time_diff.count() / 1000};
	}

	total_time_end = chrono::high_resolution_clock::now();
	total_time_diff = chrono::duration_cast<chrono::milliseconds>(total_time_end - total_time_start);
	cout << endl << "total latency: " << total_time_diff.count() / 1000 << " ms" << endl;	

	// all threads output files
	ofstream out_share;
	string out_share_path = output_dir + "/autofhe/" + model + "/all_" + dataset + "-fold" + to_string(fold) + "-" + to_string(start_image_id) + "_" + to_string(end_image_id) + ".txt";
	out_share.open(out_share_path);
	if(!out_share.is_open()) throw std::runtime_error(out_share_path + " is not open.");
	out_share << "pair,label,pred,offline,online" << endl;
	for(auto res : all_thread_results) out_share << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "," << res[4] << endl;
	out_share << endl << "total latency," << total_time_diff.count() / 1000 << ",ms" << endl;
	out_share.close();
}

void resnet_mpcnn(string &model, string &dataset, size_t fold, string &weight_dir, string &dataset_dir, string &output_dir, size_t start_image_id, size_t end_image_id)
{
	// // approximation boundary setting
	// double B = 40.0;	// approximation boundary

	// approx ReLU setting
	long alpha = 13;			// precision parameter alpha
	long comp_no = 3;		// number of compositions
	vector<int> deg = {15,15,27};		// degrees of component polynomials
	// double eta = pow(2.0,-15);		// margin
	double scaled_val = 1.7;		// scaled_val: the last scaled value
	// double max_factor = 16;		// max_factor = 1 for comparison operation. max_factor > 1 for max or ReLU function
	vector<Tree> tree;		// structure of polynomial evaluation
	evaltype ev_type = evaltype::oddbaby;
	// RR::SetOutputPrecision(25);

	// generate tree
	for(int i=0; i<comp_no; i++) 
	{
		Tree tr;
		if(ev_type == evaltype::oddbaby) upgrade_oddbaby(deg[i], tr);
		else if(ev_type == evaltype::baby) upgrade_baby(deg[i], tr);
		else std::invalid_argument("evaluation type is not correct");
		tree.emplace_back(tr);
		// tr.print();
	}

	// SEAL and bootstrapping setting
	long boundary_K = 25;
	long boot_deg = 59;
    long scale_factor = 2;
    long inverse_deg = 1; 
	long logN = 16;
	long loge = 10; 
	long logn = 15;		// full slots
	long logn_1 = 14;	// sparse slots
	long logn_2 = 13;
	long logn_3 = 12;
	int logp = 46;
	int logq = 51;
	int log_special_prime = 51;
    int log_integer_part = logq - logp - loge + 5;
	int remaining_level = 16; // Calculation required
	int boot_level = 14; // 
	int total_level = remaining_level + boot_level;

	vector<int> coeff_bit_vec;
	coeff_bit_vec.push_back(logq);
	for (int i = 0; i < remaining_level; i++) coeff_bit_vec.push_back(logp);
	for (int i = 0; i < boot_level; i++) coeff_bit_vec.push_back(logq);
	coeff_bit_vec.push_back(log_special_prime);

	cout << "Setting Parameters" << endl;
	EncryptionParameters parms(scheme_type::ckks);
	size_t poly_modulus_degree = (size_t)(1 << logN);
	parms.set_poly_modulus_degree(poly_modulus_degree);
	parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec)); 

	// added
	size_t secret_key_hamming_weight = 192;
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
	// parms.set_sparse_slots(1 << logn_1);
	double scale = pow(2.0, logp);

	SEALContext context(parms);
	// KeyGenerator keygen(context, 192);
	KeyGenerator keygen(context);
    PublicKey public_key;
	keygen.create_public_key(public_key);
	auto secret_key = keygen.secret_key();
    RelinKeys relin_keys;
	keygen.create_relin_keys(relin_keys);
	GaloisKeys gal_keys;

	CKKSEncoder encoder(context);
	Encryptor encryptor(context, public_key);
	Evaluator evaluator(context, encoder);
	Decryptor decryptor(context, secret_key);
	// ScaleInvEvaluator scale_evaluator(context, encoder, relin_keys);

	Bootstrapper bootstrapper_1(loge, logn_1, logN - 1, total_level, scale, boundary_K, boot_deg, scale_factor, inverse_deg, context, keygen, encoder, encryptor, decryptor, evaluator, relin_keys, gal_keys);
	Bootstrapper bootstrapper_2(loge, logn_2, logN - 1, total_level, scale, boundary_K, boot_deg, scale_factor, inverse_deg, context, keygen, encoder, encryptor, decryptor, evaluator, relin_keys, gal_keys);
	Bootstrapper bootstrapper_3(loge, logn_3, logN - 1, total_level, scale, boundary_K, boot_deg, scale_factor, inverse_deg, context, keygen, encoder, encryptor, decryptor, evaluator, relin_keys, gal_keys);

//	additional rotation kinds for CNN
	vector<int> rotation_kinds = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33
		,56,62,63,64,66,84,124,128,132,256,512,959,960,990,991,1008
		,1023,1024,1036,1064,1092,1952,1982,1983,2016,2044,2047,2048,2072,2078,2100,3007,3024,3040,3052,3070,3071,3072,3080,3108,4031
		,4032,4062,4063,4095,4096,5023,5024,5054,5055,5087,5118,5119,5120,6047,6078,6079,6111,6112,6142,6143,6144,7071,7102,7103,7135
		,7166,7167,7168,8095,8126,8127,8159,8190,8191,8192,9149,9183,9184,9213,9215,9216,10173,10207,10208,10237,10239,10240,11197,11231
		,11232,11261,11263,11264,12221,12255,12256,12285,12287,12288,13214,13216,13246,13278,13279,13280,13310,13311,13312,14238,14240
		,14270,14302,14303,14304,14334,14335,15262,15264,15294,15326,15327,15328,15358,15359,15360,16286,16288,16318,16350,16351,16352
		,16382,16383,16384,17311,17375,18335,18399,18432,19359,19423,20383,20447,20480,21405,21406,21437,21469,21470,21471,21501,21504
		,22429,22430,22461,22493,22494,22495,22525,22528,23453,23454,23485,23517,23518,23519,23549,24477,24478,24509,24541,24542,24543
		,24573,24576,25501,25565,25568,25600,26525,26589,26592,26624,27549,27613,27616,27648,28573,28637,28640,28672,29600,29632,29664
		,29696,30624,30656,30688,30720,31648,31680,31712,31743,31744,31774,32636,32640,32644,32672,32702,32704,32706,32735
		,32736,32737,32759,32760,32761,32762,32763,32764,32765,32766,32767
	};

	// bootstrapping preprocessing
	cout << "Generating Optimal Minimax Polynomials..." << endl;
	bootstrapper_1.prepare_mod_polynomial();
	bootstrapper_2.prepare_mod_polynomial();
	bootstrapper_3.prepare_mod_polynomial();

	cout << "Adding Bootstrapping Keys..." << endl;
	vector<int> gal_steps_vector;
	gal_steps_vector.push_back(0);
	for(int i=0; i<logN-1; i++) gal_steps_vector.push_back((1 << i));
	for(auto rot: rotation_kinds)
	{
		if(find(gal_steps_vector.begin(), gal_steps_vector.end(), rot) == gal_steps_vector.end()) gal_steps_vector.push_back(rot);
	} 
	bootstrapper_1.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);
	bootstrapper_2.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);
	bootstrapper_3.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);
	keygen.create_galois_keys(gal_steps_vector, gal_keys);

	bootstrapper_1.slot_vec.push_back(logn_1);
	bootstrapper_2.slot_vec.push_back(logn_2);
	bootstrapper_3.slot_vec.push_back(logn_3);

	cout << "Generating Linear Transformation Coefficients..." << endl;
	bootstrapper_1.generate_LT_coefficient_3();
	bootstrapper_2.generate_LT_coefficient_3();
	bootstrapper_3.generate_LT_coefficient_3();

	// end number
	size_t layer_num, end_num;
	if(model == "resnet20") layer_num = 20, end_num = 2;
	else if(model == "resnet32") layer_num = 32, end_num = 4;
	else if(model == "resnet44") layer_num = 44, end_num = 6;
	else if(model == "resnet56") layer_num = 56, end_num = 8;
	else if(model == "resnet110") layer_num = 110, end_num = 17;
	else throw std::invalid_argument(model + "is not known.");

	// load images and labels
	size_t img_s = 64;
	vector<vector<double>> image_share1;
	vector<vector<double>> image_share2;
	vector<int> label_share;
	load_dataset(dataset, dataset_dir, fold, image_share1, image_share2, label_share);
	
	// import network weights 
	double B;
	vector<double> linear_weight_share, linear_bias_share;
	vector<vector<double>> conv_weight_share, bn_bias_share, bn_running_mean_share, bn_running_var_share, bn_weight_share;
	vector<double> coef_threshold;
	weight_dir = weight_dir + "/" + "mpcnn_" + model;
	import_weights_resnet_mpcnn(dataset, fold, weight_dir, linear_weight_share, linear_bias_share, conv_weight_share, bn_bias_share, bn_running_mean_share, bn_running_var_share, bn_weight_share, coef_threshold, B, layer_num, end_num);
	double coef_a = coef_threshold[0], coef_b = coef_threshold[1], coef_c = coef_threshold[2], threshold = coef_threshold[3];

	int nthred = end_image_id - start_image_id;
	vector<vector<int>> all_thread_results;
	all_thread_results.resize(nthred);

	// time setting
	chrono::high_resolution_clock::time_point total_time_start, total_time_end;
	chrono::microseconds total_time_diff;
	total_time_start = chrono::high_resolution_clock::now();
	
	// #pragma omp parallel for num_threads(nthred) // batch inference
	for(size_t image_id = start_image_id; image_id < end_image_id; image_id++)
	{
		// each thread output result file
		ofstream output;
		string output_path = output_dir + "/mpcnn/" + model + "/" + dataset + "-fold" + to_string(fold) + "-" + to_string(image_id) + ".txt";
		output.open(output_path);
		if(!output.is_open()) throw std::runtime_error(output_path + "not open.");

		// ciphertext pool generation
		vector<Ciphertext> cipher_pool(14);

		// time setting
		chrono::high_resolution_clock::time_point offline_time_start, offline_time_end, online_time_start, online_time_end;
		chrono::microseconds offline_time_diff, online_time_diff;

		// variables
		TensorCipher cnn, temp;
		Ciphertext probe, gallery; 

		// face recognition
		// i = 0: offline, enroll
		// i = 1: online, verification
		int label = label_share[image_id];
		for(size_t i = 0; i < 2; i++){
			// deep learning parameters and import
			int co = 0, st = 0, fh = 3, fw = 3;
			long init_p = 2, n = 1<<logn;
			int stage = 0;
			double epsilon = 0.00001;
			vector<double> linear_weight(linear_weight_share.begin(), linear_weight_share.end()), linear_bias(linear_bias_share.begin(), linear_bias_share.end());
			vector<vector<double>> conv_weight(conv_weight_share.begin(), conv_weight_share.end()), bn_bias(bn_bias_share.begin(), bn_bias_share.end()); 
			vector<vector<double>> bn_running_mean(bn_running_mean_share.begin(), bn_running_mean_share.end()), bn_running_var(bn_running_var_share.begin(), bn_running_var_share.end()); 
			vector<vector<double>> bn_weight(bn_weight_share.begin(), bn_weight_share.end());

			// pack images compactly
			vector<double> image;		
			if (i == 0){
				vector<double> image1(image_share1[image_id].begin(), image_share1[image_id].end());
				for(long i=img_s*img_s*3; i<1<<logn; i++) image1.emplace_back(0.);
				for(long i=n/init_p; i<n; i++) image1[i] = image1[i%(n/init_p)];
				for(long i=0; i<n; i++) image1[i] /= B;		// for boundary [-1,1]
				image = image1;
				offline_time_start = chrono::high_resolution_clock::now();
				cout << "------------" << "offline" << "------------" << endl;
				output << "------------" << "offline" << "------------" << endl;
			}
			else{
				vector<double> image2(image_share2[image_id].begin(), image_share2[image_id].end());
				for(long i=img_s*img_s*3; i<1<<logn; i++) image2.emplace_back(0.);
				for(long i=n/init_p; i<n; i++) image2[i] = image2[i%(n/init_p)];
				for(long i=0; i<n; i++) image2[i] /= B;		// for boundary [-1,1]
				image = image2;
				online_time_start = chrono::high_resolution_clock::now();
				cout << "------------" << "online" << "------------" << endl;
				output << "------------" << "online" << "------------" << endl;
			}
			
			// generate encrypted face image
			cnn = TensorCipher(logn, 1, img_s, img_s, 3, 3, init_p, image, encryptor, encoder, logq);
			cout << "remaining level : " << context.get_context_data(cnn.cipher().parms_id())->chain_index() << endl;
			cout << "scale: " << cnn.cipher().scale() << endl;

			// modulus down
			Ciphertext ctxt;
			ctxt = cnn.cipher();
			for(int i=0; i<boot_level-3; i++) evaluator.mod_switch_to_next_inplace(ctxt);
			cnn.set_ciphertext(ctxt);

			// layer 0
			cout << "layer 0" << endl;
			output << "layer 0" << endl;
			multiplexed_parallel_convolution_print(cnn, cnn, 16, 2, fh, fw, conv_weight[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, gal_keys, cipher_pool, output, decryptor, context, stage);

			// scaling factor ~2^51 -> 2^46
			const auto &modulus = iter(context.first_context_data()->parms().coeff_modulus());
			ctxt = cnn.cipher();
			size_t cur_level = ctxt.coeff_modulus_size();
			Plaintext scaler;
			double scale_change = pow(2.0,46) * ((double)modulus[cur_level-1].value()) / ctxt.scale();
			encoder.encode(1, scale_change, scaler);
			evaluator.mod_switch_to_inplace(scaler, ctxt.parms_id());
			evaluator.multiply_plain_inplace(ctxt, scaler);
			evaluator.rescale_to_next_inplace(ctxt);
			ctxt.scale() = pow(2.0,46);
			cnn.set_ciphertext(ctxt);

			multiplexed_parallel_batch_norm_seal_print(cnn, cnn, bn_bias[stage], bn_running_mean[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, B, output, decryptor, context, stage);
			approx_ReLU_seal_print(cnn, cnn, comp_no, deg, alpha, tree, scaled_val, logp, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, B, output, context, gal_keys, stage);

			for(int j=0; j<3; j++)		// layer 1_x, 2_x, 3_x
			{
				if(j==0) co = 16;
				else if(j==1) co = 32;
				else if(j==2) co = 64;

				for(int k=0; k<=end_num; k++)	// 0 ~ 2/4/6/8/17
				{
					stage = 2*((end_num+1)*j+k)+1;
					cout << "layer " << stage << endl;
					output << "layer " << stage << endl;
					temp = cnn;
					if(j>=1 && k==0) st = 2;
					else st = 1;
					multiplexed_parallel_convolution_print(cnn, cnn, co, st, fh, fw, conv_weight[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, gal_keys, cipher_pool, output, decryptor, context, stage);
					multiplexed_parallel_batch_norm_seal_print(cnn, cnn, bn_bias[stage], bn_running_mean[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, B, output, decryptor, context, stage);
					if(j==0) bootstrap_print(cnn, cnn, bootstrapper_1, output, decryptor, encoder, context, stage);
					else if(j==1) bootstrap_print(cnn, cnn, bootstrapper_2, output, decryptor, encoder, context, stage);
					else if(j==2) bootstrap_print(cnn, cnn, bootstrapper_3, output, decryptor, encoder, context, stage);
					approx_ReLU_seal_print(cnn, cnn, comp_no, deg, alpha, tree, scaled_val, logp, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, B, output, context, gal_keys, stage);

					stage = 2*((end_num+1)*j+k)+2;
					cout << "layer " << stage << endl;
					output << "layer " << stage << endl;
					st = 1;

					multiplexed_parallel_convolution_print(cnn, cnn, co, st, fh, fw, conv_weight[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, gal_keys, cipher_pool, output, decryptor, context, stage);
					multiplexed_parallel_batch_norm_seal_print(cnn, cnn, bn_bias[stage], bn_running_mean[stage], bn_running_var[stage], bn_weight[stage], epsilon, encoder, encryptor, evaluator, B, output, decryptor, context, stage);
					if(j>=1 && k==0) multiplexed_parallel_downsampling_seal_print(temp, temp, evaluator, decryptor, encoder, context, gal_keys, output);
					cipher_add_seal_print(temp, cnn, cnn, evaluator, output, decryptor, encoder, context);
					if(j==0) bootstrap_print(cnn, cnn, bootstrapper_1, output, decryptor, encoder, context, stage);
					else if(j==1) bootstrap_print(cnn, cnn, bootstrapper_2, output, decryptor, encoder, context, stage);
					else if(j==2) bootstrap_print(cnn, cnn, bootstrapper_3, output, decryptor, encoder, context, stage);
					approx_ReLU_seal_print(cnn, cnn, comp_no, deg, alpha, tree, scaled_val, logp, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, B, output, context, gal_keys, stage);		
				}
			}
			cout << "layer " << layer_num - 1 << endl;
			output << "layer " << layer_num - 1 << endl;
			cur_level = context.get_context_data(cnn.cipher().parms_id())->chain_index();
			if (cur_level > 0){
				ctxt = cnn.cipher();
				for (size_t i = 0; i < cur_level; i++) evaluator.mod_switch_to_next_inplace(ctxt);
				cnn.set_ciphertext(ctxt);
			}
			bootstrap_print(cnn, cnn, bootstrapper_3, output, decryptor, encoder, context, stage);
			averagepooling_seal_scale_print(cnn, cnn, evaluator, gal_keys, B, output, decryptor, encoder, context);
			fully_connected_seal_print(cnn, cnn, linear_weight, linear_bias, 256, 256, evaluator, gal_keys, output, decryptor, encoder, context);
			// BatchNorm 1D
			batchnorm1d_seal_print(cnn, cnn, evaluator, bn_running_mean.back(), bn_running_var.back(), epsilon, output, decryptor, encoder, context);
			// L2 Normalization
			l2norm_seal_print(cnn, cnn, evaluator, gal_keys, relin_keys, coef_a, coef_b, coef_c, output, decryptor, encoder, context);
			// enroll or verify
			if (i == 0){
				// enroll
				// probe = cnn.cipher();
				gallery = cnn.cipher();
				offline_time_end = chrono::high_resolution_clock::now();
				offline_time_diff = chrono::duration_cast<chrono::milliseconds>(offline_time_end - offline_time_start);
				cout << endl << "offline latency: " << offline_time_diff.count() / 1000 << " ms" << endl;
				output << endl << "offline latency: " << offline_time_diff.count() / 1000 << " ms" << endl << endl << endl;
			}
			else{
				// verify
				// gallery = cnn.cipher();
				probe = cnn.cipher();
			}
		}
		
		// face matching between probe and gallery (normalized features)

		// face matching time setting
		chrono::high_resolution_clock::time_point match_time_start, match_time_end;
		chrono::microseconds match_time_diff;
		match_time_start = chrono::high_resolution_clock::now();

		// (probe - gallery)
		evaluator.sub_inplace_reduced_error(probe, gallery);

		// (probe - gallery)^2
		evaluator.multiply_inplace_reduced_error(probe, probe, relin_keys);
		evaluator.rescale_to_next_inplace(probe);

		// sum(probe - gallery)^2
		for (size_t i = 0; i < log2_long(256); i++)
		{
			gallery = probe;
			memory_save_rotate(gallery, gallery, pow2(i), evaluator, gal_keys);
			evaluator.add_inplace_reduced_error(probe, gallery);		
		}

		//  sum(probe - gallery)^2 - threshold
		vector<double> b(1<<logn, 0.);
		fill(b.begin(), b.begin()+256, threshold);
		Plaintext plain;
		encoder.encode(b, probe.scale(), plain);
		evaluator.mod_switch_to_inplace(plain, probe.parms_id());
		evaluator.sub_plain_inplace(probe, plain);

		// /*test*/
		// decrypt_and_print(probe, decryptor, encoder, 1<<logn, 256, 2);

		// match latency
		match_time_end = chrono::high_resolution_clock::now();
		match_time_diff = chrono::duration_cast<chrono::milliseconds>(match_time_end - match_time_start);
		cout << "match," << match_time_diff.count() / 1000 << endl << endl;
		output << "match," << match_time_diff.count() / 1000 << endl << endl;

		// online latency
		online_time_end = chrono::high_resolution_clock::now();
		online_time_diff = chrono::duration_cast<chrono::milliseconds>(online_time_end - online_time_start);

		// result
		decryptor.decrypt(probe, plain);
		vector<complex<double>> rtn_vec;
		encoder.decode(plain, rtn_vec);
		double score = rtn_vec[0].real();
		size_t pred = 0;
		if(score < 0) pred = 1;
		cout << "online latency: " << online_time_diff.count() / 1000 << " ms" << endl;
		output << "online latency: " << online_time_diff.count() / 1000 << " ms" << endl;
		cout << endl << "label: " << label << endl;
		cout << "pred: " << pred << endl;
		output << endl << "label: " << label << endl;
		output << "pred: " << pred << endl;
		output.close();
		all_thread_results[image_id - start_image_id] = {image_id, label, pred, offline_time_diff.count() / 1000, online_time_diff.count() / 1000};
	}

	total_time_end = chrono::high_resolution_clock::now();
	total_time_diff = chrono::duration_cast<chrono::milliseconds>(total_time_end - total_time_start);
	cout << endl << "total latency: " << total_time_diff.count() / 1000 << " ms" << endl;	

	// all threads output files
	ofstream out_share;
	string out_share_path = output_dir + "/mpcnn/" + model + "/all_" + dataset + "-fold" + to_string(fold) + "-" + to_string(start_image_id) + "_" + to_string(end_image_id) + ".txt";
	out_share.open(out_share_path);
	if(!out_share.is_open()) throw std::runtime_error(out_share_path + " is not open.");
	out_share << "pair,label,pred,offline,online" << endl;
	for(auto res : all_thread_results) out_share << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "," << res[4] << endl;
	out_share << endl << "total latency," << total_time_diff.count() / 1000 << ",ms" << endl;
	out_share.close();
} 

void patchcnn(size_t input_size, string &dataset, size_t fold, string &weight_dir, string &dataset_dir, string &output_dir, size_t start_image_id, size_t end_image_id)
{
 	// SEAL and bootstrapping setting
	long boundary_K = 25;
	long boot_deg = 59;
    long scale_factor = 2;
    long inverse_deg = 1; 
	long logN = 16;
	long loge = 10; 
	long logn = 15;		// full slots
	long logn_3 = 12;
	int logp = 46;
	int logq = 51;
	int log_special_prime = 51;
    int log_integer_part = logq - logp - loge + 5;
	int remaining_level = 16; // Calculation required
	int boot_level = 14; // 
	int total_level = remaining_level + boot_level;

	vector<int> coeff_bit_vec;
	coeff_bit_vec.push_back(logq);
	for (int i = 0; i < remaining_level; i++) coeff_bit_vec.push_back(logp);
	for (int i = 0; i < boot_level; i++) coeff_bit_vec.push_back(logq);
	coeff_bit_vec.push_back(log_special_prime);

	cout << "Setting Parameters" << endl;
	EncryptionParameters parms(scheme_type::ckks);
	size_t poly_modulus_degree = (size_t)(1 << logN);
	parms.set_poly_modulus_degree(poly_modulus_degree);
	parms.set_coeff_modulus(CoeffModulus::Create(poly_modulus_degree, coeff_bit_vec)); 

	// added
	size_t secret_key_hamming_weight = 192;
    parms.set_secret_key_hamming_weight(secret_key_hamming_weight);
	// parms.set_sparse_slots(1 << logn_1);
	double scale = pow(2.0, logp);

	SEALContext context(parms);
	// KeyGenerator keygen(context, 192);
	KeyGenerator keygen(context);
    PublicKey public_key;
	keygen.create_public_key(public_key);
	auto secret_key = keygen.secret_key();
    RelinKeys relin_keys;
	keygen.create_relin_keys(relin_keys);
	GaloisKeys gal_keys;

	CKKSEncoder encoder(context);
	Encryptor encryptor(context, public_key);
	Evaluator evaluator(context, encoder);
	Decryptor decryptor(context, secret_key);
	// ScaleInvEvaluator scale_evaluator(context, encoder, relin_keys);

	Bootstrapper bootstrapper_3(loge, logn_3, logN - 1, total_level, scale, boundary_K, boot_deg, scale_factor, inverse_deg, context, keygen, encoder, encryptor, decryptor, evaluator, relin_keys, gal_keys);

//	additional rotation kinds for CNN
	vector<int> rotation_kinds = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33
		,56,62,63,64,66,84,124,128,132,256,512,959,960,990,991,1008
		,1023,1024,1036,1064,1092,1952,1982,1983,2016,2044,2047,2048,2072,2078,2100,3007,3024,3040,3052,3070,3071,3072,3080,3108,4031
		,4032,4062,4063,4095,4096,5023,5024,5054,5055,5087,5118,5119,5120,6047,6078,6079,6111,6112,6142,6143,6144,7071,7102,7103,7135
		,7166,7167,7168,8095,8126,8127,8159,8190,8191,8192,9149,9183,9184,9213,9215,9216,10173,10207,10208,10237,10239,10240,11197,11231
		,11232,11261,11263,11264,12221,12255,12256,12285,12287,12288,13214,13216,13246,13278,13279,13280,13310,13311,13312,14238,14240
		,14270,14302,14303,14304,14334,14335,15262,15264,15294,15326,15327,15328,15358,15359,15360,16286,16288,16318,16350,16351,16352
		,16382,16383,16384,17311,17375,18335,18399,18432,19359,19423,20383,20447,20480,21405,21406,21437,21469,21470,21471,21501,21504
		,22429,22430,22461,22493,22494,22495,22525,22528,23453,23454,23485,23517,23518,23519,23549,24477,24478,24509,24541,24542,24543
		,24573,24576,25501,25565,25568,25600,26525,26589,26592,26624,27549,27613,27616,27648,28573,28637,28640,28672,29600,29632,29664
		,29696,30624,30656,30688,30720,31648,31680,31712,31743,31744,31774,32636,32640,32644,32672,32702,32704,32706,32735
		,32736,32737,32759,32760,32761,32762,32763,32764,32765,32766,32767
	};

	// bootstrapping preprocessing
	cout << "Generating Optimal Minimax Polynomials..." << endl;
	bootstrapper_3.prepare_mod_polynomial();

// 	cout << "Adding Bootstrapping Keys..." << endl;
	vector<int> gal_steps_vector;
	gal_steps_vector.push_back(0);
	for(int i=0; i<logN-1; i++) gal_steps_vector.push_back((1 << i));
	for(auto rot: rotation_kinds)
	{
		if(find(gal_steps_vector.begin(), gal_steps_vector.end(), rot) == gal_steps_vector.end()) gal_steps_vector.push_back(rot);
	} 
	bootstrapper_3.addLeftRotKeys_Linear_to_vector_3(gal_steps_vector);
	keygen.create_galois_keys(gal_steps_vector, gal_keys);
	bootstrapper_3.slot_vec.push_back(logn_3);

	cout << "Generating Linear Transformation Coefficients..." << endl;
	bootstrapper_3.generate_LT_coefficient_3();


	// load images and labels
	size_t patch_size = 32;
	double B = 120.;
	size_t num_nets = (input_size / patch_size) * (input_size / patch_size);
	vector<vector<double>> image_share1;
	vector<vector<double>> image_share2;
	vector<int> label_share;
	dataset_dir = dataset_dir + "/" + to_string(input_size);
	load_dataset(dataset, dataset_dir, fold, image_share1, image_share2, label_share);

	// import network weights 
	weight_dir = weight_dir + "/cryptoface_pcnn" + to_string(input_size);
	vector<double> linear_bias_share, output_mean_share, output_var_share, coef_threshold;
	vector<vector<double>> all_linear_weight_share, all_conv_weight_share, all_bn_weight_share, all_bn_bias_share, all_bn_running_mean_share, all_bn_running_var_share;;
	vector<vector<vector<double>>> all_layer_weight_share, all_shortcut_weight_share, all_a2_share, all_a1_share, all_a0_share, all_b1_share, all_b0_share, all_shortcut_bn_weight_share, all_shortcut_bn_bias_share, all_shortcut_bn_running_mean_share, all_shortcut_bn_running_var_share;
	import_weights_pcnn(num_nets, dataset, fold, weight_dir, all_linear_weight_share, linear_bias_share, coef_threshold, all_conv_weight_share, all_layer_weight_share, all_shortcut_weight_share, all_a2_share, all_a1_share, all_a0_share, all_b1_share, all_b0_share, all_shortcut_bn_weight_share, all_shortcut_bn_bias_share, all_shortcut_bn_running_mean_share, all_shortcut_bn_running_var_share, all_bn_weight_share, all_bn_bias_share, all_bn_running_mean_share, all_bn_running_var_share);
	double coef_a = coef_threshold[0], coef_b = coef_threshold[1], coef_c = coef_threshold[2], threshold = coef_threshold[3];

	// results
	int nthred = end_image_id - start_image_id;
	vector<vector<int>> all_thread_results;
	all_thread_results.resize(nthred);

	// time setting
	chrono::high_resolution_clock::time_point total_time_start, total_time_end;
	chrono::microseconds total_time_diff;
	total_time_start = chrono::high_resolution_clock::now();

	// #pragma omp parallel for num_threads(nthred) // batch inference (ATTEN: nested omp parallel is NOT tested)
	for(size_t image_id = start_image_id; image_id < end_image_id; image_id++)
	{
		// each thread output result file
		ofstream output;
		string output_path = output_dir + "/cryptoface/" + to_string(input_size) + "/" + dataset + "-fold" + to_string(fold) + "-" + to_string(image_id) + ".txt";
		output.open(output_path);
		if(!output.is_open()) throw std::runtime_error(output_path + " not open.");

		// time setting
		chrono::high_resolution_clock::time_point offline_time_start, offline_time_end, online_time_start, online_time_end;
		chrono::microseconds offline_time_diff, online_time_diff;

		// variables
		Ciphertext probe, gallery; 

		// face recognition
		// i = 0: offline, enroll
		// i = 1: online, verification
		int label = label_share[image_id];
		for(size_t i = 0; i < 2; i++){
			if (i == 0){
				offline_time_start = chrono::high_resolution_clock::now();
				cout << "------------" << "offline" << "------------" << endl;
				output << "------------" << "offline" << "------------" << endl;
			}
			else{
				online_time_start = chrono::high_resolution_clock::now();
				cout << "------------" << "online" << "------------" << endl;
				output << "------------" << "online" << "------------" << endl;
			}			
			
			
			// evalute subnets in parallel
			vector<Ciphertext> features(num_nets);
			#pragma omp parallel for num_threads(num_nets) // mixture of PCNNs in parallel 
			for(size_t subnet = 0; subnet < num_nets; subnet++){
				cout << endl << endl << "###########" << "subnet " << to_string(subnet) << "###########" << endl << endl;
				output << endl << endl << "###########" << "subnet " << to_string(subnet) << "###########" << endl << endl;

				// subnet weights
				vector<double> linear_weight(all_linear_weight_share[subnet].begin(), all_linear_weight_share[subnet].end());
				vector<double> linear_bias(linear_bias_share.begin(), linear_bias_share.end());
				vector<double> conv_weight(all_conv_weight_share[subnet].begin(), all_conv_weight_share[subnet].end());
				vector<double> bn_weight(all_bn_weight_share[subnet].begin(), all_bn_weight_share[subnet].end()); 
				vector<double> bn_bias(all_bn_bias_share[subnet].begin(), all_bn_bias_share[subnet].end()); 
				vector<double> bn_running_mean(all_bn_running_mean_share[subnet].begin(), all_bn_running_mean_share[subnet].end());
				vector<double> bn_running_var(all_bn_running_var_share[subnet].begin(), all_bn_running_var_share[subnet].end());
				vector<vector<double>> layer_weight(all_layer_weight_share[subnet].begin(), all_layer_weight_share[subnet].end()); 
				vector<vector<double>> shortcut_weight(all_shortcut_weight_share[subnet].begin(), all_shortcut_weight_share[subnet].end()); 
				vector<vector<double>> a2(all_a2_share[subnet].begin(), all_a2_share[subnet].end());
				vector<vector<double>> a1(all_a1_share[subnet].begin(), all_a1_share[subnet].end()); 
				vector<vector<double>> a0(all_a0_share[subnet].begin(), all_a0_share[subnet].end()); 
				vector<vector<double>> b1(all_b1_share[subnet].begin(), all_b1_share[subnet].end()); 
				vector<vector<double>> b0(all_b0_share[subnet].begin(), all_b0_share[subnet].end()); 
				vector<vector<double>> shortcut_bn_weight(all_shortcut_bn_weight_share[subnet].begin(), all_shortcut_bn_weight_share[subnet].end()); 
				vector<vector<double>> shortcut_bn_bias(all_shortcut_bn_bias_share[subnet].begin(), all_shortcut_bn_bias_share[subnet].end()); 
				vector<vector<double>> shortcut_bn_running_mean(all_shortcut_bn_running_mean_share[subnet].begin(), all_shortcut_bn_running_mean_share[subnet].end()); 
				vector<vector<double>> shortcut_bn_running_var(all_shortcut_bn_running_var_share[subnet].begin(), all_shortcut_bn_running_var_share[subnet].end());
			
				// deep learning parameters and import
				int co = 0, st = 0, fh = 3, fw = 3;
				long init_p = 8, n = 1<<logn;
				int stage = 0, coef_ind = 0;
				double epsilon = 0.00001;

				// pack images compactly
				vector<double> image;		
				if (i == 0){
					vector<double> image1(image_share1[image_id].begin()+(patch_size*patch_size*3)*subnet, image_share1[image_id].begin()+(patch_size*patch_size*3)*(subnet+1));
					for(long i=patch_size*patch_size*3; i<1<<logn; i++) image1.emplace_back(0.);
					for(long i=n/init_p; i<n; i++) image1[i] = image1[i%(n/init_p)];
					image = image1;
				}
				else{
					vector<double> image2(image_share2[image_id].begin()+(patch_size*patch_size*3)*subnet, image_share2[image_id].begin()+(patch_size*patch_size*3)*(subnet+1));
					for(long i=patch_size*patch_size*3; i<1<<logn; i++) image2.emplace_back(0.);
					for(long i=n/init_p; i<n; i++) image2[i] = image2[i%(n/init_p)];
					image = image2;
				}

				// generate encrypted face image
				TensorCipher cnn, temp;
				Ciphertext ctxt;
				cnn = TensorCipher(logn, 1, patch_size, patch_size, fh, fw, init_p, image, encryptor, encoder, logq);
				cout << "remaining level : " << context.get_context_data(cnn.cipher().parms_id())->chain_index() << endl;
				cout << "scale: " << cnn.cipher().scale() << endl;

				// ciphertext pool generation
				vector<Ciphertext> cipher_pool(14);

				// conv (2 Levels, 30 -> 28)
				co = 16, st = 1, fh = 3, fw = 3;
				multiplexed_parallel_convolution_print(cnn, cnn, co, st, fh, fw, conv_weight, encoder, encryptor, evaluator, gal_keys, cipher_pool, output, decryptor, context, stage);

				// Block 
				for(size_t j=0; j<5; j++)
				{
					if(j==0) co=16, st=1;
					else if(j==1) co=32, st=2;
					else if(j==2) co=32, st=1;
					else if(j==3) co=64, st=2;
					else if(j==4) co=64, st=1;
					else throw std::runtime_error("invalid condition j="+to_string(j));
					
					if(j==2){								
						// scaling factor ~2^51 -> 2^46
						// decrypt_and_print(cnn.cipher(), decryptor, encoder, 1<<logn, 256, 2);
						const auto &modulus = iter(context.first_context_data()->parms().coeff_modulus());
						ctxt = cnn.cipher();
						size_t cur_level = ctxt.coeff_modulus_size();
						Plaintext scaler;
						double scale_change = pow(2.0,46) * ((double)modulus[cur_level-1].value()) / ctxt.scale();
						encoder.encode(1, scale_change, scaler);
						evaluator.mod_switch_to_inplace(scaler, ctxt.parms_id());
						evaluator.multiply_plain_inplace(ctxt, scaler);
						evaluator.rescale_to_next_inplace(ctxt);
						ctxt.scale() = pow(2.0,46);
						// modulus down
						for(int k=0; k<2; k++) evaluator.mod_switch_to_next_inplace(ctxt);
						cnn.set_ciphertext(ctxt);
					}

					// conv 1
					stage++;
					herpn_print(cnn, cnn, a0[j], a1[j], encoder, encryptor, evaluator, relin_keys, output, decryptor, context);
					temp = cnn;
					multiplexed_parallel_convolution_print(cnn, cnn, co, st, fh, fw, layer_weight[2 * j], encoder, encryptor, evaluator, gal_keys, cipher_pool, output, decryptor, context, stage);
					
					// shortcut
					ctxt = temp.cipher();
					if(j==0 || j==2 || j==4) for(int k=0; k<4; k++) evaluator.mod_switch_to_next_inplace(ctxt);
					else for(int k=0; k<2; k++) evaluator.mod_switch_to_next_inplace(ctxt);
					temp.set_ciphertext(ctxt);
					channel_multiply_seal_print(temp, temp, a2[j], encoder, encryptor, evaluator, relin_keys, output, decryptor, context);
					if(j==1){
						multiplexed_parallel_convolution_print(temp, temp, co, st, 1, 1, shortcut_weight[0], shortcut_bn_running_var[0], shortcut_bn_weight[0], epsilon, encoder, encryptor, evaluator, gal_keys, cipher_pool, output, decryptor, context, stage);
						multiplexed_parallel_batch_norm_seal_print(temp, temp, shortcut_bn_bias[0], shortcut_bn_running_mean[0], shortcut_bn_running_var[0], shortcut_bn_weight[0], epsilon, encoder, encryptor, evaluator, 1, output, decryptor, context, stage);
					}
					else if(j==3){
						multiplexed_parallel_convolution_print(temp, temp, co, st, 1, 1, shortcut_weight[1], shortcut_bn_running_var[1], shortcut_bn_weight[1], epsilon, encoder, encryptor, evaluator, gal_keys, cipher_pool, output, decryptor, context, stage);
						multiplexed_parallel_batch_norm_seal_print(temp, temp, shortcut_bn_bias[1], shortcut_bn_running_mean[1], shortcut_bn_running_var[1], shortcut_bn_weight[1], epsilon, encoder, encryptor, evaluator, 1, output, decryptor, context, stage);
					}

					// conv 2
					st = 1;
					stage++;
					herpn_print(cnn, cnn, b0[j], b1[j], encoder, encryptor, evaluator, relin_keys, output, decryptor, context);
					multiplexed_parallel_convolution_print(cnn, cnn, co, st, fh, fw, layer_weight[2 * j + 1], encoder, encryptor, evaluator, gal_keys, cipher_pool, output, decryptor, context, stage);
				
					// residual addition
					cipher_add_seal_print(temp, cnn, cnn, evaluator, output, decryptor, encoder, context);

					// bootstrapping
					if(j == 3){
						// scale it to [-1, 1]
						ctxt = cnn.cipher();
						Plaintext scaler;
						encoder.encode(1./B, ctxt.scale(), scaler);
						evaluator.mod_switch_to_inplace(scaler, ctxt.parms_id());
						evaluator.multiply_plain_inplace(ctxt, scaler);
						evaluator.rescale_to_next_inplace(ctxt);
						cnn.set_ciphertext(ctxt);
						// decrypt_and_print(cnn.cipher(), decryptor, encoder, 1<<logn, 256, 2);
						// bootstrapping
						bootstrap_print(cnn, cnn, bootstrapper_3, output, decryptor, encoder, context, stage);
						// scale it back
						ctxt = cnn.cipher();
						encoder.encode(B, ctxt.scale(), scaler);
						evaluator.mod_switch_to_inplace(scaler, ctxt.parms_id());
						evaluator.multiply_plain_inplace(ctxt, scaler);
						evaluator.rescale_to_next_inplace(ctxt);
						cnn.set_ciphertext(ctxt);
						// decrypt_and_print(cnn.cipher(), decryptor, encoder, 1<<logn, 256, 2);
					}
				}

				// HerPNPool
				stage++;
				size_t j = 5;
				herpn_print(cnn, cnn, a0[j], a1[j], encoder, encryptor, evaluator, relin_keys, output, decryptor, context);
				averagepooling_seal_print(cnn, cnn, evaluator, gal_keys, output, decryptor, encoder, context);

				// BatchNorm1D
				batchnorm1d_seal_print(cnn, cnn, evaluator, a2[j], bn_weight, bn_bias, bn_running_mean, bn_running_var, epsilon, output, decryptor, encoder, context);
				
				// Linear and BatchNorm1D
				fully_connected_seal_print(cnn, cnn, linear_weight, linear_bias, 256, 256, evaluator, gal_keys, output, decryptor, encoder, context);

				// save to features
				features[subnet] = cnn.cipher();
			}

			if (i == 0){
				// enroll
				gallery = features[0];
				for(size_t subnet = 1; subnet < num_nets; subnet++) evaluator.add_inplace_reduced_error(gallery, features[subnet]);

				// L2 Normalization
				l2norm_seal_print(gallery, logn, evaluator, gal_keys, relin_keys, coef_a, coef_b, coef_c, output, decryptor, encoder, context);

				offline_time_end = chrono::high_resolution_clock::now();
				offline_time_diff = chrono::duration_cast<chrono::milliseconds>(offline_time_end - offline_time_start);
				cout << endl << "offline latency: " << offline_time_diff.count() / 1000 << " ms" << endl;
				output << endl << "offline latency: " << offline_time_diff.count() / 1000 << " ms" << endl << endl << endl;
			}
			else{
				// verify
				probe = features[0];
				for(size_t subnet = 1; subnet < num_nets; subnet++) evaluator.add_inplace_reduced_error(probe, features[subnet]);
				// L2 Normalization
				l2norm_seal_print(probe, logn, evaluator, gal_keys, relin_keys, coef_a, coef_b, coef_c, output, decryptor, encoder, context);
			}
		}

		// face matching between probe and gallery (normalized features)

		// face matching time setting
		chrono::high_resolution_clock::time_point match_time_start, match_time_end;
		chrono::microseconds match_time_diff;
		match_time_start = chrono::high_resolution_clock::now();

		// (probe - gallery)
		evaluator.sub_inplace_reduced_error(probe, gallery);

		// (probe - gallery)^2
		evaluator.multiply_inplace_reduced_error(probe, probe, relin_keys);
		evaluator.rescale_to_next_inplace(probe);

		// sum(probe - gallery)^2
		for (size_t i = 0; i < log2_long(256); i++)
		{
			gallery = probe;
			memory_save_rotate(gallery, gallery, pow2(i), evaluator, gal_keys);
			evaluator.add_inplace_reduced_error(probe, gallery);		
		}

		//  sum(probe - gallery)^2 - threshold
		vector<double> b(1<<logn, 0.);
		fill(b.begin(), b.begin()+256, threshold);
		Plaintext plain;
		encoder.encode(b, probe.scale(), plain);
		evaluator.mod_switch_to_inplace(plain, probe.parms_id());
		evaluator.sub_plain_inplace(probe, plain);

		// match latency
		match_time_end = chrono::high_resolution_clock::now();
		match_time_diff = chrono::duration_cast<chrono::milliseconds>(match_time_end - match_time_start);
		cout << "match," << match_time_diff.count() / 1000 << endl << endl;
		output << "match," << match_time_diff.count() / 1000 << endl << endl;

		// online latency
		online_time_end = chrono::high_resolution_clock::now();
		online_time_diff = chrono::duration_cast<chrono::milliseconds>(online_time_end - online_time_start);

		// result
		decryptor.decrypt(probe, plain);
		vector<complex<double>> rtn_vec;
		encoder.decode(plain, rtn_vec);
		double score = rtn_vec[0].real();
		size_t pred = 0;
		if(score < 0) pred = 1;
		cout << "online latency: " << online_time_diff.count() / 1000 << " ms" << endl;
		output << "online latency: " << online_time_diff.count() / 1000 << " ms" << endl;
		cout << endl << "label: " << label << endl;
		cout << "pred: " << pred << endl;
		output << endl << "label: " << label << endl;
		output << "pred: " << pred << endl;
		output.close();
		all_thread_results[image_id - start_image_id] = {image_id, label, pred, offline_time_diff.count() / 1000, online_time_diff.count() / 1000};
	}

	// all output
	total_time_end = chrono::high_resolution_clock::now();
	total_time_diff = chrono::duration_cast<chrono::milliseconds>(total_time_end - total_time_start);
	cout << endl << "total latency: " << total_time_diff.count() / 1000 << " ms" << endl;	
	ofstream out_share;
	string out_share_path = output_dir + "/cryptoface/" + to_string(input_size) + "/all_" + dataset + "-fold" + to_string(fold) + "-" + to_string(start_image_id) + "_" + to_string(end_image_id) + ".txt";
	out_share.open(out_share_path);
	if(!out_share.is_open()) throw std::runtime_error(out_share_path + " is not open.");
	out_share << "pair,label,pred,offline,online" << endl;
	for(auto res : all_thread_results) out_share << res[0] << "," << res[1] << "," << res[2] << "," << res[3] << "," << res[4] << endl;
	out_share << endl << "total latency," << total_time_diff.count() / 1000 << ",ms" << endl;
	out_share.close();
}