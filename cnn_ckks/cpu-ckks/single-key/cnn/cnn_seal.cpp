#include "cnn_seal.h"

TensorCipher::TensorCipher()
{
    k_=0;
    h_=0;
    w_=0;
	c_=0;
	t_=0;
    p_=0;
}
TensorCipher::TensorCipher(int logn, int k, int h, int w, int c, int t, int p, vector<double> data, Encryptor &encryptor, CKKSEncoder &encoder, int logp)
{
    if(k != 1) throw std::invalid_argument("supported k is only 1 right now");
    
	// 1 <= logn <= 16
    if(logn < 1 || logn > 16) throw std::out_of_range("the value of logn is out of range");
    if(data.size() > static_cast<long unsigned int>(1<<logn)) throw std::out_of_range("the size of data is larger than n");

    this->k_ = k;
    this->h_ = h;
	this->w_ = w;
	this->c_ = c;
    this->t_ = t;
	this->p_ = p;
	this->logn_ = logn;

	// generate vector that contains data
	vector<double> vec;
    for(int i=0; i<static_cast<int>(data.size()); i++) vec.emplace_back(data[i]);
    for(int i=data.size(); i<1<<logn; i++) vec.emplace_back(0);      // zero padding

    // vec size = n
    if(vec.size() != static_cast<long unsigned int>(1<<logn)) throw std::out_of_range("the size of vec is not n");

	// encode & encrypt
	Plaintext plain;
	Ciphertext cipher;
	double scale = pow(2.0, logp);
	encoder.encode(vec, scale, plain);
	encryptor.encrypt(plain, cipher);
	this->set_ciphertext(cipher);

}
TensorCipher::TensorCipher(int logn, int k, int h, int w, int c, int t, int p, Ciphertext cipher)
{
    this->k_ = k;
    this->h_ = h;
	this->w_ = w;
	this->c_ = c;
    this->t_ = t;
	this->p_ = p;
	this->logn_ = logn;
	this->cipher_ = cipher;
}
int TensorCipher::k() const
{
	return k_;
}
int TensorCipher::h() const
{
	return h_;
}
int TensorCipher::w() const
{
	return w_;
}
int TensorCipher::c() const
{
	return c_;
}
int TensorCipher::t() const
{
	return t_;
}
int TensorCipher::p() const
{
	return p_;
}
int TensorCipher::logn() const
{
	return logn_;
}
Ciphertext TensorCipher::cipher() const
{
	return cipher_;
}
void TensorCipher::set_ciphertext(Ciphertext cipher)
{
	cipher_ = cipher;
}
void TensorCipher::print_parms()
{
	cout << "k: " << k_ << endl;
    cout << "h: " << h_ << endl;
    cout << "w: " << w_ << endl;
	cout << "c: " << c_ << endl;
	cout << "t: " << t_ << endl;
	cout << "p: " << p_ << endl;
}

void herpn_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, const vector<double> &herpn_w0, const vector<double> &herpn_w1, const vector<double> &herpn_w2, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, RelinKeys &relin_keys, ofstream &output, Decryptor &decryptor, SEALContext &context)
{
    cout << "herpn..." << endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	cout << "before herpn" << endl;
	// decrypt_and_print(cnn_in.cipher(), decryptor, encoder, 1<<logn, 256, 2);
	channel_square_seal(cnn_in, cnn_out, herpn_w0, herpn_w1, herpn_w2, encoder, encryptor, evaluator, relin_keys);
	cout << "after herpn" << endl;
	// decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2); 
	cnn_out.print_parms();
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	cout << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;
	output << "herpn," << time_diff.count() / 1000 << "," << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << "," << cnn_out.cipher().scale() << endl << endl;
}

void herpn_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, const vector<double> &herpn_w0, const vector<double> &herpn_w1, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, RelinKeys &relin_keys, ofstream &output, Decryptor &decryptor, SEALContext &context)
{
    cout << "herpn..." << endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	cout << "before herpn" << endl;
	// decrypt_and_print(cnn_in.cipher(), decryptor, encoder, 1<<logn, 256, 2);
	channel_square_seal(cnn_in, cnn_out, herpn_w0, herpn_w1, encoder, encryptor, evaluator, relin_keys);
	cout << "after herpn" << endl;
	// decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2); 
	cnn_out.print_parms();
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	cout << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;
	output << "herpn," << time_diff.count() / 1000 << "," << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << "," << cnn_out.cipher().scale() << endl << endl;
}

void multiplexed_parallel_convolution_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, int co, int st, int fh, int fw, const vector<double> &data, vector<double> running_var, vector<double> constant_weight, double epsilon, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, GaloisKeys &gal_keys, vector<Ciphertext> &cipher_pool, ofstream &output, Decryptor &decryptor, SEALContext &context, size_t stage, bool end)
{
    cout << "multiplexed parallel convolution..." << endl;
    // output << "multiplexed parallel convolution..." << endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	// convolution_seal_sparse(cnn_in, cnn_out, hprime, st, kernel, false, data, running_var, constant_weight, epsilon, encoder, encryptor, scale_evaluator, gal_keys, cipher_pool, end);
	multiplexed_parallel_convolution_seal(cnn_in, cnn_out, co, st, fh, fw, data, running_var, constant_weight, epsilon, encoder, encryptor, evaluator, gal_keys, cipher_pool, end);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	// cout << "convolution " << stage << " result" << endl;
	// output << "time : " << time_diff.count() / 1000 << " ms" << endl;
	// output << "convolution " << stage << " result" << endl;
    // decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2); 
	cnn_out.print_parms();
	// decrypt_and_print_txt(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2, output); cnn_out.print_parms();
	cout << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;
	// output << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	// output << "scale: " << cnn_out.cipher().scale() << endl << endl;
	output << "multiplexed parallel convolution," << time_diff.count() / 1000 << "," << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << "," << cnn_out.cipher().scale() << endl << endl;
}
void multiplexed_parallel_convolution_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, int co, int st, int fh, int fw, const vector<double> &data, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, GaloisKeys &gal_keys, vector<Ciphertext> &cipher_pool, ofstream &output, Decryptor &decryptor, SEALContext &context, size_t stage, bool end)
{
    cout << "multiplexed parallel convolution..." << endl;
    // output << "multiplexed parallel convolution..." << endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	// convolution_seal_sparse(cnn_in, cnn_out, hprime, st, kernel, false, data, running_var, constant_weight, epsilon, encoder, encryptor, scale_evaluator, gal_keys, cipher_pool, end);
	multiplexed_parallel_convolution_seal(cnn_in, cnn_out, co, st, fh, fw, data, encoder, encryptor, evaluator, gal_keys, cipher_pool, end);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	// cout << "convolution " << stage << " result" << endl;
	// output << "time : " << time_diff.count() / 1000 << " ms" << endl;
	// output << "convolution " << stage << " result" << endl;
    // decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2); 
	cnn_out.print_parms();
	// decrypt_and_print_txt(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2, output); cnn_out.print_parms();
	cout << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;
	// output << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	// output << "scale: " << cnn_out.cipher().scale() << endl << endl;
	output << "multiplexed parallel convolution," << time_diff.count() / 1000 << "," << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << "," << cnn_out.cipher().scale() << endl << endl;
}
void multiplexed_parallel_batch_norm_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, vector<double> bias, vector<double> running_mean, vector<double> running_var, vector<double> weight, double epsilon, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, double B, ofstream &output, Decryptor &decryptor, SEALContext &context, size_t stage, bool end)
{
    cout << "multiplexed parallel batch normalization..." << endl;
    // output << "multiplexed parallel batch normalization..." << endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	// batch norm
	time_start = chrono::high_resolution_clock::now();
	multiplexed_parallel_batch_norm_seal(cnn_in, cnn_out, bias, running_mean, running_var, weight, epsilon, encoder, encryptor, evaluator, B, end); 
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	// cout << "batch normalization " << stage << " result" << endl;
	// output << "time : " << time_diff.count() / 1000 << " ms" << endl;
	// output << "batch normalization " << stage << " result" << endl;
	// decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2); 
	cnn_out.print_parms();
	// decrypt_and_print_txt(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2, output); cnn_out.print_parms();
	cout << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;
	// output << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	// output << "scale: " << cnn_out.cipher().scale() << endl << endl;
	output << "multiplexed parallel batch normalization," << time_diff.count() / 1000 << "," << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << "," << cnn_out.cipher().scale() << endl << endl;
}
void approx_ReLU_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, long comp_no, vector<int> deg, long alpha, vector<Tree> &tree, double scaled_val, long scalingfactor, Encryptor &encryptor, Evaluator &evaluator, Decryptor &decryptor, CKKSEncoder &encoder, PublicKey &public_key, SecretKey &secret_key, RelinKeys &relin_keys, double B, ofstream &output, SEALContext &context, GaloisKeys &gal_keys, size_t stage)
{
    cout << "approximate ReLU..." << endl;
    // output << "approximate ReLU..." << endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	ReLU_seal(cnn_in, cnn_out, comp_no, deg, alpha, tree, scaled_val, scalingfactor, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, B);
	// ReLU_remove_imaginary_seal(cnn_in, cnn_out, comp_no, deg, alpha, tree, scaled_val, scalingfactor, encryptor, evaluator, scale_evaluator, decryptor, encoder, public_key, secret_key, relin_keys, output, gal_keys, B);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	// cout << "ReLU function " << stage << " result" << endl;
	// output << "time : " << time_diff.count() / 1000 << " ms" << endl;
	// output << "ReLU function " << stage << " result" << endl;
	// decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2); cnn_out.print_parms();
	// decrypt_and_print_txt(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2, output); cnn_out.print_parms();
	cout << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl; 
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;
	// output << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	// output << "scale: " << cnn_out.cipher().scale() << endl << endl;

	output << "approximate ReLU," << time_diff.count() / 1000 << "," << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << "," << cnn_out.cipher().scale() << endl << endl;

	// cout << "intermediate decrypted values: " << endl;
	// output << "intermediate decrypted values: " << endl;
	// decrypt_and_print_txt(cnn_out.cipher(), decryptor, encoder, 1<<logn, 4, 1, output); // cnn_out.print_parms();
}
void evorelu_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, string &dir, Encryptor &encryptor, Evaluator &evaluator, Decryptor &decryptor, CKKSEncoder &encoder, PublicKey &public_key, SecretKey &secret_key, RelinKeys &relin_keys, ofstream &output, SEALContext &context, GaloisKeys &gal_keys, double Bin, double Bout, size_t stage)
{
    cout << "AutoFHE EvoReLU..." << endl;
    // output << "AutoFHE EvoReLU..." << endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();

	ifstream in_evorelu;
	in_evorelu.open(dir + "/evorelus/" + to_string(stage) + ".txt");
	if(!in_evorelu.is_open()) throw std::runtime_error(dir + "/" + to_string(stage) + ".txt is not open.");
	int depth;
	in_evorelu >> depth;
	if (depth == 2)
	{
		vector<vector<double>> weight;
		weight.resize(3);
		string line;
		string w_str;
		for(size_t i=0; i<3; i++){
			in_evorelu >> line;
			std::istringstream iss_coef(line);
			if (i==0) while (std::getline(iss_coef, w_str, ',')) weight[i].emplace_back(atof(w_str.c_str()) * Bout);
			if (i==1) while (std::getline(iss_coef, w_str, ',')) weight[i].emplace_back(atof(w_str.c_str()) * Bout * Bin);
			if (i==2) while (std::getline(iss_coef, w_str, ',')) weight[i].emplace_back(atof(w_str.c_str()) * Bout * Bin * Bin);
		}
		channel_square_seal(cnn_in, cnn_out, weight[0], weight[1], weight[2], encoder, encryptor, evaluator, relin_keys);
	}
	else if (depth > 2)
	{
		long comp_no; in_evorelu >> comp_no;
		vector<int> deg;
		int tmp;
		for (size_t i = 0; i < comp_no; i++) in_evorelu >> tmp, deg.push_back(tmp); 
		vector<Tree> tree;	
		evaltype ev_type = evaltype::oddbaby;
		for(size_t i = 0; i < comp_no; i++) 
		{
			Tree tr;
			if(ev_type == evaltype::oddbaby) upgrade_oddbaby(deg[i], tr);
			else if(ev_type == evaltype::baby) upgrade_baby(deg[i], tr);
			else std::invalid_argument("evaluation type is not correct");
			tree.emplace_back(tr);
		}

		vector<vector<double>> coef(comp_no, vector<double>(0));
		string line;
		string w_str;
		for(size_t i = 0; i < comp_no; i++){
			in_evorelu >> line;
			std::istringstream iss_coef(line);
			while (std::getline(iss_coef, w_str, ','))  {coef[i].emplace_back(atof(w_str.c_str())); };
		}
		RR s(1.0);
		vector<vector<double>> cheby_coef(comp_no, vector<double>(0));
		for(size_t i = 0; i < comp_no; i++){
			poly_decompose_integrate(deg[i], 0, s, coef[i], cheby_coef[i], tree[i], ev_type, 1, s);
		}
		evorelu_seal(cnn_in, cnn_out, comp_no, deg, cheby_coef, tree, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, Bout);

	}
	in_evorelu.close();
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	cout << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl; 
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;
	output << "AutoFHE EvoReLU," << time_diff.count() / 1000 << "," << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << "," << cnn_out.cipher().scale() << endl << endl;
	// decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2); 
	cnn_out.print_parms();
}
void bootstrap_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, Bootstrapper &bootstrapper, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context, size_t stage)
{
    cout << "bootstrapping..." << endl;
	Ciphertext ctxt, rtn;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	ctxt = cnn_in.cipher();
	time_start = chrono::high_resolution_clock::now();
	cout << "before bootstrapping... " << endl;
	// decrypt_and_print(cnn_in.cipher(), decryptor, encoder, 1<<logn, 256, 2);
	bootstrapper.bootstrap_real_3(rtn, ctxt);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	cnn_out.set_ciphertext(rtn);
	cout << "after boostrapping... " << endl;
	// decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2);
	cnn_out.print_parms();
	cout << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;

	output << "bootstrapping," << time_diff.count() / 1000 << "," << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << "," << cnn_out.cipher().scale() << endl << endl;
}
void channel_multiply_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, const vector<double> &weight, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, RelinKeys &relin_keys, ofstream &output, Decryptor &decryptor, SEALContext &context)
{
    cout << "channel-wise multiply..." << endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	cout << "before channel-wise mulitply" << endl;
	// decrypt_and_print(cnn_in.cipher(), decryptor, encoder, 1<<logn, 256, 2);
	channel_multiply_seal(cnn_in, cnn_out, weight, encoder, encryptor, evaluator, relin_keys);
	cout << "after channel-wise mulitply" << endl;
	// decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2); 
	cnn_out.print_parms();
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	cout << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;
	output << "channel-wise multiply," << time_diff.count() / 1000 << "," << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << "," << cnn_out.cipher().scale() << endl << endl;
}
void cipher_add_seal_print(const TensorCipher &cnn1, const TensorCipher &cnn2, TensorCipher &destination, Evaluator &evaluator, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context)
{
    cout << "residual add..." << endl;
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;
	time_start = chrono::high_resolution_clock::now();
	int logn = cnn1.logn();
	// decrypt_and_print(cnn1.cipher(), decryptor, encoder, 1<<logn, 256, 2);
	// decrypt_and_print(cnn2.cipher(), decryptor, encoder, 1<<logn, 256, 2);
	cnn_add_seal(cnn1, cnn2, destination, evaluator);
	// decrypt_and_print(destination.cipher(), decryptor, encoder, 1<<logn, 256, 2);
	destination.print_parms();
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	cout << "remaining level : " << context.get_context_data(destination.cipher().parms_id())->chain_index() << endl;
	cout << "scale: " << destination.cipher().scale() << endl << endl;
	output << "residual add," << time_diff.count() / 1000 << "," << context.get_context_data(destination.cipher().parms_id())->chain_index() << "," << destination.cipher().scale() << endl << endl;
}
void multiplexed_parallel_downsampling_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context, GaloisKeys &gal_keys, ofstream &output)
{
    cout << "multiplexed parallel downsampling..." << endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	// multiplexed_parallel_downsampling_seal(cnn_in, cnn_out, evaluator, gal_keys);
	if(cnn_in.t()==4) multiplexed_parallel_downsampling_seal2(cnn_in, cnn_out, evaluator, gal_keys);
	else if(cnn_in.t()==2) multiplexed_parallel_downsampling_seal3(cnn_in, cnn_out, evaluator, gal_keys, decryptor, encoder); 
	else multiplexed_parallel_downsampling_seal(cnn_in, cnn_out, evaluator, gal_keys);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	cout << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl;
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;
	output << "multiplexed parallel downsampling," << time_diff.count() / 1000 << "," << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << "," << cnn_out.cipher().scale() << endl << endl;
}
void averagepooling_seal_scale_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys, double B, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context)
{
    cout << "average pooling..." << endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	averagepooling_seal_scale(cnn_in, cnn_out, evaluator, gal_keys, B, encoder, decryptor, output);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	// decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	cout << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl; 
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;
	output << "average pooling," << time_diff.count() / 1000 << "," << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << "," << cnn_out.cipher().scale() << endl << endl;

}
void averagepooling_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context)
{
    cout << "average pooling..." << endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	averagepooling_seal(cnn_in, cnn_out, evaluator, gal_keys, encoder, decryptor, output);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	// decrypt_and_print(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	cout << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl; 
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;
	output << "average pooling," << time_diff.count() / 1000 << "," << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << "," << cnn_out.cipher().scale() << endl << endl;

}
void fully_connected_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, vector<double> matrix, vector<double> bias, int q, int r, Evaluator &evaluator, GaloisKeys &gal_keys, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context)
{
    cout << "fully connected layer..." << endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	matrix_multiplication_seal(cnn_in, cnn_out, matrix, bias, q, r, evaluator, gal_keys, encoder);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	cout << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl; 
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;

	output << "fully connected layer," << time_diff.count() / 1000 << "," << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << "," << cnn_out.cipher().scale() << endl << endl;
}
void batchnorm1d_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, vector<double> &running_mean, vector<double> &running_var, double epsilon, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context)
{
    cout << "Batchnorm 1D..." << endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	batchnorm1d_seal(cnn_in, cnn_out, evaluator, encoder, running_mean, running_var, epsilon);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	cout << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl; 
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;

	output << "Batchnorm 1D," << time_diff.count() / 1000 << "," << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << "," << cnn_out.cipher().scale() << endl << endl;
}
void batchnorm1d_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator,  vector<double> &a, vector<double> &weight, vector<double> &bias, vector<double> &running_mean, vector<double> &running_var, double epsilon, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context)
{
    cout << "Batchnorm 1D..." << endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	batchnorm1d_seal(cnn_in, cnn_out, evaluator, encoder, a, weight, bias, running_mean, running_var, epsilon);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	cout << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl; 
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;

	output << "Batchnorm 1D," << time_diff.count() / 1000 << "," << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << "," << cnn_out.cipher().scale() << endl << endl;
}
void l2norm_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys, double &a, double &b, double &c, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context)
{
    cout << "L2 Norm ..." << endl;
	int logn = cnn_in.logn();
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	l2norm_seal(cnn_in, cnn_out, evaluator, gal_keys, relin_keys, a, b, c, decryptor, encoder);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	cout << "remaining level : " << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << endl; 
	cout << "scale: " << cnn_out.cipher().scale() << endl << endl;

	output << "L2 Norm," << time_diff.count() / 1000 << "," << context.get_context_data(cnn_out.cipher().parms_id())->chain_index() << "," << cnn_out.cipher().scale() << endl << endl;
}
void l2norm_seal_print(Ciphertext &ct, size_t logn, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys, double &a, double &b, double &c, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context)
{
    cout << "L2 Norm ..." << endl;
	chrono::high_resolution_clock::time_point time_start, time_end;
	chrono::microseconds time_diff;

	time_start = chrono::high_resolution_clock::now();
	l2norm_seal(ct, logn, evaluator, gal_keys, relin_keys, a, b, c, decryptor, encoder);
	time_end = chrono::high_resolution_clock::now();
	time_diff = chrono::duration_cast<chrono::milliseconds>(time_end - time_start);
	cout << "time : " << time_diff.count() / 1000 << " ms" << endl;
	cout << "remaining level : " << context.get_context_data(ct.parms_id())->chain_index() << endl; 
	cout << "scale: " << ct.scale() << endl << endl;
	output << "L2 Norm," << time_diff.count() / 1000 << "," << context.get_context_data(ct.parms_id())->chain_index() << "," << ct.scale() << endl << endl;
}
void channel_square_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, const vector<double> &weight0, const vector<double> &weight1, const vector<double> &weight2, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, RelinKeys &relin_keys)
{
	// set parameters
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	long n = 1 << logn;

	// HerPN coefficients
	vector<double> w0(n, 0), w1(n, 0), w2(n, 0);
	for (size_t x = 0; x < ti; x++)
	{
		for (size_t y = 0; y < ki * hi; y++)
		{
			for (size_t z = 0; z < ki * wi; z++)
			{
				w0[x*ki*wi*ki*hi+y*ki*wi+z] = weight0[z%ki+(y%ki)*ki+x*ki*ki];
				w1[x*ki*wi*ki*hi+y*ki*wi+z] = weight1[z%ki+(y%ki)*ki+x*ki*ki];
				w2[x*ki*wi*ki*hi+y*ki*wi+z] = weight2[z%ki+(y%ki)*ki+x*ki*ki];
			}
		}
	}
	for (long i=n/pi; i<n; i++) {w0[i] = w0[i%(n/pi)]; w1[i] = w1[i%(n/pi)]; w2[i] = w2[i%(n/pi)];}
	

	Ciphertext input, square;
	input = cnn_in.cipher();
	square = cnn_in.cipher();
	// w2 * x^2
	evaluator.square_inplace(square);
	evaluator.relinearize_inplace(square, relin_keys);
	evaluator.rescale_to_next_inplace(square);
	evaluator.multiply_vector_inplace_reduced_error(square, w2);
	evaluator.rescale_to_next_inplace(square);
	// w1 * x
	evaluator.multiply_vector_inplace_reduced_error(input, w1);
	evaluator.rescale_to_next_inplace(input);
	// w2 * x^2 + w1 * x
	evaluator.add_inplace_reduced_error(input, square);
	// w2 * x^2 + w1 * x + w0
	Plaintext plain;
	encoder.encode(w0, input.scale(), plain);
	evaluator.mod_switch_to_inplace(plain, input.parms_id());
	evaluator.add_plain_inplace(input, plain);

	cnn_out = TensorCipher(logn, ki, hi, wi, ci, ti, pi, input);

}

void channel_multiply_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, const vector<double> &weight1, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, RelinKeys &relin_keys)
{
	// set parameters
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	long n = 1 << logn;
	vector<double> w1(n, 0);
	for (size_t x = 0; x < ti; x++)
	{
		for (size_t y = 0; y < ki * hi; y++)
		{
			for (size_t z = 0; z < ki * wi; z++)
			{
				w1[x*ki*wi*ki*hi+y*ki*wi+z] = weight1[z%ki+(y%ki)*ki+x*ki*ki];
			}
		}
	}
	for (long i=n/pi; i<n; i++) {w1[i] = w1[i%(n/pi)];}
	Ciphertext input;
	input = cnn_in.cipher();
	// w1 * x
	evaluator.multiply_vector_inplace_reduced_error(input, w1);
	evaluator.rescale_to_next_inplace(input);
	cnn_out = TensorCipher(logn, ki, hi, wi, ci, ti, pi, input);
}

void channel_square_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, const vector<double> &weight0, const vector<double> &weight1, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, RelinKeys &relin_keys)
{
	// set parameters
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	long n = 1 << logn;

	// HerPN coefficients
	vector<double> w0(n, 0), w1(n, 0);
	for (size_t x = 0; x < ti; x++)
	{
		for (size_t y = 0; y < ki * hi; y++)
		{
			for (size_t z = 0; z < ki * wi; z++)
			{
				w0[x*ki*wi*ki*hi+y*ki*wi+z] = weight0[z%ki+(y%ki)*ki+x*ki*ki];
				w1[x*ki*wi*ki*hi+y*ki*wi+z] = weight1[z%ki+(y%ki)*ki+x*ki*ki];
			}
		}
	}
	for (long i=n/pi; i<n; i++) {w0[i] = w0[i%(n/pi)]; w1[i] = w1[i%(n/pi)];}
	

	Ciphertext input, square;
	input = cnn_in.cipher();
	square = cnn_in.cipher();
	// x^2
	evaluator.square_inplace(square);
	evaluator.relinearize_inplace(square, relin_keys);
	evaluator.rescale_to_next_inplace(square);
	// w1 * x
	evaluator.multiply_vector_inplace_reduced_error(input, w1);
	evaluator.rescale_to_next_inplace(input);
	// x^2 + w1 * x
	evaluator.add_inplace_reduced_error(input, square);
	// x^2 + w1 * x + w0
	Plaintext plain;
	encoder.encode(w0, input.scale(), plain);
	evaluator.mod_switch_to_inplace(plain, input.parms_id());
	evaluator.add_plain_inplace(input, plain);

	cnn_out = TensorCipher(logn, ki, hi, wi, ci, ti, pi, input);

}

void multiplexed_parallel_convolution_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, int co, int st, int fh, int fw, const vector<double> &data, vector<double> running_var, vector<double> constant_weight, double epsilon, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, GaloisKeys &gal_keys, vector<Ciphertext> &cipher_pool, bool end)
{
	// set parameters
    vector<double> conv_data;
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = 0, ho = 0, wo = 0, to = 0, po = 0;

	// error check
	if(st != 1 && st != 2) throw invalid_argument("supported st is only 1 or 2");		// check if st is 1 or 2
    if(static_cast<int>(data.size()) != fh*fw*ci*co) throw std::invalid_argument("the size of data vector is not ker x ker x h x h");	// check if the size of data vector is kernel x kernel x h x h'
	if(log2_long(ki) == -1) throw std::invalid_argument("ki is not power of two");

	if(static_cast<int>(running_var.size())!=co || static_cast<int>(constant_weight.size())!=co) throw std::invalid_argument("the size of running_var or weight is not correct");
	// for(auto num : running_var) if(num<pow(10,-16) && num>-pow(10,-16)) throw std::invalid_argument("the size of running_var is too small. nearly zero.");

	// set ho, wo, ko
	if(st == 1) 
	{
		ho = hi;
		wo = wi;
		ko = ki;
	}
	else if(st == 2) 
	{
		if(hi%2 == 1 || wi%2 == 1) throw std::invalid_argument("hi or wi is not even");
		ho = hi/2;
		wo = wi/2;
		ko = 2*ki;
	}

	// set to, po, q
	long n = 1<<logn;
	to = (co+ko*ko-1) / (ko*ko);
	po =  pow2(floor_to_int(log(static_cast<double>(n)/static_cast<double>(ko*ko*ho*wo*to)) / log(2.0)));
	long q = (co+pi-1)/pi;

	// check if pi, po | n
	if(n%pi != 0) throw std::out_of_range("n is not divisible by pi");
	if(n%po != 0) throw std::out_of_range("n is not divisible by po");

	// check if ki^2 hi wi ti pi <= n and ko^2 ho wo to po <= n
	if(ki*ki*hi*wi*ti*pi > n) throw std::out_of_range("ki^2 hi wi ti pi is larger than n");
	if(ko*ko*ho*wo*to*po > (1<<logn)) throw std::out_of_range("ko^2 ho wo to po is larger than n");

	// variable
	vector<vector<vector<vector<double>>>> weight(fh, vector<vector<vector<double>>>(fw, vector<vector<double>>(ci, vector<double>(co, 0.0))));		// weight tensor
	vector<vector<vector<vector<double>>>> compact_weight_vec(fh, vector<vector<vector<double>>>(fw, vector<vector<double>>(q, vector<double>(n, 0.0))));	// multiplexed parallel shifted weight tensor
	vector<vector<vector<vector<double>>>> select_one(co, vector<vector<vector<double>>>(ko*ho, vector<vector<double>>(ko*wo, vector<double>(to, 0.0))));
	vector<vector<double>> select_one_vec(co, vector<double>(1<<logn, 0.0));

	// weight setting
	for(int i1=0; i1<fh; i1++)
	{
		for(int i2=0; i2<fw; i2++)
		{
			for(int j3=0; j3<ci; j3++)
			{
				for(int j4=0; j4<co; j4++)
				{
					weight[i1][i2][j3][j4] = data[fh*fw*ci*j4 + fh*fw*j3 + fw*i1 + i2];
				}
			}
		}
	}

	// compact shifted weight vector setting
	for(int i1=0; i1<fh; i1++)
	{
		for(int i2=0; i2<fw; i2++)
		{
			for(int i9=0; i9<q; i9++)
			{
				for(int j8=0; j8<n; j8++)
				{
					int j5 = ((j8%(n/pi))%(ki*ki*hi*wi))/(ki*wi), j6 = (j8%(n/pi))%(ki*wi), i7 = (j8%(n/pi))/(ki*ki*hi*wi), i8 = j8/(n/pi);
					if(j8%(n/pi)>=ki*ki*hi*wi*ti || i8+pi*i9>=co || ki*ki*i7+ki*(j5%ki)+j6%ki>=ci || (j6/ki)-(fw-1)/2+i2 < 0 || (j6/ki)-(fw-1)/2+i2 > wi-1 || (j5/ki)-(fh-1)/2+i1 < 0 || (j5/ki)-(fh-1)/2+i1 > hi-1)
						compact_weight_vec[i1][i2][i9][j8] = 0.0;
					else
					{
						compact_weight_vec[i1][i2][i9][j8] = weight[i1][i2][ki*ki*i7+ki*(j5%ki)+j6%ki][i8+pi*i9];
					}
				}
			}
		}
	}

	// select one setting
	for(int j4=0; j4<co; j4++)
	{
		for(int v1=0; v1<ko*ho; v1++)
		{
			for(int v2=0; v2<ko*wo; v2++)
			{
				for(int u3=0; u3<to; u3++)
				{
					if(ko*ko*u3 + ko*(v1%ko) + v2%ko == j4)	select_one[j4][v1][v2][u3] = constant_weight[j4] / sqrt(running_var[j4]+epsilon);
					else select_one[j4][v1][v2][u3] = 0.0;
				}
			}
		}
	}

	// select one vector setting
	for(int j4=0; j4<co; j4++)
	{
		for(int v1=0; v1<ko*ho; v1++)
		{
			for(int v2=0; v2<ko*wo; v2++)
			{
				for(int u3=0; u3<to; u3++)
				{
					select_one_vec[j4][ko*ko*ho*wo*u3 + ko*wo*v1 + v2] = select_one[j4][v1][v2][u3];
				}
			}
		}
	}

	// ciphertext variables
	Ciphertext *ctxt_in=&cipher_pool[0], *ct_zero=&cipher_pool[1], *temp=&cipher_pool[2], *sum=&cipher_pool[3], *total_sum=&cipher_pool[4], *var=&cipher_pool[5];

	// ciphertext input
	*ctxt_in = cnn_in.cipher();

	// rotated input precomputation
	vector<vector<Ciphertext*>> ctxt_rot(fh, vector<Ciphertext*>(fw));
	// if(fh != 3 || fw != 3) throw std::invalid_argument("fh and fw should be 3");
	if(fh%2 == 0 || fw%2 == 0) throw std::invalid_argument("fh and fw should be odd");
	for(int i1=0; i1<fh; i1++)
	{
		for(int i2=0; i2<fw; i2++)
		{
			if(i1==(fh-1)/2 && i2==(fw-1)/2) ctxt_rot[i1][i2] = ctxt_in;		// i1=(fh-1)/2, i2=(fw-1)/2 means ctxt_in
			else if((i1==(fh-1)/2 && i2>(fw-1)/2) || i1>(fh-1)/2) ctxt_rot[i1][i2] = &cipher_pool[6+fw*i1+i2-1];
			else ctxt_rot[i1][i2] = &cipher_pool[6+fw*i1+i2];
		}
	}
	// ctxt_rot[0][0] = &cipher_pool[6];	ctxt_rot[0][1] = &cipher_pool[7];	ctxt_rot[0][2] = &cipher_pool[8];	
	// ctxt_rot[1][0] = &cipher_pool[9];	ctxt_rot[1][1] = ctxt_in;			ctxt_rot[1][2] = &cipher_pool[10];		// i1=1, i2=1 means ctxt_in
	// ctxt_rot[2][0] = &cipher_pool[11];	ctxt_rot[2][1] = &cipher_pool[12];	ctxt_rot[2][2] = &cipher_pool[13];
	for(int i1=0; i1<fh; i1++)
	{
		for(int i2=0; i2<fw; i2++)
		{
			*ctxt_rot[i1][i2] = *ctxt_in;
			memory_save_rotate(*ctxt_rot[i1][i2], *ctxt_rot[i1][i2], ki*ki*wi*(i1-(fh-1)/2) + ki*(i2-(fw-1)/2), evaluator, gal_keys);
		}
	}

	// generate zero ciphertext 
	vector<double> zero(1<<logn, 0.0);
	Plaintext plain;
	encoder.encode(zero, ctxt_in->scale(), plain);
	encryptor.encrypt(plain, *ct_zero);		// ct_zero: original scaling factor

	for(int i9=0; i9<q; i9++)
	{
		// weight multiplication
		// cout << "multiplication by filter coefficients" << endl;
		for(int i1=0; i1<fh; i1++)
		{
			for(int i2=0; i2<fw; i2++)
			{
				// *temp = *ctxt_in;
				// memory_save_rotate(*temp, *temp, k*k*l*(i1-(kernel-1)/2) + k*(i2-(kernel-1)/2), scale_evaluator, gal_keys);
				// scale_evaluator.multiply_vector_inplace_scaleinv(*temp, compact_weight_vec[i1][i2][i9]);		// temp: double scaling factor
				evaluator.multiply_vector_reduced_error(*ctxt_rot[i1][i2], compact_weight_vec[i1][i2][i9], *temp);		// temp: double scaling factor
				if(i1==0 && i2==0) *sum = *temp;	// sum: double scaling factor
				else evaluator.add_inplace_reduced_error(*sum, *temp);
			}
		}
		evaluator.rescale_to_next_inplace(*sum);
		*var = *sum;

		// summation for all input channels
		// cout << "summation for all input channels" << endl;
		int d = log2_long(ki), c = log2_long(ti);
		for(int x=0; x<d; x++)
		{
			*temp = *var;
		//	scale_evaluator.rotate_vector(temp, pow2(x), gal_keys, temp);
			memory_save_rotate(*temp, *temp, pow2(x), evaluator, gal_keys);
			evaluator.add_inplace_reduced_error(*var, *temp);
		}
		for(int x=0; x<d; x++)
		{
			*temp = *var;
		//	scale_evaluator.rotate_vector(temp, pow2(x)*k*l, gal_keys, temp);
			memory_save_rotate(*temp, *temp, pow2(x)*ki*wi, evaluator, gal_keys);
			evaluator.add_inplace_reduced_error(*var, *temp);
		}
		if(c==-1)
		{
			*sum = *ct_zero;
			for(int x=0; x<ti; x++)
			{
				*temp = *var;
			//	scale_evaluator.rotate_vector(temp, k*k*l*l*x, gal_keys, temp);
				memory_save_rotate(*temp, *temp, ki*ki*hi*wi*x, evaluator, gal_keys);
				evaluator.add_inplace_reduced_error(*sum, *temp);
			}
			*var = *sum;
		}
		else
		{
			for(int x=0; x<c; x++)
			{
				*temp = *var;
			//	scale_evaluator.rotate_vector(temp, pow2(x)*k*k*l*l, gal_keys, temp);
				memory_save_rotate(*temp, *temp, pow2(x)*ki*ki*hi*wi, evaluator, gal_keys);
				evaluator.add_inplace_reduced_error(*var, *temp);
			}
		}

		// collecting valid values into one ciphertext.
		// cout << "collecting valid values into one ciphertext." << endl;
		for(int i8=0; i8<pi && pi*i9+i8<co; i8++)
		{
			int j4 = pi*i9+i8;
			if(j4 >= co) throw std::out_of_range("the value of j4 is out of range!");

			*temp = *var;
			memory_save_rotate(*temp, *temp, (n/pi)*(j4%pi) - j4%ko - (j4/(ko*ko))*ko*ko*ho*wo - ((j4%(ko*ko))/ko)*ko*wo, evaluator, gal_keys);
			evaluator.multiply_vector_inplace_reduced_error(*temp, select_one_vec[j4]);		// temp: double scaling factor
			if(i8==0 && i9==0) *total_sum = *temp;	// total_sum: double scaling factor
			else evaluator.add_inplace_reduced_error(*total_sum, *temp);
		}
	}
	evaluator.rescale_to_next_inplace(*total_sum);
	*var = *total_sum;

	// po copies
	if(end == false)
	{
		// cout << "po copies" << endl;
		*sum = *ct_zero;
		for(int u6=0; u6<po; u6++)
		{
			*temp = *var;
			memory_save_rotate(*temp, *temp, -u6*(n/po), evaluator, gal_keys);
			evaluator.add_inplace_reduced_error(*sum, *temp);		// sum: original scaling factor.
		}
		*var = *sum;
	}

	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, po, *var);

}
void multiplexed_parallel_convolution_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, int co, int st, int fh, int fw, const vector<double> &data, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, GaloisKeys &gal_keys, vector<Ciphertext> &cipher_pool, bool end)
{
	// set parameters
    vector<double> conv_data;
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = 0, ho = 0, wo = 0, to = 0, po = 0;

	// error check
	if(st != 1 && st != 2) throw invalid_argument("supported st is only 1 or 2");		// check if st is 1 or 2
    if(static_cast<int>(data.size()) != fh*fw*ci*co) throw std::invalid_argument("the size of data vector is not ker x ker x h x h");	// check if the size of data vector is kernel x kernel x h x h'
	if(log2_long(ki) == -1) throw std::invalid_argument("ki is not power of two");

	// for(auto num : running_var) if(num<pow(10,-16) && num>-pow(10,-16)) throw std::invalid_argument("the size of running_var is too small. nearly zero.");

	// set ho, wo, ko
	if(st == 1) 
	{
		ho = hi;
		wo = wi;
		ko = ki;
	}
	else if(st == 2) 
	{
		if(hi%2 == 1 || wi%2 == 1) throw std::invalid_argument("hi or wi is not even");
		ho = hi/2;
		wo = wi/2;
		ko = 2*ki;
	}

	// set to, po, q
	long n = 1<<logn;
	to = (co+ko*ko-1) / (ko*ko);
	po =  pow2(floor_to_int(log(static_cast<double>(n)/static_cast<double>(ko*ko*ho*wo*to)) / log(2.0)));
	long q = (co+pi-1)/pi;

	// check if pi, po | n
	if(n%pi != 0) throw std::out_of_range("n is not divisible by pi");
	if(n%po != 0) throw std::out_of_range("n is not divisible by po");

	// check if ki^2 hi wi ti pi <= n and ko^2 ho wo to po <= n
	if(ki*ki*hi*wi*ti*pi > n) throw std::out_of_range("ki^2 hi wi ti pi is larger than n");
	if(ko*ko*ho*wo*to*po > (1<<logn)) throw std::out_of_range("ko^2 ho wo to po is larger than n");

	// variable
	vector<vector<vector<vector<double>>>> weight(fh, vector<vector<vector<double>>>(fw, vector<vector<double>>(ci, vector<double>(co, 0.0))));		// weight tensor
	vector<vector<vector<vector<double>>>> compact_weight_vec(fh, vector<vector<vector<double>>>(fw, vector<vector<double>>(q, vector<double>(n, 0.0))));	// multiplexed parallel shifted weight tensor
	vector<vector<vector<vector<double>>>> select_one(co, vector<vector<vector<double>>>(ko*ho, vector<vector<double>>(ko*wo, vector<double>(to, 0.0))));
	vector<vector<double>> select_one_vec(co, vector<double>(1<<logn, 0.0));

	// weight setting
	for(int i1=0; i1<fh; i1++)
	{
		for(int i2=0; i2<fw; i2++)
		{
			for(int j3=0; j3<ci; j3++)
			{
				for(int j4=0; j4<co; j4++)
				{
					weight[i1][i2][j3][j4] = data[fh*fw*ci*j4 + fh*fw*j3 + fw*i1 + i2];
				}
			}
		}
	}

	// compact shifted weight vector setting
	for(int i1=0; i1<fh; i1++)
	{
		for(int i2=0; i2<fw; i2++)
		{
			for(int i9=0; i9<q; i9++)
			{
				for(int j8=0; j8<n; j8++)
				{
					int j5 = ((j8%(n/pi))%(ki*ki*hi*wi))/(ki*wi), j6 = (j8%(n/pi))%(ki*wi), i7 = (j8%(n/pi))/(ki*ki*hi*wi), i8 = j8/(n/pi);
					if(j8%(n/pi)>=ki*ki*hi*wi*ti || i8+pi*i9>=co || ki*ki*i7+ki*(j5%ki)+j6%ki>=ci || (j6/ki)-(fw-1)/2+i2 < 0 || (j6/ki)-(fw-1)/2+i2 > wi-1 || (j5/ki)-(fh-1)/2+i1 < 0 || (j5/ki)-(fh-1)/2+i1 > hi-1)
						compact_weight_vec[i1][i2][i9][j8] = 0.0;
					else
					{
						compact_weight_vec[i1][i2][i9][j8] = weight[i1][i2][ki*ki*i7+ki*(j5%ki)+j6%ki][i8+pi*i9];
					}
				}
			}
		}
	}

	// select one setting
	for(int j4=0; j4<co; j4++)
	{
		for(int v1=0; v1<ko*ho; v1++)
		{
			for(int v2=0; v2<ko*wo; v2++)
			{
				for(int u3=0; u3<to; u3++)
				{
					if(ko*ko*u3 + ko*(v1%ko) + v2%ko == j4)	select_one[j4][v1][v2][u3] = 1;
					else select_one[j4][v1][v2][u3] = 0.0;
				}
			}
		}
	}

	// select one vector setting
	for(int j4=0; j4<co; j4++)
	{
		for(int v1=0; v1<ko*ho; v1++)
		{
			for(int v2=0; v2<ko*wo; v2++)
			{
				for(int u3=0; u3<to; u3++)
				{
					select_one_vec[j4][ko*ko*ho*wo*u3 + ko*wo*v1 + v2] = select_one[j4][v1][v2][u3];
				}
			}
		}
	}

	// ciphertext variables
	Ciphertext *ctxt_in=&cipher_pool[0], *ct_zero=&cipher_pool[1], *temp=&cipher_pool[2], *sum=&cipher_pool[3], *total_sum=&cipher_pool[4], *var=&cipher_pool[5];

	// ciphertext input
	*ctxt_in = cnn_in.cipher();

	// rotated input precomputation
	vector<vector<Ciphertext*>> ctxt_rot(fh, vector<Ciphertext*>(fw));
	// if(fh != 3 || fw != 3) throw std::invalid_argument("fh and fw should be 3");
	if(fh%2 == 0 || fw%2 == 0) throw std::invalid_argument("fh and fw should be odd");
	for(int i1=0; i1<fh; i1++)
	{
		for(int i2=0; i2<fw; i2++)
		{
			if(i1==(fh-1)/2 && i2==(fw-1)/2) ctxt_rot[i1][i2] = ctxt_in;		// i1=(fh-1)/2, i2=(fw-1)/2 means ctxt_in
			else if((i1==(fh-1)/2 && i2>(fw-1)/2) || i1>(fh-1)/2) ctxt_rot[i1][i2] = &cipher_pool[6+fw*i1+i2-1];
			else ctxt_rot[i1][i2] = &cipher_pool[6+fw*i1+i2];
		}
	}
	// ctxt_rot[0][0] = &cipher_pool[6];	ctxt_rot[0][1] = &cipher_pool[7];	ctxt_rot[0][2] = &cipher_pool[8];	
	// ctxt_rot[1][0] = &cipher_pool[9];	ctxt_rot[1][1] = ctxt_in;			ctxt_rot[1][2] = &cipher_pool[10];		// i1=1, i2=1 means ctxt_in
	// ctxt_rot[2][0] = &cipher_pool[11];	ctxt_rot[2][1] = &cipher_pool[12];	ctxt_rot[2][2] = &cipher_pool[13];
	for(int i1=0; i1<fh; i1++)
	{
		for(int i2=0; i2<fw; i2++)
		{
			*ctxt_rot[i1][i2] = *ctxt_in;
			memory_save_rotate(*ctxt_rot[i1][i2], *ctxt_rot[i1][i2], ki*ki*wi*(i1-(fh-1)/2) + ki*(i2-(fw-1)/2), evaluator, gal_keys);
		}
	}

	// generate zero ciphertext 
	vector<double> zero(1<<logn, 0.0);
	Plaintext plain;
	encoder.encode(zero, ctxt_in->scale(), plain);
	encryptor.encrypt(plain, *ct_zero);		// ct_zero: original scaling factor

	for(int i9=0; i9<q; i9++)
	{
		// weight multiplication
		// cout << "multiplication by filter coefficients" << endl;
		for(int i1=0; i1<fh; i1++)
		{
			for(int i2=0; i2<fw; i2++)
			{
				// *temp = *ctxt_in;
				// memory_save_rotate(*temp, *temp, k*k*l*(i1-(kernel-1)/2) + k*(i2-(kernel-1)/2), scale_evaluator, gal_keys);
				// scale_evaluator.multiply_vector_inplace_scaleinv(*temp, compact_weight_vec[i1][i2][i9]);		// temp: double scaling factor
				evaluator.multiply_vector_reduced_error(*ctxt_rot[i1][i2], compact_weight_vec[i1][i2][i9], *temp);		// temp: double scaling factor
				if(i1==0 && i2==0) *sum = *temp;	// sum: double scaling factor
				else evaluator.add_inplace_reduced_error(*sum, *temp);
			}
		}
		evaluator.rescale_to_next_inplace(*sum);
		*var = *sum;

		// summation for all input channels
		// cout << "summation for all input channels" << endl;
		int d = log2_long(ki), c = log2_long(ti);
		for(int x=0; x<d; x++)
		{
			*temp = *var;
		//	scale_evaluator.rotate_vector(temp, pow2(x), gal_keys, temp);
			memory_save_rotate(*temp, *temp, pow2(x), evaluator, gal_keys);
			evaluator.add_inplace_reduced_error(*var, *temp);
		}
		for(int x=0; x<d; x++)
		{
			*temp = *var;
		//	scale_evaluator.rotate_vector(temp, pow2(x)*k*l, gal_keys, temp);
			memory_save_rotate(*temp, *temp, pow2(x)*ki*wi, evaluator, gal_keys);
			evaluator.add_inplace_reduced_error(*var, *temp);
		}
		if(c==-1)
		{
			*sum = *ct_zero;
			for(int x=0; x<ti; x++)
			{
				*temp = *var;
			//	scale_evaluator.rotate_vector(temp, k*k*l*l*x, gal_keys, temp);
				memory_save_rotate(*temp, *temp, ki*ki*hi*wi*x, evaluator, gal_keys);
				evaluator.add_inplace_reduced_error(*sum, *temp);
			}
			*var = *sum;
		}
		else
		{
			for(int x=0; x<c; x++)
			{
				*temp = *var;
			//	scale_evaluator.rotate_vector(temp, pow2(x)*k*k*l*l, gal_keys, temp);
				memory_save_rotate(*temp, *temp, pow2(x)*ki*ki*hi*wi, evaluator, gal_keys);
				evaluator.add_inplace_reduced_error(*var, *temp);
			}
		}

		// collecting valid values into one ciphertext.
		// cout << "collecting valid values into one ciphertext." << endl;
		for(int i8=0; i8<pi && pi*i9+i8<co; i8++)
		{
			int j4 = pi*i9+i8;
			if(j4 >= co) throw std::out_of_range("the value of j4 is out of range!");

			*temp = *var;
			memory_save_rotate(*temp, *temp, (n/pi)*(j4%pi) - j4%ko - (j4/(ko*ko))*ko*ko*ho*wo - ((j4%(ko*ko))/ko)*ko*wo, evaluator, gal_keys);
			evaluator.multiply_vector_inplace_reduced_error(*temp, select_one_vec[j4]);		// temp: double scaling factor
			if(i8==0 && i9==0) *total_sum = *temp;	// total_sum: double scaling factor
			else evaluator.add_inplace_reduced_error(*total_sum, *temp);
		}
	}
	evaluator.rescale_to_next_inplace(*total_sum);
	*var = *total_sum;

	// po copies
	if(end == false)
	{
		// cout << "po copies" << endl;
		*sum = *ct_zero;
		for(int u6=0; u6<po; u6++)
		{
			*temp = *var;
			memory_save_rotate(*temp, *temp, -u6*(n/po), evaluator, gal_keys);
			evaluator.add_inplace_reduced_error(*sum, *temp);		// sum: original scaling factor.
		}
		*var = *sum;
	}

	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, po, *var);

}
void multiplexed_parallel_batch_norm_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, vector<double> bias, vector<double> running_mean, vector<double> running_var, vector<double> weight, double epsilon, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, double B, bool end)
{
	// parameter setting
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = ki, ho = hi, wo = wi, co = ci, to = ti, po = pi;

	// error check
	if(static_cast<int>(bias.size())!=ci || static_cast<int>(running_mean.size())!=ci || static_cast<int>(running_var.size())!=ci || static_cast<int>(weight.size())!=ci) throw std::invalid_argument("the size of bias, running_mean, running_var, or weight are not correct");
	// for(auto num : running_var) if(num<pow(10,-16) && num>-pow(10,-16)) throw std::invalid_argument("the size of running_var is too small. nearly zero.");
	if(hi*wi*ci > 1<<logn) throw std::invalid_argument("hi*wi*ci should not be larger than n");

	// generate g vector
	vector<double> g(1<<logn, 0.0);

	// set f value
	long n = 1<<logn;

	// check if pi | n
	if(n%pi != 0) throw std::out_of_range("n is not divisible by pi");

	// set g vector
	for(int v4=0; v4<n; v4++)
	{
		int v1 = ((v4%(n/pi))%(ki*ki*hi*wi))/(ki*wi), v2 = (v4%(n/pi))%(ki*wi), u3 = (v4%(n/pi))/(ki*ki*hi*wi);
		if(ki*ki*u3+ki*(v1%ki)+v2%ki>=ci || v4%(n/pi)>=ki*ki*hi*wi*ti) g[v4] = 0.0;
		else 
		{
			int idx = ki*ki*u3 + ki*(v1%ki) + v2%ki;
			g[v4] = (running_mean[idx] * weight[idx] / sqrt(running_var[idx]+epsilon) - bias[idx])/B;
		}
	}

	// encode & encrypt
	Plaintext plain;
	Ciphertext cipher_g;
	Ciphertext temp;
	temp = cnn_in.cipher();
	encoder.encode(g, temp.scale(), plain);
	encryptor.encrypt(plain, cipher_g);

	// batch norm
	evaluator.sub_inplace_reduced_error(temp, cipher_g);

	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, po, temp);

}
void ReLU_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, long comp_no, vector<int> deg, long alpha, vector<Tree> &tree, double scaled_val, long scalingfactor, Encryptor &encryptor, Evaluator &evaluator, Decryptor &decryptor, CKKSEncoder &encoder, PublicKey &public_key, SecretKey &secret_key, RelinKeys &relin_keys, double scale)
{
	// parameter setting
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = ki, ho = hi, wo = wi, co = ci, to = ti, po = pi;

	// error check
	if(hi*wi*ci > 1<<logn) throw std::invalid_argument("hi*wi*ci should not be larger than n");

	// ReLU
	Ciphertext temp;
	temp = cnn_in.cipher();
	minimax_ReLU_seal(comp_no, deg, alpha, tree, scaled_val, scalingfactor, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, temp, temp);

	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, po, temp);
}
void evorelu_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, long comp_no, vector<int> deg, vector<vector<double>> &decomp_coeff, vector<Tree> &tree, Encryptor &encryptor, Evaluator &evaluator, Decryptor &decryptor, CKKSEncoder &encoder, PublicKey &public_key, SecretKey &secret_key, RelinKeys &relin_keys, double B)
{
	// parameter setting
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = ki, ho = hi, wo = wi, co = ci, to = ti, po = pi;

	// error check
	if(hi*wi*ci > 1<<logn) throw std::invalid_argument("hi*wi*ci should not be larger than n");

	// ReLU
	Ciphertext temp;
	temp = cnn_in.cipher();
	autofhe_ReLU_seal(comp_no, deg, decomp_coeff, tree, B, encryptor, evaluator, decryptor, encoder, public_key, secret_key, relin_keys, temp, temp);

	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, po, temp);
}
void cnn_add_seal(const TensorCipher &cnn1, const TensorCipher &cnn2, TensorCipher &destination, Evaluator &evaluator)
{
	// parameter setting
	int k1 = cnn1.k(), h1 = cnn1.h(), w1 = cnn1.w(), c1 = cnn1.c(), t1 = cnn1.t(), p1 = cnn1.p(), logn1 = cnn1.logn();
	int k2 = cnn2.k(), h2 = cnn2.h(), w2 = cnn2.w(), c2 = cnn2.c(), t2 = cnn2.t(), p2 = cnn2.p(), logn2 = cnn2.logn();

	// error check
	if(k1!=k2 || h1!=h2 || w1!=w2 || c1!=c2 || t1!=t2 || p1!=p2 || logn1!=logn2) throw std::invalid_argument("the parameters of cnn1 and cnn2 are not the same");

	// addition
	Ciphertext temp1, temp2;
	temp1 = cnn1.cipher();
	temp2 = cnn2.cipher();
	evaluator.add_inplace_reduced_error(temp1, temp2);

	destination = TensorCipher(logn1, k1, h1, w1, c1, t1, p1, temp1);
}
void multiplexed_parallel_downsampling_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys)
{
	// parameter setting
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = 0, ho = 0, wo = 0, co = 0, to = 0, po = 0;

	// parameter setting
	long n = 1<<logn;
	ko = 2*ki;
	ho = hi/2;
	wo = wi/2;
	to = ti/2;
	co = 2*ci;
	po = pow2(floor_to_int(log(static_cast<double>(n)/static_cast<double>(ko*ko*ho*wo*to)) / log(2.0)));

	// error check
	if(ti%8 != 0) throw std::invalid_argument("ti is not multiple of 8");
	if(hi%2 != 0) throw std::invalid_argument("hi is not even");
	if(wi%2 != 0) throw std::invalid_argument("wi is not even");
	if(n%po != 0) throw std::out_of_range("n is not divisible by po");		// check if po | n

	// variables
	vector<vector<vector<double>>> select_one_vec(ki, vector<vector<double>>(ti, vector<double>(1<<logn, 0.0)));
	Ciphertext ct, sum, temp;
	ct = cnn_in.cipher();

	// selecting tensor vector setting
	for(int w1=0; w1<ki; w1++)
	{
		for(int w2=0; w2<ti; w2++)
		{
			for(int v4=0; v4<1<<logn; v4++)
			{
				int j5 = (v4%(ki*ki*hi*wi))/(ki*wi), j6 = v4%(ki*wi), i7 = v4/(ki*ki*hi*wi);
				if(v4<ki*ki*hi*wi*ti && (j5/ki)%2 == 0 && (j6/ki)%2 == 0 && (j5%ki) == w1 && i7 == w2) select_one_vec[w1][w2][v4] = 1.0;
				else select_one_vec[w1][w2][v4] = 0.0;
			}
		}
	}

	for(int w1=0; w1<ki; w1++)
	{
		for(int w2=0; w2<ti; w2++)
		{
			temp = ct;
			evaluator.multiply_vector_inplace_reduced_error(temp, select_one_vec[w1][w2]);

			int w3 = ((ki*w2+w1)%(2*ko))/2, w4 = (ki*w2+w1)%2, w5 = (ki*w2+w1)/(2*ko);
			memory_save_rotate(temp, temp, ki*ki*hi*wi*w2 + ki*wi*w1 - ko*ko*ho*wo*w5 - ko*wo*w3 - ki*w4 - ko*ko*ho*wo*(ti/8), evaluator, gal_keys);
			if(w1==0 && w2==0) sum = temp;
			else evaluator.add_inplace_reduced_error(sum, temp);
			
		}
	}
	evaluator.rescale_to_next_inplace(sum);		// added
	ct = sum;

	// for fprime packing
	sum = ct;
	for(int u6=1; u6<po; u6++)
	{
		temp = ct;
		memory_save_rotate(temp, temp, -(n/po)*u6, evaluator, gal_keys);
		evaluator.add_inplace_reduced_error(sum, temp);
	}
	ct = sum;

	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, po, ct);

}
void multiplexed_parallel_downsampling_seal2(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys)
{
	// parameter setting
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = 0, ho = 0, wo = 0, co = 0, to = 0, po = 0;

	// parameter setting
	long n = 1<<logn;
	ko = 2*ki;
	ho = hi/2;
	wo = wi/2;
	to = ti/2;
	co = 2*ci;
	po = pow2(floor_to_int(log(static_cast<double>(n)/static_cast<double>(ko*ko*ho*wo*to)) / log(2.0)));

	// error check
	if(hi%2 != 0) throw std::invalid_argument("hi is not even");
	if(wi%2 != 0) throw std::invalid_argument("wi is not even");
	if(n%po != 0) throw std::out_of_range("n is not divisible by po");		// check if po | n

	// load select one vector
	ifstream in;
	in.open("../result/downsample1.txt");
	if(!in.is_open()) throw std::runtime_error("../result/downsample1.txt is not open.");
	vector<vector<vector<double>>> select_one_vec(ki, vector<vector<double>>(ti, vector<double>(1<<logn, 0.0)));
    for (size_t n1 = 0; n1 < ki; n1++)
    {
        double tmp;
        for (size_t n2 = 0; n2 < 4096; n2++)
        {
            in >> tmp;
            for (size_t n3 = 0; n3 < ti; n3++)
            {
                select_one_vec[n1][n3][n3 * 4096 + n2] = tmp;
            }
        }
    }
    in.close();

	Ciphertext ct, sum, temp;
	ct = cnn_in.cipher();
	vector<int> rotate_steps = {-128, 3904, 4096, 8128, -66, 3966, 4158, 8190};
	for (size_t n1 = 0; n1 < ki; n1++)
    {
        for (size_t n3 = 0; n3 < ti; n3++)
        {
			temp = ct;
			evaluator.multiply_vector_inplace_reduced_error(temp, select_one_vec[n1][n3]);
			int step = rotate_steps[n1*ti + n3];
			memory_save_rotate(temp, temp, step, evaluator, gal_keys);
			if(n1==0 && n3==0) sum = temp;
			else evaluator.add_inplace_reduced_error(sum, temp);
        }
    }
	evaluator.rescale_to_next_inplace(sum);		// added
	ct = sum;

	// for fprime packing
	sum = ct;
	for(int u6=1; u6<po; u6++)
	{
		temp = ct;
		memory_save_rotate(temp, temp, -(n/po)*u6, evaluator, gal_keys);
		evaluator.add_inplace_reduced_error(sum, temp);
	}
	ct = sum;

	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, po, ct);

}
void multiplexed_parallel_downsampling_seal3(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys, Decryptor &decryptor, CKKSEncoder &encoder)
{
	// parameter setting
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = 0, ho = 0, wo = 0, co = 0, to = 0, po = 0;

	// parameter setting
	long n = 1<<logn;
	ko = 2*ki;
	ho = hi/2;
	wo = wi/2;
	to = ti/2;
	co = 2*ci;
	po = pow2(floor_to_int(log(static_cast<double>(n)/static_cast<double>(ko*ko*ho*wo*to)) / log(2.0)));

	// error check
	if(hi%2 != 0) throw std::invalid_argument("hi is not even");
	if(wi%2 != 0) throw std::invalid_argument("wi is not even");
	if(n%po != 0) throw std::out_of_range("n is not divisible by po");		// check if po | n

	// load select one vector
	ifstream in;
	in.open("../result/downsample2.txt");
	if(!in.is_open()) throw std::runtime_error("../result/downsample2.txt is not open.");
	vector<vector<vector<double>>> select_one_vec(ki, vector<vector<double>>(ti, vector<double>(1<<logn, 0.0)));
    for (size_t n1 = 0; n1 < ki; n1++)
    {
        double tmp;
        for (size_t n2 = 0; n2 < 4096; n2++)
        {
            in >> tmp;
            for (size_t n3 = 0; n3 < ti; n3++)
            {
                select_one_vec[n1][n3][n3 * 4096 + n2] = tmp;
            }
        }
    }
    in.close();

	Ciphertext ct, sum, temp;
	ct = cnn_in.cipher();
	vector<int> rotate_steps = {-128, 3840, -68, 3900, -64, 3904, -4, 3964};
	for (size_t n1 = 0; n1 < ki; n1++)
    {
        for (size_t n3 = 0; n3 < ti; n3++)
        {
			temp = ct;
			// decrypt_and_output(temp, decryptor, encoder, 1<<logn, 256, 2);
			evaluator.multiply_vector_inplace_reduced_error(temp, select_one_vec[n1][n3]);
			// decrypt_and_output(temp, decryptor, encoder, 1<<logn, 256, 2);
			int step = rotate_steps[n1*ti + n3];
			memory_save_rotate(temp, temp, step, evaluator, gal_keys);
			// decrypt_and_output(temp, decryptor, encoder, 1<<logn, 256, 2);
			if(n1==0 && n3==0) sum = temp;
			else evaluator.add_inplace_reduced_error(sum, temp);
			// decrypt_and_output(sum, decryptor, encoder, 1<<logn, 256, 2);
        }
    }
	evaluator.rescale_to_next_inplace(sum);		// added
	ct = sum;

	// for fprime packing
	sum = ct;
	for(int u6=1; u6<po; u6++)
	{
		temp = ct;
		// decrypt_and_output(temp, decryptor, encoder, 1<<logn, 256, 2);
		memory_save_rotate(temp, temp, -(n/po)*u6, evaluator, gal_keys);
		// decrypt_and_output(temp, decryptor, encoder, 1<<logn, 256, 2);
		evaluator.add_inplace_reduced_error(sum, temp);
		// decrypt_and_output(sum, decryptor, encoder, 1<<logn, 256, 2);
	}
	ct = sum;

	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, po, ct);
	// decrypt_and_output(cnn_out.cipher(), decryptor, encoder, 1<<logn, 256, 2);
}
void averagepooling_seal_scale(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys, double B, CKKSEncoder &encoder, Decryptor &decryptor, ofstream &output)
{
	// parameter setting
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = 1, ho = 1, wo = 1, co = ci, to = ti;

	if(log2_long(hi) == -1) throw std::invalid_argument("hi is not power of two");
	if(log2_long(wi) == -1) throw std::invalid_argument("wi is not power of two");

	Ciphertext ct, temp, sum1, sum2, sum3, sum4;
	ct = cnn_in.cipher();
	double avg = B / static_cast<double>(hi*wi) * 4.;

	// (2,2) feature 
	for(int x=0; x<log2_long(wi/2); x++)
	{
		temp = ct;
		memory_save_rotate(temp, temp, pow2(x)*ki, evaluator, gal_keys);
		evaluator.add_inplace_reduced_error(ct, temp);
	}
	for(int x=0; x<log2_long(hi/2); x++)
	{
		temp = ct;
		memory_save_rotate(temp, temp, pow2(x)*ki*ki*wi, evaluator, gal_keys);
		evaluator.add_inplace_reduced_error(ct, temp);
	}

	// sum1
	vector<double> select_one(1<<logn, 0.0);
	for(int s=0; s<ki; s++)
	{
		for(int u=0; u<ti; u++)
		{
			int p=ki*u+s;
			temp = ct;
			memory_save_rotate(temp, temp, -p*ki + ki*ki*hi*wi*u + ki*wi*s, evaluator, gal_keys);
			fill(select_one.begin(), select_one.end(), 0.);
			for(int i=0; i<ki; i++) select_one[(ki*u+s)*ki+i] = avg;
			evaluator.multiply_vector_inplace_reduced_error(temp, select_one);
			if(u==0 && s==0) sum1 = temp;	
			else evaluator.add_inplace_reduced_error(sum1, temp);
		}
	}

	// sum2
	for(int s=0; s<ki; s++)
	{
		for(int u=0; u<ti; u++)
		{
			int p=ki*u+s;
			temp = ct;
			memory_save_rotate(temp, temp, -p*ki + ki*ki*hi*wi*u + ki*wi*s, evaluator, gal_keys);
			fill(select_one.begin(), select_one.end(), 0.);
			for(int i=0; i<ki; i++) select_one[(ki*u+s)*ki+i+wi/2*ki] = avg;
			evaluator.multiply_vector_inplace_reduced_error(temp, select_one);
			if(u==0 && s==0) sum2 = temp;	
			else evaluator.add_inplace_reduced_error(sum2, temp);
		}
	}
	memory_save_rotate(sum2, sum2, -32, evaluator, gal_keys);

	// sum3 
	for(int s=0; s<ki; s++)
	{
		for(int u=0; u<ti; u++)
		{
			int p=ki*u+s;
			temp = ct;
			memory_save_rotate(temp, temp, -p*ki + ki*ki*hi*wi*u + ki*wi*s, evaluator, gal_keys);
			fill(select_one.begin(), select_one.end(), 0.);
			for(int i=0; i<ki; i++) select_one[(ki*u+s)*ki+i+ki*ki*wi*hi/2] = avg;
			evaluator.multiply_vector_inplace_reduced_error(temp, select_one);
			if(u==0 && s==0) sum3 = temp;	
			else evaluator.add_inplace_reduced_error(sum3, temp);
		}
	}
	memory_save_rotate(sum3, sum3, ki*ki*wi*hi/2-128, evaluator, gal_keys);

	// sum4
	for(int s=0; s<ki; s++)
	{
		for(int u=0; u<ti; u++)
		{
			int p=ki*u+s;
			temp = ct;
			memory_save_rotate(temp, temp, -p*ki + ki*ki*hi*wi*u + ki*wi*s, evaluator, gal_keys);
			fill(select_one.begin(), select_one.end(), 0.);
			for(int i=0; i<ki; i++) select_one[(ki*u+s)*ki+i+ki*ki*wi*hi/2+wi/2*ki] = avg;
			evaluator.multiply_vector_inplace_reduced_error(temp, select_one);
			if(u==0 && s==0) sum4 = temp;	
			else evaluator.add_inplace_reduced_error(sum4, temp);
		}
	}
	memory_save_rotate(sum4, sum4, ki*ki*wi*hi/2+32-192, evaluator, gal_keys);

	// add
	evaluator.add_inplace_reduced_error(sum1, sum2);
	evaluator.add_inplace_reduced_error(sum1, sum3);
	evaluator.add_inplace_reduced_error(sum1, sum4);

	evaluator.rescale_to_next_inplace(sum1);

	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, 1, sum1);
	
}
void averagepooling_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys, CKKSEncoder &encoder, Decryptor &decryptor, ofstream &output)
{
	// parameter setting
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = 1, ho = 1, wo = 1, co = ci, to = ti;

	if(log2_long(hi) == -1) throw std::invalid_argument("hi is not power of two");
	if(log2_long(wi) == -1) throw std::invalid_argument("wi is not power of two");

	Ciphertext ct, temp, sum1, sum2, sum3, sum4;
	ct = cnn_in.cipher();
	double avg = 1. / static_cast<double>(hi*wi) * 4.;

	// (2,2) feature 
	for(int x=0; x<log2_long(wi/2); x++)
	{
		temp = ct;
		memory_save_rotate(temp, temp, pow2(x)*ki, evaluator, gal_keys);
		evaluator.add_inplace_reduced_error(ct, temp);
	}
	for(int x=0; x<log2_long(hi/2); x++)
	{
		temp = ct;
		memory_save_rotate(temp, temp, pow2(x)*ki*ki*wi, evaluator, gal_keys);
		evaluator.add_inplace_reduced_error(ct, temp);
	}

	// sum1
	vector<double> select_one(1<<logn, 0.0);
	for(int s=0; s<ki; s++)
	{
		for(int u=0; u<ti; u++)
		{
			int p=ki*u+s;
			temp = ct;
			memory_save_rotate(temp, temp, -p*ki + ki*ki*hi*wi*u + ki*wi*s, evaluator, gal_keys);
			fill(select_one.begin(), select_one.end(), 0.);
			for(int i=0; i<ki; i++) select_one[(ki*u+s)*ki+i] = avg;
			evaluator.multiply_vector_inplace_reduced_error(temp, select_one);
			if(u==0 && s==0) sum1 = temp;	
			else evaluator.add_inplace_reduced_error(sum1, temp);
		}
	}

	// sum2
	for(int s=0; s<ki; s++)
	{
		for(int u=0; u<ti; u++)
		{
			int p=ki*u+s;
			temp = ct;
			memory_save_rotate(temp, temp, -p*ki + ki*ki*hi*wi*u + ki*wi*s, evaluator, gal_keys);
			fill(select_one.begin(), select_one.end(), 0.);
			for(int i=0; i<ki; i++) select_one[(ki*u+s)*ki+i+wi/2*ki] = avg;
			evaluator.multiply_vector_inplace_reduced_error(temp, select_one);
			if(u==0 && s==0) sum2 = temp;	
			else evaluator.add_inplace_reduced_error(sum2, temp);
		}
	}
	memory_save_rotate(sum2, sum2, -48, evaluator, gal_keys);

	// sum3 
	for(int s=0; s<ki; s++)
	{
		for(int u=0; u<ti; u++)
		{
			int p=ki*u+s;
			temp = ct;
			memory_save_rotate(temp, temp, -p*ki + ki*ki*hi*wi*u + ki*wi*s, evaluator, gal_keys);
			fill(select_one.begin(), select_one.end(), 0.);
			for(int i=0; i<ki; i++) select_one[(ki*u+s)*ki+i+ki*ki*wi*hi/2] = avg;
			evaluator.multiply_vector_inplace_reduced_error(temp, select_one);
			if(u==0 && s==0) sum3 = temp;	
			else evaluator.add_inplace_reduced_error(sum3, temp);
		}
	}
	memory_save_rotate(sum3, sum3, ki*ki*wi*hi/2-128, evaluator, gal_keys);

	// sum4
	for(int s=0; s<ki; s++)
	{
		for(int u=0; u<ti; u++)
		{
			int p=ki*u+s;
			temp = ct;
			memory_save_rotate(temp, temp, -p*ki + ki*ki*hi*wi*u + ki*wi*s, evaluator, gal_keys);
			fill(select_one.begin(), select_one.end(), 0.);
			for(int i=0; i<ki; i++) select_one[(ki*u+s)*ki+i+ki*ki*wi*hi/2+wi/2*ki] = avg;
			evaluator.multiply_vector_inplace_reduced_error(temp, select_one);
			if(u==0 && s==0) sum4 = temp;	
			else evaluator.add_inplace_reduced_error(sum4, temp);
		}
	}
	memory_save_rotate(sum4, sum4, ki*ki*wi*hi/2+16-192, evaluator, gal_keys);

	// add
	evaluator.add_inplace_reduced_error(sum1, sum2);
	evaluator.add_inplace_reduced_error(sum1, sum3);
	evaluator.add_inplace_reduced_error(sum1, sum4);

	evaluator.rescale_to_next_inplace(sum1);

	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, 1, sum1);
	
}
void batchnorm1d_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, CKKSEncoder &encoder, vector<double> &running_mean, vector<double> &running_var, double epsilon)
{
	// parameter setting
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = 1, ho = 1, wo = 1, co = ci, to = ti;

	Ciphertext ct = cnn_in.cipher();

	vector<double> mean(1<<logn, 0.0);
	vector<double> std(1<<logn, 0.0);
	for (size_t i = 0; i < 256; i++)
	{
		mean[i] = running_mean[i] / sqrt(running_var[i] + epsilon);
		std[i] = 1. / sqrt(running_var[i] + epsilon);
	}
	evaluator.multiply_vector_inplace_reduced_error(ct, std);
	evaluator.rescale_to_next_inplace(ct);
	Plaintext plain;
	encoder.encode(mean, ct.scale(), plain);
	evaluator.mod_switch_to_inplace(plain, ct.parms_id());
	evaluator.sub_plain_inplace(ct, plain);
	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, 1, ct);
}
void batchnorm1d_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, CKKSEncoder &encoder, vector<double> &a0, vector<double> &weight0, vector<double> &bias0, vector<double> &running_mean0, vector<double> &running_var0, double epsilon)
{
	// parameter setting
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = 1, ho = 1, wo = 1, co = ci, to = ti;

	Ciphertext ct = cnn_in.cipher();

	vector<double> a(256, 0.), weight(256, 0.), bias(256, 0.), running_mean(256, 0.), running_var(256, 0.);
	for(size_t i = 0; i < 64; i++){
		for(size_t j=0; j<4; j++){
			a[i+64*j] = a0[i];
			weight[64*j+i] = weight0[i*4+j];
			bias[64*j+i] = bias0[i*4+j];
			running_mean[64*j+i] = running_mean0[i*4+j];
			running_var[64*j+i] = running_var0[i*4+j];
		}
	}

	vector<double> mean(1<<logn, 0.0);
	vector<double> std(1<<logn, 0.0);
	for (size_t i = 0; i < 256; i++)
	{
		mean[i] = running_mean[i] * weight[i] / sqrt(running_var[i] + epsilon) - bias[i];
		std[i] = a[i] * weight[i] / sqrt(running_var[i] + epsilon);
	}
	evaluator.multiply_vector_inplace_reduced_error(ct, std);
	evaluator.rescale_to_next_inplace(ct);
	Plaintext plain;
	encoder.encode(mean, ct.scale(), plain);
	evaluator.mod_switch_to_inplace(plain, ct.parms_id());
	evaluator.sub_plain_inplace(ct, plain);
	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, 1, ct);
}
void l2norm_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys, double &a, double &b, double &c, Decryptor &decryptor, CKKSEncoder &encoder)
{
	// parameter setting
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = 1, ho = 1, wo = 1, co = ci, to = ti;
	Ciphertext ct = cnn_in.cipher(); 
	Ciphertext temp, sum;
	temp = ct;

	evaluator.multiply_inplace_reduced_error(temp, temp, relin_keys);
	evaluator.rescale_to_next_inplace(temp);

	sum = temp;
	memory_save_rotate(temp, temp, -256, evaluator, gal_keys);
	evaluator.add_inplace_reduced_error(sum, temp);

	for (size_t i = 0; i < log2_long(256); i++)
	{
		temp = sum;
		memory_save_rotate(temp, temp, pow2(i), evaluator, gal_keys);
		evaluator.add_inplace_reduced_error(sum, temp);		
	}

	// l2 norm
	Plaintext plain;
	vector<double> coef(1<<logn, 0.);
	temp = sum;
	evaluator.multiply_inplace_reduced_error(temp, temp, relin_keys);
	evaluator.rescale_to_next_inplace(temp);
	
	fill(coef.begin(), coef.begin()+256, a);
	encoder.encode(coef, temp.scale(), plain);
	evaluator.mod_switch_to_inplace(plain, temp.parms_id());
	evaluator.multiply_plain_inplace(temp, plain);
	evaluator.rescale_to_next_inplace(temp);

	fill(coef.begin(), coef.begin()+256, b);
	encoder.encode(coef, sum.scale(), plain);
	evaluator.mod_switch_to_inplace(plain, sum.parms_id());
	evaluator.multiply_plain_inplace(sum, plain);
	evaluator.rescale_to_next_inplace(sum);

	sum.scale() = temp.scale();
	ct.scale() = temp.scale();
	evaluator.mod_switch_to_inplace(sum, temp.parms_id());
	evaluator.mod_switch_to_inplace(ct, temp.parms_id());

	fill(coef.begin(), coef.begin()+256, c);
	encoder.encode(coef, sum.scale(), plain);
	evaluator.mod_switch_to_inplace(plain, sum.parms_id());
	evaluator.add_inplace_reduced_error(sum, temp);
	evaluator.add_plain_inplace(sum, plain);

	// normalization
	evaluator.multiply_inplace_reduced_error(ct, sum, relin_keys);
	evaluator.rescale_to_next_inplace(ct);

	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, 1, ct);
}
void l2norm_seal(Ciphertext &ct, size_t logn, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys, double &a, double &b, double &c, Decryptor &decryptor, CKKSEncoder &encoder)
{
	Ciphertext temp, sum;
	temp = ct;

	evaluator.multiply_inplace_reduced_error(temp, temp, relin_keys);
	evaluator.rescale_to_next_inplace(temp);

	sum = temp;
	memory_save_rotate(temp, temp, -256, evaluator, gal_keys);
	evaluator.add_inplace_reduced_error(sum, temp);

	for (size_t i = 0; i < log2_long(256); i++)
	{
		temp = sum;
		memory_save_rotate(temp, temp, pow2(i), evaluator, gal_keys);
		evaluator.add_inplace_reduced_error(sum, temp);		
	}

	// l2 norm
	Plaintext plain;
	vector<double> coef(1<<logn, 0.);
	temp = sum;
	evaluator.multiply_inplace_reduced_error(temp, temp, relin_keys);
	evaluator.rescale_to_next_inplace(temp);
	
	fill(coef.begin(), coef.begin()+256, a);
	encoder.encode(coef, temp.scale(), plain);
	evaluator.mod_switch_to_inplace(plain, temp.parms_id());
	evaluator.multiply_plain_inplace(temp, plain);
	evaluator.rescale_to_next_inplace(temp);

	fill(coef.begin(), coef.begin()+256, b);
	encoder.encode(coef, sum.scale(), plain);
	evaluator.mod_switch_to_inplace(plain, sum.parms_id());
	evaluator.multiply_plain_inplace(sum, plain);
	evaluator.rescale_to_next_inplace(sum);

	sum.scale() = temp.scale();
	ct.scale() = temp.scale();
	evaluator.mod_switch_to_inplace(sum, temp.parms_id());
	evaluator.mod_switch_to_inplace(ct, temp.parms_id());

	fill(coef.begin(), coef.begin()+256, c);
	encoder.encode(coef, sum.scale(), plain);
	evaluator.mod_switch_to_inplace(plain, sum.parms_id());
	evaluator.add_inplace_reduced_error(sum, temp);
	evaluator.add_plain_inplace(sum, plain);

	// normalization
	evaluator.multiply_inplace_reduced_error(ct, sum, relin_keys);
	evaluator.rescale_to_next_inplace(ct);
}
void matrix_multiplication_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, vector<double> matrix, vector<double> bias, int q, int r, Evaluator &evaluator, GaloisKeys &gal_keys, CKKSEncoder &encoder)
{
	// parameter setting
	int ki = cnn_in.k(), hi = cnn_in.h(), wi = cnn_in.w(), ci = cnn_in.c(), ti = cnn_in.t(), pi = cnn_in.p(), logn = cnn_in.logn();
	int ko = ki, ho = hi, wo = wi, co = ci, to = ti, po = pi;

	if(static_cast<int>(matrix.size()) != q*r) throw std::invalid_argument("the size of matrix is not q*r");
	if(static_cast<int>(bias.size()) != q) throw std::invalid_argument("the size of bias is not q");

	// generate matrix and bias
	vector<vector<double>> W(q+r-1, vector<double>(1<<logn, 0.0));
	vector<double> b(1<<logn, 0.0);

	for(int z=0; z<q; z++) b[z] = bias[z];
	for(int i=0; i<q; i++)
	{
		for(int j=0; j<r; j++)
		{
			W[i-j+r-1][i] = matrix[i*r+j];
			if(i-j+r-1<0 || i-j+r-1>=q+r-1) throw std::out_of_range("i-j+r-1 is out of range");
			if(i*r+j<0 || i*r+j>=static_cast<int>(matrix.size())) throw std::out_of_range("i*r+j is out of range");
		}
	}

	// matrix multiplication
	Ciphertext ct, temp, sum;
	ct = cnn_in.cipher();
	for(int s=0; s<q+r-1; s++)
	{
		temp = ct;
	//	scale_evaluator.rotate_vector_inplace(temp, r-1-s, gal_keys);
		memory_save_rotate(temp, temp, r-1-s, evaluator, gal_keys);
		evaluator.multiply_vector_inplace_reduced_error(temp, W[s]);
		if(s==0) sum = temp;
		else evaluator.add_inplace_reduced_error(sum, temp);
	}
	evaluator.rescale_to_next_inplace(sum);

	// add bias
	Plaintext plain;
	encoder.encode(b, sum.scale(), plain);
	evaluator.mod_switch_to_inplace(plain, sum.parms_id());
	evaluator.add_plain_inplace(sum, plain);

	cnn_out = TensorCipher(logn, ko, ho, wo, co, to, po, sum);

}
void memory_save_rotate(const Ciphertext &cipher_in, Ciphertext &cipher_out, int steps, Evaluator &evaluator, GaloisKeys &gal_keys)
{
	long n = cipher_in.poly_modulus_degree() / 2;
	Ciphertext temp = cipher_in;
	steps = (steps+n)%n;	// 0 ~ n-1
	int first_step = 0;

	if(34<=steps && steps<=55) first_step = 33;
	else if(57<=steps && steps<=61) first_step = 33;
	else first_step = 0;
	if(steps == 0) return;		// no rotation

	if(first_step == 0) evaluator.rotate_vector_inplace(temp, steps, gal_keys);
	else
	{
		evaluator.rotate_vector_inplace(temp, first_step, gal_keys);
		evaluator.rotate_vector_inplace(temp, steps-first_step, gal_keys);
	}

	cipher_out = temp;
//	else scale_evaluator.rotate_vector(cipher_in, steps, gal_keys, cipher_out);
}

