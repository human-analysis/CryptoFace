#pragma	once
#include "seal/seal.h"
#include "SEALcomp.h"
#include "MinicompFunc.h"
#include "func.h"
#include "PolyUpdate.h"
#include "program.h"
#include "Bootstrapper.h"
#include <omp.h>
#include <NTL/RR.h>
#include <fstream>
#include <vector>
#include <chrono>
#include <istream>

using namespace std;
using namespace seal;
using namespace minicomp;

class TensorCipher
{
private:
	int k_;		// k: gap
	int h_;		// w: height
	int w_;		// w: width
	int c_;		// c: number of channels
	int t_;		// t: \lfloor c/k^2 \rfloor
	int p_;		// p: 2^log2(nt/k^2hwt)
	int logn_;
	Ciphertext cipher_;

public:
	TensorCipher();
	TensorCipher(int logn, int k, int h, int w, int c, int t, int p, vector<double> data, Encryptor &encryptor, CKKSEncoder &encoder, int logp); 	// data vector contains hxwxc real numbers. 
	TensorCipher(int logn, int k, int h, int w, int c, int t, int p, Ciphertext cipher);
	int k() const;
    int h() const;
    int w() const;
	int c() const;
	int t() const;
	int p() const;
    int logn() const;
	Ciphertext cipher() const;
	void set_ciphertext(Ciphertext cipher);
	void print_parms();
};

void multiplexed_parallel_convolution_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, int co, int st, int fh, int fw, const vector<double> &data, vector<double> running_var, vector<double> constant_weight, double epsilon, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, GaloisKeys &gal_keys, vector<Ciphertext> &cipher_pool, ofstream &output, Decryptor &decryptor, SEALContext &context, size_t stage, bool end = false);
void multiplexed_parallel_convolution_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, int co, int st, int fh, int fw, const vector<double> &data, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, GaloisKeys &gal_keys, vector<Ciphertext> &cipher_pool, ofstream &output, Decryptor &decryptor, SEALContext &context, size_t stage, bool end = false);
void multiplexed_parallel_batch_norm_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, vector<double> bias, vector<double> running_mean, vector<double> running_var, vector<double> weight, double epsilon, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, double B, ofstream &output, Decryptor &decryptor, SEALContext &context, size_t stage, bool end = false);
void approx_ReLU_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, long comp_no, vector<int> deg, long alpha, vector<Tree> &tree, double scaled_val, long scalingfactor, Encryptor &encryptor, Evaluator &evaluator, Decryptor &decryptor, CKKSEncoder &encoder, PublicKey &public_key, SecretKey &secret_key, RelinKeys &relin_keys, double B, ofstream &output, SEALContext &context, GaloisKeys &gal_keys, size_t stage);
void bootstrap_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, Bootstrapper &bootstrapper, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context, size_t stage);
void cipher_add_seal_print(const TensorCipher &cnn1, const TensorCipher &cnn2, TensorCipher &destination, Evaluator &evaluator, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context);
void multiplexed_parallel_downsampling_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context, GaloisKeys &gal_keys, ofstream &output);
void averagepooling_seal_scale_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys, double B, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context);
void averagepooling_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context);
void fully_connected_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, vector<double> matrix, vector<double> bias, int q, int r, Evaluator &evaluator, GaloisKeys &gal_keys, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context);
void multiplexed_parallel_convolution_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, int co, int st, int fh, int fw, const vector<double> &data, vector<double> running_var, vector<double> constant_weight, double epsilon, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, GaloisKeys &gal_keys, vector<Ciphertext> &cipher_pool, bool end = false);
void multiplexed_parallel_convolution_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, int co, int st, int fh, int fw, const vector<double> &data, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, GaloisKeys &gal_keys, vector<Ciphertext> &cipher_pool, bool end = false);
void multiplexed_parallel_batch_norm_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, vector<double> bias, vector<double> running_mean, vector<double> running_var, vector<double> weight, double epsilon, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, double B, bool end = false);
void ReLU_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, long comp_no, vector<int> deg, long alpha, vector<Tree> &tree, double scaled_val, long scalingfactor, Encryptor &encryptor, Evaluator &evaluator, Decryptor &decryptor, CKKSEncoder &encoder, PublicKey &public_key, SecretKey &secret_key, RelinKeys &relin_keys, double scale = 1.0);
void cnn_add_seal(const TensorCipher &cnn1, const TensorCipher &cnn2, TensorCipher &destination, Evaluator &evaluator);
void multiplexed_parallel_downsampling_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys);
void multiplexed_parallel_downsampling_seal2(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys);
void multiplexed_parallel_downsampling_seal3(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys, Decryptor &decryptor, CKKSEncoder &encoder);
void averagepooling_seal_scale(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys, double B, CKKSEncoder &encoder, Decryptor &decryptor, ofstream &output);
void averagepooling_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys, CKKSEncoder &encoder, Decryptor &decryptor, ofstream &output);
void matrix_multiplication_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, vector<double> matrix, vector<double> bias, int q, int r, Evaluator &evaluator, GaloisKeys &gal_keys,  CKKSEncoder &encoder);
void memory_save_rotate(const Ciphertext &cipher_in, Ciphertext &cipher_out, int steps, Evaluator &evaluator, GaloisKeys &gal_keys);

void herpn_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, const vector<double> &herpn_w0, const vector<double> &herpn_w1, const vector<double> &herpn_w2, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, RelinKeys &relin_keys, ofstream &output, Decryptor &decryptor, SEALContext &context);
void herpn_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, const vector<double> &herpn_w0, const vector<double> &herpn_w1, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, RelinKeys &relin_keys, ofstream &output, Decryptor &decryptor, SEALContext &context);
void channel_square_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, const vector<double> &weight0, const vector<double> &weight1, const vector<double> &weight2, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, RelinKeys &relin_keys);
void channel_square_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, const vector<double> &weight0, const vector<double> &weight1, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, RelinKeys &relin_keys);
void channel_multiply_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, const vector<double> &weight1, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, RelinKeys &relin_keys);
void channel_multiply_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, const vector<double> &weight, CKKSEncoder &encoder, Encryptor &encryptor, Evaluator &evaluator, RelinKeys &relin_keys, ofstream &output, Decryptor &decryptor, SEALContext &context);

void evorelu_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, string &dir, Encryptor &encryptor, Evaluator &evaluator, Decryptor &decryptor, CKKSEncoder &encoder, PublicKey &public_key, SecretKey &secret_key, RelinKeys &relin_keys, ofstream &output, SEALContext &context, GaloisKeys &gal_keys, double Bin, double Bout, size_t stage);
void evorelu_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, long comp_no, vector<int> deg, vector<vector<double>> &decomp_coeff, vector<Tree> &tree, Encryptor &encryptor, Evaluator &evaluator, Decryptor &decryptor, CKKSEncoder &encoder, PublicKey &public_key, SecretKey &secret_key, RelinKeys &relin_keys, double B);

void batchnorm1d_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, vector<double> &running_mean, vector<double> &running_var, double epsilon, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context);
void batchnorm1d_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, CKKSEncoder &encoder, vector<double> &running_mean, vector<double> &running_var, double epsilon);
void batchnorm1d_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator,  vector<double> &a, vector<double> &weight, vector<double> &bias, vector<double> &running_mean, vector<double> &running_var, double epsilon, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context);
void batchnorm1d_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, CKKSEncoder &encoder, vector<double> &a0, vector<double> &weight0, vector<double> &bias0, vector<double> &running_mean0, vector<double> &running_var0, double epsilon);

void l2norm_seal(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys, double &a, double &b, double &c, Decryptor &decryptor, CKKSEncoder &encoder);
void l2norm_seal_print(const TensorCipher &cnn_in, TensorCipher &cnn_out, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys, double &a, double &b, double &c, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context);
void l2norm_seal(Ciphertext &ct, size_t logn, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys, double &a, double &b, double &c, Decryptor &decryptor, CKKSEncoder &encoder);
void l2norm_seal_print(Ciphertext &ct, size_t logn, Evaluator &evaluator, GaloisKeys &gal_keys, RelinKeys &relin_keys, double &a, double &b, double &c, ofstream &output, Decryptor &decryptor, CKKSEncoder &encoder, SEALContext &context);