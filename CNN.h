#pragma once
#define SEG 3	// セグメント数
#define CONV 2	// 畳み込み数
#define CHMAX 32 // チャンネルの最大値()

#define KS 3	// CNNのカーネルサイズ
#define PS 2	// Pooling Size (2*2)

// 正則化
void CBatchNormalF(double bnet[][H][W], double anet[][H][W], double* mean, double* var, double gamma, double beta, int hsize, int wsize);
void CBatchNormalB(double delta[][H][W], double bnet[][H][W], double mean, double var, double* gamma, double* beta, double eta, int hsize, int wsize);

// Forward
void InConv(int ich, int och, double bias[], double woi[][ICH][IKS][IKS], double in[][H][W], double out[][H][W], int ksize, double gamma[], double beta[], 
	double mmean[], double mvar[]);
void BInConv(int ich, int och, double bias[], double woi[][ICH][IKS][IKS], double in[][BS][H][W], double out[][BS][H][W], int ksize,
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double mmean[], double mvar[]);
void Conv(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double in[][H][W], double out[][H][W], int hsize, int wsize, int st, int ksize,
	double gamma[], double beta[], double mmean[], double mvar[]);
void BConv(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double in[][BS][H][W], double out[][BS][H][W],
	int hsize, int wsize, int st, int ksize, double bnet[][BS][H][W], double mean[], double var[], double gamma[],
	double beta[], double mmean[], double mvar[]);
void SkipForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W],
	double mout[], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][H][W],
	double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS]);
void BSkipForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W],
	double mout[][NMAX], double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX],
	double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][BS][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][BS][H][W],
	double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS]);

// BackProp
void InBackConv(int ich, int och, double delta[][H][W], int hsize, int wsize,
	int st, int ksize, double in[][H][W], double dBias[], double dWoi[][ICH][IKS][IKS], double out[][H][W]);
void BInBackConv(int ich, int och, double delta[][BS][H][W], int hsize, int wsize,
	int st, int ksize, double in[][BS][H][W], double dBias[], double dWoi[][ICH][IKS][IKS], double out[][BS][H][W],
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double eta);
void BackConv(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double indelta[][H][W], double delta[][H][W], int hsize, int wsize,
	int st, int ksize, double in[][H][W], double dBias[], double dWoi[][CHMAX][KS][KS], double out[][H][W]);
void BBackConv(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double indelta[][BS][H][W], double delta[][BS][H][W], int hsize, int wsize,
	int st, int ksize, double in[][BS][H][W], double out[][BS][H][W], double dBias[], double dWoi[][CHMAX][KS][KS], double bnet[][BS][H][W], double mean[], double var[], double gamma[],
	double beta[], double eta);
void SkipBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W],
	double delta[][CONV + 1][CHMAX][H][W], double mdelta[], double dBias[][CONV][CHMAX], double dWoi[][CONV][CHMAX][CHMAX][KS][KS], double eta,
	double in[ICH][H][W], double inBias[], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][H][W], double skipDelta[SEG][CHMAX][H][W],
	double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS], double dSkipBias[SEG][CHMAX], double dSkipWoi[SEG][CHMAX][CHMAX][KS][KS],
	double dInBias[CHMAX], double dInWoi[CHMAX][ICH][IKS][IKS]);
void BSkipBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W],
	double delta[][CONV + 1][CHMAX][BS][H][W], double mdelta[][BS], double dBias[][CONV][CHMAX], double dWoi[][CONV][CHMAX][CHMAX][KS][KS], double eta,
	double in[ICH][BS][H][W], double inBias[], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][BS][H][W], double skipDelta[SEG][CHMAX][BS][H][W],
	double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX],
	double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS], double dSkipBias[SEG][CHMAX], double dSkipWoi[SEG][CHMAX][CHMAX][KS][KS],
	double dInBias[CHMAX], double dInWoi[CHMAX][ICH][IKS][IKS]);