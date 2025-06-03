#pragma once
// アルゴリズム
#define POINTMODE 0 // 誘導点のモード　0:全て, 1:削除, 2:結合, 3:追加
#define POINTINIT 0 // 誘導点の初期化モード　0:ランダム, 1:正方形初期化, 2:誘導点数の変更(9,6,6 入力層のみ正方形初期化)
#define SINGROWTH 0	// Sine Growth  0:OFF, 1:ON

// パラメータ
#define SD 2		// 次元
#define NPMIN 1		// 誘導点数の最小値
#define NPMAX 32	// 誘導点数の最大値
#define NPINIT 9	// 誘導点数の初期値
#define NPINITRAND 6// 誘導点数の初期値（チャンネルごとに誘導点数を変更する場合）
#define MU 0.5		// 平均      Box-Muller
#define SIGMA 0.2	// 分散(0.1) Box-Muller
#define R 0.12		// 半径(0.12)
#define LOWESTR 0.0001	// 半径の最低値(クリッピング) 
#define HIGHESTR 1		// 半径の最高値(クリッピング)

// 提案手法実行範囲(Epoc)
#define SOIPSTART 2	// 増減開始タイミング(Epoc)　※EPOC範囲内
#define SOIPEND 4	// 増減終了タイミング(Epoc)　※EPOC範囲内

// 提案手法実行周期
#define SOIPTIME 50	// 誘導点の増減タイミング(逐次学習)　 ※SOIPTIME回に1回実行する
#define BSOIPTIME 5	// 誘導点の増減タイミング(バッチ学習) ※BSOIPTIME回に1回実行する

// 誘導点の削除
#define DELCOUNT 1	// 誘導点削除範囲に含まれた回数(1:即削除)
#define DELW 0.025	// 点を削除する閾値(重み) 

// 誘導点の結合
#define MERDIFF (R/2.0)	// 半径閾値(半径の差分距離)(0.05)
#define MERRATE 0.8		// 距離倍率(半径の倍率変動)(0.1)
#define MERSIMI 0.2		// 重み閾値(重みのコサイン類似度)(0.5)

// 誘導点の増加
#define ADDRATEBASIS 0.001	// 追加閾値の最低ライン(0.00001)
#define ADDRATERANGE 0.03	// 追加閾値の幅(閾値の範囲：ADDRATEBASIS～ADDRATEBASIS+ADDRATERANGE)
#define ADDDEL (-4)			// 削除猶予

void SMPSkipForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W],
	double mout[], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][H][W], double skipBias[SEG][CHMAX],
	double skipWoi[SEG][CHMAX][CHMAX][KS][KS], double pi[SEG][CONV][CHMAX][NPMAX][SD], double ri[SEG][CONV][CHMAX][NPMAX], double wi[SEG][CONV][CHMAX][CHMAX][NPMAX],
	int np[SEG][CONV][CHMAX], double cie[SEG][H][W][SD], double dis[SEG][CONV + 1][H][W][CHMAX][NPMAX], double g[SEG][CONV + 1][H][W][CHMAX][NPMAX],
	int sumCount[SEG][CONV + 1][H][W][CHMAX]);

void SMPSkipBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W],
	double delta[][CONV + 1][CHMAX][H][W], double mdelta[], double dBias[][CONV][CHMAX], double dWoi[][CONV][CHMAX][CHMAX][KS][KS], double eta,
	double in[ICH][H][W], double inBias[], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][H][W], double skipDelta[SEG][CHMAX][H][W],
	double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS], double dSkipBias[SEG][CHMAX], double dSkipWoi[SEG][CHMAX][CHMAX][KS][KS],
	double dInBias[CHMAX], double dInWoi[CHMAX][ICH][IKS][IKS], double pi[SEG][CONV][CHMAX][NPMAX][SD], double ri[SEG][CONV][CHMAX][NPMAX],
	double wi[SEG][CONV][CHMAX][CHMAX][NPMAX], int np[SEG][CONV][CHMAX], double cie[SEG][H][W][SD], double dis[SEG][CONV + 1][H][W][CHMAX][NPMAX],
	double g[SEG][CONV + 1][H][W][CHMAX][NPMAX], int sumCount[SEG][CONV + 1][H][W][CHMAX], double dPi[SEG][CONV][CHMAX][NPMAX][SD],
	double dRi[SEG][CONV][CHMAX][NPMAX], double dWi[SEG][CONV][CHMAX][CHMAX][NPMAX], int wUseCount[SEG][CONV][CHMAX][CHMAX][NPMAX],
	int prUseCount[SEG][CONV][CHMAX][NPMAX], double dPsum[SEG][CONV][CHMAX][NPMAX][SD], double dRsum[SEG][CONV][CHMAX][NPMAX]);

void BSMPSkipForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W],
	double mout[][NMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][BS][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][BS][H][W], double skipBias[SEG][CHMAX],
	double skipWoi[SEG][CHMAX][CHMAX][KS][KS], double pi[SEG][CONV][CHMAX][NPMAX][SD], double ri[SEG][CONV][CHMAX][NPMAX], double wi[SEG][CONV][CHMAX][CHMAX][NPMAX],
	int np[SEG][CONV][CHMAX], double cie[SEG][H][W][SD], double dis[SEG][CONV + 1][H][W][CHMAX][NPMAX], double g[SEG][CONV + 1][H][W][CHMAX][NPMAX],
	int sumCount[SEG][CONV + 1][H][W][CHMAX], double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX]);

void BSMPSkipBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W],
	double delta[][CONV + 1][CHMAX][BS][H][W], double mdelta[][BS], double dBias[][CONV][CHMAX], double dWoi[][CONV][CHMAX][CHMAX][KS][KS], double eta,
	double in[ICH][BS][H][W], double inBias[], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][BS][H][W], double skipDelta[SEG][CHMAX][BS][H][W],
	double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS], double dSkipBias[SEG][CHMAX], double dSkipWoi[SEG][CHMAX][CHMAX][KS][KS],
	double dInBias[CHMAX], double dInWoi[CHMAX][ICH][IKS][IKS], double pi[SEG][CONV][CHMAX][NPMAX][SD], double ri[SEG][CONV][CHMAX][NPMAX],
	double wi[SEG][CONV][CHMAX][CHMAX][NPMAX], int np[SEG][CONV][CHMAX], double cie[SEG][H][W][SD], double dis[SEG][CONV + 1][H][W][CHMAX][NPMAX],
	double g[SEG][CONV + 1][H][W][CHMAX][NPMAX], int sumCount[SEG][CONV + 1][H][W][CHMAX], double dPi[SEG][CONV][CHMAX][NPMAX][SD],
	double dRi[SEG][CONV][CHMAX][NPMAX], double dWi[SEG][CONV][CHMAX][CHMAX][NPMAX], int wUseCount[SEG][CONV][CHMAX][CHMAX][NPMAX],
	int prUseCount[SEG][CONV][CHMAX][NPMAX], double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX],
	double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double dPsum[SEG][CONV][CHMAX][NPMAX][SD],
	double dRsum[SEG][CONV][CHMAX][NPMAX]);

void SOIPInit(double dPsum[SEG][CONV][CHMAX][NPMAX][SD], double dRsum[SEG][CONV][CHMAX][NPMAX]);

void SOIP(int chN[SEG][CONV + 1], double pi[SEG][CONV][CHMAX][NPMAX][SD], double ri[SEG][CONV][CHMAX][NPMAX],
	double wi[SEG][CONV][CHMAX][CHMAX][NPMAX], int np[SEG][CONV][CHMAX], int* allNP, int delC[SEG][CONV][CHMAX][NPMAX],
	double dPsum[SEG][CONV][CHMAX][NPMAX][SD], double dRsum[SEG][CONV][CHMAX][NPMAX], bool add[SEG][CONV][CHMAX], double delW, double eta,
	int* delN, int* merN, int* addN);

