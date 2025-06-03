#pragma once
#define L 3			// 層の数
#define ETA (0.004)	// 学習率(0.04)
#define BETA (0.1)	// 学習率(0.1)
#define RELU 1		// 中間層までの活性化関数(0.シグモイド関数, 1.ReLU)

#define NETMODEL 2		// 0:MLP, 1:SkipCNN, 2:SkipSMP, 3:SkipSMP-提案①
#define LEARNRULE 1		// 0:通常 1:cosine decay
#define WARMUP 1		// ウォームアップ(Epoc数)
#define WARMUPRATE 0.5	// ウォームアップの係数
#define RANDMODE 1		// 入力IDの生成方法　0:通常, 1:配列

// ラベルスムージング
#define LS 1		// 0:OFF, 1:ON
#define LSEPS 0.0045// パラメータ

// バッチ学習
#define BATCHLEARN 1// 0.通常学習, 1.バッチ学習
#define BS 100		// バッチサイズ(32)
#define EPS 0.0000001// ε(正則化で使用)
#define MRATE 0.9	 // 移動平均時の慣性

/* ------------------------------ ↓ 学習データに依存(※Mainも変更) ↓ ------------------------------ */
/* Optdigits */
//#define K 10		// 出力数
//#define IND 64	// 入力データの次元
//#define SIND 64	// 入力データの次元
//#define RIND 8	// 入力データの次元の平方根
//#define NMAX 128	// 各層のノードの最大数
//#define ICH 1		// 入力データのチャンネル数
//#define H 8		// 縦幅
//#define W 8		// 横幅

/* MNIST */
//#define K 10	// 出力数
//#define IND 784	// 入力データの次元数
//#define SIND 784// 入力データの次元数
//#define RIND 28	// 入力データの次元の平方根
//#define NMAX 784// 各層のノードの最大数
//#define ICH 1	// 入力データのチャンネル数
//#define H 28	// 縦幅
//#define W 28	// 横幅
//#define IKS 5	// 5×5 No-Padding

/* Cifar-10 */
#define K 10		// 出力数
#define IND 3072	// 入力データの次元数
#define SIND 1024	// 1次元の入力データ
#define RIND 32		// 入力データの次元の平方根
#define NMAX 3072	// 各層のノードの最大数
#define ICH 3		// 入力データのチャンネル数
#define H 32		// 縦幅
#define W 32		// 横幅
#define IKS 1		// 1×1 No-Padding
/* ------------------------------ ↑ 学習データに依存(※Mainも変更) ↑ ------------------------------ */

void Forward(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][NMAX], double mean[][NMAX],
	double var[][NMAX], double gamma[][NMAX], double beta[][NMAX]);
void BForward(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][BS][NMAX], double bnet[][NMAX][BS],
	double mean[][NMAX], double var[][NMAX], double gamma[][NMAX], double beta[][NMAX], double mmean[][NMAX], double mvar[][NMAX]);
void BackProp(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][NMAX], double tk[], double delta[][NMAX], double eta);
void BBackProp(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][BS][NMAX], double tk[][K], double delta[][NMAX][BS],
	double bnet[][NMAX][BS], double mean[][NMAX], double var[][NMAX], double gamma[][NMAX], double beta[][NMAX], double eta);