#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "MLP.h"
#include "CNN.h"
#include "SMP.h"

// モード切替
int SMPBatchNormal = 0;// バッチ正則化　0:off, 1:on ※CNNもONに

/* -------------------- ↓ Forward ↓ -------------------- */
// SMPConv
void SMPConv2D(int ich, int och, double in[CHMAX][H][W], double out[CHMAX][H][W], double pi[CHMAX][NPMAX][SD], double ri[CHMAX][NPMAX],
	double wi[CHMAX][CHMAX][NPMAX], int np[CHMAX], double cie[H][W][SD], double dis[H][W][CHMAX][NPMAX], double g[H][W][CHMAX][NPMAX],
	int sumCount[H][W][CHMAX], int hsize, int wsize, double gamma[], double beta[], double mmean[], double mvar[]) {
	int i, j, h, w, m, d;
	double temp, sum;
	double hd, wd;
	// 距離関数計算
	for (h = 0; h < hsize; h++) {
		for (w = 0; w < wsize; w++) {
			for (i = 0; i < ich; i++) {
				for (sumCount[h][w][i] = 0, m = 0; m < np[i]; m++) {
					// L1距離
					for (dis[h][w][i][m] = 0, d = 0; d < SD; d++) {// 次元数分
						temp = cie[h][w][d] - pi[i][m][d];	// 距離計算(L1距離)
						if (temp < 0)temp = -temp;			// 符号反転(負→正)
						dis[h][w][i][m] += temp;			// 距離の合計(次元数分)
					}
					// 距離関数:g(x,p,r)
					g[h][w][i][m] = 1.0 - dis[h][w][i][m] / ri[i][m];	// 距離関数:g(x,p,r)
					if (g[h][w][i][m] <= 0)g[h][w][i][m] = 0;			// 範囲外なら0
					else sumCount[h][w][i]++;							// 影響点の数をカウント
				}
			}
		}
	}

	// 出力計算
	for (j = 0; j < och; j++) {// 出力チャンネル
		for (h = 0; h < hsize; h++) {
			for (w = 0; w < wsize; w++) {
				for (out[j][h][w] = 0, i = 0; i < ich; i++) {// 入力チャンネル
					// 距離関数×重み
					if (sumCount[h][w][i] > 0) {
						for (sum = 0, m = 0; m < np[i]; m++) {
							sum += g[h][w][i][m] * wi[j][i][m];// 距離関数×重み
						}
						out[j][h][w] += in[i][h][w] * (sum / sumCount[h][w][i]);// 出力計算(合成積)
					}
				}
			}
		}
	}

	// バッチ正規化
	if (BATCHLEARN && SMPBatchNormal) {
		for (j = 0; j < och; j++)// 出力ノード数
			for (h = 0; h < hsize; h++)// 画像サイズ(高さ)
				for (w = 0; w < wsize; w++)// 画像サイズ(横幅)
					out[j][h][w] = (out[j][h][w] - mmean[j]) / sqrt(mvar[j] + EPS) * gamma[j] + beta[j];
	}

	// ReLU
	for (j = 0; j < och; j++) {
		for (h = 0; h < hsize; h++) {
			for (w = 0; w < wsize; w++) {
				if (out[j][h][w] < 0)out[j][h][w] = 0;
			}
		}
	}
}

// SMPForward
void SMPSkipForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W],
	double mout[], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][H][W], double skipBias[SEG][CHMAX],
	double skipWoi[SEG][CHMAX][CHMAX][KS][KS], double pi[SEG][CONV][CHMAX][NPMAX][SD], double ri[SEG][CONV][CHMAX][NPMAX], double wi[SEG][CONV][CHMAX][CHMAX][NPMAX],
	int np[SEG][CONV][CHMAX], double cie[SEG][H][W][SD], double dis[SEG][CONV + 1][H][W][CHMAX][NPMAX], double g[SEG][CONV + 1][H][W][CHMAX][NPMAX],
	int sumCount[SEG][CONV + 1][H][W][CHMAX]) {
	int i, j, s, c, w, h, n, fn, hsize = H, wsize = W, st = 1;// st:ストライドサイズ
	double sum, max;
	//Input
	InConv(ICH, chN[0][0], inBias, inWoi, in, out[0][0], IKS, gamma[0][0], beta[0][0], mmean[0][0], mvar[0][0]);
	hsize = H - IKS + 1; wsize = W - IKS + 1;// padding アリの場合

	// 順伝播
	for (s = 0; s < SEG; s++) {// segment
		// 入力座標作成
		for (h = 0; h < hsize; h++) {
			for (w = 0; w < wsize; w++) {
				cie[s][h][w][0] = (1.0 / hsize) * h + (1.0 / hsize) / 2.0; // 縦座標
				cie[s][h][w][1] = (1.0 / wsize) * w + (1.0 / wsize) / 2.0; // 横座標
			}
		}

		// 畳み込み+ReLU(1セグメント分)
		for (c = 0; c < CONV; c++) {
			SMPConv2D(chN[s][c], chN[s][c + 1], out[s][c], out[s][c + 1], pi[s][c], ri[s][c], wi[s][c], np[s][c], cie[s], dis[s][c], g[s][c], sumCount[s][c],
				hsize, wsize, gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1]);
		}

		// Skip-Connection
		for (n = 0; n < chN[s][c]; n++) {
			for (h = 0; h < hsize; h++)
				for (w = 0; w < wsize; w++)
					skipOut[s][n][h][w] = out[s][c][n][h][w] + out[s][0][n][h][w];// Skip-Connection
		}

		// Max-Pooling(ストライドPSの畳み込み)
		if (s == SEG - 1) {// Max-Pooling
			for (fn = 0, n = 0; n < chN[s][c]; n++) {
				for (h = 0; h < hsize; h += PS) {
					for (w = 0; w < wsize; w += PS) {
						// max計算
						for (max = 0, i = 0; i < PS; i++)
							for (j = 0; j < PS; j++)
								if (max < skipOut[s][n][h + i][w + j])
									max = skipOut[s][n][h + i][w + j];
						// 全結合の入力
						mout[fn] = max;
						fn++;
					}
				}
			}
		}
		else {// ストライドPSの畳み込み
			Conv(chN[s][c], chN[s + 1][0], skipBias[s], skipWoi[s], skipOut[s], out[s + 1][0], hsize, wsize, PS, KS,
				gamma[s + 1][0], beta[s + 1][0], mmean[s + 1][0], mvar[s + 1][0]);

			// Skip-Connection
			for (n = 0; n < chN[s][c]; n++) {
				for (h = 0; h < hsize; h += PS) {
					for (w = 0; w < wsize; w += PS) {
						// 平均
						for (sum = 0, j = 0; j < PS; j++)for (i = 0; i < PS; i++)
							sum += skipOut[s][n][h + j][w + i];
						// Skip-Connection
						out[s + 1][0][n][h / PS][w / PS] += sum / (PS * PS);
					}
				}
			}
		}
		hsize /= PS;// 画像サイズ
		wsize /= PS;// 画像サイズ
	}
}

// BatchSMPConv
void BSMPConv2D(int ich, int och, double in[CHMAX][BS][H][W], double out[CHMAX][BS][H][W], double pi[CHMAX][NPMAX][SD],
	double ri[CHMAX][NPMAX], double wi[CHMAX][CHMAX][NPMAX], int np[CHMAX], double cie[H][W][SD], double dis[H][W][CHMAX][NPMAX],
	double g[H][W][CHMAX][NPMAX], int sumCount[H][W][CHMAX], int hsize, int wsize, double bnet[][BS][H][W],
	double mean[], double var[], double gamma[], double beta[], double mmean[], double mvar[]) {
	int i, j, h, w, m, d, b;
	double temp, sum;

	// 距離関数計算
	for (h = 0; h < hsize; h++) {
		for (w = 0; w < wsize; w++) {
			for (i = 0; i < ich; i++) {
				for (sumCount[h][w][i] = 0, m = 0; m < np[i]; m++) {
					// L1距離
					for (dis[h][w][i][m] = 0, d = 0; d < SD; d++) {// 次元数分
						temp = cie[h][w][d] - pi[i][m][d];	// 距離計算(L1距離)
						if (temp < 0)temp = -temp;			// 符号反転(負→正)
						dis[h][w][i][m] += temp;			// 距離の合計(次元数分)
					}
					// 距離関数:g(x,p,r)
					g[h][w][i][m] = 1.0 - dis[h][w][i][m] / ri[i][m];	// 距離関数:g(x,p,r)
					if (g[h][w][i][m] <= 0)g[h][w][i][m] = 0;			// 範囲外なら0
					else sumCount[h][w][i]++;							// 影響点の数をカウント
				}
			}
		}
	}

	// 出力計算
	for (j = 0; j < och; j++) {// 出力チャンネル
		for (b = 0; b < BS; b++) {// バッチ数
			for (h = 0; h < hsize; h++) {// 画像サイズ(高さ)
				for (w = 0; w < wsize; w++) {// 画像サイズ(横幅)
					for (bnet[j][b][h][w] = 0, i = 0; i < ich; i++) {// 入力チャンネル
						// 距離関数×重み
						if (sumCount[h][w][i] > 0) {
							for (sum = 0, m = 0; m < np[i]; m++) {
								sum += g[h][w][i][m] * wi[j][i][m];// 距離関数×重み
							}
							bnet[j][b][h][w] += in[i][b][h][w] * (sum / sumCount[h][w][i]);// 出力計算(合成積)
						}
					}
				}
			}
		}
	}

	// バッチ正規化
	if (SMPBatchNormal) {
		if (RELU) {// ReLU
			for (j = 0; j < och; j++) {// 出力ノード数
				// バッチ正規化
				CBatchNormalF(bnet[j], out[j], &mean[j], &var[j], gamma[j], beta[j], hsize, wsize);
				// ReLU
				for (b = 0; b < BS; b++) // バッチ数
					for (h = 0; h < hsize; h++)// 画像サイズ(高さ)
						for (w = 0; w < wsize; w++)// 画像サイズ(横幅)
							if (out[j][b][h][w] <= 0)out[j][b][h][w] = 0;
				// 移動平均・移動分散
				mmean[j] = mmean[j] * MERRATE + (1.0 - MERRATE) * mean[j];	// 移動平均
				mvar[j] = mvar[j] * MERRATE + (1.0 - MERRATE) * var[j];		// 移動分散
			}
		}
	}
	else {
		// ReLU
		for (j = 0; j < och; j++) {// 出力ノード数
			for (b = 0; b < BS; b++)// バッチ数
				for (h = 0; h < hsize; h++)// 画像サイズ(高さ)
					for (w = 0; w < wsize; w++) {// 画像サイズ(横幅)
						if (bnet[j][b][h][w] <= 0)out[j][b][h][w] = 0;
						else out[j][b][h][w] = bnet[j][b][h][w];
					}
		}
	}
}

// BatchSMPForward
void BSMPSkipForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W],
	double mout[][NMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][BS][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][BS][H][W], double skipBias[SEG][CHMAX],
	double skipWoi[SEG][CHMAX][CHMAX][KS][KS], double pi[SEG][CONV][CHMAX][NPMAX][SD], double ri[SEG][CONV][CHMAX][NPMAX], double wi[SEG][CONV][CHMAX][CHMAX][NPMAX],
	int np[SEG][CONV][CHMAX], double cie[SEG][H][W][SD], double dis[SEG][CONV + 1][H][W][CHMAX][NPMAX], double g[SEG][CONV + 1][H][W][CHMAX][NPMAX],
	int sumCount[SEG][CONV + 1][H][W][CHMAX], double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX]) {
	int i, j, s, c, w, h, n, b, fn, hsize = H, wsize = W, st = 1;// st:ストライドサイズ
	double sum, max;
	//BatchInput
	BInConv(ICH, chN[0][0], inBias, inWoi, in, out[0][0], IKS, bnet[0][0], mean[0][0], var[0][0], gamma[0][0], beta[0][0], mean[0][0], mvar[0][0]);
	hsize = H - IKS + 1; wsize = W - IKS + 1;// padding アリの場合

	// 順伝播
	for (s = 0; s < SEG; s++) {// segment
		// 入力座標作成
		for (h = 0; h < hsize; h++) {
			for (w = 0; w < wsize; w++) {
				cie[s][h][w][0] = (1.0 / hsize) * h + (1.0 / hsize) / 2.0; // 縦座標
				cie[s][h][w][1] = (1.0 / wsize) * w + (1.0 / wsize) / 2.0; // 横座標
			}
		}

		// 畳み込み(BSMPConv)
		for (c = 0; c < CONV; c++) {
			BSMPConv2D(chN[s][c], chN[s][c + 1], out[s][c], out[s][c + 1], pi[s][c], ri[s][c], wi[s][c], np[s][c], cie[s], dis[s][c], g[s][c], sumCount[s][c],
				hsize, wsize, bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1]);
		}

		// Skip-Connection
		for (n = 0; n < chN[s][c]; n++) {
			for (b = 0; b < BS; b++) {
				for (h = 0; h < hsize; h++) {
					for (w = 0; w < wsize; w++) {
						skipOut[s][n][b][h][w] = out[s][c][n][b][h][w] + out[s][0][n][b][h][w];// Skip-Connection
					}
				}
			}
		}

		// Max-Pooling(ストライドPSの畳み込み)
		if (s == SEG - 1) {// out[s][c] -> mout 全結合に入れる 
			for (b = 0; b < BS; b++) {
				for (fn = 0, n = 0; n < chN[s][c]; n++) {
					for (h = 0; h < hsize; h += PS) {
						for (w = 0; w < wsize; w += PS) {
							// max計算
							for (max = 0, i = 0; i < PS; i++)
								for (j = 0; j < PS; j++)
									if (max < skipOut[s][n][b][h + i][w + j])
										max = skipOut[s][n][b][h + i][w + j];
							// 全結合の入力
							mout[b][fn] = max;
							fn++;
						}
					}
				}
			}
		}
		else {// ストライドPSの畳み込み
			BConv(chN[s][c], chN[s + 1][0], skipBias[s], skipWoi[s], skipOut[s], out[s + 1][0], hsize, wsize, PS, KS,
				bnet[s + 1][0], mean[s + 1][0], var[s + 1][0], gamma[s + 1][0], beta[s + 1][0], mmean[s + 1][0], mvar[s + 1][0]);
			// Skip-Connection
			for (n = 0; n < chN[s][c]; n++) {
				for (b = 0; b < BS; b++) {
					for (h = 0; h < hsize; h += PS) {
						for (w = 0; w < wsize; w += PS) {
							// 平均
							for (sum = 0, j = 0; j < PS; j++)for (i = 0; i < PS; i++)
								sum += skipOut[s][n][b][h + j][w + i];
							// Skip-Connection
							out[s + 1][0][n][b][h / PS][w / PS] += sum / (PS * PS);
						}
					}
				}
			}
		}
		hsize /= PS;// 画像サイズ
		wsize /= PS;// 画像サイズ
	}
}


/* -------------------- ↓ BackProp ↓ -------------------- */
// SMPBackConv
void SMPBackConv2D(int ich, int och, double in[CHMAX][H][W], double out[CHMAX][H][W], double pi[CHMAX][NPMAX][SD], double ri[CHMAX][NPMAX],
	double wi[CHMAX][CHMAX][NPMAX], int np[CHMAX], double inDelta[CHMAX][H][W], double outDelta[CHMAX][H][W],
	double cie[H][W][SD], double dis[H][W][CHMAX][NPMAX], double g[H][W][CHMAX][NPMAX], int sumCount[H][W][CHMAX],
	int hsize, int wsize, double dPi[CHMAX][NPMAX][SD], double dRi[CHMAX][NPMAX], double dWi[CHMAX][CHMAX][NPMAX],
	int wUseCount[CHMAX][CHMAX][NPMAX], int prUseCount[CHMAX][NPMAX]) {
	int i, j, k, x, m, d, h, w;
	double sum;

	// delta初期化
	for (i = 0; i < ich; i++) {// 入力チャンネル
		for (h = 0; h < hsize; h++) {// 縦サイズ
			for (w = 0; w < wsize; w++) {// 横サイズ
				inDelta[i][h][w] = 0;
			}
		}
	}

	// ReLU
	for (j = 0; j < och; j++) {
		for (h = 0; h < hsize; h++) {
			for (w = 0; w < wsize; w++) {
				if (out[j][h][w] <= 0)outDelta[j][h][w] = 0;
			}
		}
	}

	// 誤差逆伝搬
	for (j = 0; j < och; j++) {// 出力チャンネル
		for (h = 0; h < hsize; h++) {// 縦サイズ
			for (w = 0; w < wsize; w++) {// 横サイズ
				if (outDelta[j][h][w] != 0) {
					for (i = 0; i < ich; i++) {// 入力チャンネル
						if (sumCount[h][w][i] > 0) {// 近傍誘導点の数が存在する
							for (sum = 0, m = 0; m < np[i]; m++) {// 誘導点数
								if (g[h][w][i][m] > 0) {// 距離関数=0は近傍範囲外のため使用しない
									sum += g[h][w][i][m] * wi[j][i][m];
									// 座標の変化量・半径の変化量
									for (d = 0; d < SD; d++) {
										if (cie[h][w][d] - pi[i][m][d] >= 0) {
											dPi[i][m][d] += outDelta[j][h][w] * in[i][h][w] * (wi[j][i][m] / sumCount[h][w][i]) * (1.0 / ri[i][m]);// 座標
											//dRi[i][m] += outDelta[j][h][w] * in[i][h][w] * (wi[j][i][m] / sumCount[h][w][i]) * (dis[h][w][i][m] / (ri[i][m] * ri[i][m]));// 半径
										}
										else {
											dPi[i][m][d] -= outDelta[j][h][w] * in[i][h][w] * (wi[j][i][m] / sumCount[h][w][i]) * (1.0 / ri[i][m]);// 座標
											//dRi[i][m] -= outDelta[j][h][w] * in[i][h][w] * (wi[j][i][m] / sumCount[h][w][i]) * (dis[h][w][i][m] / (ri[i][m] * ri[i][m]));// 半径
										}
									}
									dRi[i][m] += outDelta[j][h][w] * in[i][h][w] * (wi[j][i][m] / sumCount[h][w][i]) * (dis[h][w][i][m] / (ri[i][m] * ri[i][m]));// 半径
									// 重みの変化量
									dWi[j][i][m] += outDelta[j][h][w] * in[i][h][w] * g[h][w][i][m] / sumCount[h][w][i];
									// 誘導点の使用回数増加
									wUseCount[j][i][m]++;
									prUseCount[i][m]++;
								}
							}
							// delta計算
							inDelta[i][h][w] += outDelta[j][h][w] * sum / sumCount[h][w][i];
						}
					}
				}
			}
		}
	}
}

// SMPBackProp
void SMPSkipBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W],
	double delta[][CONV + 1][CHMAX][H][W], double mdelta[], double dBias[][CONV][CHMAX], double dWoi[][CONV][CHMAX][CHMAX][KS][KS], double eta,
	double in[ICH][H][W], double inBias[], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][H][W], double skipDelta[SEG][CHMAX][H][W],
	double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS], double dSkipBias[SEG][CHMAX], double dSkipWoi[SEG][CHMAX][CHMAX][KS][KS],
	double dInBias[CHMAX], double dInWoi[CHMAX][ICH][IKS][IKS], double pi[SEG][CONV][CHMAX][NPMAX][SD], double ri[SEG][CONV][CHMAX][NPMAX],
	double wi[SEG][CONV][CHMAX][CHMAX][NPMAX], int np[SEG][CONV][CHMAX], double cie[SEG][H][W][SD], double dis[SEG][CONV + 1][H][W][CHMAX][NPMAX],
	double g[SEG][CONV + 1][H][W][CHMAX][NPMAX], int sumCount[SEG][CONV + 1][H][W][CHMAX], double dPi[SEG][CONV][CHMAX][NPMAX][SD],
	double dRi[SEG][CONV][CHMAX][NPMAX], double dWi[SEG][CONV][CHMAX][CHMAX][NPMAX], int wUseCount[SEG][CONV][CHMAX][CHMAX][NPMAX],
	int prUseCount[SEG][CONV][CHMAX][NPMAX], double dPsum[SEG][CONV][CHMAX][NPMAX][SD], double dRsum[SEG][CONV][CHMAX][NPMAX]) {
	int i, j, s, c, hsize = H / pow(PS, SEG), wsize = W / pow(PS, SEG), st = 1, iN, oN, h, w, nh, nw, och, n, m, d;
	double sum, max;
	int x, y;

	// 重み修正量を0初期化
	hsize = (H - IKS + 1) / pow(PS, SEG);
	wsize = (W - IKS + 1) / pow(PS, SEG);

	/* ---------- CNN初期化 ---------- */
	// 重み・バイアスの変化量
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++) {
		for (j = 0; j < CHMAX; j++)
			for (dBias[s][c][j] = 0, i = 0; i < CHMAX; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					dWoi[s][c][j][i][y][x] = 0; //初期化
	}
	// スキップ接続の重み・バイアスの変化量
	for (c = CONV, s = 0; s < SEG; s++) {
		for (j = 0; j < CHMAX; j++)for (dSkipBias[s][j] = 0, i = 0; i < CHMAX; i++)
			for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
				dSkipWoi[s][j][i][y][x] = 0; //初期化
	}
	// 入力畳み込みの重み・バイアスの変化量
	for (j = 0; j < CHMAX; j++) {
		for (dInBias[j] = 0, i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				dInWoi[j][i][y][x] = 0; //初期化
	}

	/* ---------- SMP初期化 ---------- */
	// 座標・半径・使用回数初期化
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++) {
		for (i = 0; i < CHMAX; i++) {
			for (m = 0; m < NPMAX; m++) {
				for (d = 0; d < SD; d++) {
					dPi[s][c][i][m][d] = 0;
				}
				dRi[s][c][i][m] = 0;
				prUseCount[s][c][i][m] = 0;
			}
		}
	}
	// 重み・使用回数初期化
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++) {
		for (j = 0; j < CHMAX; j++) {
			for (i = 0; i < CHMAX; i++) {
				for (m = 0; m < NPMAX; m++) {
					dWi[s][c][j][i][m] = 0;
					wUseCount[s][c][j][i][m] = 0;
				}
			}
		}
	}

	/* ---------- SMPBackProp ---------- */
	for (s = SEG - 1; s >= 0; s--) {// segment
		// サイズ戻し
		hsize *= PS;// 画像サイズ
		wsize *= PS;// 画像サイズ
		c = CONV;
		// Max Pooling
		if (s == SEG - 1) {// 全結合のデルタを入れる delta[s][CONV]=mdelta[]
			for (oN = 0, iN = 0; iN < chN[s][c]; iN++) {
				for (h = 0; h < hsize; h += PS) {
					for (w = 0; w < wsize; w += PS, oN++) {
						// max計算
						for (nh = h, nw = w, max = skipOut[s][iN][h][w], i = 0; i < PS; i++) {
							for (j = 0; j < PS; j++) {
								skipDelta[s][iN][h + i][w + j] = 0;
								if (max < skipOut[s][iN][h + i][w + j]) {
									max = skipOut[s][iN][h + i][w + j];
									nh = h + i;
									nw = w + j;
								}
							}
						}
						// delta代入
						skipDelta[s][iN][nh][nw] = mdelta[oN];
					}
				}
			}
		}
		else {// out[s][c] -> out[s+1][0] 次のセグメントの0層に入れる
			BackConv(chN[s][c], chN[s + 1][0], skipBias[s], skipWoi[s], skipDelta[s], delta[s + 1][0],
				hsize, wsize, PS, KS, skipOut[s], dSkipBias[s], dSkipWoi[s], out[s + 1][0]);
			// Skip-Connection
			for (n = 0; n < chN[s][c]; n++) {
				for (h = 0; h < hsize; h += PS) {
					for (w = 0; w < wsize; w += PS) {
						// Skip-Connection
						for (sum = 0, j = 0; j < PS; j++)for (i = 0; i < PS; i++)
							skipDelta[s][n][h + j][w + i] += delta[s + 1][0][n][h / PS][w / PS] / (PS * PS);
					}
				}
			}
		}

		// デルタ代入
		for (n = 0; n < chN[s][c]; n++) {
			for (h = 0; h < hsize; h++)
				for (w = 0; w < wsize; w++)
					delta[s][c][n][h][w] = skipDelta[s][n][h][w];
		}

		// 畳み込み
		for (c = CONV - 1; c >= 0; c--) {
			SMPBackConv2D(chN[s][c], chN[s][c + 1], out[s][c], out[s][c + 1], pi[s][c], ri[s][c], wi[s][c], np[s][c], delta[s][c], delta[s][c + 1], cie[s], dis[s][c], g[s][c], sumCount[s][c], hsize, wsize,
				dPi[s][c], dRi[s][c], dWi[s][c], wUseCount[s][c], prUseCount[s][c]);
		}

		// スキップ接続
		for (n = 0; n < chN[s][0]; n++) {
			for (h = 0; h < hsize; h++)
				for (w = 0; w < wsize; w++)
					delta[s][0][n][h][w] += skipDelta[s][n][h][w];
		}
	}
	// InBackConv
	InBackConv(ICH, chN[0][0], delta[0][0], H, W, st, IKS, in, dInBias, dInWoi, out[0][0]);

	/* ---------- 更新 ---------- */
	// SkipConv重み・バイアス
	for (hsize = H - IKS + 1, wsize = W - IKS + 1, s = 0; s < SEG - 1; s++, hsize /= PS, wsize /= PS) {
		c = CONV;
		w = hsize * wsize / (PS * PS);
		for (j = 0; j < chN[s + 1][0]; j++) {
			for (skipBias[s][j] += -eta * (dSkipBias[s][j] / w), i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					skipWoi[s][j][i][y][x] += -eta * (dSkipWoi[s][j][i][y][x] / w);
		}
	}
	// InConv重み・バイアス
	w = (H - IKS + 1) * (W - IKS + 1);
	for (j = 0; j < chN[0][0]; j++) {
		for (inBias[j] += -eta * (dInBias[j] / w), i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				inWoi[j][i][y][x] += -eta * (dInWoi[j][i][y][x] / w);
	}
	// SMP重み
	for (s = 0; s < SEG; s++) {
		for (c = 0; c < CONV; c++)
			for (j = 0; j < chN[s][c + 1]; j++)
				for (i = 0; i < chN[s][c]; i++)
					for (m = 0; m < np[s][c][i]; m++) {
						if (wUseCount[s][c][j][i][m] > 0) {
							wi[s][c][j][i][m] += -eta * dWi[s][c][j][i][m] / wUseCount[s][c][j][i][m];
						}
					}
	}
	// SMP座標・半径
	for (s = 0; s < SEG; s++) {
		for (c = 0; c < CONV; c++) {
			for (i = 0; i < chN[s][c]; i++) {
				for (m = 0; m < np[s][c][i]; m++) {
					// 座標・半径更新
					if (prUseCount[s][c][i][m] > 0) {
						for (d = 0; d < SD; d++) {
							pi[s][c][i][m][d] += -eta * dPi[s][c][i][m][d] / prUseCount[s][c][i][m];
							dPsum[s][c][i][m][d] += dPi[s][c][i][m][d] / prUseCount[s][c][i][m];// 変化量を足しこむ(提案手法用)
						}
						ri[s][c][i][m] += -eta * dRi[s][c][i][m] / prUseCount[s][c][i][m];
						dRsum[s][c][i][m] += dRi[s][c][i][m] / prUseCount[s][c][i][m];// 変化量を足しこむ(提案手法用)
					}
					// 範囲内にクリッピング
					if (ri[s][c][i][m] < LOWESTR)ri[s][c][i][m] = LOWESTR;
					else if (ri[s][c][i][m] > HIGHESTR)ri[s][c][i][m] = HIGHESTR;
				}
			}
		}
	}
}

// BatchSMPBackConv
void BSMPBackConv2D(int ich, int och, double in[CHMAX][BS][H][W], double out[CHMAX][BS][H][W], double pi[CHMAX][NPMAX][SD],
	double ri[CHMAX][NPMAX], double wi[CHMAX][CHMAX][NPMAX], int np[CHMAX], double inDelta[CHMAX][BS][H][W],
	double outDelta[CHMAX][BS][H][W], double cie[H][W][SD], double dis[H][W][CHMAX][NPMAX], double g[H][W][CHMAX][NPMAX],
	int sumCount[H][W][CHMAX], int hsize, int wsize, double dPi[CHMAX][NPMAX][SD], double dRi[CHMAX][NPMAX],
	double dWi[CHMAX][CHMAX][NPMAX], int wUseCount[CHMAX][CHMAX][NPMAX], int prUseCount[CHMAX][NPMAX],
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double eta) {
	int i, j, k, x, m, d, h, w, b;
	double sum;

	// delta初期化
	for (i = 0; i < ich; i++) {// 入力チャンネル
		for (b = 0; b < BS; b++)
			for (h = 0; h < hsize; h++)// 縦サイズ
				for (w = 0; w < wsize; w++)// 横サイズ
					inDelta[i][b][h][w] = 0;
	}

	// ReLU
	for (j = 0; j < och; j++) {
		for (b = 0; b < BS; b++) {
			for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
				if (out[j][b][h][w] <= 0)outDelta[j][b][h][w] = 0;// ReLU
			// バッチ正規化
			if (SMPBatchNormal)CBatchNormalB(outDelta[j], bnet[j], mean[j], var[j], &gamma[j], &beta[j], eta, hsize, wsize);
		}
	}

	// 誤差逆伝搬
	for (j = 0; j < och; j++) {// 出力チャンネル
		for (b = 0; b < BS; b++) {
			for (h = 0; h < hsize; h++) {// 縦サイズ
				for (w = 0; w < wsize; w++) {// 横サイズ
					if (outDelta[j][b][h][w] != 0) {
						for (i = 0; i < ich; i++) {// 入力チャンネル
							if (sumCount[h][w][i] > 0) {// 近傍誘導点の数が存在する
								for (sum = 0, m = 0; m < np[i]; m++) {// 誘導点数
									if (g[h][w][i][m] > 0) {// 距離関数=0は近傍範囲外のため使用しない
										sum += g[h][w][i][m] * wi[j][i][m];
										// 座標の変化量・半径の変化量
										for (d = 0; d < SD; d++) {
											if (cie[h][w][d] - pi[i][m][d] >= 0) {
												dPi[i][m][d] += outDelta[j][b][h][w] * in[i][b][h][w] * (wi[j][i][m] / sumCount[h][w][i]) * (1.0 / ri[i][m]);// 座標
											}
											else {
												dPi[i][m][d] -= outDelta[j][b][h][w] * in[i][b][h][w] * (wi[j][i][m] / sumCount[h][w][i]) * (1.0 / ri[i][m]);// 座標
											}
										}
										dRi[i][m] += outDelta[j][b][h][w] * in[i][b][h][w] * (wi[j][i][m] / sumCount[h][w][i]) * (dis[h][w][i][m] / (ri[i][m] * ri[i][m]));// 半径
										// 重みの変化量
										dWi[j][i][m] += outDelta[j][b][h][w] * in[i][b][h][w] * g[h][w][i][m] / sumCount[h][w][i];
										// 誘導点の使用回数増加
										wUseCount[j][i][m]++;
										prUseCount[i][m]++;
									}
								}
								// delta計算
								inDelta[i][b][h][w] += outDelta[j][b][h][w] * sum / sumCount[h][w][i];
							}
						}
					}
				}
			}
		}
	}
}

// BatchSMPBackProp
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
	double dRsum[SEG][CONV][CHMAX][NPMAX]) {
	int i, j, s, c, hsize = H / pow(PS, SEG), wsize = W / pow(PS, SEG), st = 1, iN, oN, h, w, nh, nw, och, n, m, d, b;
	double sum, max;
	int x, y;

	// 重み修正量を0初期化
	hsize = (H - IKS + 1) / pow(PS, SEG);
	wsize = (W - IKS + 1) / pow(PS, SEG);

	/* ---------- CNN初期化 ---------- */
	// 重み・バイアスの変化量
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++) {
		for (j = 0; j < CHMAX; j++)
			for (dBias[s][c][j] = 0, i = 0; i < CHMAX; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					dWoi[s][c][j][i][y][x] = 0; //初期化
	}
	// スキップ接続の重み・バイアスの変化量
	for (c = CONV, s = 0; s < SEG; s++) {
		for (j = 0; j < CHMAX; j++)for (dSkipBias[s][j] = 0, i = 0; i < CHMAX; i++)
			for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
				dSkipWoi[s][j][i][y][x] = 0; //初期化
	}
	// 入力畳み込みの重み・バイアスの変化量
	for (j = 0; j < CHMAX; j++) {
		for (dInBias[j] = 0, i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				dInWoi[j][i][y][x] = 0; //初期化
	}

	/* ---------- SMP初期化 ---------- */
	// 座標・半径・使用回数初期化
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++) {
		for (i = 0; i < CHMAX; i++) {
			for (m = 0; m < NPMAX; m++) {
				for (d = 0; d < SD; d++) {
					dPi[s][c][i][m][d] = 0;
				}
				dRi[s][c][i][m] = 0;
				prUseCount[s][c][i][m] = 0;
			}
		}
	}
	// 重み・使用回数初期化
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++) {
		for (j = 0; j < CHMAX; j++)
			for (i = 0; i < CHMAX; i++)
				for (m = 0; m < NPMAX; m++) {
					dWi[s][c][j][i][m] = 0;
					wUseCount[s][c][j][i][m] = 0;
				}
	}

	/* ---------- SMPBackProp ---------- */
	// SMPBackProp
	for (s = SEG - 1; s >= 0; s--) {// segment
		// サイズ戻し
		hsize *= PS;// 画像サイズ
		wsize *= PS;// 画像サイズ
		c = CONV;
		// Max Pooling
		if (s == SEG - 1) {// 全結合のデルタを入れる delta[s][CONV]=mdelta[]
			for (b = 0; b < BS; b++) {
				for (oN = 0, iN = 0; iN < chN[s][c]; iN++) {
					for (h = 0; h < hsize; h += PS) {
						for (w = 0; w < wsize; w += PS, oN++) {
							// max計算
							for (nh = h, nw = w, max = skipOut[s][iN][b][h][w], i = 0; i < PS; i++) {
								for (j = 0; j < PS; j++) {
									skipDelta[s][iN][b][h + i][w + j] = 0;
									if (max < skipOut[s][iN][b][h + i][w + j]) {
										max = skipOut[s][iN][b][h + i][w + j];
										nh = h + i;
										nw = w + j;
									}
								}
							}
							// delta代入
							skipDelta[s][iN][b][nh][nw] = mdelta[oN][b];
						}
					}
				}
			}
		}
		else {// out[s][c] -> out[s+1][0] 次のセグメントの0層に入れる
			BBackConv(chN[s][c], chN[s + 1][0], skipBias[s], skipWoi[s], skipDelta[s], delta[s + 1][0],
				hsize, wsize, PS, KS, skipOut[s], out[s + 1][0], dSkipBias[s], dSkipWoi[s], bnet[s + 1][0], mean[s + 1][0], var[s + 1][0], gamma[s + 1][0], beta[s + 1][0], eta);
			// Skip-Connection
			for (n = 0; n < chN[s][c]; n++) {
				for (b = 0; b < BS; b++) {
					for (h = 0; h < hsize; h += PS) {
						for (w = 0; w < wsize; w += PS) {
							// Skip-Connection
							for (sum = 0, j = 0; j < PS; j++)for (i = 0; i < PS; i++)
								skipDelta[s][n][b][h + j][w + i] += delta[s + 1][0][n][b][h / PS][w / PS] / (PS * PS);
						}
					}
				}
			}
		}

		// デルタ代入
		for (n = 0; n < chN[s][c]; n++) {
			for (b = 0; b < BS; b++) {
				for (h = 0; h < hsize; h++) {
					for (w = 0; w < wsize; w++) {
						delta[s][c][n][b][h][w] = skipDelta[s][n][b][h][w];
					}
				}
			}
		}

		// 畳み込み(SMPBackConv)
		for (c = CONV - 1; c >= 0; c--) {
			BSMPBackConv2D(chN[s][c], chN[s][c + 1], out[s][c], out[s][c + 1], pi[s][c], ri[s][c], wi[s][c], np[s][c], delta[s][c], delta[s][c + 1], cie[s], dis[s][c], g[s][c], sumCount[s][c], hsize, wsize,
				dPi[s][c], dRi[s][c], dWi[s][c], wUseCount[s][c], prUseCount[s][c], bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], eta);
		}

		// スキップ接続
		for (n = 0; n < chN[s][0]; n++) {
			for (b = 0; b < BS; b++) {
				for (h = 0; h < hsize; h++) {
					for (w = 0; w < wsize; w++) {
						delta[s][0][n][b][h][w] += skipDelta[s][n][b][h][w];
					}
				}
			}
		}
	}
	// InBackConv
	BInBackConv(ICH, chN[0][0], delta[0][0], H, W, st, IKS, in, dInBias, dInWoi, out[0][0], bnet[0][0], mean[0][0], var[0][0], gamma[0][0], beta[0][0], eta);

	/* ---------- 更新 ---------- */
	// SkipConv重み・バイアス
	for (hsize = H - IKS + 1, wsize = W - IKS + 1, s = 0; s < SEG - 1; s++, hsize /= PS, wsize /= PS) {
		c = CONV;
		w = (hsize * wsize * BS) / (PS * PS);
		for (j = 0; j < chN[s + 1][0]; j++) {
			for (skipBias[s][j] += -eta * (dSkipBias[s][j] / w), i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					skipWoi[s][j][i][y][x] += -eta * (dSkipWoi[s][j][i][y][x] / w);
		}
	}
	// InConv重み・バイアス
	w = (H - IKS + 1) * (W - IKS + 1) * BS;
	for (j = 0; j < chN[0][0]; j++) {
		for (inBias[j] += -eta * (dInBias[j] / w), i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				inWoi[j][i][y][x] += -eta * (dInWoi[j][i][y][x] / w);
	}
	// SMP重み
	for (s = 0; s < SEG; s++) {
		for (c = 0; c < CONV; c++)
			for (j = 0; j < chN[s][c + 1]; j++)
				for (i = 0; i < chN[s][c]; i++)
					for (m = 0; m < np[s][c][i]; m++) {
						if (wUseCount[s][c][j][i][m] != 0) {
							wi[s][c][j][i][m] += -eta * dWi[s][c][j][i][m] / wUseCount[s][c][j][i][m];
						}
					}
	}
	// SMP座標・半径
	for (s = 0; s < SEG; s++) {
		for (c = 0; c < CONV; c++) {
			for (i = 0; i < chN[s][c]; i++) {
				for (m = 0; m < np[s][c][i]; m++) {
					// 座標・半径更新
					if (prUseCount[s][c][i][m] != 0) {
						for (d = 0; d < SD; d++) {
							pi[s][c][i][m][d] += -eta * dPi[s][c][i][m][d] / prUseCount[s][c][i][m];
							dPsum[s][c][i][m][d] += dPi[s][c][i][m][d] / prUseCount[s][c][i][m];// 変化量を足しこむ(提案手法用)
						}
						ri[s][c][i][m] += -eta * dRi[s][c][i][m] / prUseCount[s][c][i][m];
						dRsum[s][c][i][m] += dRi[s][c][i][m] / prUseCount[s][c][i][m];// 変化量を足しこむ(提案手法用)
					}
					// 範囲内にクリッピング
					if (ri[s][c][i][m] < LOWESTR)ri[s][c][i][m] = LOWESTR;
					else if (ri[s][c][i][m] > HIGHESTR)ri[s][c][i][m] = HIGHESTR;
				}
			}
		}
	}
}

/* -------------------- ↓ 提案手法 ↓ -------------------- */
// 初期化
void SOIPInit(double dPsum[SEG][CONV][CHMAX][NPMAX][SD], double dRsum[SEG][CONV][CHMAX][NPMAX]) {
	int s, c, i, m, d;
	for (s = 0; s < SEG; s++) {
		for (c = 0; c < CONV; c++)
			for (i = 0; i < CHMAX; i++)
				for (m = 0; m < NPMAX; m++) {
					// 初期化
					for (d = 0; d < SD; d++)dPsum[s][c][i][m][d] = 0;
					dRsum[s][c][i][m] = 0;
				}
	}
}

// 誘導点の増減
void SOIP(int chN[SEG][CONV + 1], double pi[SEG][CONV][CHMAX][NPMAX][SD], double ri[SEG][CONV][CHMAX][NPMAX],
	double wi[SEG][CONV][CHMAX][CHMAX][NPMAX], int np[SEG][CONV][CHMAX], int* allNP, int delC[SEG][CONV][CHMAX][NPMAX],
	double dPsum[SEG][CONV][CHMAX][NPMAX][SD], double dRsum[SEG][CONV][CHMAX][NPMAX], bool add[SEG][CONV][CHMAX], double delW, double eta,
	int* delN, int* merN, int* addN) {
	int i, j, k, s, c, m, n, d, id, minp;
	double temp, dis, max, r, sum, wmSum, wnSum, similarity, addRate, min;

	/* 誘導点の削除 */
	if (POINTMODE == 0 || POINTMODE == 1) {
		for (s = 0; s < SEG; s++) {// セグメント数
			for (c = 0; c < CONV; c++)// 畳み込み数
				for (i = 0; i < chN[s][c]; i++)// 入力数
					if (np[s][c][i] > NPMIN) {
						// 重みの平均値計算
						for (sum = 0, m = 0, j = 0; j < chN[s][c + 1]; j++)
							if (wi[s][c][j][i][m] < 0)sum -= wi[s][c][j][i][m];
							else sum += wi[s][c][j][i][m];// 絶対値
						min = sum;
						for (minp = 0, m = 1; m < np[s][c][i]; m++) {// 誘導点数
							// 重みの平均計算
							for (sum = 0, j = 0; j < chN[s][c + 1]; j++)
								if (wi[s][c][j][i][m] < 0)sum -= wi[s][c][j][i][m];
								else sum += wi[s][c][j][i][m];// 絶対値
							if (min > sum) {
								min = sum;
								minp = m;
							}
						}
						// 削除条件
						if (min / chN[s][c + 1]* ri[s][c][i][minp] <= delW) {// 重み・半径が閾値以下
							m = minp;
							delC[s][c][i][m]++;						// カウント増加
							if (delC[s][c][i][m] >= DELCOUNT) {		// 特定回数選択されたら削除
								n = np[s][c][i] - 1;				// 最後の要素番号
								ri[s][c][i][m] = ri[s][c][i][n];	// 半径削除(上書き)							
								for (d = 0; d < SD; d++)pi[s][c][i][m][d] = pi[s][c][i][n][d];			 // 座標削除(上書き)
								for (j = 0; j < chN[s][c + 1]; j++)wi[s][c][j][i][m] = wi[s][c][j][i][n];// 重み削除(上書き)	
								for (d = 0; d < SD; d++)dPsum[s][c][i][m][d] = dPsum[s][c][i][n][d];	 // 座標変化量(上書き)
								dRsum[s][c][i][m] = dRsum[s][c][i][n];									 // 半径変化量(上書き)
								delC[s][c][i][m] = delC[s][c][i][n];// カウント数削除(上書き)
								np[s][c][i]--;						// 誘導点数をデクリメント
								m--;								// 上書きした箇所を再度計算
								(*delN)++;							// 削除カウント(モデル評価用)
							}
						}
						else {
							if (delC[s][c][i][m] >= 0)delC[s][c][i][m] = 0;// カウントリセット(追加時の猶予を0にしないようにする)
						}

					}
		}
	}

	/* 誘導点の結合 */
	if (POINTMODE == 0 || POINTMODE == 2) {
		for (s = 0; s < SEG; s++) {// セグメント数
			for (c = 0; c < CONV; c++)// 畳み込み数
				for (i = 0; i < chN[s][c]; i++)// 入力数
					if (np[s][c][i] > NPMIN) {// 誘導点の最小数よりも多い
						for (m = 0; m < np[s][c][i]; m++) {// 誘導点数(基準)
							for (n = m + 1; n < np[s][c][i]; n++) {// 誘導点数(比較)
								if (ri[s][c][i][m] - ri[s][c][i][n] >= -MERDIFF && ri[s][c][i][m] - ri[s][c][i][n] <= MERDIFF) {// 条件①：2点の半径の差分が閾値以下
									// L2距離
									for (dis = 0, d = 0; d < SD; d++) {
										temp = (pi[s][c][i][m][d] - pi[s][c][i][n][d]) * (pi[s][c][i][m][d] - pi[s][c][i][n][d]);
										dis += temp;
									}
									dis = sqrt(dis);
									if (dis < MERRATE * ri[s][c][i][m] && dis < MERRATE * ri[s][c][i][n]) {// 条件②：距離が一定値以内
										// コサイン類似度
										for (sum = 0, wmSum = 0, wnSum = 0, j = 0; j < chN[s][c + 1]; j++) {// 出力数
											sum += wi[s][c][j][i][m] * wi[s][c][j][i][n];// 重みベクトルm * 重みベクトルn
											wmSum += wi[s][c][j][i][m] * wi[s][c][j][i][m];// 重みベクトルmの2乗の合計
											wnSum += wi[s][c][j][i][n] * wi[s][c][j][i][n];// 重みベクトルnの2乗の合計
										}
										wmSum = sqrt(wmSum); wnSum = sqrt(wnSum);// 平方根
										similarity = sum / (wmSum * wnSum);		// コサイン類似度計算
										if (similarity >= MERSIMI) {// 条件③：重みに類似性がある
											/* ノード結合 */
											// 重み
											for (j = 0; j < chN[s][c + 1]; j++) {// 出力数
												//wi[s][c][j][i][m] = (wi[s][c][j][i][m] + wi[s][c][j][i][n]) / (2.0 - (1.0 - similarity));// 平均(類似度が高ければ割る値が大きくなる:2.0～1.0)
												wi[s][c][j][i][m] = wi[s][c][j][i][m] + wi[s][c][j][i][n];// 平均(類似度が高ければ割る値が大きくなる:2.0～1.0)
											}
											wmSum /= chN[s][c + 1]; wnSum /= chN[s][c + 1];// 重みの平均
											// 半径
											if (ri[s][c][i][m] < ri[s][c][i][n])ri[s][c][i][m] = ri[s][c][i][n];
											// 座標
											for (d = 0; d < SD; d++) {
												//pi[s][c][i][m][d] += (pi[s][c][i][n][d] - pi[s][c][i][m][d]) * wi[s][c][j][i][n] / (wi[s][c][j][i][m] + wi[s][c][j][i][n]);
												pi[s][c][i][m][d] += (pi[s][c][i][n][d] - pi[s][c][i][m][d]) * wnSum / (wmSum + wnSum);
											}
											delC[s][c][i][m] = ADDDEL;// 削除カウント
											/* ノード削除 */
											k = np[s][c][i] - 1;
											for (j = 0; j < chN[s][c + 1]; j++)wi[s][c][j][i][n] = wi[s][c][j][i][k];// 重み
											ri[s][c][i][n] = ri[s][c][i][k];										// 半径
											for (d = 0; d < SD; d++)pi[s][c][i][n][d] = pi[s][c][i][k][d];			// 座標
											for (d = 0; d < SD; d++)dPsum[s][c][i][n][d] = dPsum[s][c][i][k][d];	// 座標変化量(上書き)
											dRsum[s][c][i][n] = dRsum[s][c][i][k];									// 半径変化量(上書き)
											delC[s][c][i][n] = delC[s][c][i][k];
											np[s][c][i]--;		// 誘導点数をデクリメント
											(*merN)++;			// 結合カウント(モデル評価用)
											break;
										}
									}
								}
							}
						}
					}
		}
	}

	/* 誘導点の追加 */
	if (POINTMODE == 0 || POINTMODE == 3) {
		for (s = 0; s < SEG; s++) {// セグメント数
			for (c = 0; c < CONV; c++)// 畳み込み数
				for (i = 0; i < chN[s][c]; i++) {// 入力数
					if (np[s][c][i] < NPMAX) {// 追加条件を満たす
						// 閾値計算
						addRate = ADDRATEBASIS;
						//addRate = ADDRATEBASIS + (ADDRATERANGE * (np[s][c][i] - NPMIN) / (NPMAX - NPMIN));
						// 座標の変化量が最大の番号を保持
						for (id = -1, max = 0, m = 0; m < np[s][c][i]; m++) {// 誘導点数
							// 距離計算
							for (dis = 0, d = 0; d < SD; d++) dis += dPsum[s][c][i][m][d] * dPsum[s][c][i][m][d];
							dis = sqrt(dis);// ユークリッド距離
							// 最大値計算
							if (max <= dis) {
								max = dis;
								id = m;// 最大値の番号
							}
						}
						if (id != -1 && ((BATCHLEARN == 0 && max >= addRate * SOIPTIME) || (BATCHLEARN == 1 && max >= addRate * BSOIPTIME))) {
							if (-dRsum[s][c][i][id] > 0) {// 半径を増やすように更新している場合
								// 重み
								for (j = 0; j < chN[s][c + 1]; j++) {
									wi[s][c][j][i][np[s][c][i]] = (rand() / (RAND_MAX + 1.0) - 0.5) * 0.6;

									//wi[s][c][j][i][np[s][c][i]] = delW + delW * ADDW;										// 削除最小値+削除最小値×レート
									//if (wi[s][c][j][i][id] < 0)wi[s][c][j][i][np[s][c][i]] = -wi[s][c][j][i][np[s][c][i]];	// 符号反転
								}
								// 半径(初期化)
								ri[s][c][i][np[s][c][i]] = R;
								// 座標
								for (d = 0; d < SD; d++) {
									pi[s][c][i][np[s][c][i]][d] = pi[s][c][i][id][d] - eta * dPsum[s][c][i][id][d];
									
									// 追加処理
									//if (eta * dPsum[s][c][i][id][d] >= 0)pi[s][c][i][np[s][c][i]][d] += (MERRATE * ri[s][c][i][np[s][c][i]]) / sqrt(2);	// 結合範囲外にする
									//else pi[s][c][i][np[s][c][i]][d] -= (MERRATE * ri[s][c][i][np[s][c][i]]) / sqrt(2);									// 結合範囲外にする
								}
								// 誘導点増減パラメータ
								delC[s][c][i][np[s][c][i]] = ADDDEL;// 削除カウント
								np[s][c][i]++;						// 誘導点数増加
								(*addN)++;							// 追加カウント(モデル評価用)
							}
						}
					}
				}
		}
	}

	// 誘導点数の合計(描画用)
	for (*allNP = 0, s = 0; s < SEG; s++) {// セグメント数
		for (c = 0; c < CONV; c++)// 畳み込み数
			for (i = 0; i < chN[s][c]; i++)// 入力数
				*allNP += np[s][c][i];
	}
}
