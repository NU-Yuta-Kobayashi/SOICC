#include<math.h> 
#include"MLP.h"
#include"CNN.h"

double Pin[CHMAX][H + KS - 1][W + KS - 1];			// パディング後の入力データ
double Pdelta[CHMAX][H + KS - 1][W + KS - 1];		// パディング後のデルタ 
double BPin[CHMAX][BS][H + KS - 1][W + KS - 1];		// パディング後の入力データ(バッチ用)
double BPdelta[CHMAX][BS][H + KS - 1][W + KS - 1];	// パディング後のデルタ(バッチ用)
double D3a[BS][H][W];								// 途中式

int CBatchNormal = 0;// バッチ正則化　0:off, 1:on

/* -------------------- ↓ Forward ↓ -------------------- */
void CBatchNormalF(double bnet[][H][W], double anet[][H][W], double* mean, double* var, double gamma, double beta, int hsize, int wsize) {
	double sqvar, m = BS * hsize * wsize;
	int h, w, b;

	// 平均計算
	for (*mean = 0, b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) *mean += bnet[b][h][w];
	*mean /= m;
	// 分散
	for (*var = 0, b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) *var += (bnet[b][h][w] - *mean) * (bnet[b][h][w] - *mean);
	*var /= m;
	// 正規化
	for (sqvar = sqrt(*var + EPS), b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) anet[b][h][w] = (bnet[b][h][w] - *mean) / sqvar;
	// 変倍・移動
	for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) anet[b][h][w] = gamma * anet[b][h][w] + beta;
}

void InConv(int ich, int och, double bias[], double woi[][ICH][IKS][IKS], double in[][H][W], double out[][H][W], int ksize, double gamma[], double beta[], double mmean[], double mvar[]) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw;
	double sum;
	int oh = H - ksize + 1, ow = W - ksize + 1;

	// convolution
	for (j = 0; j < och; j++) {//出力側
		for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
			for (out[j][h][w] = bias[j], i = 0; i < ich; i++) //入力側
				for (y = 0; y < ksize; y++)for (x = 0; x < ksize; x++)//ここにｙとｘのループ
					out[j][h][w] += woi[j][i][y][x] * in[i][h + y][w + x];
	}
	//BNしているなら正則化しながら
	if (BATCHLEARN && CBatchNormal)
		for (j = 0; j < och; j++) //出力側をReLU
			for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
				out[j][h][w] = (out[j][h][w] - mmean[j]) / sqrt(mvar[j] + EPS) * gamma[j] + beta[j];
	//ReLU
	for (j = 0; j < och; j++) //出力側をReLU
		for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
			if (out[j][h][w] <= 0) out[j][h][w] = 0;
}

void BInConv(int ich, int och, double bias[], double woi[][ICH][IKS][IKS], double in[][BS][H][W], double out[][BS][H][W], int ksize,
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double mmean[], double mvar[]) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw, b;
	double sum;
	int oh = H - ksize + 1, ow = W - ksize + 1;

	// convolution
	for (j = 0; j < och; j++) {//出力側
		for (b = 0; b < BS; b++)// バッチ数
			for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
				for (bnet[j][b][h][w] = bias[j], i = 0; i < ich; i++) //入力側
					for (y = 0; y < ksize; y++)for (x = 0; x < ksize; x++)//ここにｙとｘのループ
						bnet[j][b][h][w] += woi[j][i][y][x] * in[i][b][h + y][w + x];
	}

	//ReLU
	if (CBatchNormal) {
		if (RELU) {// ReLU
			for (j = 0; j < och; j++) {// 出力ノード数
				// バッチ正規化
				CBatchNormalF(bnet[j], out[j], &mean[j], &var[j], gamma[j], beta[j], oh, ow);
				// ReLU
				for (b = 0; b < BS; b++) // バッチ数
					for (h = 0; h < oh; h++)// 画像サイズ(高さ)
						for (w = 0; w < ow; w++)// 画像サイズ(横幅)
							if (out[j][b][h][w] <= 0)out[j][b][h][w] = 0;
				// 移動平均・移動分散
				mmean[j] = mmean[j] * MRATE + (1.0 - MRATE) * mean[j];	// 移動平均
				mvar[j] = mvar[j] * MRATE + (1.0 - MRATE) * var[j];		// 移動分散
			}
		}
	}
	else {
		if (RELU)// ReLU
			for (j = 0; j < och; j++)// 出力ノード数
				for (b = 0; b < BS; b++)// バッチ数
					for (h = 0; h < oh; h++)// 画像サイズ(高さ)
						for (w = 0; w < ow; w++) {// 画像サイズ(横幅)
							if (bnet[j][b][h][w] <= 0)out[j][b][h][w] = 0;
							else out[j][b][h][w] = bnet[j][b][h][w];
						}
	}
}

void Conv(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double in[][H][W], double out[][H][W], int hsize, int wsize, int st, int ksize,
	double gamma[], double beta[], double mmean[], double mvar[]) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw;
	double sum;

	// 0 padding
	ph = hsize + ksize - 1;
	pw = wsize + ksize - 1;
	for (i = 0; i < ich; i++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) Pin[i][h][w] = 0;							// 0-padding
	for (i = 0; i < ich; i++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) Pin[i][h + p][w + p] = in[i][h][w];	// データ入力

	// convolution
	for (j = 0; j < och; j++) {// 出力ノード数
		for (h = 0; h < hsize; h += st) {// 画像サイズ(高さ)
			for (w = 0; w < wsize; w += st) {// 画像サイズ(横幅)			
				for (sum = bias[j], i = 0; i < ich; i++) {// 入力ノード数
				// 畳み込み
					for (y = 0; y < ksize; y++) for (x = 0; x < ksize; x++)// カーネルサイズ 
						sum += Pin[i][h + y][w + x] * woi[j][i][y][x];// 畳み込み
				}
				out[j][h / st][w / st] = sum;
			}
		}
	}

	// バッチ正規化
	if (BATCHLEARN && CBatchNormal) {
		for (j = 0; j < och; j++)// 出力ノード数
			for (h = 0; h < hsize / st; h++)// 画像サイズ(高さ)
				for (w = 0; w < wsize / st; w++)// 画像サイズ(横幅)
					out[j][h][w] = (out[j][h][w] - mmean[j]) / sqrt(mvar[j] + EPS) * gamma[j] + beta[j];
	}

	// ReLU
	if (RELU)
		for (j = 0; j < och; j++)// 出力ノード数
			for (h = 0; h < hsize / st; h++)// 画像サイズ(高さ)
				for (w = 0; w < wsize / st; w++)// 画像サイズ(横幅)
					if (out[j][h][w] <= 0)
						out[j][h][w] = 0;
}

void BConv(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double in[][BS][H][W], double out[][BS][H][W],
	int hsize, int wsize, int st, int ksize, double bnet[][BS][H][W], double mean[], double var[], double gamma[],
	double beta[], double mmean[], double mvar[]) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw, b;
	double sum;

	ph = hsize + ksize - 1; pw = wsize + ksize - 1;
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)
		for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) BPin[i][b][h][w] = 0;
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)
		for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) BPin[i][b][h + p][w + p] = in[i][b][h][w];

	// convolution
	for (j = 0; j < och; j++) {// 出力ノード数
		for (b = 0; b < BS; b++) {// バッチ数
			for (h = 0; h < hsize; h += st) {// 画像サイズ(高さ)
				for (w = 0; w < wsize; w += st) {// 画像サイズ(横幅)			
					for (sum = bias[j], i = 0; i < ich; i++) {// 入力ノード数
						// 畳み込み
						for (y = 0; y < ksize; y++) for (x = 0; x < ksize; x++)// カーネルサイズ 
							sum += BPin[i][b][h + y][w + x] * woi[j][i][y][x];// 畳み込み
					}
					bnet[j][b][h / st][w / st] = sum;
				}
			}
		}
	}

	// バッチ正規化
	if (CBatchNormal) {
		if (RELU) {// ReLU
			for (j = 0; j < och; j++) {// 出力ノード数
				// バッチ正規化
				CBatchNormalF(bnet[j], out[j], &mean[j], &var[j], gamma[j], beta[j], hsize / st, wsize / st);
				// ReLU
				for (b = 0; b < BS; b++) // バッチ数
					for (h = 0; h < hsize / st; h++)// 画像サイズ(高さ)
						for (w = 0; w < wsize / st; w++)// 画像サイズ(横幅)
							if (out[j][b][h][w] <= 0)out[j][b][h][w] = 0;
				// 移動平均・移動分散
				mmean[j] = mmean[j] * MRATE + (1.0 - MRATE) * mean[j];	// 移動平均
				mvar[j] = mvar[j] * MRATE + (1.0 - MRATE) * var[j];		// 移動分散
			}
		}
	}
	else {
		if (RELU)// ReLU
			for (j = 0; j < och; j++)// 出力ノード数
				for (b = 0; b < BS; b++)// バッチ数
					for (h = 0; h < hsize / st; h++)// 画像サイズ(高さ)
						for (w = 0; w < wsize / st; w++) {// 画像サイズ(横幅)
							if (bnet[j][b][h][w] <= 0)out[j][b][h][w] = 0;
							else out[j][b][h][w] = bnet[j][b][h][w];
						}
	}
}

void SkipForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W],
	double mout[], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX], 
	double in[ICH][H][W],double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][H][W],
	double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS]) {
	int i, j, s, c, w, h, n, fn, hsize = H, wsize = W, st = 1;// st:ストライドサイズ
	double sum, max;
	
	// 入力
	InConv(ICH, chN[0][0], inBias, inWoi, in, out[0][0], IKS, gamma[0][0], beta[0][0], mmean[0][0], mvar[0][0]);
	hsize = H - IKS + 1; wsize = W - IKS + 1;// padding アリの場合

	// 順伝播
	for (s = 0; s < SEG; s++) {// segment
		// 畳み込み+ReLU(1セグメント分)
		for (c = 0; c < CONV; c++) {
			Conv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS, 
				gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1]);
			hsize /= st;// ストライドに応じてサイズ縮小
			wsize /= st;// ストライドに応じてサイズ縮小
		}

		// Skip-Connection
		for (n = 0; n < chN[s][c]; n++) {
			for (h = 0; h < hsize; h++)
				for (w = 0; w < wsize; w++)
					skipOut[s][n][h][w] = out[s][c][n][h][w] + out[s][0][n][h][w];// Skip-Connection
		}

		// Max-Pooling
		if (s == SEG - 1) {// out[s][c] -> mout 全結合に入れる 
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
		else {// 次のセグメントの0層にmax代入
			// 畳み込み
			Conv(chN[s][c], chN[s + 1][0], skipBias[s], skipWoi[s], skipOut[s], out[s + 1][0], hsize, wsize, PS, KS, gamma[s + 1][0], beta[s + 1][0], mmean[s + 1][0], mvar[s + 1][0]);

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

void BSkipForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W],
	double mout[][NMAX], double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], 
	double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][BS][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][BS][H][W],
	double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS]) {
	int i, j, s, c, w, h, n, fn, b, hsize = H, wsize = W, st = 1;// st:ストライドサイズ
	double sum, max;

	// 入力
	BInConv(ICH, chN[0][0], inBias, inWoi, in, out[0][0], IKS, bnet[0][0], mean[0][0], var[0][0], gamma[0][0], beta[0][0], 
		mmean[0][0], mvar[0][0]);
	hsize = H - IKS + 1; wsize = W - IKS + 1;// padding アリの場合

	// 順伝播
	for (s = 0; s < SEG; s++) {// segment
		// 畳み込み+ReLU(1セグメント分)
		for (c = 0; c < CONV; c++) {
			/*BConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS,
				bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1]);*/
			BConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS,
				bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1]);


			hsize /= st;// ストライドに応じてサイズ縮小
			wsize /= st;// ストライドに応じてサイズ縮小
		}
		// Skip-Connection
		for (n = 0; n < chN[s][c]; n++) {
			for (b = 0; b < BS; b++) {
				for (h = 0; h < hsize; h++) {
					for (w = 0; w < wsize; w++) {
						//skipOut[s][n][h][w] = out[s][c][n][h][w];
						skipOut[s][n][b][h][w] = out[s][c][n][b][h][w] + out[s][0][n][b][h][w];// Skip-Connection
					}
				}
			}
		}

		// Max-Pooling
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
		else {// 次のセグメントの0層にmax代入
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
void CBatchNormalB(double delta[][H][W], double bnet[][H][W], double mean, double var, double* gamma, double* beta, double eta, int hsize, int wsize) {
	int i, h, w, b;
	double d3aMean, f3d12a;
	double dBeta, dGamma, f7;
	double m = BS * hsize * wsize;

	// δ更新の途中式
	for (f3d12a = 0, b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) f3d12a += (bnet[b][h][w] - mean) * (*gamma) * delta[b][h][w];
	f3d12a /= m;
	f7 = 1.0 / sqrt(var + EPS);

	for (d3aMean = 0, b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) {
		D3a[b][h][w] = f7 * ((*gamma * delta[b][h][w]) - (bnet[b][h][w] - mean) / (var + EPS) * f3d12a);
		d3aMean += D3a[b][h][w];
	}
	d3aMean /= m;

	// β更新
	for (dBeta = 0, b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) dBeta += delta[b][h][w];
	*beta += -eta * dBeta / m;

	// γ更新
	for (dGamma = 0, b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) dGamma += (bnet[b][h][w] - mean) * f7 * delta[b][h][w]; // f9〇d15
	*gamma += -eta * dGamma / m;

	// δ更新
	for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) delta[b][h][w] = D3a[b][h][w] - d3aMean;
}

void InBackConv(int ich, int och, double delta[][H][W], int hsize, int wsize,
	int st, int ksize, double in[][H][W], double dBias[], double dWoi[][ICH][IKS][IKS], double out[][H][W]) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw;
	double sum;
	int oh = (H - ksize + 1) / st, ow = (W - ksize + 1) / st;

	//作られた入力側の誤差信号に活性化関数の微分を適用(outDeltaのReLU戻し)
	for (j = 0; j < och; j++) {
		for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
			if (out[j][h][w] <= 0) delta[j][h][w] *= 0;
	}

	for (j = 0; j < och; j++) {//出力側
		for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
			for (dBias[j] += delta[j][h][w], i = 0; i < ich; i++) //入力側
				for (y = 0; y < ksize; y++)for (x = 0; x < ksize; x++)//ここにｙとｘのループ
					dWoi[j][i][y][x] += in[i][h + y][w + x] * delta[j][h / st][w / st];// 畳み込み
	}
}

void BInBackConv(int ich, int och, double delta[][BS][H][W], int hsize, int wsize,
	int st, int ksize, double in[][BS][H][W], double dBias[], double dWoi[][ICH][IKS][IKS], double out[][BS][H][W],
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double eta) {
	int i, j, h, w, y, x, b, p = (ksize - 1) / 2, ph, pw;
	double sum;
	int oh = (H - ksize + 1) / st, ow = (W - ksize + 1) / st;

	//作られた入力側の誤差信号に活性化関数の微分を適用(outDeltaのReLU戻し)
	for (j = 0; j < och; j++) {
		for (b = 0; b < BS; b++) {
			for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
				if (out[j][b][h][w] <= 0) delta[j][b][h][w] = 0;
		}
		// バッチ正則化
		if (CBatchNormal) CBatchNormalB(delta[j], bnet[j], mean[j], var[j], &gamma[j], &beta[j], eta, oh, ow);
	}

	for (j = 0; j < och; j++) {//出力側
		for (b = 0; b < BS; b++) {
			for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
				for (dBias[j] += delta[j][b][h][w], i = 0; i < ich; i++) //入力側
					for (y = 0; y < ksize; y++)for (x = 0; x < ksize; x++)//ここにｙとｘのループ
						dWoi[j][i][y][x] += in[i][b][h + y][w + x] * delta[j][b][h / st][w / st];// 畳み込み
		}
	}
}

void BackConv(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double indelta[][H][W], double delta[][H][W], int hsize, int wsize,
	int st, int ksize, double in[][H][W], double dBias[], double dWoi[][CHMAX][KS][KS],double out[][H][W]) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw;
	double sum;
	int oh = hsize / st, ow = wsize / st;
	//作られた入力側の誤差信号に活性化関数の微分を適用(	outDeltaのReLU戻し)
	for (j = 0; j < och; j++) {
		for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
			if (out[j][h][w] <= 0) delta[j][h][w] *= 0;
	}

	// 0 padding
	ph = hsize + ksize - 1;
	pw = wsize + ksize - 1;
	for (i = 0; i < ich; i++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) Pdelta[i][h][w] = 0;						// デルタ 0-padding
	for (i = 0; i < ich; i++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) Pin[i][h][w] = 0;							// 入力 0-padding
	for (i = 0; i < ich; i++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) Pin[i][h + p][w + p] = in[i][h][w];	// データ入力

	// convolution
	for (j = 0; j < och; j++) {// 出力ノード数
		for (h = 0; h < hsize; h += st) {// 画像サイズ(高さ)
			for (w = 0; w < wsize; w += st) {// 画像サイズ(横幅)			
				for (dBias[j] += delta[j][h/st][w/st], i = 0; i < ich; i++) {// 入力ノード数
					// 畳み込み
					for (y = 0; y < ksize; y++) for (x = 0; x < ksize; x++) {// カーネルサイズ 
						Pdelta[i][h + y][w + x] += woi[j][i][y][x] * delta[j][h / st][w / st];// 畳み込み
						dWoi[j][i][y][x] += Pin[i][h + y][w + x] * delta[j][h / st][w / st];// 畳み込み
					}
				}
			}
		}
	}

	// ReLU戻し
	for (i = 0; i < ich; i++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
		if (in[i][h][w] > 0)indelta[i][h][w] = Pdelta[i][h + p][w + p];	// デルタのパディング削除
		else indelta[i][h][w] = 0;
}

void BBackConv(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double indelta[][BS][H][W], double delta[][BS][H][W], int hsize, int wsize,
	int st, int ksize, double in[][BS][H][W], double out[][BS][H][W], double dBias[], double dWoi[][CHMAX][KS][KS], double bnet[][BS][H][W], double mean[], double var[], double gamma[],
	double beta[], double eta) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw, b;
	double sum;

	// ReLU戻し
	for (j = 0; j < och; j++) {
		for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
			if (out[j][b][h][w] <= 0)delta[j][b][h][w] = 0;// ReLU
		// バッチ正規化
		if (CBatchNormal)CBatchNormalB(delta[j], bnet[j], mean[j], var[j], &gamma[j], &beta[j], eta, hsize / st, wsize / st);
	}

	// 0 padding
	ph = hsize + ksize - 1;
	pw = wsize + ksize - 1;
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) BPdelta[i][b][h][w] = 0;						// デルタ 0-padding
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) BPin[i][b][h][w] = 0;							// 入力 0-padding
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) BPin[i][b][h + p][w + p] = in[i][b][h][w];	// データ入力

	// convolution
	for (j = 0; j < och; j++) {// 出力ノード数
		for (b = 0; b < BS; b++) {
			for (h = 0; h < hsize; h += st) {// 画像サイズ(高さ)
				for (w = 0; w < wsize; w += st) {// 画像サイズ(横幅)			
					for (dBias[j] += delta[j][b][h / st][w / st], i = 0; i < ich; i++) {// 入力ノード数
						// 畳み込み
						for (y = 0; y < ksize; y++) for (x = 0; x < ksize; x++) {// カーネルサイズ 
							BPdelta[i][b][h + y][w + x] += woi[j][i][y][x] * delta[j][b][h / st][w / st];// 畳み込み
							dWoi[j][i][y][x] += BPin[i][b][h + y][w + x] * delta[j][b][h / st][w / st];// 畳み込み
						}
					}
				}
			}
		}
	}

	// パディング戻し
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) {
		indelta[i][b][h][w] = BPdelta[i][b][h + p][w + p];	// デルタのパディング削除(戻す)
	}
}

void SkipBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W],
	double delta[][CONV + 1][CHMAX][H][W], double mdelta[], double dBias[][CONV][CHMAX], double dWoi[][CONV][CHMAX][CHMAX][KS][KS], double eta,
	double in[ICH][H][W], double inBias[], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][H][W], double skipDelta[SEG][CHMAX][H][W],
	double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS],double dSkipBias[SEG][CHMAX], double dSkipWoi[SEG][CHMAX][CHMAX][KS][KS],
	double dInBias[CHMAX], double dInWoi[CHMAX][ICH][IKS][IKS]) {
	int i, j, s, c, hsize = H / pow(PS, SEG), wsize = W / pow(PS, SEG), st = 1, iN, oN, h, w, nh, nw, och, n;
	double sum, max;
	int x, y;

	// 重み修正量を0初期化
	hsize = (H - IKS + 1) / pow(PS, SEG);
	wsize = (W - IKS + 1) / pow(PS, SEG);

	// 初期化
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++) {
		for (j = 0; j < chN[s][c + 1]; j++)
			for (dBias[s][c][j] = 0, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					dWoi[s][c][j][i][y][x] = 0; //初期化
	}
	for (c = CONV, s = 0; s < SEG; s++) {
		for (j = 0; j < CHMAX; j++)for (dSkipBias[s][j] = 0, i = 0; i < chN[s][c]; i++)
			for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
				dSkipWoi[s][j][i][y][x] = 0; //初期化
	}
	for (j = 0; j < chN[0][0]; j++) {
		for (dInBias[j] = 0, i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				dInWoi[j][i][y][x] = 0; //初期化
	}

	for (s = SEG - 1; s >= 0; s--) {// segment
		// サイズ戻し
		hsize *= PS;// 画像サイズ
		wsize *= PS;// 画像サイズ
		c = CONV;
		// Max Pooling
		if (s == SEG - 1) {// 全結合のデルタを入れる delta[s][CONV]=mdelta[]
			for (oN = 0, iN = 0; iN < chN[s][c]; iN++) {
				for (h = 0; h < hsize; h += PS) {
					for (w = 0; w < wsize; w += PS,oN++) {
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

		// Skip-Connection
		for (n = 0; n < chN[s][c]; n++) {
			for (h = 0; h < hsize; h++) {
				for (w = 0; w < wsize; w++) {
					delta[s][c][n][h][w] = skipDelta[s][n][h][w];
				}
			}
		}

		// 畳み込み+ReLU(CONV回分)
		for (c = CONV - 1; c >= 0; c--) {
			hsize *= st;
			wsize *= st;
			BackConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS, out[s][c], dBias[s][c], dWoi[s][c], out[s][c + 1]);

		}
		for (n = 0; n < chN[s][0]; n++) {
			for (h = 0; h < hsize; h++) {
				for (w = 0; w < wsize; w++) {
					delta[s][0][n][h][w] += skipDelta[s][n][h][w];
				}
			}
		}
	}

	InBackConv(ICH, chN[0][0], delta[0][0], H, W, st, IKS, in, dInBias, dInWoi, out[0][0]);

	// 重みバイアス修正(自分で追加)
	for (hsize = H-IKS+1, wsize = W-IKS+1, s = 0; s < SEG; s++, hsize /= PS, wsize /= PS) {
		for (w = hsize * wsize, c = 0; c < CONV; c++) {
			for (j = 0; j < chN[s][c + 1]; j++)
				for (bias[s][c][j] += -eta * (dBias[s][c][j] / w), i = 0; i < chN[s][c]; i++)
					for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
						woi[s][c][j][i][y][x] += -eta * (dWoi[s][c][j][i][y][x] / w);
		}
		w /= PS * PS; 

		if (s < SEG - 1) {
			for (j = 0; j < chN[s + 1][0]; j++) {
				for (skipBias[s][j] += -eta * (dSkipBias[s][j] / w), i = 0; i < chN[s][c]; i++)
					for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
						skipWoi[s][j][i][y][x] += -eta * (dSkipWoi[s][j][i][y][x] / w);
			}
		}
	}

	w = (H - IKS + 1) * (W - IKS + 1);
	for (j = 0; j < chN[0][0]; j++)
		for (inBias[j] += -eta * (dInBias[j] / w), i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				inWoi[j][i][y][x] += -eta * (dInWoi[j][i][y][x] / w);
}

void BSkipBackProp(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W],
	double delta[][CONV + 1][CHMAX][BS][H][W], double mdelta[][BS], double dBias[][CONV][CHMAX], double dWoi[][CONV][CHMAX][CHMAX][KS][KS], double eta,
	double in[ICH][BS][H][W], double inBias[], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][BS][H][W], double skipDelta[SEG][CHMAX][BS][H][W],
	double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX],
	double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS], double dSkipBias[SEG][CHMAX], double dSkipWoi[SEG][CHMAX][CHMAX][KS][KS],
	double dInBias[CHMAX], double dInWoi[CHMAX][ICH][IKS][IKS]) {
	int i, j, s, c, hsize = H / pow(PS, SEG), wsize = W / pow(PS, SEG), st = 1, iN, oN, h, w, nh, nw, och, n, b;
	double sum, max;
	int x, y;

	// 重み修正量を0初期化
	hsize = (H - IKS + 1) / pow(PS, SEG);
	wsize = (W - IKS + 1) / pow(PS, SEG);

	// 初期化
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++) {
		for (j = 0; j < chN[s][c + 1]; j++)
			for (dBias[s][c][j] = 0, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					dWoi[s][c][j][i][y][x] = 0; //初期化
	}
	for (c = CONV, s = 0; s < SEG; s++) {
		for (j = 0; j < CHMAX; j++)for (dSkipBias[s][j] = 0, i = 0; i < chN[s][c]; i++)
			for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
				dSkipWoi[s][j][i][y][x] = 0; //初期化
	}
	for (j = 0; j < chN[0][0]; j++) {
		for (dInBias[j] = 0, i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				dInWoi[j][i][y][x] = 0; //初期化
	}

	// 処理
	for (s = SEG - 1; s >= 0; s--) {
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
				hsize, wsize, PS, KS, skipOut[s], out[s + 1][0], dSkipBias[s], dSkipWoi[s], bnet[s + 1][0],
				mean[s + 1][0], var[s + 1][0], gamma[s + 1][0], beta[s + 1][0], eta);
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
		// 畳み込み+ReLU(CONV回分)
		for (c = CONV - 1; c >= 0; c--) {
			hsize *= st;
			wsize *= st;
			BBackConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS,
				out[s][c], out[s][c + 1], dBias[s][c], dWoi[s][c], bnet[s][c + 1], mean[s][c + 1], var[s][c + 1],
				gamma[s][c + 1], beta[s][c + 1], eta);
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

	// InConv
	BInBackConv(ICH, chN[0][0], delta[0][0], H, W, st, IKS, in, dInBias, dInWoi, out[0][0], bnet[0][0], mean[0][0], var[0][0], gamma[0][0], beta[0][0], eta);

	// 重みバイアス修正(自分で追加)
	for (hsize = H - IKS + 1, wsize = W - IKS + 1, s = 0; s < SEG; s++, hsize /= PS, wsize /= PS) {
		for (w = hsize * wsize * BS, c = 0; c < CONV; c++) {
			for (j = 0; j < chN[s][c + 1]; j++)
				for (bias[s][c][j] += -eta * (dBias[s][c][j] / w), i = 0; i < chN[s][c]; i++)
					for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
						woi[s][c][j][i][y][x] += -eta * (dWoi[s][c][j][i][y][x] / w);
		}
		w /= PS * PS;

		if (s < SEG - 1) {
			for (j = 0; j < chN[s + 1][0]; j++) {
				for (skipBias[s][j] += -eta * (dSkipBias[s][j] / w), i = 0; i < chN[s][c]; i++)
					for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
						skipWoi[s][j][i][y][x] += -eta * (dSkipWoi[s][j][i][y][x] / w);
			}
		}
	}
	w = (H - IKS + 1) * (W - IKS + 1) * BS;
	for (j = 0; j < chN[0][0]; j++) {
		for (inBias[j] += -eta * (dInBias[j] / w), i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				inWoi[j][i][y][x] += -eta * (dInWoi[j][i][y][x] / w);
	}
}
