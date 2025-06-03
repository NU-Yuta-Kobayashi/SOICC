#include <stdlib.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "MLP.h"
#include "CNN.h"
#include "SMP.h"

// ���[�h�ؑ�
int SMPBatchNormal = 0;// �o�b�`�������@0:off, 1:on ��CNN��ON��

/* -------------------- �� Forward �� -------------------- */
// SMPConv
void SMPConv2D(int ich, int och, double in[CHMAX][H][W], double out[CHMAX][H][W], double pi[CHMAX][NPMAX][SD], double ri[CHMAX][NPMAX],
	double wi[CHMAX][CHMAX][NPMAX], int np[CHMAX], double cie[H][W][SD], double dis[H][W][CHMAX][NPMAX], double g[H][W][CHMAX][NPMAX],
	int sumCount[H][W][CHMAX], int hsize, int wsize, double gamma[], double beta[], double mmean[], double mvar[]) {
	int i, j, h, w, m, d;
	double temp, sum;
	double hd, wd;
	// �����֐��v�Z
	for (h = 0; h < hsize; h++) {
		for (w = 0; w < wsize; w++) {
			for (i = 0; i < ich; i++) {
				for (sumCount[h][w][i] = 0, m = 0; m < np[i]; m++) {
					// L1����
					for (dis[h][w][i][m] = 0, d = 0; d < SD; d++) {// ��������
						temp = cie[h][w][d] - pi[i][m][d];	// �����v�Z(L1����)
						if (temp < 0)temp = -temp;			// �������](������)
						dis[h][w][i][m] += temp;			// �����̍��v(��������)
					}
					// �����֐�:g(x,p,r)
					g[h][w][i][m] = 1.0 - dis[h][w][i][m] / ri[i][m];	// �����֐�:g(x,p,r)
					if (g[h][w][i][m] <= 0)g[h][w][i][m] = 0;			// �͈͊O�Ȃ�0
					else sumCount[h][w][i]++;							// �e���_�̐����J�E���g
				}
			}
		}
	}

	// �o�͌v�Z
	for (j = 0; j < och; j++) {// �o�̓`�����l��
		for (h = 0; h < hsize; h++) {
			for (w = 0; w < wsize; w++) {
				for (out[j][h][w] = 0, i = 0; i < ich; i++) {// ���̓`�����l��
					// �����֐��~�d��
					if (sumCount[h][w][i] > 0) {
						for (sum = 0, m = 0; m < np[i]; m++) {
							sum += g[h][w][i][m] * wi[j][i][m];// �����֐��~�d��
						}
						out[j][h][w] += in[i][h][w] * (sum / sumCount[h][w][i]);// �o�͌v�Z(������)
					}
				}
			}
		}
	}

	// �o�b�`���K��
	if (BATCHLEARN && SMPBatchNormal) {
		for (j = 0; j < och; j++)// �o�̓m�[�h��
			for (h = 0; h < hsize; h++)// �摜�T�C�Y(����)
				for (w = 0; w < wsize; w++)// �摜�T�C�Y(����)
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
	int i, j, s, c, w, h, n, fn, hsize = H, wsize = W, st = 1;// st:�X�g���C�h�T�C�Y
	double sum, max;
	//Input
	InConv(ICH, chN[0][0], inBias, inWoi, in, out[0][0], IKS, gamma[0][0], beta[0][0], mmean[0][0], mvar[0][0]);
	hsize = H - IKS + 1; wsize = W - IKS + 1;// padding �A���̏ꍇ

	// ���`�d
	for (s = 0; s < SEG; s++) {// segment
		// ���͍��W�쐬
		for (h = 0; h < hsize; h++) {
			for (w = 0; w < wsize; w++) {
				cie[s][h][w][0] = (1.0 / hsize) * h + (1.0 / hsize) / 2.0; // �c���W
				cie[s][h][w][1] = (1.0 / wsize) * w + (1.0 / wsize) / 2.0; // �����W
			}
		}

		// ��ݍ���+ReLU(1�Z�O�����g��)
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

		// Max-Pooling(�X�g���C�hPS�̏�ݍ���)
		if (s == SEG - 1) {// Max-Pooling
			for (fn = 0, n = 0; n < chN[s][c]; n++) {
				for (h = 0; h < hsize; h += PS) {
					for (w = 0; w < wsize; w += PS) {
						// max�v�Z
						for (max = 0, i = 0; i < PS; i++)
							for (j = 0; j < PS; j++)
								if (max < skipOut[s][n][h + i][w + j])
									max = skipOut[s][n][h + i][w + j];
						// �S�����̓���
						mout[fn] = max;
						fn++;
					}
				}
			}
		}
		else {// �X�g���C�hPS�̏�ݍ���
			Conv(chN[s][c], chN[s + 1][0], skipBias[s], skipWoi[s], skipOut[s], out[s + 1][0], hsize, wsize, PS, KS,
				gamma[s + 1][0], beta[s + 1][0], mmean[s + 1][0], mvar[s + 1][0]);

			// Skip-Connection
			for (n = 0; n < chN[s][c]; n++) {
				for (h = 0; h < hsize; h += PS) {
					for (w = 0; w < wsize; w += PS) {
						// ����
						for (sum = 0, j = 0; j < PS; j++)for (i = 0; i < PS; i++)
							sum += skipOut[s][n][h + j][w + i];
						// Skip-Connection
						out[s + 1][0][n][h / PS][w / PS] += sum / (PS * PS);
					}
				}
			}
		}
		hsize /= PS;// �摜�T�C�Y
		wsize /= PS;// �摜�T�C�Y
	}
}

// BatchSMPConv
void BSMPConv2D(int ich, int och, double in[CHMAX][BS][H][W], double out[CHMAX][BS][H][W], double pi[CHMAX][NPMAX][SD],
	double ri[CHMAX][NPMAX], double wi[CHMAX][CHMAX][NPMAX], int np[CHMAX], double cie[H][W][SD], double dis[H][W][CHMAX][NPMAX],
	double g[H][W][CHMAX][NPMAX], int sumCount[H][W][CHMAX], int hsize, int wsize, double bnet[][BS][H][W],
	double mean[], double var[], double gamma[], double beta[], double mmean[], double mvar[]) {
	int i, j, h, w, m, d, b;
	double temp, sum;

	// �����֐��v�Z
	for (h = 0; h < hsize; h++) {
		for (w = 0; w < wsize; w++) {
			for (i = 0; i < ich; i++) {
				for (sumCount[h][w][i] = 0, m = 0; m < np[i]; m++) {
					// L1����
					for (dis[h][w][i][m] = 0, d = 0; d < SD; d++) {// ��������
						temp = cie[h][w][d] - pi[i][m][d];	// �����v�Z(L1����)
						if (temp < 0)temp = -temp;			// �������](������)
						dis[h][w][i][m] += temp;			// �����̍��v(��������)
					}
					// �����֐�:g(x,p,r)
					g[h][w][i][m] = 1.0 - dis[h][w][i][m] / ri[i][m];	// �����֐�:g(x,p,r)
					if (g[h][w][i][m] <= 0)g[h][w][i][m] = 0;			// �͈͊O�Ȃ�0
					else sumCount[h][w][i]++;							// �e���_�̐����J�E���g
				}
			}
		}
	}

	// �o�͌v�Z
	for (j = 0; j < och; j++) {// �o�̓`�����l��
		for (b = 0; b < BS; b++) {// �o�b�`��
			for (h = 0; h < hsize; h++) {// �摜�T�C�Y(����)
				for (w = 0; w < wsize; w++) {// �摜�T�C�Y(����)
					for (bnet[j][b][h][w] = 0, i = 0; i < ich; i++) {// ���̓`�����l��
						// �����֐��~�d��
						if (sumCount[h][w][i] > 0) {
							for (sum = 0, m = 0; m < np[i]; m++) {
								sum += g[h][w][i][m] * wi[j][i][m];// �����֐��~�d��
							}
							bnet[j][b][h][w] += in[i][b][h][w] * (sum / sumCount[h][w][i]);// �o�͌v�Z(������)
						}
					}
				}
			}
		}
	}

	// �o�b�`���K��
	if (SMPBatchNormal) {
		if (RELU) {// ReLU
			for (j = 0; j < och; j++) {// �o�̓m�[�h��
				// �o�b�`���K��
				CBatchNormalF(bnet[j], out[j], &mean[j], &var[j], gamma[j], beta[j], hsize, wsize);
				// ReLU
				for (b = 0; b < BS; b++) // �o�b�`��
					for (h = 0; h < hsize; h++)// �摜�T�C�Y(����)
						for (w = 0; w < wsize; w++)// �摜�T�C�Y(����)
							if (out[j][b][h][w] <= 0)out[j][b][h][w] = 0;
				// �ړ����ρE�ړ����U
				mmean[j] = mmean[j] * MERRATE + (1.0 - MERRATE) * mean[j];	// �ړ�����
				mvar[j] = mvar[j] * MERRATE + (1.0 - MERRATE) * var[j];		// �ړ����U
			}
		}
	}
	else {
		// ReLU
		for (j = 0; j < och; j++) {// �o�̓m�[�h��
			for (b = 0; b < BS; b++)// �o�b�`��
				for (h = 0; h < hsize; h++)// �摜�T�C�Y(����)
					for (w = 0; w < wsize; w++) {// �摜�T�C�Y(����)
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
	int i, j, s, c, w, h, n, b, fn, hsize = H, wsize = W, st = 1;// st:�X�g���C�h�T�C�Y
	double sum, max;
	//BatchInput
	BInConv(ICH, chN[0][0], inBias, inWoi, in, out[0][0], IKS, bnet[0][0], mean[0][0], var[0][0], gamma[0][0], beta[0][0], mean[0][0], mvar[0][0]);
	hsize = H - IKS + 1; wsize = W - IKS + 1;// padding �A���̏ꍇ

	// ���`�d
	for (s = 0; s < SEG; s++) {// segment
		// ���͍��W�쐬
		for (h = 0; h < hsize; h++) {
			for (w = 0; w < wsize; w++) {
				cie[s][h][w][0] = (1.0 / hsize) * h + (1.0 / hsize) / 2.0; // �c���W
				cie[s][h][w][1] = (1.0 / wsize) * w + (1.0 / wsize) / 2.0; // �����W
			}
		}

		// ��ݍ���(BSMPConv)
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

		// Max-Pooling(�X�g���C�hPS�̏�ݍ���)
		if (s == SEG - 1) {// out[s][c] -> mout �S�����ɓ���� 
			for (b = 0; b < BS; b++) {
				for (fn = 0, n = 0; n < chN[s][c]; n++) {
					for (h = 0; h < hsize; h += PS) {
						for (w = 0; w < wsize; w += PS) {
							// max�v�Z
							for (max = 0, i = 0; i < PS; i++)
								for (j = 0; j < PS; j++)
									if (max < skipOut[s][n][b][h + i][w + j])
										max = skipOut[s][n][b][h + i][w + j];
							// �S�����̓���
							mout[b][fn] = max;
							fn++;
						}
					}
				}
			}
		}
		else {// �X�g���C�hPS�̏�ݍ���
			BConv(chN[s][c], chN[s + 1][0], skipBias[s], skipWoi[s], skipOut[s], out[s + 1][0], hsize, wsize, PS, KS,
				bnet[s + 1][0], mean[s + 1][0], var[s + 1][0], gamma[s + 1][0], beta[s + 1][0], mmean[s + 1][0], mvar[s + 1][0]);
			// Skip-Connection
			for (n = 0; n < chN[s][c]; n++) {
				for (b = 0; b < BS; b++) {
					for (h = 0; h < hsize; h += PS) {
						for (w = 0; w < wsize; w += PS) {
							// ����
							for (sum = 0, j = 0; j < PS; j++)for (i = 0; i < PS; i++)
								sum += skipOut[s][n][b][h + j][w + i];
							// Skip-Connection
							out[s + 1][0][n][b][h / PS][w / PS] += sum / (PS * PS);
						}
					}
				}
			}
		}
		hsize /= PS;// �摜�T�C�Y
		wsize /= PS;// �摜�T�C�Y
	}
}


/* -------------------- �� BackProp �� -------------------- */
// SMPBackConv
void SMPBackConv2D(int ich, int och, double in[CHMAX][H][W], double out[CHMAX][H][W], double pi[CHMAX][NPMAX][SD], double ri[CHMAX][NPMAX],
	double wi[CHMAX][CHMAX][NPMAX], int np[CHMAX], double inDelta[CHMAX][H][W], double outDelta[CHMAX][H][W],
	double cie[H][W][SD], double dis[H][W][CHMAX][NPMAX], double g[H][W][CHMAX][NPMAX], int sumCount[H][W][CHMAX],
	int hsize, int wsize, double dPi[CHMAX][NPMAX][SD], double dRi[CHMAX][NPMAX], double dWi[CHMAX][CHMAX][NPMAX],
	int wUseCount[CHMAX][CHMAX][NPMAX], int prUseCount[CHMAX][NPMAX]) {
	int i, j, k, x, m, d, h, w;
	double sum;

	// delta������
	for (i = 0; i < ich; i++) {// ���̓`�����l��
		for (h = 0; h < hsize; h++) {// �c�T�C�Y
			for (w = 0; w < wsize; w++) {// ���T�C�Y
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

	// �덷�t�`��
	for (j = 0; j < och; j++) {// �o�̓`�����l��
		for (h = 0; h < hsize; h++) {// �c�T�C�Y
			for (w = 0; w < wsize; w++) {// ���T�C�Y
				if (outDelta[j][h][w] != 0) {
					for (i = 0; i < ich; i++) {// ���̓`�����l��
						if (sumCount[h][w][i] > 0) {// �ߖT�U���_�̐������݂���
							for (sum = 0, m = 0; m < np[i]; m++) {// �U���_��
								if (g[h][w][i][m] > 0) {// �����֐�=0�͋ߖT�͈͊O�̂��ߎg�p���Ȃ�
									sum += g[h][w][i][m] * wi[j][i][m];
									// ���W�̕ω��ʁE���a�̕ω���
									for (d = 0; d < SD; d++) {
										if (cie[h][w][d] - pi[i][m][d] >= 0) {
											dPi[i][m][d] += outDelta[j][h][w] * in[i][h][w] * (wi[j][i][m] / sumCount[h][w][i]) * (1.0 / ri[i][m]);// ���W
											//dRi[i][m] += outDelta[j][h][w] * in[i][h][w] * (wi[j][i][m] / sumCount[h][w][i]) * (dis[h][w][i][m] / (ri[i][m] * ri[i][m]));// ���a
										}
										else {
											dPi[i][m][d] -= outDelta[j][h][w] * in[i][h][w] * (wi[j][i][m] / sumCount[h][w][i]) * (1.0 / ri[i][m]);// ���W
											//dRi[i][m] -= outDelta[j][h][w] * in[i][h][w] * (wi[j][i][m] / sumCount[h][w][i]) * (dis[h][w][i][m] / (ri[i][m] * ri[i][m]));// ���a
										}
									}
									dRi[i][m] += outDelta[j][h][w] * in[i][h][w] * (wi[j][i][m] / sumCount[h][w][i]) * (dis[h][w][i][m] / (ri[i][m] * ri[i][m]));// ���a
									// �d�݂̕ω���
									dWi[j][i][m] += outDelta[j][h][w] * in[i][h][w] * g[h][w][i][m] / sumCount[h][w][i];
									// �U���_�̎g�p�񐔑���
									wUseCount[j][i][m]++;
									prUseCount[i][m]++;
								}
							}
							// delta�v�Z
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

	// �d�ݏC���ʂ�0������
	hsize = (H - IKS + 1) / pow(PS, SEG);
	wsize = (W - IKS + 1) / pow(PS, SEG);

	/* ---------- CNN������ ---------- */
	// �d�݁E�o�C�A�X�̕ω���
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++) {
		for (j = 0; j < CHMAX; j++)
			for (dBias[s][c][j] = 0, i = 0; i < CHMAX; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					dWoi[s][c][j][i][y][x] = 0; //������
	}
	// �X�L�b�v�ڑ��̏d�݁E�o�C�A�X�̕ω���
	for (c = CONV, s = 0; s < SEG; s++) {
		for (j = 0; j < CHMAX; j++)for (dSkipBias[s][j] = 0, i = 0; i < CHMAX; i++)
			for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
				dSkipWoi[s][j][i][y][x] = 0; //������
	}
	// ���͏�ݍ��݂̏d�݁E�o�C�A�X�̕ω���
	for (j = 0; j < CHMAX; j++) {
		for (dInBias[j] = 0, i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				dInWoi[j][i][y][x] = 0; //������
	}

	/* ---------- SMP������ ---------- */
	// ���W�E���a�E�g�p�񐔏�����
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
	// �d�݁E�g�p�񐔏�����
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
		// �T�C�Y�߂�
		hsize *= PS;// �摜�T�C�Y
		wsize *= PS;// �摜�T�C�Y
		c = CONV;
		// Max Pooling
		if (s == SEG - 1) {// �S�����̃f���^������ delta[s][CONV]=mdelta[]
			for (oN = 0, iN = 0; iN < chN[s][c]; iN++) {
				for (h = 0; h < hsize; h += PS) {
					for (w = 0; w < wsize; w += PS, oN++) {
						// max�v�Z
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
						// delta���
						skipDelta[s][iN][nh][nw] = mdelta[oN];
					}
				}
			}
		}
		else {// out[s][c] -> out[s+1][0] ���̃Z�O�����g��0�w�ɓ����
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

		// �f���^���
		for (n = 0; n < chN[s][c]; n++) {
			for (h = 0; h < hsize; h++)
				for (w = 0; w < wsize; w++)
					delta[s][c][n][h][w] = skipDelta[s][n][h][w];
		}

		// ��ݍ���
		for (c = CONV - 1; c >= 0; c--) {
			SMPBackConv2D(chN[s][c], chN[s][c + 1], out[s][c], out[s][c + 1], pi[s][c], ri[s][c], wi[s][c], np[s][c], delta[s][c], delta[s][c + 1], cie[s], dis[s][c], g[s][c], sumCount[s][c], hsize, wsize,
				dPi[s][c], dRi[s][c], dWi[s][c], wUseCount[s][c], prUseCount[s][c]);
		}

		// �X�L�b�v�ڑ�
		for (n = 0; n < chN[s][0]; n++) {
			for (h = 0; h < hsize; h++)
				for (w = 0; w < wsize; w++)
					delta[s][0][n][h][w] += skipDelta[s][n][h][w];
		}
	}
	// InBackConv
	InBackConv(ICH, chN[0][0], delta[0][0], H, W, st, IKS, in, dInBias, dInWoi, out[0][0]);

	/* ---------- �X�V ---------- */
	// SkipConv�d�݁E�o�C�A�X
	for (hsize = H - IKS + 1, wsize = W - IKS + 1, s = 0; s < SEG - 1; s++, hsize /= PS, wsize /= PS) {
		c = CONV;
		w = hsize * wsize / (PS * PS);
		for (j = 0; j < chN[s + 1][0]; j++) {
			for (skipBias[s][j] += -eta * (dSkipBias[s][j] / w), i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					skipWoi[s][j][i][y][x] += -eta * (dSkipWoi[s][j][i][y][x] / w);
		}
	}
	// InConv�d�݁E�o�C�A�X
	w = (H - IKS + 1) * (W - IKS + 1);
	for (j = 0; j < chN[0][0]; j++) {
		for (inBias[j] += -eta * (dInBias[j] / w), i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				inWoi[j][i][y][x] += -eta * (dInWoi[j][i][y][x] / w);
	}
	// SMP�d��
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
	// SMP���W�E���a
	for (s = 0; s < SEG; s++) {
		for (c = 0; c < CONV; c++) {
			for (i = 0; i < chN[s][c]; i++) {
				for (m = 0; m < np[s][c][i]; m++) {
					// ���W�E���a�X�V
					if (prUseCount[s][c][i][m] > 0) {
						for (d = 0; d < SD; d++) {
							pi[s][c][i][m][d] += -eta * dPi[s][c][i][m][d] / prUseCount[s][c][i][m];
							dPsum[s][c][i][m][d] += dPi[s][c][i][m][d] / prUseCount[s][c][i][m];// �ω��ʂ𑫂�����(��Ď�@�p)
						}
						ri[s][c][i][m] += -eta * dRi[s][c][i][m] / prUseCount[s][c][i][m];
						dRsum[s][c][i][m] += dRi[s][c][i][m] / prUseCount[s][c][i][m];// �ω��ʂ𑫂�����(��Ď�@�p)
					}
					// �͈͓��ɃN���b�s���O
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

	// delta������
	for (i = 0; i < ich; i++) {// ���̓`�����l��
		for (b = 0; b < BS; b++)
			for (h = 0; h < hsize; h++)// �c�T�C�Y
				for (w = 0; w < wsize; w++)// ���T�C�Y
					inDelta[i][b][h][w] = 0;
	}

	// ReLU
	for (j = 0; j < och; j++) {
		for (b = 0; b < BS; b++) {
			for (h = 0; h < hsize; h++) for (w = 0; w < wsize; w++)
				if (out[j][b][h][w] <= 0)outDelta[j][b][h][w] = 0;// ReLU
			// �o�b�`���K��
			if (SMPBatchNormal)CBatchNormalB(outDelta[j], bnet[j], mean[j], var[j], &gamma[j], &beta[j], eta, hsize, wsize);
		}
	}

	// �덷�t�`��
	for (j = 0; j < och; j++) {// �o�̓`�����l��
		for (b = 0; b < BS; b++) {
			for (h = 0; h < hsize; h++) {// �c�T�C�Y
				for (w = 0; w < wsize; w++) {// ���T�C�Y
					if (outDelta[j][b][h][w] != 0) {
						for (i = 0; i < ich; i++) {// ���̓`�����l��
							if (sumCount[h][w][i] > 0) {// �ߖT�U���_�̐������݂���
								for (sum = 0, m = 0; m < np[i]; m++) {// �U���_��
									if (g[h][w][i][m] > 0) {// �����֐�=0�͋ߖT�͈͊O�̂��ߎg�p���Ȃ�
										sum += g[h][w][i][m] * wi[j][i][m];
										// ���W�̕ω��ʁE���a�̕ω���
										for (d = 0; d < SD; d++) {
											if (cie[h][w][d] - pi[i][m][d] >= 0) {
												dPi[i][m][d] += outDelta[j][b][h][w] * in[i][b][h][w] * (wi[j][i][m] / sumCount[h][w][i]) * (1.0 / ri[i][m]);// ���W
											}
											else {
												dPi[i][m][d] -= outDelta[j][b][h][w] * in[i][b][h][w] * (wi[j][i][m] / sumCount[h][w][i]) * (1.0 / ri[i][m]);// ���W
											}
										}
										dRi[i][m] += outDelta[j][b][h][w] * in[i][b][h][w] * (wi[j][i][m] / sumCount[h][w][i]) * (dis[h][w][i][m] / (ri[i][m] * ri[i][m]));// ���a
										// �d�݂̕ω���
										dWi[j][i][m] += outDelta[j][b][h][w] * in[i][b][h][w] * g[h][w][i][m] / sumCount[h][w][i];
										// �U���_�̎g�p�񐔑���
										wUseCount[j][i][m]++;
										prUseCount[i][m]++;
									}
								}
								// delta�v�Z
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

	// �d�ݏC���ʂ�0������
	hsize = (H - IKS + 1) / pow(PS, SEG);
	wsize = (W - IKS + 1) / pow(PS, SEG);

	/* ---------- CNN������ ---------- */
	// �d�݁E�o�C�A�X�̕ω���
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++) {
		for (j = 0; j < CHMAX; j++)
			for (dBias[s][c][j] = 0, i = 0; i < CHMAX; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					dWoi[s][c][j][i][y][x] = 0; //������
	}
	// �X�L�b�v�ڑ��̏d�݁E�o�C�A�X�̕ω���
	for (c = CONV, s = 0; s < SEG; s++) {
		for (j = 0; j < CHMAX; j++)for (dSkipBias[s][j] = 0, i = 0; i < CHMAX; i++)
			for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
				dSkipWoi[s][j][i][y][x] = 0; //������
	}
	// ���͏�ݍ��݂̏d�݁E�o�C�A�X�̕ω���
	for (j = 0; j < CHMAX; j++) {
		for (dInBias[j] = 0, i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				dInWoi[j][i][y][x] = 0; //������
	}

	/* ---------- SMP������ ---------- */
	// ���W�E���a�E�g�p�񐔏�����
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
	// �d�݁E�g�p�񐔏�����
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
		// �T�C�Y�߂�
		hsize *= PS;// �摜�T�C�Y
		wsize *= PS;// �摜�T�C�Y
		c = CONV;
		// Max Pooling
		if (s == SEG - 1) {// �S�����̃f���^������ delta[s][CONV]=mdelta[]
			for (b = 0; b < BS; b++) {
				for (oN = 0, iN = 0; iN < chN[s][c]; iN++) {
					for (h = 0; h < hsize; h += PS) {
						for (w = 0; w < wsize; w += PS, oN++) {
							// max�v�Z
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
							// delta���
							skipDelta[s][iN][b][nh][nw] = mdelta[oN][b];
						}
					}
				}
			}
		}
		else {// out[s][c] -> out[s+1][0] ���̃Z�O�����g��0�w�ɓ����
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

		// �f���^���
		for (n = 0; n < chN[s][c]; n++) {
			for (b = 0; b < BS; b++) {
				for (h = 0; h < hsize; h++) {
					for (w = 0; w < wsize; w++) {
						delta[s][c][n][b][h][w] = skipDelta[s][n][b][h][w];
					}
				}
			}
		}

		// ��ݍ���(SMPBackConv)
		for (c = CONV - 1; c >= 0; c--) {
			BSMPBackConv2D(chN[s][c], chN[s][c + 1], out[s][c], out[s][c + 1], pi[s][c], ri[s][c], wi[s][c], np[s][c], delta[s][c], delta[s][c + 1], cie[s], dis[s][c], g[s][c], sumCount[s][c], hsize, wsize,
				dPi[s][c], dRi[s][c], dWi[s][c], wUseCount[s][c], prUseCount[s][c], bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], eta);
		}

		// �X�L�b�v�ڑ�
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

	/* ---------- �X�V ---------- */
	// SkipConv�d�݁E�o�C�A�X
	for (hsize = H - IKS + 1, wsize = W - IKS + 1, s = 0; s < SEG - 1; s++, hsize /= PS, wsize /= PS) {
		c = CONV;
		w = (hsize * wsize * BS) / (PS * PS);
		for (j = 0; j < chN[s + 1][0]; j++) {
			for (skipBias[s][j] += -eta * (dSkipBias[s][j] / w), i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					skipWoi[s][j][i][y][x] += -eta * (dSkipWoi[s][j][i][y][x] / w);
		}
	}
	// InConv�d�݁E�o�C�A�X
	w = (H - IKS + 1) * (W - IKS + 1) * BS;
	for (j = 0; j < chN[0][0]; j++) {
		for (inBias[j] += -eta * (dInBias[j] / w), i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				inWoi[j][i][y][x] += -eta * (dInWoi[j][i][y][x] / w);
	}
	// SMP�d��
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
	// SMP���W�E���a
	for (s = 0; s < SEG; s++) {
		for (c = 0; c < CONV; c++) {
			for (i = 0; i < chN[s][c]; i++) {
				for (m = 0; m < np[s][c][i]; m++) {
					// ���W�E���a�X�V
					if (prUseCount[s][c][i][m] != 0) {
						for (d = 0; d < SD; d++) {
							pi[s][c][i][m][d] += -eta * dPi[s][c][i][m][d] / prUseCount[s][c][i][m];
							dPsum[s][c][i][m][d] += dPi[s][c][i][m][d] / prUseCount[s][c][i][m];// �ω��ʂ𑫂�����(��Ď�@�p)
						}
						ri[s][c][i][m] += -eta * dRi[s][c][i][m] / prUseCount[s][c][i][m];
						dRsum[s][c][i][m] += dRi[s][c][i][m] / prUseCount[s][c][i][m];// �ω��ʂ𑫂�����(��Ď�@�p)
					}
					// �͈͓��ɃN���b�s���O
					if (ri[s][c][i][m] < LOWESTR)ri[s][c][i][m] = LOWESTR;
					else if (ri[s][c][i][m] > HIGHESTR)ri[s][c][i][m] = HIGHESTR;
				}
			}
		}
	}
}

/* -------------------- �� ��Ď�@ �� -------------------- */
// ������
void SOIPInit(double dPsum[SEG][CONV][CHMAX][NPMAX][SD], double dRsum[SEG][CONV][CHMAX][NPMAX]) {
	int s, c, i, m, d;
	for (s = 0; s < SEG; s++) {
		for (c = 0; c < CONV; c++)
			for (i = 0; i < CHMAX; i++)
				for (m = 0; m < NPMAX; m++) {
					// ������
					for (d = 0; d < SD; d++)dPsum[s][c][i][m][d] = 0;
					dRsum[s][c][i][m] = 0;
				}
	}
}

// �U���_�̑���
void SOIP(int chN[SEG][CONV + 1], double pi[SEG][CONV][CHMAX][NPMAX][SD], double ri[SEG][CONV][CHMAX][NPMAX],
	double wi[SEG][CONV][CHMAX][CHMAX][NPMAX], int np[SEG][CONV][CHMAX], int* allNP, int delC[SEG][CONV][CHMAX][NPMAX],
	double dPsum[SEG][CONV][CHMAX][NPMAX][SD], double dRsum[SEG][CONV][CHMAX][NPMAX], bool add[SEG][CONV][CHMAX], double delW, double eta,
	int* delN, int* merN, int* addN) {
	int i, j, k, s, c, m, n, d, id, minp;
	double temp, dis, max, r, sum, wmSum, wnSum, similarity, addRate, min;

	/* �U���_�̍폜 */
	if (POINTMODE == 0 || POINTMODE == 1) {
		for (s = 0; s < SEG; s++) {// �Z�O�����g��
			for (c = 0; c < CONV; c++)// ��ݍ��ݐ�
				for (i = 0; i < chN[s][c]; i++)// ���͐�
					if (np[s][c][i] > NPMIN) {
						// �d�݂̕��ϒl�v�Z
						for (sum = 0, m = 0, j = 0; j < chN[s][c + 1]; j++)
							if (wi[s][c][j][i][m] < 0)sum -= wi[s][c][j][i][m];
							else sum += wi[s][c][j][i][m];// ��Βl
						min = sum;
						for (minp = 0, m = 1; m < np[s][c][i]; m++) {// �U���_��
							// �d�݂̕��όv�Z
							for (sum = 0, j = 0; j < chN[s][c + 1]; j++)
								if (wi[s][c][j][i][m] < 0)sum -= wi[s][c][j][i][m];
								else sum += wi[s][c][j][i][m];// ��Βl
							if (min > sum) {
								min = sum;
								minp = m;
							}
						}
						// �폜����
						if (min / chN[s][c + 1]* ri[s][c][i][minp] <= delW) {// �d�݁E���a��臒l�ȉ�
							m = minp;
							delC[s][c][i][m]++;						// �J�E���g����
							if (delC[s][c][i][m] >= DELCOUNT) {		// ����񐔑I�����ꂽ��폜
								n = np[s][c][i] - 1;				// �Ō�̗v�f�ԍ�
								ri[s][c][i][m] = ri[s][c][i][n];	// ���a�폜(�㏑��)							
								for (d = 0; d < SD; d++)pi[s][c][i][m][d] = pi[s][c][i][n][d];			 // ���W�폜(�㏑��)
								for (j = 0; j < chN[s][c + 1]; j++)wi[s][c][j][i][m] = wi[s][c][j][i][n];// �d�ݍ폜(�㏑��)	
								for (d = 0; d < SD; d++)dPsum[s][c][i][m][d] = dPsum[s][c][i][n][d];	 // ���W�ω���(�㏑��)
								dRsum[s][c][i][m] = dRsum[s][c][i][n];									 // ���a�ω���(�㏑��)
								delC[s][c][i][m] = delC[s][c][i][n];// �J�E���g���폜(�㏑��)
								np[s][c][i]--;						// �U���_�����f�N�������g
								m--;								// �㏑�������ӏ����ēx�v�Z
								(*delN)++;							// �폜�J�E���g(���f���]���p)
							}
						}
						else {
							if (delC[s][c][i][m] >= 0)delC[s][c][i][m] = 0;// �J�E���g���Z�b�g(�ǉ����̗P�\��0�ɂ��Ȃ��悤�ɂ���)
						}

					}
		}
	}

	/* �U���_�̌��� */
	if (POINTMODE == 0 || POINTMODE == 2) {
		for (s = 0; s < SEG; s++) {// �Z�O�����g��
			for (c = 0; c < CONV; c++)// ��ݍ��ݐ�
				for (i = 0; i < chN[s][c]; i++)// ���͐�
					if (np[s][c][i] > NPMIN) {// �U���_�̍ŏ�����������
						for (m = 0; m < np[s][c][i]; m++) {// �U���_��(�)
							for (n = m + 1; n < np[s][c][i]; n++) {// �U���_��(��r)
								if (ri[s][c][i][m] - ri[s][c][i][n] >= -MERDIFF && ri[s][c][i][m] - ri[s][c][i][n] <= MERDIFF) {// �����@�F2�_�̔��a�̍�����臒l�ȉ�
									// L2����
									for (dis = 0, d = 0; d < SD; d++) {
										temp = (pi[s][c][i][m][d] - pi[s][c][i][n][d]) * (pi[s][c][i][m][d] - pi[s][c][i][n][d]);
										dis += temp;
									}
									dis = sqrt(dis);
									if (dis < MERRATE * ri[s][c][i][m] && dis < MERRATE * ri[s][c][i][n]) {// �����A�F���������l�ȓ�
										// �R�T�C���ގ��x
										for (sum = 0, wmSum = 0, wnSum = 0, j = 0; j < chN[s][c + 1]; j++) {// �o�͐�
											sum += wi[s][c][j][i][m] * wi[s][c][j][i][n];// �d�݃x�N�g��m * �d�݃x�N�g��n
											wmSum += wi[s][c][j][i][m] * wi[s][c][j][i][m];// �d�݃x�N�g��m��2��̍��v
											wnSum += wi[s][c][j][i][n] * wi[s][c][j][i][n];// �d�݃x�N�g��n��2��̍��v
										}
										wmSum = sqrt(wmSum); wnSum = sqrt(wnSum);// ������
										similarity = sum / (wmSum * wnSum);		// �R�T�C���ގ��x�v�Z
										if (similarity >= MERSIMI) {// �����B�F�d�݂ɗގ���������
											/* �m�[�h���� */
											// �d��
											for (j = 0; j < chN[s][c + 1]; j++) {// �o�͐�
												//wi[s][c][j][i][m] = (wi[s][c][j][i][m] + wi[s][c][j][i][n]) / (2.0 - (1.0 - similarity));// ����(�ގ��x��������Ί���l���傫���Ȃ�:2.0�`1.0)
												wi[s][c][j][i][m] = wi[s][c][j][i][m] + wi[s][c][j][i][n];// ����(�ގ��x��������Ί���l���傫���Ȃ�:2.0�`1.0)
											}
											wmSum /= chN[s][c + 1]; wnSum /= chN[s][c + 1];// �d�݂̕���
											// ���a
											if (ri[s][c][i][m] < ri[s][c][i][n])ri[s][c][i][m] = ri[s][c][i][n];
											// ���W
											for (d = 0; d < SD; d++) {
												//pi[s][c][i][m][d] += (pi[s][c][i][n][d] - pi[s][c][i][m][d]) * wi[s][c][j][i][n] / (wi[s][c][j][i][m] + wi[s][c][j][i][n]);
												pi[s][c][i][m][d] += (pi[s][c][i][n][d] - pi[s][c][i][m][d]) * wnSum / (wmSum + wnSum);
											}
											delC[s][c][i][m] = ADDDEL;// �폜�J�E���g
											/* �m�[�h�폜 */
											k = np[s][c][i] - 1;
											for (j = 0; j < chN[s][c + 1]; j++)wi[s][c][j][i][n] = wi[s][c][j][i][k];// �d��
											ri[s][c][i][n] = ri[s][c][i][k];										// ���a
											for (d = 0; d < SD; d++)pi[s][c][i][n][d] = pi[s][c][i][k][d];			// ���W
											for (d = 0; d < SD; d++)dPsum[s][c][i][n][d] = dPsum[s][c][i][k][d];	// ���W�ω���(�㏑��)
											dRsum[s][c][i][n] = dRsum[s][c][i][k];									// ���a�ω���(�㏑��)
											delC[s][c][i][n] = delC[s][c][i][k];
											np[s][c][i]--;		// �U���_�����f�N�������g
											(*merN)++;			// �����J�E���g(���f���]���p)
											break;
										}
									}
								}
							}
						}
					}
		}
	}

	/* �U���_�̒ǉ� */
	if (POINTMODE == 0 || POINTMODE == 3) {
		for (s = 0; s < SEG; s++) {// �Z�O�����g��
			for (c = 0; c < CONV; c++)// ��ݍ��ݐ�
				for (i = 0; i < chN[s][c]; i++) {// ���͐�
					if (np[s][c][i] < NPMAX) {// �ǉ������𖞂���
						// 臒l�v�Z
						addRate = ADDRATEBASIS;
						//addRate = ADDRATEBASIS + (ADDRATERANGE * (np[s][c][i] - NPMIN) / (NPMAX - NPMIN));
						// ���W�̕ω��ʂ��ő�̔ԍ���ێ�
						for (id = -1, max = 0, m = 0; m < np[s][c][i]; m++) {// �U���_��
							// �����v�Z
							for (dis = 0, d = 0; d < SD; d++) dis += dPsum[s][c][i][m][d] * dPsum[s][c][i][m][d];
							dis = sqrt(dis);// ���[�N���b�h����
							// �ő�l�v�Z
							if (max <= dis) {
								max = dis;
								id = m;// �ő�l�̔ԍ�
							}
						}
						if (id != -1 && ((BATCHLEARN == 0 && max >= addRate * SOIPTIME) || (BATCHLEARN == 1 && max >= addRate * BSOIPTIME))) {
							if (-dRsum[s][c][i][id] > 0) {// ���a�𑝂₷�悤�ɍX�V���Ă���ꍇ
								// �d��
								for (j = 0; j < chN[s][c + 1]; j++) {
									wi[s][c][j][i][np[s][c][i]] = (rand() / (RAND_MAX + 1.0) - 0.5) * 0.6;

									//wi[s][c][j][i][np[s][c][i]] = delW + delW * ADDW;										// �폜�ŏ��l+�폜�ŏ��l�~���[�g
									//if (wi[s][c][j][i][id] < 0)wi[s][c][j][i][np[s][c][i]] = -wi[s][c][j][i][np[s][c][i]];	// �������]
								}
								// ���a(������)
								ri[s][c][i][np[s][c][i]] = R;
								// ���W
								for (d = 0; d < SD; d++) {
									pi[s][c][i][np[s][c][i]][d] = pi[s][c][i][id][d] - eta * dPsum[s][c][i][id][d];
									
									// �ǉ�����
									//if (eta * dPsum[s][c][i][id][d] >= 0)pi[s][c][i][np[s][c][i]][d] += (MERRATE * ri[s][c][i][np[s][c][i]]) / sqrt(2);	// �����͈͊O�ɂ���
									//else pi[s][c][i][np[s][c][i]][d] -= (MERRATE * ri[s][c][i][np[s][c][i]]) / sqrt(2);									// �����͈͊O�ɂ���
								}
								// �U���_�����p�����[�^
								delC[s][c][i][np[s][c][i]] = ADDDEL;// �폜�J�E���g
								np[s][c][i]++;						// �U���_������
								(*addN)++;							// �ǉ��J�E���g(���f���]���p)
							}
						}
					}
				}
		}
	}

	// �U���_���̍��v(�`��p)
	for (*allNP = 0, s = 0; s < SEG; s++) {// �Z�O�����g��
		for (c = 0; c < CONV; c++)// ��ݍ��ݐ�
			for (i = 0; i < chN[s][c]; i++)// ���͐�
				*allNP += np[s][c][i];
	}
}
