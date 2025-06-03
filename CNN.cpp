#include<math.h> 
#include"MLP.h"
#include"CNN.h"

double Pin[CHMAX][H + KS - 1][W + KS - 1];			// �p�f�B���O��̓��̓f�[�^
double Pdelta[CHMAX][H + KS - 1][W + KS - 1];		// �p�f�B���O��̃f���^ 
double BPin[CHMAX][BS][H + KS - 1][W + KS - 1];		// �p�f�B���O��̓��̓f�[�^(�o�b�`�p)
double BPdelta[CHMAX][BS][H + KS - 1][W + KS - 1];	// �p�f�B���O��̃f���^(�o�b�`�p)
double D3a[BS][H][W];								// �r����

int CBatchNormal = 0;// �o�b�`�������@0:off, 1:on

/* -------------------- �� Forward �� -------------------- */
void CBatchNormalF(double bnet[][H][W], double anet[][H][W], double* mean, double* var, double gamma, double beta, int hsize, int wsize) {
	double sqvar, m = BS * hsize * wsize;
	int h, w, b;

	// ���όv�Z
	for (*mean = 0, b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) *mean += bnet[b][h][w];
	*mean /= m;
	// ���U
	for (*var = 0, b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) *var += (bnet[b][h][w] - *mean) * (bnet[b][h][w] - *mean);
	*var /= m;
	// ���K��
	for (sqvar = sqrt(*var + EPS), b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) anet[b][h][w] = (bnet[b][h][w] - *mean) / sqvar;
	// �ϔ{�E�ړ�
	for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) anet[b][h][w] = gamma * anet[b][h][w] + beta;
}

void InConv(int ich, int och, double bias[], double woi[][ICH][IKS][IKS], double in[][H][W], double out[][H][W], int ksize, double gamma[], double beta[], double mmean[], double mvar[]) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw;
	double sum;
	int oh = H - ksize + 1, ow = W - ksize + 1;

	// convolution
	for (j = 0; j < och; j++) {//�o�͑�
		for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
			for (out[j][h][w] = bias[j], i = 0; i < ich; i++) //���͑�
				for (y = 0; y < ksize; y++)for (x = 0; x < ksize; x++)//�����ɂ��Ƃ��̃��[�v
					out[j][h][w] += woi[j][i][y][x] * in[i][h + y][w + x];
	}
	//BN���Ă���Ȃ琳�������Ȃ���
	if (BATCHLEARN && CBatchNormal)
		for (j = 0; j < och; j++) //�o�͑���ReLU
			for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
				out[j][h][w] = (out[j][h][w] - mmean[j]) / sqrt(mvar[j] + EPS) * gamma[j] + beta[j];
	//ReLU
	for (j = 0; j < och; j++) //�o�͑���ReLU
		for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
			if (out[j][h][w] <= 0) out[j][h][w] = 0;
}

void BInConv(int ich, int och, double bias[], double woi[][ICH][IKS][IKS], double in[][BS][H][W], double out[][BS][H][W], int ksize,
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double mmean[], double mvar[]) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw, b;
	double sum;
	int oh = H - ksize + 1, ow = W - ksize + 1;

	// convolution
	for (j = 0; j < och; j++) {//�o�͑�
		for (b = 0; b < BS; b++)// �o�b�`��
			for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
				for (bnet[j][b][h][w] = bias[j], i = 0; i < ich; i++) //���͑�
					for (y = 0; y < ksize; y++)for (x = 0; x < ksize; x++)//�����ɂ��Ƃ��̃��[�v
						bnet[j][b][h][w] += woi[j][i][y][x] * in[i][b][h + y][w + x];
	}

	//ReLU
	if (CBatchNormal) {
		if (RELU) {// ReLU
			for (j = 0; j < och; j++) {// �o�̓m�[�h��
				// �o�b�`���K��
				CBatchNormalF(bnet[j], out[j], &mean[j], &var[j], gamma[j], beta[j], oh, ow);
				// ReLU
				for (b = 0; b < BS; b++) // �o�b�`��
					for (h = 0; h < oh; h++)// �摜�T�C�Y(����)
						for (w = 0; w < ow; w++)// �摜�T�C�Y(����)
							if (out[j][b][h][w] <= 0)out[j][b][h][w] = 0;
				// �ړ����ρE�ړ����U
				mmean[j] = mmean[j] * MRATE + (1.0 - MRATE) * mean[j];	// �ړ�����
				mvar[j] = mvar[j] * MRATE + (1.0 - MRATE) * var[j];		// �ړ����U
			}
		}
	}
	else {
		if (RELU)// ReLU
			for (j = 0; j < och; j++)// �o�̓m�[�h��
				for (b = 0; b < BS; b++)// �o�b�`��
					for (h = 0; h < oh; h++)// �摜�T�C�Y(����)
						for (w = 0; w < ow; w++) {// �摜�T�C�Y(����)
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
	for (i = 0; i < ich; i++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) Pin[i][h + p][w + p] = in[i][h][w];	// �f�[�^����

	// convolution
	for (j = 0; j < och; j++) {// �o�̓m�[�h��
		for (h = 0; h < hsize; h += st) {// �摜�T�C�Y(����)
			for (w = 0; w < wsize; w += st) {// �摜�T�C�Y(����)			
				for (sum = bias[j], i = 0; i < ich; i++) {// ���̓m�[�h��
				// ��ݍ���
					for (y = 0; y < ksize; y++) for (x = 0; x < ksize; x++)// �J�[�l���T�C�Y 
						sum += Pin[i][h + y][w + x] * woi[j][i][y][x];// ��ݍ���
				}
				out[j][h / st][w / st] = sum;
			}
		}
	}

	// �o�b�`���K��
	if (BATCHLEARN && CBatchNormal) {
		for (j = 0; j < och; j++)// �o�̓m�[�h��
			for (h = 0; h < hsize / st; h++)// �摜�T�C�Y(����)
				for (w = 0; w < wsize / st; w++)// �摜�T�C�Y(����)
					out[j][h][w] = (out[j][h][w] - mmean[j]) / sqrt(mvar[j] + EPS) * gamma[j] + beta[j];
	}

	// ReLU
	if (RELU)
		for (j = 0; j < och; j++)// �o�̓m�[�h��
			for (h = 0; h < hsize / st; h++)// �摜�T�C�Y(����)
				for (w = 0; w < wsize / st; w++)// �摜�T�C�Y(����)
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
	for (j = 0; j < och; j++) {// �o�̓m�[�h��
		for (b = 0; b < BS; b++) {// �o�b�`��
			for (h = 0; h < hsize; h += st) {// �摜�T�C�Y(����)
				for (w = 0; w < wsize; w += st) {// �摜�T�C�Y(����)			
					for (sum = bias[j], i = 0; i < ich; i++) {// ���̓m�[�h��
						// ��ݍ���
						for (y = 0; y < ksize; y++) for (x = 0; x < ksize; x++)// �J�[�l���T�C�Y 
							sum += BPin[i][b][h + y][w + x] * woi[j][i][y][x];// ��ݍ���
					}
					bnet[j][b][h / st][w / st] = sum;
				}
			}
		}
	}

	// �o�b�`���K��
	if (CBatchNormal) {
		if (RELU) {// ReLU
			for (j = 0; j < och; j++) {// �o�̓m�[�h��
				// �o�b�`���K��
				CBatchNormalF(bnet[j], out[j], &mean[j], &var[j], gamma[j], beta[j], hsize / st, wsize / st);
				// ReLU
				for (b = 0; b < BS; b++) // �o�b�`��
					for (h = 0; h < hsize / st; h++)// �摜�T�C�Y(����)
						for (w = 0; w < wsize / st; w++)// �摜�T�C�Y(����)
							if (out[j][b][h][w] <= 0)out[j][b][h][w] = 0;
				// �ړ����ρE�ړ����U
				mmean[j] = mmean[j] * MRATE + (1.0 - MRATE) * mean[j];	// �ړ�����
				mvar[j] = mvar[j] * MRATE + (1.0 - MRATE) * var[j];		// �ړ����U
			}
		}
	}
	else {
		if (RELU)// ReLU
			for (j = 0; j < och; j++)// �o�̓m�[�h��
				for (b = 0; b < BS; b++)// �o�b�`��
					for (h = 0; h < hsize / st; h++)// �摜�T�C�Y(����)
						for (w = 0; w < wsize / st; w++) {// �摜�T�C�Y(����)
							if (bnet[j][b][h][w] <= 0)out[j][b][h][w] = 0;
							else out[j][b][h][w] = bnet[j][b][h][w];
						}
	}
}

void SkipForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][H][W],
	double mout[], double gamma[][CONV + 1][CHMAX], double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX], 
	double in[ICH][H][W],double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][H][W],
	double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS]) {
	int i, j, s, c, w, h, n, fn, hsize = H, wsize = W, st = 1;// st:�X�g���C�h�T�C�Y
	double sum, max;
	
	// ����
	InConv(ICH, chN[0][0], inBias, inWoi, in, out[0][0], IKS, gamma[0][0], beta[0][0], mmean[0][0], mvar[0][0]);
	hsize = H - IKS + 1; wsize = W - IKS + 1;// padding �A���̏ꍇ

	// ���`�d
	for (s = 0; s < SEG; s++) {// segment
		// ��ݍ���+ReLU(1�Z�O�����g��)
		for (c = 0; c < CONV; c++) {
			Conv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS, 
				gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1]);
			hsize /= st;// �X�g���C�h�ɉ����ăT�C�Y�k��
			wsize /= st;// �X�g���C�h�ɉ����ăT�C�Y�k��
		}

		// Skip-Connection
		for (n = 0; n < chN[s][c]; n++) {
			for (h = 0; h < hsize; h++)
				for (w = 0; w < wsize; w++)
					skipOut[s][n][h][w] = out[s][c][n][h][w] + out[s][0][n][h][w];// Skip-Connection
		}

		// Max-Pooling
		if (s == SEG - 1) {// out[s][c] -> mout �S�����ɓ���� 
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
		else {// ���̃Z�O�����g��0�w��max���
			// ��ݍ���
			Conv(chN[s][c], chN[s + 1][0], skipBias[s], skipWoi[s], skipOut[s], out[s + 1][0], hsize, wsize, PS, KS, gamma[s + 1][0], beta[s + 1][0], mmean[s + 1][0], mvar[s + 1][0]);

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

void BSkipForward(int chN[][CONV + 1], double bias[][CONV][CHMAX], double woi[][CONV][CHMAX][CHMAX][KS][KS], double out[][CONV + 1][CHMAX][BS][H][W],
	double mout[][NMAX], double bnet[][CONV + 1][CHMAX][BS][H][W], double mean[][CONV + 1][CHMAX], double var[][CONV + 1][CHMAX], double gamma[][CONV + 1][CHMAX], 
	double beta[][CONV + 1][CHMAX], double mmean[][CONV + 1][CHMAX], double mvar[][CONV + 1][CHMAX],
	double in[ICH][BS][H][W], double inBias[CHMAX], double inWoi[CHMAX][ICH][IKS][IKS], double skipOut[SEG][CHMAX][BS][H][W],
	double skipBias[SEG][CHMAX], double skipWoi[SEG][CHMAX][CHMAX][KS][KS]) {
	int i, j, s, c, w, h, n, fn, b, hsize = H, wsize = W, st = 1;// st:�X�g���C�h�T�C�Y
	double sum, max;

	// ����
	BInConv(ICH, chN[0][0], inBias, inWoi, in, out[0][0], IKS, bnet[0][0], mean[0][0], var[0][0], gamma[0][0], beta[0][0], 
		mmean[0][0], mvar[0][0]);
	hsize = H - IKS + 1; wsize = W - IKS + 1;// padding �A���̏ꍇ

	// ���`�d
	for (s = 0; s < SEG; s++) {// segment
		// ��ݍ���+ReLU(1�Z�O�����g��)
		for (c = 0; c < CONV; c++) {
			/*BConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS,
				bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1]);*/
			BConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], out[s][c], out[s][c + 1], hsize, wsize, st, KS,
				bnet[s][c + 1], mean[s][c + 1], var[s][c + 1], gamma[s][c + 1], beta[s][c + 1], mmean[s][c + 1], mvar[s][c + 1]);


			hsize /= st;// �X�g���C�h�ɉ����ăT�C�Y�k��
			wsize /= st;// �X�g���C�h�ɉ����ăT�C�Y�k��
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
		else {// ���̃Z�O�����g��0�w��max���
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
void CBatchNormalB(double delta[][H][W], double bnet[][H][W], double mean, double var, double* gamma, double* beta, double eta, int hsize, int wsize) {
	int i, h, w, b;
	double d3aMean, f3d12a;
	double dBeta, dGamma, f7;
	double m = BS * hsize * wsize;

	// �X�V�̓r����
	for (f3d12a = 0, b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) f3d12a += (bnet[b][h][w] - mean) * (*gamma) * delta[b][h][w];
	f3d12a /= m;
	f7 = 1.0 / sqrt(var + EPS);

	for (d3aMean = 0, b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) {
		D3a[b][h][w] = f7 * ((*gamma * delta[b][h][w]) - (bnet[b][h][w] - mean) / (var + EPS) * f3d12a);
		d3aMean += D3a[b][h][w];
	}
	d3aMean /= m;

	// ���X�V
	for (dBeta = 0, b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) dBeta += delta[b][h][w];
	*beta += -eta * dBeta / m;

	// ���X�V
	for (dGamma = 0, b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) dGamma += (bnet[b][h][w] - mean) * f7 * delta[b][h][w]; // f9�Zd15
	*gamma += -eta * dGamma / m;

	// �X�V
	for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) delta[b][h][w] = D3a[b][h][w] - d3aMean;
}

void InBackConv(int ich, int och, double delta[][H][W], int hsize, int wsize,
	int st, int ksize, double in[][H][W], double dBias[], double dWoi[][ICH][IKS][IKS], double out[][H][W]) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw;
	double sum;
	int oh = (H - ksize + 1) / st, ow = (W - ksize + 1) / st;

	//���ꂽ���͑��̌덷�M���Ɋ������֐��̔�����K�p(outDelta��ReLU�߂�)
	for (j = 0; j < och; j++) {
		for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
			if (out[j][h][w] <= 0) delta[j][h][w] *= 0;
	}

	for (j = 0; j < och; j++) {//�o�͑�
		for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
			for (dBias[j] += delta[j][h][w], i = 0; i < ich; i++) //���͑�
				for (y = 0; y < ksize; y++)for (x = 0; x < ksize; x++)//�����ɂ��Ƃ��̃��[�v
					dWoi[j][i][y][x] += in[i][h + y][w + x] * delta[j][h / st][w / st];// ��ݍ���
	}
}

void BInBackConv(int ich, int och, double delta[][BS][H][W], int hsize, int wsize,
	int st, int ksize, double in[][BS][H][W], double dBias[], double dWoi[][ICH][IKS][IKS], double out[][BS][H][W],
	double bnet[][BS][H][W], double mean[], double var[], double gamma[], double beta[], double eta) {
	int i, j, h, w, y, x, b, p = (ksize - 1) / 2, ph, pw;
	double sum;
	int oh = (H - ksize + 1) / st, ow = (W - ksize + 1) / st;

	//���ꂽ���͑��̌덷�M���Ɋ������֐��̔�����K�p(outDelta��ReLU�߂�)
	for (j = 0; j < och; j++) {
		for (b = 0; b < BS; b++) {
			for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
				if (out[j][b][h][w] <= 0) delta[j][b][h][w] = 0;
		}
		// �o�b�`������
		if (CBatchNormal) CBatchNormalB(delta[j], bnet[j], mean[j], var[j], &gamma[j], &beta[j], eta, oh, ow);
	}

	for (j = 0; j < och; j++) {//�o�͑�
		for (b = 0; b < BS; b++) {
			for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
				for (dBias[j] += delta[j][b][h][w], i = 0; i < ich; i++) //���͑�
					for (y = 0; y < ksize; y++)for (x = 0; x < ksize; x++)//�����ɂ��Ƃ��̃��[�v
						dWoi[j][i][y][x] += in[i][b][h + y][w + x] * delta[j][b][h / st][w / st];// ��ݍ���
		}
	}
}

void BackConv(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double indelta[][H][W], double delta[][H][W], int hsize, int wsize,
	int st, int ksize, double in[][H][W], double dBias[], double dWoi[][CHMAX][KS][KS],double out[][H][W]) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw;
	double sum;
	int oh = hsize / st, ow = wsize / st;
	//���ꂽ���͑��̌덷�M���Ɋ������֐��̔�����K�p(	outDelta��ReLU�߂�)
	for (j = 0; j < och; j++) {
		for (h = 0; h < oh; h++)for (w = 0; w < ow; w++)
			if (out[j][h][w] <= 0) delta[j][h][w] *= 0;
	}

	// 0 padding
	ph = hsize + ksize - 1;
	pw = wsize + ksize - 1;
	for (i = 0; i < ich; i++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) Pdelta[i][h][w] = 0;						// �f���^ 0-padding
	for (i = 0; i < ich; i++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) Pin[i][h][w] = 0;							// ���� 0-padding
	for (i = 0; i < ich; i++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) Pin[i][h + p][w + p] = in[i][h][w];	// �f�[�^����

	// convolution
	for (j = 0; j < och; j++) {// �o�̓m�[�h��
		for (h = 0; h < hsize; h += st) {// �摜�T�C�Y(����)
			for (w = 0; w < wsize; w += st) {// �摜�T�C�Y(����)			
				for (dBias[j] += delta[j][h/st][w/st], i = 0; i < ich; i++) {// ���̓m�[�h��
					// ��ݍ���
					for (y = 0; y < ksize; y++) for (x = 0; x < ksize; x++) {// �J�[�l���T�C�Y 
						Pdelta[i][h + y][w + x] += woi[j][i][y][x] * delta[j][h / st][w / st];// ��ݍ���
						dWoi[j][i][y][x] += Pin[i][h + y][w + x] * delta[j][h / st][w / st];// ��ݍ���
					}
				}
			}
		}
	}

	// ReLU�߂�
	for (i = 0; i < ich; i++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
		if (in[i][h][w] > 0)indelta[i][h][w] = Pdelta[i][h + p][w + p];	// �f���^�̃p�f�B���O�폜
		else indelta[i][h][w] = 0;
}

void BBackConv(int ich, int och, double bias[], double woi[][CHMAX][KS][KS], double indelta[][BS][H][W], double delta[][BS][H][W], int hsize, int wsize,
	int st, int ksize, double in[][BS][H][W], double out[][BS][H][W], double dBias[], double dWoi[][CHMAX][KS][KS], double bnet[][BS][H][W], double mean[], double var[], double gamma[],
	double beta[], double eta) {
	int i, j, h, w, y, x, p = (ksize - 1) / 2, ph, pw, b;
	double sum;

	// ReLU�߂�
	for (j = 0; j < och; j++) {
		for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++)
			if (out[j][b][h][w] <= 0)delta[j][b][h][w] = 0;// ReLU
		// �o�b�`���K��
		if (CBatchNormal)CBatchNormalB(delta[j], bnet[j], mean[j], var[j], &gamma[j], &beta[j], eta, hsize / st, wsize / st);
	}

	// 0 padding
	ph = hsize + ksize - 1;
	pw = wsize + ksize - 1;
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) BPdelta[i][b][h][w] = 0;						// �f���^ 0-padding
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)for (h = 0; h < ph; h++)for (w = 0; w < pw; w++) BPin[i][b][h][w] = 0;							// ���� 0-padding
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) BPin[i][b][h + p][w + p] = in[i][b][h][w];	// �f�[�^����

	// convolution
	for (j = 0; j < och; j++) {// �o�̓m�[�h��
		for (b = 0; b < BS; b++) {
			for (h = 0; h < hsize; h += st) {// �摜�T�C�Y(����)
				for (w = 0; w < wsize; w += st) {// �摜�T�C�Y(����)			
					for (dBias[j] += delta[j][b][h / st][w / st], i = 0; i < ich; i++) {// ���̓m�[�h��
						// ��ݍ���
						for (y = 0; y < ksize; y++) for (x = 0; x < ksize; x++) {// �J�[�l���T�C�Y 
							BPdelta[i][b][h + y][w + x] += woi[j][i][y][x] * delta[j][b][h / st][w / st];// ��ݍ���
							dWoi[j][i][y][x] += BPin[i][b][h + y][w + x] * delta[j][b][h / st][w / st];// ��ݍ���
						}
					}
				}
			}
		}
	}

	// �p�f�B���O�߂�
	for (i = 0; i < ich; i++)for (b = 0; b < BS; b++)for (h = 0; h < hsize; h++)for (w = 0; w < wsize; w++) {
		indelta[i][b][h][w] = BPdelta[i][b][h + p][w + p];	// �f���^�̃p�f�B���O�폜(�߂�)
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

	// �d�ݏC���ʂ�0������
	hsize = (H - IKS + 1) / pow(PS, SEG);
	wsize = (W - IKS + 1) / pow(PS, SEG);

	// ������
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++) {
		for (j = 0; j < chN[s][c + 1]; j++)
			for (dBias[s][c][j] = 0, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					dWoi[s][c][j][i][y][x] = 0; //������
	}
	for (c = CONV, s = 0; s < SEG; s++) {
		for (j = 0; j < CHMAX; j++)for (dSkipBias[s][j] = 0, i = 0; i < chN[s][c]; i++)
			for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
				dSkipWoi[s][j][i][y][x] = 0; //������
	}
	for (j = 0; j < chN[0][0]; j++) {
		for (dInBias[j] = 0, i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				dInWoi[j][i][y][x] = 0; //������
	}

	for (s = SEG - 1; s >= 0; s--) {// segment
		// �T�C�Y�߂�
		hsize *= PS;// �摜�T�C�Y
		wsize *= PS;// �摜�T�C�Y
		c = CONV;
		// Max Pooling
		if (s == SEG - 1) {// �S�����̃f���^������ delta[s][CONV]=mdelta[]
			for (oN = 0, iN = 0; iN < chN[s][c]; iN++) {
				for (h = 0; h < hsize; h += PS) {
					for (w = 0; w < wsize; w += PS,oN++) {
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

		// Skip-Connection
		for (n = 0; n < chN[s][c]; n++) {
			for (h = 0; h < hsize; h++) {
				for (w = 0; w < wsize; w++) {
					delta[s][c][n][h][w] = skipDelta[s][n][h][w];
				}
			}
		}

		// ��ݍ���+ReLU(CONV��)
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

	// �d�݃o�C�A�X�C��(�����Œǉ�)
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

	// �d�ݏC���ʂ�0������
	hsize = (H - IKS + 1) / pow(PS, SEG);
	wsize = (W - IKS + 1) / pow(PS, SEG);

	// ������
	for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++) {
		for (j = 0; j < chN[s][c + 1]; j++)
			for (dBias[s][c][j] = 0, i = 0; i < chN[s][c]; i++)
				for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
					dWoi[s][c][j][i][y][x] = 0; //������
	}
	for (c = CONV, s = 0; s < SEG; s++) {
		for (j = 0; j < CHMAX; j++)for (dSkipBias[s][j] = 0, i = 0; i < chN[s][c]; i++)
			for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
				dSkipWoi[s][j][i][y][x] = 0; //������
	}
	for (j = 0; j < chN[0][0]; j++) {
		for (dInBias[j] = 0, i = 0; i < ICH; i++)
			for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
				dInWoi[j][i][y][x] = 0; //������
	}

	// ����
	for (s = SEG - 1; s >= 0; s--) {
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
		// ��ݍ���+ReLU(CONV��)
		for (c = CONV - 1; c >= 0; c--) {
			hsize *= st;
			wsize *= st;
			BBackConv(chN[s][c], chN[s][c + 1], bias[s][c], woi[s][c], delta[s][c], delta[s][c + 1], hsize, wsize, st, KS,
				out[s][c], out[s][c + 1], dBias[s][c], dWoi[s][c], bnet[s][c + 1], mean[s][c + 1], var[s][c + 1],
				gamma[s][c + 1], beta[s][c + 1], eta);
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

	// InConv
	BInBackConv(ICH, chN[0][0], delta[0][0], H, W, st, IKS, in, dInBias, dInWoi, out[0][0], bnet[0][0], mean[0][0], var[0][0], gamma[0][0], beta[0][0], eta);

	// �d�݃o�C�A�X�C��(�����Œǉ�)
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
