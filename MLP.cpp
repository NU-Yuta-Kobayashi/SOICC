#include"MLP.h"
#include<stdio.h>
#include<stdlib.h>
#include<math.h>

// �w�K�ݒ�
int Softmax = 1;	// �o�͑w�̊������֐�(0.�V�O���C�h�֐�, 1.Softmax)
int BatchNormal = 1;// �o�b�`���K��

// �O���[�o���ϐ�
double Anet[L][NMAX][BS];// �o�b�`��������̏o��


void BatchNormalF(double bnet[], double anet[], double* mean, double* var, double gamma, double beta);// �A�h���X�n�����Ă���҂͒l������������

void BatchNormalB(double delta[], double bnet[], double mean, double var, double* gamma, double* beta, double eta);// �A�h���X�n�������Ă��Ȃ����̂͒l���Q�Ƃ��邾��

void Forward(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][NMAX], 
	double mean[][NMAX], double var[][NMAX], double gamma[][NMAX], double beta[][NMAX]) {
	int i, j, l;
	double sum;

	// ���`�d
	for (l = 0; l < L - 2; l++) {// ��ԊO���̒��ԑw�܂�
		for (j = 0; j < nodeN[l + 1]; j++) {
			for (out[l + 1][j] = bias[l][j], i = 0; i < nodeN[l]; i++)// �d�݁~���͒l�̑�������
				out[l + 1][j] += out[l][i] * woi[l][j][i];
			// �o�b�`���K��
			if (BATCHLEARN && BatchNormal) out[l + 1][j] = (out[l + 1][j] - mean[l + 1][j]) / sqrt(var[l + 1][j] + EPS) * gamma[l + 1][j] + beta[l + 1][j];
			// �������֐�
			if (RELU) {
				if (out[l + 1][j] <= 0)// 0�ȉ��Ȃ�0, ����ȊO�͂��̂܂�
					out[l + 1][j] = 0;
			}
			else
				out[l + 1][j] = 1.0 / (1.0 + exp(-out[l + 1][j])); // �V�O���C�h�֐�
		}
	}

	// �o�͑w�̌v�Z
	for (j = 0; j < nodeN[l + 1]; j++) {
		for (out[l + 1][j] = bias[l][j], i = 0; i < nodeN[l]; i++)
			out[l + 1][j] += out[l][i] * woi[l][j][i];
	}

	// �������֐�
	if (Softmax) {// Softmax
		for (sum = 0, j = 0; j < nodeN[l + 1]; j++)
			sum += exp(out[l + 1][j]);
		for (j = 0; j < nodeN[l + 1]; j++)
			out[l + 1][j] = exp(out[l + 1][j]) / sum;
	}
	else {// Sigmoid
		for (j = 0; j < nodeN[l + 1]; j++)
			out[l + 1][j] = 1.0 / (1.0 + exp(-out[l + 1][j]));
	}
}

void BForward(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][BS][NMAX], double bnet[][NMAX][BS],
	double mean[][NMAX], double var[][NMAX], double gamma[][NMAX], double beta[][NMAX], double mmean[][NMAX], double mvar[][NMAX]) {
	int i, j, l, b;
	double sum;

	// ���`�d
	for (l = 0; l < L - 2; l++) {// ��ԊO���̒��ԑw�܂�
		for (j = 0; j < nodeN[l + 1]; j++) {
			for (b = 0; b < BS; b++) {
				for (bnet[l + 1][j][b] = bias[l][j], i = 0; i < nodeN[l]; i++) {
					bnet[l + 1][j][b] += out[l][b][i] * woi[l][j][i];
				}
			}

			// �o�b�`������
			BatchNormalF(bnet[l + 1][j], Anet[l + 1][j], &mean[l + 1][j], &var[l + 1][j], gamma[l + 1][j], beta[l + 1][j]);
			mmean[l + 1][j] = mmean[l + 1][j] * MRATE + (1.0 - MRATE) * mean[l + 1][j];
			mvar[l + 1][j] = mvar[l + 1][j] * MRATE + (1.0 - MRATE) * var[l + 1][j];

			// �������֐�
			for (b = 0; b < BS; b++) {// �o�b�`���Ń��[�v������
				if (RELU) {
					if (Anet[l + 1][j][b] <= 0)// 0�ȉ��Ȃ�0, ����ȊO�͂��̂܂�
						out[l + 1][b][j] = 0;
					else
						out[l + 1][b][j] = Anet[l + 1][j][b];
				}
				else
					out[l + 1][b][j] = 1.0 / (1.0 + exp(-Anet[l + 1][j][b])); // �V�O���C�h�֐�
			}
		}
	}

	// �o�͑w�̌v�Z
	for (b = 0; b < BS; b++) {// �o�b�`����
		for (j = 0; j < nodeN[l + 1]; j++)
			for (out[l + 1][b][j] = bias[l][j], i = 0; i < nodeN[l]; i++)
				out[l + 1][b][j] += out[l][b][i] * woi[l][j][i];
	}

	// �������֐�
	if (Softmax) {// Softmax
		for (b = 0; b < BS; b++) {// �o�b�`����
			for (sum = 0, j = 0; j < nodeN[l + 1]; j++)
				sum += exp(out[l + 1][b][j]);
			for (j = 0; j < nodeN[l + 1]; j++)
				out[l + 1][b][j] = exp(out[l + 1][b][j]) / sum;
		}
	}
	else {// Sigmoid
		for (b = 0; b < BS; b++)// �o�b�`����
			for (j = 0; j < nodeN[l + 1]; j++)
				out[l + 1][b][j] = 1.0 / (1.0 + exp(-out[l + 1][b][j]));
	}
}

void BackProp(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][NMAX], double tk[], double delta[][NMAX], double eta) {
	int  i, j, k, l;
	double sum;

	// �o�͑w����B��w
	if (Softmax) {// Softmax
		for (k = 0; k < nodeN[L - 1]; k++)// Softmax+CrossEntropy�̔���
			delta[L - 1][k] = out[L - 1][k] - tk[k];
	}
	else {// Sigmoid
		for (k = 0; k < nodeN[L - 1]; k++) // Sigmoid+���덷
			delta[L - 1][k] = (out[L - 1][k] - tk[k]) * (1 - out[L - 1][k]) * out[L - 1][k];
	}

	// �B��w����B��w or �B��w������͑w
	for (l = L - 2; l > 0; l--) {
		for (j = 0; j < nodeN[l]; j++) {
			for (sum = 0, k = 0; k < nodeN[l + 1]; k++)
				sum += delta[l + 1][k] * woi[l][k][j];
			if (RELU) {// ReLU
				if (out[l][j] > 0) delta[l][j] = sum;
				else delta[l][j] = 0;
			}
			else// Sigmoid
				delta[l][j] = sum * (1.0 - out[l][j]) * out[l][j];
		}
	}

	// 0�w��Delta�쐬
	if (NETMODEL == 1 || NETMODEL == 2 || NETMODEL == 3) {// CNN
		for (j = 0; j < nodeN[l]; j++) {
			for (sum = 0, k = 0; k < nodeN[l + 1]; k++)
				sum += delta[l + 1][k] * woi[l][k][j];
			if (RELU) {// ReLU
				if (out[l][j] > 0) delta[l][j] = sum;
				else delta[l][j] = 0;
			}
			else// Sigmoid
				delta[l][j] = sum * (1.0 - out[l][j]) * out[l][j];
		}
	}

	// �d�݁E�o�C�A�X
	for (l = L - 2; l >= 0; l--) {
		for (j = 0; j < nodeN[l + 1]; j++) {
			bias[l][j] += -eta * delta[l + 1][j];
			for (i = 0; i < nodeN[l]; i++)
				woi[l][j][i] += -eta * delta[l + 1][j] * out[l][i];
		}
	}
}

void BBackProp(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][BS][NMAX], double tk[][K], double delta[][NMAX][BS],
	double bnet[][NMAX][BS], double mean[][NMAX], double var[][NMAX], double gamma[][NMAX], double beta[][NMAX], double eta) {
	int  i, j, k, l, b;
	double sum;

	// �o�͑w����B��w
	if (Softmax) {// Softmax
		for (k = 0; k < nodeN[L - 1]; k++)	// Softmax+CrossEntropy�̔���
			for (b = 0; b < BS; b++)		// �o�b�`����
				delta[L - 1][k][b] = out[L - 1][b][k] - tk[b][k];
	}
	else {//Sigmoid
		for (k = 0; k < nodeN[L - 1]; k++)	// sigmoid+���덷
			for (b = 0; b < BS; b++)		// �o�b�`����
				delta[L - 1][k][b] = (out[L - 1][b][k] - tk[b][k]) * (1 - out[L - 1][b][k]) * out[L - 1][b][k];
	}

	// �B��w����B��w or �B��w������͑w
	for (l = L - 2; l > 0; l--) {//l-2;>0
		for (j = 0; j < nodeN[l]; j++) {
			for (b = 0; b < BS; b++) {// �o�b�`����
				for (sum = 0, k = 0; k < nodeN[l + 1]; k++)
					sum += delta[l + 1][k][b] * woi[l][k][j];
				if (RELU) {// ReLU
					if (out[l][b][j] > 0)
						delta[l][j][b] = sum * 1;
					else
						delta[l][j][b] = sum * 0;
				}
				else// Sigmoid
					delta[l][j][b] = sum * (1 - out[l][b][j]) * out[l][b][j];
			}
			BatchNormalB(delta[l][j], bnet[l][j], mean[l][j], var[l][j], &gamma[l][j], &beta[l][j], eta);
		}
	}

	// 0�w��Delta�쐬
	if (NETMODEL == 1|| NETMODEL == 2 || NETMODEL == 3) {// Delta��0�w�܂ō��
		for (j = 0; j < nodeN[l]; j++) {
			for (b = 0; b < BS; b++) {// �o�b�`����
				for (sum = 0, k = 0; k < nodeN[l + 1]; k++)
					sum += delta[l + 1][k][b] * woi[l][k][j];
				if (RELU) {// ReLU
					if (out[l][b][j] > 0)
						delta[l][j][b] = sum;
					else
						delta[l][j][b] = 0;
				}
				else// Sigmoid
					delta[l][j][b] = sum * (1 - out[l][b][j]) * out[l][b][j];
			}
			//BatchNormalB(delta[l][j], bnet[l][j], mean[l][j], var[l][j], &gamma[l][j], &beta[l][j], eta);
		}
	}
	
	// �d�݁E�o�C�A�X
	for (l = L - 2; l >= 0; l--) {
		for (j = 0; j < nodeN[l + 1]; j++) {
			for (sum = 0, b = 0; b < BS; b++)// Batchi�����~��
				sum += -eta * delta[l + 1][j][b];
			bias[l][j] += sum / BS;// Batchi�����~�ς�����1�񂾂��X�V
			for (i = 0; i < nodeN[l]; i++) {
				for (sum = 0, b = 0; b < BS; b++)// Batchi�����~��
					sum += -eta * delta[l + 1][j][b] * out[l][b][i];
				woi[l][j][i] += sum / BS;// Batchi�����~�ς�����1�񂾂��X�V
			}
		}
	}
}

void BatchNormalF(double bnet[], double anet[], double* mean, double* var, double gamma, double beta) {
	int b;

	// ���όv�Z
	for (*mean = 0, b = 0; b < BS; b++) *mean += bnet[b];
	*mean /= BS;

	// ���U
	for (*var = 0, b = 0; b < BS; b++) *var += (bnet[b] - *mean) * (bnet[b] - *mean);
	*var /= BS;

	// ���K��
	for (b = 0; b < BS; b++) anet[b] = (bnet[b] - *mean) / (sqrt(*var + EPS));

	// �ϔ{�E�ړ�
	for (b = 0; b < BS; b++) anet[b] = gamma * anet[b] + beta;
}

void BatchNormalB(double delta[], double bnet[], double mean, double var, double* gamma, double* beta, double eta) {
	int i, b;
	double d3a[BS], d3aMean, f3d12a;
	double dBeta, dGamma, f7;

	// �X�V�̓r����
	for (f3d12a = 0, b = 0; b < BS; b++) f3d12a += (bnet[b] - mean) * (*gamma) * delta[b];
	f3d12a /= BS;
	f7 = 1.0 / sqrt(var + EPS);

	for (d3aMean = 0, b = 0; b < BS; b++) {
		d3a[b] = f7 * ((*gamma * delta[b]) - (bnet[b] - mean) / (var + EPS) * f3d12a);
		d3aMean += d3a[b];
	}
	d3aMean /= BS;

	// ���X�V
	for (dBeta = 0, b = 0; b < BS; b++) dBeta += delta[b];
	*beta += -eta * dBeta / BS;

	// ���X�V
	for (dGamma = 0, b = 0; b < BS; b++) dGamma += (bnet[b] - mean) * f7 * delta[b]; // f9�Zd15
	*gamma += -eta * dGamma / BS;

	// �X�V
	for (b = 0; b < BS; b++) delta[b] = d3a[b] - d3aMean;
}
