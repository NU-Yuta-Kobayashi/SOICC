#pragma once
#define L 3			// �w�̐�
#define ETA (0.004)	// �w�K��(0.04)
#define BETA (0.1)	// �w�K��(0.1)
#define RELU 1		// ���ԑw�܂ł̊������֐�(0.�V�O���C�h�֐�, 1.ReLU)

#define NETMODEL 2		// 0:MLP, 1:SkipCNN, 2:SkipSMP, 3:SkipSMP-��ć@
#define LEARNRULE 1		// 0:�ʏ� 1:cosine decay
#define WARMUP 1		// �E�H�[���A�b�v(Epoc��)
#define WARMUPRATE 0.5	// �E�H�[���A�b�v�̌W��
#define RANDMODE 1		// ����ID�̐������@�@0:�ʏ�, 1:�z��

// ���x���X���[�W���O
#define LS 1		// 0:OFF, 1:ON
#define LSEPS 0.0045// �p�����[�^

// �o�b�`�w�K
#define BATCHLEARN 1// 0.�ʏ�w�K, 1.�o�b�`�w�K
#define BS 100		// �o�b�`�T�C�Y(32)
#define EPS 0.0000001// ��(�������Ŏg�p)
#define MRATE 0.9	 // �ړ����ώ��̊���

/* ------------------------------ �� �w�K�f�[�^�Ɉˑ�(��Main���ύX) �� ------------------------------ */
/* Optdigits */
//#define K 10		// �o�͐�
//#define IND 64	// ���̓f�[�^�̎���
//#define SIND 64	// ���̓f�[�^�̎���
//#define RIND 8	// ���̓f�[�^�̎����̕�����
//#define NMAX 128	// �e�w�̃m�[�h�̍ő吔
//#define ICH 1		// ���̓f�[�^�̃`�����l����
//#define H 8		// �c��
//#define W 8		// ����

/* MNIST */
//#define K 10	// �o�͐�
//#define IND 784	// ���̓f�[�^�̎�����
//#define SIND 784// ���̓f�[�^�̎�����
//#define RIND 28	// ���̓f�[�^�̎����̕�����
//#define NMAX 784// �e�w�̃m�[�h�̍ő吔
//#define ICH 1	// ���̓f�[�^�̃`�����l����
//#define H 28	// �c��
//#define W 28	// ����
//#define IKS 5	// 5�~5 No-Padding

/* Cifar-10 */
#define K 10		// �o�͐�
#define IND 3072	// ���̓f�[�^�̎�����
#define SIND 1024	// 1�����̓��̓f�[�^
#define RIND 32		// ���̓f�[�^�̎����̕�����
#define NMAX 3072	// �e�w�̃m�[�h�̍ő吔
#define ICH 3		// ���̓f�[�^�̃`�����l����
#define H 32		// �c��
#define W 32		// ����
#define IKS 1		// 1�~1 No-Padding
/* ------------------------------ �� �w�K�f�[�^�Ɉˑ�(��Main���ύX) �� ------------------------------ */

void Forward(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][NMAX], double mean[][NMAX],
	double var[][NMAX], double gamma[][NMAX], double beta[][NMAX]);
void BForward(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][BS][NMAX], double bnet[][NMAX][BS],
	double mean[][NMAX], double var[][NMAX], double gamma[][NMAX], double beta[][NMAX], double mmean[][NMAX], double mvar[][NMAX]);
void BackProp(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][NMAX], double tk[], double delta[][NMAX], double eta);
void BBackProp(int nodeN[], double bias[][NMAX], double woi[][NMAX][NMAX], double out[][BS][NMAX], double tk[][K], double delta[][NMAX][BS],
	double bnet[][NMAX][BS], double mean[][NMAX], double var[][NMAX], double gamma[][NMAX], double beta[][NMAX], double eta);