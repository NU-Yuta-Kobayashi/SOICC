#pragma once
// �A���S���Y��
#define POINTMODE 0 // �U���_�̃��[�h�@0:�S��, 1:�폜, 2:����, 3:�ǉ�
#define POINTINIT 0 // �U���_�̏��������[�h�@0:�����_��, 1:�����`������, 2:�U���_���̕ύX(9,6,6 ���͑w�̂ݐ����`������)
#define SINGROWTH 0	// Sine Growth  0:OFF, 1:ON

// �p�����[�^
#define SD 2		// ����
#define NPMIN 1		// �U���_���̍ŏ��l
#define NPMAX 32	// �U���_���̍ő�l
#define NPINIT 9	// �U���_���̏����l
#define NPINITRAND 6// �U���_���̏����l�i�`�����l�����ƂɗU���_����ύX����ꍇ�j
#define MU 0.5		// ����      Box-Muller
#define SIGMA 0.2	// ���U(0.1) Box-Muller
#define R 0.12		// ���a(0.12)
#define LOWESTR 0.0001	// ���a�̍Œ�l(�N���b�s���O) 
#define HIGHESTR 1		// ���a�̍ō��l(�N���b�s���O)

// ��Ď�@���s�͈�(Epoc)
#define SOIPSTART 2	// �����J�n�^�C�~���O(Epoc)�@��EPOC�͈͓�
#define SOIPEND 4	// �����I���^�C�~���O(Epoc)�@��EPOC�͈͓�

// ��Ď�@���s����
#define SOIPTIME 50	// �U���_�̑����^�C�~���O(�����w�K)�@ ��SOIPTIME���1����s����
#define BSOIPTIME 5	// �U���_�̑����^�C�~���O(�o�b�`�w�K) ��BSOIPTIME���1����s����

// �U���_�̍폜
#define DELCOUNT 1	// �U���_�폜�͈͂Ɋ܂܂ꂽ��(1:���폜)
#define DELW 0.025	// �_���폜����臒l(�d��) 

// �U���_�̌���
#define MERDIFF (R/2.0)	// ���a臒l(���a�̍�������)(0.05)
#define MERRATE 0.8		// �����{��(���a�̔{���ϓ�)(0.1)
#define MERSIMI 0.2		// �d��臒l(�d�݂̃R�T�C���ގ��x)(0.5)

// �U���_�̑���
#define ADDRATEBASIS 0.001	// �ǉ�臒l�̍Œ჉�C��(0.00001)
#define ADDRATERANGE 0.03	// �ǉ�臒l�̕�(臒l�͈̔́FADDRATEBASIS�`ADDRATEBASIS+ADDRATERANGE)
#define ADDDEL (-4)			// �폜�P�\

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

