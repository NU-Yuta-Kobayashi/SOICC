#include "DxLib.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "MLP.h"
#include "CNN.h"
#include "SMP.h"

// �F
#define BLACK 0         // ��
#define BLUE 255        // ��
#define GREEN 65280     // ��
#define RED 16711680    // ��
#define WHITE 16777215  // ��
#define PURPLE 14431708	// ��

// �\���֌W
#define FLAMEX 1200	// X���̕`��T�C�Y
#define FLAMEY 800	// Y���̕`��T�C�Y
#define DSIPLAY 10	// �`��J�n�ʒu(X,Y����)
#define MENUX 128	// ���j���[��X���W
#define MENUY 64	// ���j���[��Y���W
#define OPTIONX 30	// ���j���[��X���W
#define OPTIONY 440	// ���j���[��Y���W
#define MAPCHIP 32	// �}�b�v�`�b�v�T�C�Y
#define RD H		// 1�ӂ̒���
#define BOXS 64		// ���`��T�C�Y
#define MARGIN 46	// �]��
#define MAXW 5		// �`��\�ȃZ�O�����g��
#define MAXH 6		// �`��\�ȃ`�����l����
#define DISPSEG 2	// �`�悷��Z�O�����g

// �w�K�֌W
#define EPOC 30		// �w�K��
#define TRIALS 3	// ������
#define DEBUG_MODE 0// �f�o�b�O���[�h 0:OFF 1:ON

/* ------------------------------ �� �w�K�f�[�^�Ɉˑ�(��MLP.h, CCNN.h���ύX) �� ------------------------------ */
/* Optdigit */
//#define TRA 3823// �w�K�f�[�^��
//#define TES 1797// �e�X�g�f�[�^��
//#define MAXV 16	// ���̓f�[�^�̍ő�l
//char TrainFileName[] = "optdigits_tra.csv", TestFileName[] = "optdigits_tes.csv";
//int NodeN[L] = { IND,NMAX,K };
//int SkipChN[SEG][CONV + 1] = {
//	{ CHMAX / 4,CHMAX / 4,CHMAX / 4},
//	{ CHMAX / 2,CHMAX / 2,CHMAX / 2},
//	{ CHMAX,CHMAX,CHMAX}
//};// CNN�`�����l����

/* MNIST */
//#define TRA 60000	// �w�K�f�[�^
//#define TES 10000	// �e�X�g�f�[�^
//#define MAXV 255	// �f�[�^�̍ő�l
//char TrainFileName[] = "dataset_mnist_train.csv", TestFileName[] = "dataset_mnist_test.csv";
//int NodeN[L] = { IND,160,K };
//int SkipChN[SEG][CONV + 1] = {
//	{ CHMAX / 4,CHMAX / 4,CHMAX / 4},
//	{ CHMAX / 2,CHMAX / 2,CHMAX / 2},
//	{ CHMAX,CHMAX,CHMAX}
//};// CNN�`�����l����

/* Cifar-10 */
#define TRA 500	// �w�K�f�[�^
#define TES 100	// �e�X�g�f�[�^
#define MAXV 255	// �f�[�^�̍ő�l
char TrainFileName[] = "Dataset\\cifar10_train.csv", TestFileName[] = "Dataset\\cifar10_test.csv";// �f�[�^�t�@�C����
int NodeN[L] = { IND,400,K }; // MLP�`�����l����
int SkipChN[SEG][CONV + 1] = {
	{ CHMAX / 4,CHMAX / 4,CHMAX / 4},
	{ CHMAX / 2,CHMAX / 2,CHMAX / 2},
	{ CHMAX,CHMAX,CHMAX}
};// CNN�`�����l����
/* ------------------------------ �� �w�K�f�[�^�Ɉˑ�(��MLP.h, CCNN.h���ύX) �� ------------------------------ */

// �t�@�C���o��
char SaveFileName[] = "LogFile\\AccLog_CIFAR10_SkipSMP_1_Time.csv";			// �t�@�C���ۑ���
char SaveParamDWsumFileName[] = "LogFile\\Parametrer_CIFAR10_SMP_DWsum.csv";// �d�݂̍��v�i�f�o�b�O�p�j
char SaveParamDPsumFileName[] = "LogFile\\Parametrer_CIFAR10_SMP_DPsum.csv";// ���W�̍��v�i�f�o�b�O�p�j
char SaveParamDRsumFileName[] = "LogFile\\Parametrer_CIFAR10_SMP_DRsum.csv";// ���a�̍��v�i�f�o�b�O�p�j
char SaveParamWFileName[] = "LogFile\\Parametrer_CIFAR10_SMP_W.csv";// �d�݁i�f�o�b�O�p�j
char SaveParamPFileName[] = "LogFile\\Parametrer_CIFAR10_SMP_P.csv";// ���W�i�f�o�b�O�p�j
char SaveParamRFileName[] = "LogFile\\Parametrer_CIFAR10_SMP_R.csv";// ���a�i�f�o�b�O�p�j

// �`����W
int InCIE[CONV + 1][CHMAX][2], MapCIE[CONV][CHMAX * CHMAX][2];// ���o�͂̕`����W, �����}�b�v�̕`����W

// �f�[�^�֘A
unsigned char TraData[TRA][IND], TesData[TES][IND];	// 0�`255�܂�
int TraNum[TRA];									// ����ID�ێ�
int TraClass[TRA], TesClass[TES];					// ���x��
int In, Epoc, Loop;									// ���́A�w�K��
double TrainErr, TestErr, TrainAcc, TestAcc;		// �덷�A������
time_t Start, End, LStart, AccTime;					// �^�C�}�[

// MLP�p
double Out[L][NMAX], Tk[K], Delta[L][NMAX];		// �o�́A���t�A�덷
double Bias[L][NMAX], Woi[L - 1][NMAX][NMAX];	// NN�̏d��

// MLP�p �o�b�`�o��
double BOut[L][BS][NMAX], BTk[BS][K], BDelta[L][NMAX][BS];

// MLP�p �o�b�`������
double BNet[L][NMAX][BS], Mean[L][NMAX], Var[L][NMAX];
double Gamma[L][NMAX], Beta[L][NMAX], MMean[L][NMAX], MVar[L][NMAX];

// CNN�p
double COut[SEG][CONV + 1][CHMAX][H][W], CDelta[SEG][CONV + 1][CHMAX][H][W];
double CBias[SEG][CONV][CHMAX], CWoi[SEG][CONV][CHMAX][CHMAX][KS][KS];		// CNN�̏d��
double DCBias[SEG][CONV][CHMAX], DCWoi[SEG][CONV][CHMAX][CHMAX][KS][KS];	// �C���ʂ�ێ�����ϐ�
double BCOut[SEG][CONV + 1][CHMAX][BS][H][W], BCDelta[SEG][CONV + 1][CHMAX][BS][H][W];

// CNN�p 
double CBNet[SEG][CONV + 1][CHMAX][BS][H][W], CMean[SEG][CONV + 1][CHMAX], CVar[SEG][CONV + 1][CHMAX];
double CGamma[SEG][CONV + 1][CHMAX], CBeta[SEG][CONV + 1][CHMAX], CMMean[SEG][CONV + 1][CHMAX], CMVar[SEG][CONV + 1][CHMAX];

// Skip-CNN�p
double InOut[ICH][H][W], InBias[CHMAX], InWoi[CHMAX][ICH][IKS][IKS];
double DInBias[CHMAX], DInWoi[CHMAX][ICH][IKS][IKS];
double SkipOut[SEG][CHMAX][H][W], SkipDelta[SEG][CHMAX][H][W];
double SkipBias[SEG][CHMAX], SkipWoi[SEG][CHMAX][CHMAX][KS][KS];
double DSkipBias[SEG][CHMAX], DSkipWoi[SEG][CHMAX][CHMAX][KS][KS];

// Skip-CNN�p �o�b�`������
double BInOut[ICH][BS][H][W], BSkipOut[SEG][CHMAX][BS][H][W], BSkipDelta[SEG][CHMAX][BS][H][W];

// Skip-SMP�p
int NP[SEG][CONV][CHMAX];// �e�w�̗U���_��
double Pi[SEG][CONV][CHMAX][NPMAX][SD], Ri[SEG][CONV][CHMAX][NPMAX], Wi[SEG][CONV][CHMAX][CHMAX][NPMAX];	// ���W�E���a�E�d��
double CIE[SEG][H][W][SD], Dis[SEG][CONV + 1][H][W][CHMAX][NPMAX], G[SEG][CONV + 1][H][W][CHMAX][NPMAX];	// ���͍��W�E�����E�����֐�
double DPi[SEG][CONV][CHMAX][NPMAX][SD], DRi[SEG][CONV][CHMAX][NPMAX], DWi[SEG][CONV][CHMAX][CHMAX][NPMAX];	// ���W�̕ω��ʁE���a�̕ω��ʁE�d�݂̕ω���
int SumCount[SEG][CONV + 1][H][W][CHMAX], WUseCount[SEG][CONV][CHMAX][CHMAX][NPMAX], PRUseCount[SEG][CONV][CHMAX][NPMAX];// �������݉񐔁E�d�݂̎g�p�񐔁E���W�Ɣ��a�̎g�p��


// ��Ď�@
bool Add[SEG][CONV][CHMAX];					// �m�[�h�̒ǉ�����
int DelC[SEG][CONV][CHMAX][NPMAX];			// �폜�J�E���g
double DelW;								// �폜臒l(�d��)
double DPsum[SEG][CONV][CHMAX][NPMAX][SD];	// ���W�̕ω���
double DRsum[SEG][CONV][CHMAX][NPMAX];		// ���a�̕ω���
int SOIPCount;								// ��Ď�@���s�^�C�~���O
int AllNP;									// �U���_���̍��v

// �w�K��(cosine decay)
double Eta;

// �f�o�b�O�p�ϐ��E�֐�
double DebugDPsum[SEG][CONV][CHMAX][NPMAX][SD];		// ���W�̕ω���
double DebugDRsum[SEG][CONV][CHMAX][NPMAX];			// ���a�̕ω���
double DebugDWsum[SEG][CONV][CHMAX][CHMAX][NPMAX];	// �d�݂̕ω���
double DebugDPave[SEG][CONV];
double DebugDRave[SEG][CONV];
double DebugDWave[SEG][CONV];

// �֐�
void Display();											// �`��
void ReadFileData();									// �t�@�C���ǂݍ���
void Initialize();										// ������
void Input(unsigned char indata[], double out[]);		// MLP����
void CInput(unsigned char indata[], double out[][H][W]);// CNN����
void MLP();												// NN
void STest(int delN, int merN, int addN);				// �e�X�g
void DebugInit();										// �������i�f�o�b�O�j
void DebugSum();										// �X�V�ʁi�f�o�b�O�j

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {// ���C������
	int MouseInput, MouseX, MouseY, k, i;

	/* ----------------- ������ ----------------- */
	//srand((unsigned int)time(NULL));
	srand(1);		// �V�[�h�l
	ReadFileData();	// �w�K�f�[�^�ǂݍ���
	Initialize();	// ������

	/* ----------- �c�w���C�u�����ݒ� ----------- */
	SetGraphMode(1200, 800, 32);		// ��ʃ��[�h�̐ݒ�(�𑜓x�E�J���[�r�b�g��)
	SetBackgroundColor(255, 255, 255);	// �w�i�F�ݒ�
	ChangeWindowMode(TRUE);				// �E�C���h�E���[�h�ɕύX(��t���X�N���[��)
	SetAlwaysRunFlag(TRUE);				// �E�B���h�E����A�N�e�B�u���ɂ����s
	SetMouseDispFlag(TRUE);				// �}�E�X��\����Ԃɂ���
	if (DxLib_Init() == -1) return -1;	// �c�w���C�u��������������(�G���[���ɏI��)

	/* --------------- ���C������ --------------- */
	Display();
	while (1) {
		if (ProcessMessage() == -1) break;	// �G���[���ɏI��
		MouseInput = GetMouseInput();		// �}�E�X�̓��͂�҂�			
		if ((MouseInput & MOUSE_INPUT_LEFT) != 0) {	// ���{�^�������ꂽ
			GetMousePoint(&MouseX, &MouseY);		// �}�E�X�̈ʒu���擾
			if (MouseX < MENUX) {					// Menu area click
				if (MouseY < MENUY) break;			// �I��
				else if (MouseY < MENUY * 2) Initialize();//NN������
				else if (MouseY < MENUY * 3) { //Input
					In = (rand() / (RAND_MAX + 1.0)) * TRA;
					Input(TraData[In], Out[0]);
					for (k = 0; k < K; k++) Tk[k] = 0;
					Tk[TraClass[In]] = 1.0;//Hot One Vecter
				}
				else if (MouseY < MENUY * 4)Forward(NodeN, Bias, Woi, Out, MMean, MVar, Gamma, Beta); //�ړ����ς���
				else if (MouseY < MENUY * 5)MLP();
				else if (MouseY < MENUY * 6) {// �����p
					for (i = 0; i < TRIALS; i++) {
						Initialize();	// ������
						MLP();			// MLP
					}
				}
			}
			Display();
		}
		WaitTimer(100);
	}
	DxLib_End();// �c�w���C�u�����g�p�̏I������
	return 0;	// �\�t�g�̏I�� 
}

void Display() {// �f�B�X�v���C�\��
	int i, j, k, mx, my, cr, size = BOXS / RD, b, d, c, dispS = DISPSEG, l, dsize = 8, in, out, rate, p, temp, x, y, s, h, w, disY, poiY;
	double max;
	char fileName[50];
	char menu[][20] = { "END", "Init", "Input", "Forward", "Learn","n-Learn" };

	ClearDrawScreen();
	/* ���j���[�\�� */
	for (i = 0; i < 6; i++)DrawLine(0, MENUY * (i + 1), MENUX, MENUY * (i + 1), BLACK, 2);// �g�\��
	DrawLine(MENUX, 0, MENUX, MENUY * 6, BLACK, 2);

	for (i = 0; i < 6; i++)DrawString(20 + DSIPLAY, i * MENUY + 12 + DSIPLAY, menu[i], BLACK);// ���j���[����	

	/* �w�K�A���S���Y���\�� */
	if (NETMODEL == 1) DrawString(OPTIONX, OPTIONY, "CNN-Skip", BLACK);
	else if (NETMODEL == 2) DrawString(OPTIONX, OPTIONY, "SMP-Skip", BLACK);
	else if (NETMODEL == 3) DrawString(OPTIONX, OPTIONY, "SMP-Skip-Prop", BLACK);
	else DrawString(OPTIONX, OPTIONY, "MLP", BLACK);
	if (BATCHLEARN)DrawFormatString(OPTIONX, OPTIONY + 20, BLACK, "BatchLearn %d", BS);

	/* �w�K���\�� */
	DrawFormatString(OPTIONX, OPTIONY + 20 * 2, BLACK, "%s", TrainFileName);
	DrawFormatString(OPTIONX, OPTIONY + 20 * 3, BLACK, "Learning Rate %.3lf", Eta);
	DrawFormatString(OPTIONX, OPTIONY + 20 * 4, BLACK, "Epoc %3d   Time %3d", Epoc, End - Start);
	DrawFormatString(OPTIONX, OPTIONY + 20 * 5, BLACK, "TraAcc %.3lf", TrainAcc);
	DrawFormatString(OPTIONX + 120, OPTIONY + 20 * 5, BLACK, "TesAcc %.3lf", TestAcc);
	DrawFormatString(OPTIONX, OPTIONY + 20 * 12, BLACK, "Loop %d", Loop);
	mx = MENUX + 190; my = MENUY;

	/* �S�����w���̓f�[�^�ƃA���S���Y���̏ڍו\�� */
	if (NETMODEL == 2 || NETMODEL == 3) {
		// �Z�O�����g��, ��ݍ��ݐ�, �U���_��
		DrawFormatString(OPTIONX, OPTIONY + 20 * 6, BLACK, "Seg:%d  Conv:%d", SEG, CONV);
		DrawFormatString(OPTIONX, OPTIONY + 20 * 7, BLACK, "NPINIT:%d  NPMAX:%d", NPINIT, NPMAX);
		DrawFormatString(OPTIONX, OPTIONY + 20 * 8, BLACK, "AllNP:%d", AllNP);
		// ���a�A���U
		DrawFormatString(OPTIONX, OPTIONY + 20 * 9, BLACK, "Rad:%.3lf  Sig:%.3lf", R, SIGMA);
		// �`�����l����
		DrawString(OPTIONX, OPTIONY + 20 * 10, "Ch:", BLACK);
		DrawFormatString(OPTIONX + 30, OPTIONY + 20 * 10, BLACK, "%3d", SkipChN[0][0]);
		for (s = 1; s < SEG; s++)DrawFormatString(OPTIONX + 20 + 40 * s, OPTIONY + 20 * 10, BLACK, ",%3d", SkipChN[s][0]);
	}
	else {
		for (i = 0; i < RD; i++) for (j = 0; j < RD && (i * RD + j) < NodeN[0]; j++) {
			if (Out[0][i * RD + j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
			else {
				cr = (int)(Out[0][i * RD + j] * 255);
				DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
			}
		}
		for (k = 0; k < K; k++) if (Tk[k] > 0) DrawFormatString(mx + 66, my + 50, BLACK, "%d", k);				// ���t�f�[�^�̕\��
		for (k = 0; k < K; k++) DrawFormatString(mx, my + size * RD + k * 20, BLACK, "%.2lf", Out[L - 1][k]);	// �o�͂̕\��
	}

	/* �e���[�h���Ƃ̌��ʕ\�� */
	if (NETMODEL == 1) {//��ݍ��ݑw���̓f�[�^�̏��
		for (my = 300, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {
			if (COut[0][0][0][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
			else {
				/*cr = (int)(COut[0][0][0][i][j] * 255);*/
				//DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
				cr = GetColor((int)(COut[0][0][0][i][j] * 255), (int)(COut[0][0][0][i][j] * 255), (int)(COut[0][0][0][i][j] * 255));
				DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, cr, TRUE);
			}
		}
		for (my = 400, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//1���ݍ��݌���̓f�[�^�̏��
			if (COut[0][1][0][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
			else {
				cr = (int)(COut[0][1][0][i][j] * 255);
				DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
			}
		}
		for (my = 500, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//�v�[�����O����̓f�[�^�̏��
			if (COut[1][0][0][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
			else {
				cr = (int)(COut[1][0][0][i][j] * 255);
				DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
			}
		}
		if (SEG > 2)
			for (my = 600, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//�v�[�����O����̓f�[�^�̏��
				if (COut[2][0][0][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
				else {
					cr = (int)(COut[2][0][0][i][j] * 255);
					DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
				}
			}
	}
	if ((NETMODEL == 2 || NETMODEL == 3) && !BATCHLEARN) {
		mx += 10; poiY = 14; disY = 6;
		/* ���o�͕`����W���v�Z�E�`�� */
		for (c = 0; c < CONV + 1; c++) {
			for (i = 0; i < MAXH - 1 && i < SkipChN[dispS][c]; i++) {
				// ���W�v�Z
				InCIE[c][i][0] = mx + (BOXS + MARGIN) * c * 2;// x���W
				InCIE[c][i][1] = (FLAMEY / MAXH / 2 - BOXS / 2) + (FLAMEY / MAXH) * i - disY * i + poiY;// y���W
				// �t���[���`��
				DrawBox(InCIE[c][i][0], InCIE[c][i][1], InCIE[c][i][0] + BOXS, InCIE[c][i][1] + BOXS, WHITE, TRUE);
				DrawBox(InCIE[c][i][0], InCIE[c][i][1], InCIE[c][i][0] + BOXS, InCIE[c][i][1] + BOXS, BLACK, FALSE);
				DrawFormatString(InCIE[c][i][0] + 8, InCIE[c][i][1] + BOXS + 4, BLACK, "Ch[%d]", i);
				// �o�͕`��
				for (h = 0; h < RD; h++) {
					for (w = 0; w < RD; w++) {
						if (COut[dispS][c][i][h][w] == 0) DrawBox(InCIE[c][i][0] + w * size, InCIE[c][i][1] + h * size, InCIE[c][i][0] + (w + 1) * size, InCIE[c][i][1] + (h + 1) * size, 255, TRUE);
						else {
							cr = (int)(COut[dispS][c][i][h][w] * 255);
							DrawBox(InCIE[c][i][0] + w * size, InCIE[c][i][1] + h * size, InCIE[c][i][0] + (w + 1) * size, InCIE[c][i][1] + (h + 1) * size, GetColor(cr, cr, cr), TRUE);
						}
					}
				}
			}
			if (i < SkipChN[dispS][c]) {// �c��̕`�搔���`��\�͈͂��傫���Ȃ�A�Ō�̒l��\������
				DrawLine(InCIE[c][0][0] - 10, InCIE[c][i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS + 2, InCIE[c][0][0] + BOXS + 10, InCIE[c][i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS + 2, PURPLE);
				DrawLine(InCIE[c][0][0] - 10, InCIE[c][i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS - 2, InCIE[c][0][0] + BOXS + 10, InCIE[c][i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS - 2, PURPLE);
				InCIE[c][i][0] = mx + (BOXS + MARGIN) * c * 2;// w���W
				InCIE[c][i][1] = (FLAMEY / MAXH / 2 - BOXS / 2) + (FLAMEY / MAXH) * i - disY * i + poiY;// h���W
				// �t���[���`��
				DrawBox(InCIE[c][i][0], InCIE[c][i][1], InCIE[c][i][0] + BOXS, InCIE[c][i][1] + BOXS, WHITE, TRUE);
				DrawBox(InCIE[c][i][0], InCIE[c][i][1], InCIE[c][i][0] + BOXS, InCIE[c][i][1] + BOXS, BLACK, FALSE);
				DrawFormatString(InCIE[c][i][0] + 8, InCIE[c][i][1] + BOXS + 4, BLACK, "Ch[%d]", SkipChN[dispS][c] - 1);
				// �o�͕`��
				for (h = 0; h < RD; h++) {
					for (w = 0; w < RD; w++) {
						if (COut[dispS][c][SkipChN[dispS][c] - 1][h][w] == 0) DrawBox(InCIE[c][i][0] + w * size, InCIE[c][i][1] + h * size, InCIE[c][i][0] + (w + 1) * size, InCIE[c][i][1] + (h + 1) * size, 255, TRUE);
						else {
							cr = (int)(COut[dispS][c][SkipChN[dispS][c] - 1][h][w] * 255);
							DrawBox(InCIE[c][i][0] + w * size, InCIE[c][i][1] + h * size, InCIE[c][i][0] + (w + 1) * size, InCIE[c][i][1] + (h + 1) * size, GetColor(cr, cr, cr), TRUE);
						}
					}
				}
			}
			// �w�ԍ��`��
			DrawFormatString(InCIE[c][0][0] - 4, 14 + poiY, BLACK, "[%d][%d]�w", dispS, c);
		}
		/* �����}�b�v�`�� */
		for (c = 0; c < CONV; c++) {
			for (j = 0; j < SkipChN[dispS][c + 1]; j++) {
				for (i = 0; i < SkipChN[dispS][c]; i++) {
					if (j * SkipChN[dispS][c] + i >= MAXH - 1)break;
					// ���W�v�Z
					MapCIE[c][j * SkipChN[dispS][c] + i][0] = mx + (BOXS + MARGIN) * c * 2 + (BOXS + MARGIN);// x���W
					MapCIE[c][j * SkipChN[dispS][c] + i][1] = (FLAMEY / MAXH / 2 - BOXS / 2) + (FLAMEY / MAXH) * (j * SkipChN[dispS][c] + i) - disY * i + poiY;// y���W
					// �g�`��
					DrawBox(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1], MapCIE[c][j * SkipChN[dispS][c] + i][0] + BOXS, MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS, WHITE, TRUE);
					DrawBox(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1], MapCIE[c][j * SkipChN[dispS][c] + i][0] + BOXS, MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS, BLACK, FALSE);
					DrawFormatString(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS + 4, BLACK, "W[%d][%d]", j, i);
					// �d�ݕ\����
					DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 8, MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 8, BLACK);				// �d�ݕ\�����F����
					DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 13, MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 2, BLACK);							// �d�ݕ\�����F�c��(�n�_)
					DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 13, MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 2, BLACK);	// �d�ݕ\�����F����(�I�_)
					// �d�݂̐��l�\��(�_)
					for (p = 0; p < NP[dispS][c][i]; p++) {
						if (Wi[dispS][c][j][i][p] >= 0) {// �d�݂����̓_
							cr = 255;
							DrawPixel(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][i][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][i][p][1] * (RD * size - 1), GetColor(cr, 0, 0));// ���W
							DrawCircle(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][i][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][i][p][1] * (RD * size - 1), Ri[dispS][c][i][p] * (RD * size - 1), GetColor(cr, 0, 0), FALSE);// �͈�
							DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Wi[dispS][c][j][i][p] * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 11, MapCIE[c][j * SkipChN[dispS][c] + i][0] + Wi[dispS][c][j][i][p] * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 4, GetColor(cr, 0, 0));// �d�ݕ\����
						}
						else {// �d�݂����̓_
							cr = 255;
							DrawPixel(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][i][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][i][p][1] * (RD * size - 1), GetColor(0, 0, cr));// ���W
							DrawCircle(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][i][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][i][p][1] * (RD * size - 1), Ri[dispS][c][i][p] * (RD * size - 1), GetColor(0, 0, cr), FALSE);// �͈�
							DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + (-Wi[dispS][c][j][i][p]) * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 11, MapCIE[c][j * SkipChN[dispS][c] + i][0] + (-Wi[dispS][c][j][i][p]) * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 4, GetColor(0, 0, cr));// �d�ݕ\����
						}
					}
				}
				if (j * SkipChN[dispS][c] + i >= MAXH - 1)break;
			}
			if (j * SkipChN[dispS][c] + i < SkipChN[dispS][c] * SkipChN[dispS][c] - 1) {// �c��̕`�搔���`��\�͈͂��傫���Ȃ�A�Ō�̒l��\������
				// �������C���\��
				DrawLine(MapCIE[c][0][0] - 10, MapCIE[c][j * SkipChN[dispS][c] + i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS + 2, MapCIE[c][0][0] + BOXS + 10, MapCIE[c][j * SkipChN[dispS][c] + i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS + 2, PURPLE);
				DrawLine(MapCIE[c][0][0] - 10, MapCIE[c][j * SkipChN[dispS][c] + i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS - 2, MapCIE[c][0][0] + BOXS + 10, MapCIE[c][j * SkipChN[dispS][c] + i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS - 2, PURPLE);
				// �����v�Z
				MapCIE[c][j * SkipChN[dispS][c] + i][0] = mx + (BOXS + MARGIN) * c * 2 + (BOXS + MARGIN);// x���W
				MapCIE[c][j * SkipChN[dispS][c] + i][1] = (FLAMEY / MAXH / 2 - BOXS / 2) + (FLAMEY / MAXH) * (j * SkipChN[dispS][c] + i) - disY * i + poiY;// y���W
				// �g�\��
				DrawBox(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1], MapCIE[c][j * SkipChN[dispS][c] + i][0] + BOXS, MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS, WHITE, TRUE);
				DrawBox(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1], MapCIE[c][j * SkipChN[dispS][c] + i][0] + BOXS, MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS, BLACK, FALSE);
				DrawFormatString(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS + 4, BLACK, "W[%d][%d]", SkipChN[dispS][c + 1] - 1, SkipChN[dispS][c] - 1);
				// �U���_�`��
				DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 8, MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 8, BLACK);				// �d�ݕ\�����F����
				DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 13, MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 2, BLACK);							// �d�ݕ\�����F�c��(�n�_)
				DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 13, MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 2, BLACK);	// �d�ݕ\�����F����(�I�_)
				for (p = 0; p < NP[dispS][c][i]; p++) {
					if (Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p] >= 0) {// �d�݂����̓_
						cr = 255;
						DrawPixel(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][1] * (RD * size - 1), GetColor(cr, 0, 0));// ���W
						DrawCircle(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][1] * (RD * size - 1), Ri[dispS][c][SkipChN[dispS][c] - 1][p] * (RD * size - 1), GetColor(cr, 0, 0), FALSE);// �͈�
						DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p] * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 11, MapCIE[c][j * SkipChN[dispS][c] + i][0] + Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p] * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 4, GetColor(cr, 0, 0));// �d�ݕ\����
					}
					else {// �d�݂����̓_
						cr = 255;
						DrawPixel(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][1] * (RD * size - 1), GetColor(0, 0, cr));// ���W
						DrawCircle(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][1] * (RD * size - 1), Ri[dispS][c][SkipChN[dispS][c] - 1][p] * (RD * size - 1), GetColor(0, 0, cr), FALSE);// �͈�
						DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + (-Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p]) * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 11, MapCIE[c][j * SkipChN[dispS][c] + i][0] + (-Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p]) * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 4, GetColor(0, 0, cr));// �d�ݕ\����
					}

				}
			}
		}
	}

	/* �o�b�`�w�K�\�� */
	if (BATCHLEARN) {
		if ((NETMODEL == 2 || NETMODEL == 3)) {
			/* �����}�b�v�`�� */
			mx += 10; poiY = 14; disY = 6;
			for (c = 0; c < CONV; c++) {
				for (j = 0; j < SkipChN[dispS][c + 1]; j++) {
					for (i = 0; i < SkipChN[dispS][c]; i++) {
						if (j * SkipChN[dispS][c] + i >= MAXH - 1)break;
						// ���W�v�Z
						MapCIE[c][j * SkipChN[dispS][c] + i][0] = mx + (BOXS + MARGIN) * c * 2 + (BOXS + MARGIN);// x���W
						MapCIE[c][j * SkipChN[dispS][c] + i][1] = (FLAMEY / MAXH / 2 - BOXS / 2) + (FLAMEY / MAXH) * (j * SkipChN[dispS][c] + i) - disY * i + poiY;// y���W
						// �g�`��
						DrawBox(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1], MapCIE[c][j * SkipChN[dispS][c] + i][0] + BOXS, MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS, WHITE, TRUE);
						DrawBox(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1], MapCIE[c][j * SkipChN[dispS][c] + i][0] + BOXS, MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS, BLACK, FALSE);
						DrawFormatString(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS + 4, BLACK, "W[%d][%d]", j, i);
						// �d�ݕ\����
						DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 8, MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 8, BLACK);				// �d�ݕ\�����F����
						DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 13, MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 2, BLACK);							// �d�ݕ\�����F�c��(�n�_)
						DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 13, MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 2, BLACK);	// �d�ݕ\�����F����(�I�_)
						// �d�݂̐��l�\��(�_)
						for (p = 0; p < NP[dispS][c][i]; p++) {
							if (Wi[dispS][c][j][i][p] >= 0) {// �d�݂����̓_
								cr = 255;
								DrawPixel(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][i][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][i][p][1] * (RD * size - 1), GetColor(cr, 0, 0));// ���W
								DrawCircle(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][i][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][i][p][1] * (RD * size - 1), Ri[dispS][c][i][p] * (RD * size - 1), GetColor(cr, 0, 0), FALSE);// �͈�
								DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Wi[dispS][c][j][i][p] * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 11, MapCIE[c][j * SkipChN[dispS][c] + i][0] + Wi[dispS][c][j][i][p] * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 4, GetColor(cr, 0, 0));// �d�ݕ\����
							}
							else {// �d�݂����̓_
								cr = 255;
								DrawPixel(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][i][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][i][p][1] * (RD * size - 1), GetColor(0, 0, cr));// ���W
								DrawCircle(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][i][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][i][p][1] * (RD * size - 1), Ri[dispS][c][i][p] * (RD * size - 1), GetColor(0, 0, cr), FALSE);// �͈�
								DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + (-Wi[dispS][c][j][i][p]) * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 11, MapCIE[c][j * SkipChN[dispS][c] + i][0] + (-Wi[dispS][c][j][i][p]) * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 4, GetColor(0, 0, cr));// �d�ݕ\����
							}
						}
					}
					if (j * SkipChN[dispS][c] + i >= MAXH - 1)break;
				}
				if (j * SkipChN[dispS][c] + i < SkipChN[dispS][c] * SkipChN[dispS][c] - 1) {// �c��̕`�搔���`��\�͈͂��傫���Ȃ�A�Ō�̒l��\������
					// �������C���\��
					DrawLine(MapCIE[c][0][0] - 10, MapCIE[c][j * SkipChN[dispS][c] + i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS + 2, MapCIE[c][0][0] + BOXS + 10, MapCIE[c][j * SkipChN[dispS][c] + i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS + 2, PURPLE);
					DrawLine(MapCIE[c][0][0] - 10, MapCIE[c][j * SkipChN[dispS][c] + i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS - 2, MapCIE[c][0][0] + BOXS + 10, MapCIE[c][j * SkipChN[dispS][c] + i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS - 2, PURPLE);
					// �����v�Z
					MapCIE[c][j * SkipChN[dispS][c] + i][0] = mx + (BOXS + MARGIN) * c * 2 + (BOXS + MARGIN);// x���W
					MapCIE[c][j * SkipChN[dispS][c] + i][1] = (FLAMEY / MAXH / 2 - BOXS / 2) + (FLAMEY / MAXH) * (j * SkipChN[dispS][c] + i) - disY * i + poiY;// y���W
					// �g�\��
					DrawBox(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1], MapCIE[c][j * SkipChN[dispS][c] + i][0] + BOXS, MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS, WHITE, TRUE);
					DrawBox(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1], MapCIE[c][j * SkipChN[dispS][c] + i][0] + BOXS, MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS, BLACK, FALSE);
					DrawFormatString(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS + 4, BLACK, "W[%d][%d]", SkipChN[dispS][c + 1] - 1, SkipChN[dispS][c] - 1);
					// �U���_�`��
					DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 8, MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 8, BLACK);				// �d�ݕ\�����F����
					DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 13, MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 2, BLACK);							// �d�ݕ\�����F�c��(�n�_)
					DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 13, MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 2, BLACK);	// �d�ݕ\�����F����(�I�_)
					for (p = 0; p < NP[dispS][c][i]; p++) {
						if (Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p] >= 0) {// �d�݂����̓_
							cr = 255;
							DrawPixel(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][1] * (RD * size - 1), GetColor(cr, 0, 0));// ���W
							DrawCircle(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][1] * (RD * size - 1), Ri[dispS][c][SkipChN[dispS][c] - 1][p] * (RD * size - 1), GetColor(cr, 0, 0), FALSE);// �͈�
							DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p] * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 11, MapCIE[c][j * SkipChN[dispS][c] + i][0] + Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p] * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 4, GetColor(cr, 0, 0));// �d�ݕ\����
						}
						else {// �d�݂����̓_
							cr = 255;
							DrawPixel(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][1] * (RD * size - 1), GetColor(0, 0, cr));// ���W
							DrawCircle(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][1] * (RD * size - 1), Ri[dispS][c][SkipChN[dispS][c] - 1][p] * (RD * size - 1), GetColor(0, 0, cr), FALSE);// �͈�
							DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + (-Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p]) * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 11, MapCIE[c][j * SkipChN[dispS][c] + i][0] + (-Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p]) * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 4, GetColor(0, 0, cr));// �d�ݕ\����
						}
					}
				}
			}
		}
		else {
			for (mx = MENUX + 269, b = 0; b < BS && b < 10; b++, mx += 79) {
				for (my = MENUY, i = 0; i < RD; i++) for (j = 0; j < RD && (i * RD + j) < NodeN[0]; j++) {//�S�����w���̓f�[�^�̏��
					if (BOut[0][b][i * RD + j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
					else {
						cr = (int)(BOut[0][b][i * RD + j] * 255);
						DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
					}
				}
				for (k = 0; k < K; k++) if (BTk[b][k] > 0) DrawFormatString(mx + 58, my + 50, BLACK, "%d", k);
				for (k = 0; k < K; k++) // �o�͂̕\��
					DrawFormatString(mx, my + size * RD + k * 20, BLACK, "%.2lf", BOut[L - 1][b][k]);
				if (NETMODEL == 10) {//��ݍ��ݑw���̓f�[�^�̏��
					for (my = 300, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {
						if (BCOut[0][0][0][b][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
						else {
							cr = (int)(BCOut[0][0][0][b][i][j] * 255);
							DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
						}
					}
					for (my = 360, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//1���ݍ��݌���̓f�[�^�̏��
						if (BCOut[0][1][0][b][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
						else {
							cr = (int)(BCOut[0][1][0][b][i][j] * 255);
							DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
						}
					}
					for (my = 420, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//�v�[�����O����̓f�[�^�̏��
						if (BCOut[1][0][0][b][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
						else {
							cr = (int)(BCOut[1][0][0][b][i][j] * 255);
							DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
						}
					}
					if (SEG > 2) {
						for (my = 480, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//�v�[�����O����̓f�[�^�̏��
							if (BCOut[2][0][0][b][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
							else {
								cr = (int)(BCOut[2][0][0][b][i][j] * 255);
								DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
							}
						}
					}
				}
			}
		}
	}
	ScreenFlip(); WaitTimer(10);
}

void ReadFileData() {// �w�K�E�e�X�g�f�[�^�ǂݍ���
	FILE* fp;
	int i, n;

	/* --------- �w�K�f�[�^�ǂݍ��� --------- */
	fopen_s(&fp, TrainFileName, "r");
	for (n = 0; n < TRA; n++) {
		for (i = 0; i < IND; i++) {
			fscanf_s(fp, "%d,", &TraData[n][i]);
		}
		fscanf_s(fp, "%d\n", &TraClass[n]);
	}
	fclose(fp);

	/* -------- �e�X�g�f�[�^�ǂݍ��� -------- */
	fopen_s(&fp, TestFileName, "r");
	for (n = 0; n < TES; n++) {
		for (i = 0; i < IND; i++) {
			fscanf_s(fp, "%d,", &TesData[n][i]);
		}
		fscanf_s(fp, "%d\n", &TesClass[n]);
	}
	fclose(fp);
}

void Initialize() {// �p�����[�^�̏�����
	int i, j, k, l, s, c, y, x, d, m, och, r = sqrt(NPINIT);
	double p1, p2, he_wp;

	/* ---------- MLP���͑w���̏����� ---------- */
	if (NETMODEL == 1) NodeN[0] = ((H - IKS + 1) / pow(PS, SEG)) * ((W - IKS + 1) / pow(PS, SEG)) * CHMAX;
	else if (NETMODEL == 2 || NETMODEL == 3) NodeN[0] = ((H - IKS + 1) / pow(PS, SEG)) * ((W - IKS + 1) / pow(PS, SEG)) * CHMAX;

	/* --------------- MLP������ --------------- */
	for (l = 0; l < L; l++)for (i = 0; i < NodeN[l]; i++) {
		Gamma[l][i] = 1.0; Beta[l][i] = 0;
		MMean[l][i] = 0; MVar[l][i] = 1;
	}
	if (RELU) { //He������
		for (l = 0; l < L - 2; l++) {//���ԑw�O���@ReLU����ꍇ������
			he_wp = 2 * sqrt(6.0 / NodeN[l]);
			for (j = 0; j < NodeN[l + 1]; j++)//�o�͑�
				for (Bias[l][j] = 0, i = 0; i < NodeN[l]; i++) //���͑�
					Woi[l][j][i] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp;
		}
	}
	else {
		for (l = 0; l < L - 2; l++) //���ԑw�O���@ReLU����ꍇ������
			for (j = 0; j < NodeN[l + 1]; j++)//�o�͑�
				for (Bias[l][j] = 0, i = 0; i < NodeN[l]; i++) //���͑�
					Woi[l][j][i] = (double)rand() / (RAND_MAX + 1.0) - 0.5; //+-0.5
	}
	for (j = 0; j < NodeN[l + 1]; j++) {//�o�͑�
		for (Bias[l][j] = 0, i = 0; i < NodeN[l]; i++) //���͑�
			Woi[l][j][i] = (double)rand() / (RAND_MAX + 1.0) - 0.5;
	}

	/* ---------- Skip-Connection-CNN ---------- */
	if (NETMODEL == 1) {//CNN(Skip) �d�ݏ�����
		for (he_wp = 2 * sqrt(6.0 / (ICH * IKS * IKS)), j = 0; j < SkipChN[0][0]; j++) {
			for (InBias[j] = 0, i = 0; i < ICH; i++)
				for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
					InWoi[j][i][y][x] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He������
		}

		for (s = 0; s < SEG - 1; s++) {
			for (he_wp = 2 * sqrt(6.0 / (SkipChN[s][CONV] * KS * KS)), j = 0; j < SkipChN[s + 1][0]; j++)
				for (SkipBias[s][j] = 0, i = 0; i < SkipChN[s][CONV]; i++)
					for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
						SkipWoi[s][j][i][y][x] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He������
		}

		// �o�b�`���K��
		for (s = 0; s < SEG; s++) for (c = 0; c <= CONV; c++)for (j = 0; j < CHMAX; j++) {
			CGamma[s][c][j] = 1.0; CBeta[s][c][j] = 0;
			CMMean[s][c][j] = 0; CMVar[s][c][j] = 1.0;
		}

		// �d�݁E�o�C�A�X
		for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
			for (he_wp = 2 * sqrt(6.0 / (SkipChN[s][c] * KS * KS)), j = 0; j < SkipChN[s][c + 1]; j++)
				for (CBias[s][c][j] = 0, i = 0; i < SkipChN[s][c]; i++)
					for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
						CWoi[s][c][j][i][y][x] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He������
	}

	/* ---------- Skip-Connection-SMP ---------- */
	if (NETMODEL == 2 || NETMODEL == 3) {
		/* ------ CNN ------ */
		// InConv�d�݁E�o�C�A�X
		for (he_wp = 2 * sqrt(6.0 / (ICH * IKS * IKS)), j = 0; j < SkipChN[0][0]; j++) {
			for (InBias[j] = 0, i = 0; i < ICH; i++)
				for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
					InWoi[j][i][y][x] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He������
		}
		// SkipOut�d�݁E�o�C�A�X
		for (s = 0; s < SEG - 1; s++) {
			for (he_wp = 2 * sqrt(6.0 / (SkipChN[s][CONV] * KS * KS)), j = 0; j < SkipChN[s + 1][0]; j++)
				for (SkipBias[s][j] = 0, i = 0; i < SkipChN[s][CONV]; i++)
					for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
						SkipWoi[s][j][i][y][x] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He������
		}
		// �o�b�`���K��
		for (s = 0; s < SEG; s++) for (c = 0; c <= CONV; c++)for (j = 0; j < CHMAX; j++) {
			CGamma[s][c][j] = 1.0; CBeta[s][c][j] = 0;
			CMMean[s][c][j] = 0; CMVar[s][c][j] = 1;
		}
		// �d�݁E�o�C�A�X
		for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++) {
			for (he_wp = 2 * sqrt(6.0 / (SkipChN[s][c] * KS * KS)), j = 0; j < SkipChN[s][c + 1]; j++)
				for (CBias[s][c][j] = 0, i = 0; i < SkipChN[s][c]; i++)
					for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
						CWoi[s][c][j][i][y][x] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He������
		}
		/* ------ SMP ------ */
		// �U���_��������
		for (AllNP = 0, s = 0; s < SEG; s++) {// �Z�O�����g��
			for (c = 0; c < CONV; c++)// ��ݍ��ݐ�
				for (j = 0; j < SkipChN[s][c]; j++) {// �`�����l��
					if (POINTINIT == 2) {// �`�����l�����ƂɗU���_����ύX����
						if (s == 0) {
							NP[s][c][j] = NPINIT;
							AllNP += NPINIT;
						}
						else {
							NP[s][c][j] = NPINITRAND;
							AllNP += NPINITRAND;
						}
					}
					else {
						NP[s][c][j] = NPINIT;
						AllNP += NPINIT;
					}
				}
		}
		for (s = 0; s < SEG; s++) {// �Z�O�����g��
			for (c = 0; c < CONV; c++) {// ��ݍ��ݐ�
				// �d�݁E���a�E���W������
				for (j = 0; j < CHMAX; j++) {// �o�̓`�����l��
					for (i = 0; i < CHMAX; i++)// ���̓`�����l��
						for (m = 0; m < NPMAX; m++) {// �U���_��
							Wi[s][c][j][i][m] = (double)rand() / (RAND_MAX + 1.0) - 0.5; //+-0.5
						}
				}
				for (i = 0; i < CHMAX; i++) {
					for (m = 0; m < NPMAX; m++) {// �U���_��
						// �폜�J�E���g������
						DelC[s][c][i][m] = 0;
						// ���W������
						if (POINTINIT == 1 && r * r == NPINIT) {
							/* �����` */
							Pi[s][c][i][m][0] = (1.0 / r) * (m % r) + (1.0 / r) / 2.0;
							Pi[s][c][i][m][1] = (1.0 / r) * (m / r) + (1.0 / r) / 2.0;
						}
						else if (POINTINIT == 2 && r * r == NPINIT && s == 0) {
							/* �����` */
							Pi[s][c][i][m][0] = (1.0 / r) * (m % r) + (1.0 / r) / 2.0;
							Pi[s][c][i][m][1] = (1.0 / r) * (m / r) + (1.0 / r) / 2.0;
						}
						else {
							/* �K�E�X���z */
							for (d = 0; d < SD; d++) {// ������
								while (1) {
									// �K�E�X���z�ɂ��m��(Box-Muller�@)
									p1 = (double)rand() / (RAND_MAX);// ��l��������
									p2 = (double)rand() / (RAND_MAX);// ��l��������
									//p1 = sqrt(-2.0 * log(p1)) * sin(2 * M_PI * p2);
									p1 = sqrt(-2.0 * log(p1)) * cos(2 * M_PI * p2);// Box-Muller�@
									p1 = MU + SIGMA * p1;// ���`�ϊ�
									if (p1 > 0 && p1 < 1) {// �������ꂽ�m�����͈͓�
										Pi[s][c][i][m][d] = p1;
										break;
									}
								}
							}
						}
						// ���a������
						Ri[s][c][i][m] = R;
					}
				}
			}
		}
		// �ω��ʂ̍��v�l������
		SOIPInit(DPsum, DRsum);
	}
}

void Input(unsigned char indata[], double out[]) {
	int i;
	for (i = 0; i < IND; i++) out[i] = indata[i] / (double)MAXV;// ���K��
}

void CInput(unsigned char indata[], double out[][H][W]) {
	int i, h, w, n;
	for (n = 0, i = 0; i < ICH; i++)
		for (h = 0; h < H; h++)for (w = 0; w < W; w++, n++)
			out[i][h][w] = indata[n] / (double)MAXV;// ���K��
}

void MLP() {
	int d, in, k, count, maxk, b, s, c, j, i, temp, h, w;
	int addN, delN, merN;
	double a;
	FILE* fp;
	fopen_s(&fp, SaveFileName, "a");
	fprintf_s(fp, "\nEpoc,TrainAcc,TestAcc,TrainErr,TestErr,Time,LTime,Eta,Point,Del,Mer,Add\n");
	fclose(fp);

	// �w�K��������
	if (BATCHLEARN == 1)Eta = BETA;
	else Eta = ETA;
	// �폜����������
	if (SINGROWTH == 1)DelW = 0;
	else DelW = DELW;
	// ���ԏ�����
	Start = time(NULL);

	/* ---------------------- �w�K ---------------------- */
	if (DEBUG_MODE)DebugInit();// Debug
	for (AccTime = 0, SOIPCount = 0, Epoc = 0; Epoc < EPOC; Epoc++) {
		// ������
		LStart = time(NULL);
		delN = 0; merN = 0; addN = 0;
		// �w�K�f�[�^�쐬
		if (RANDMODE == 1) {
			for (i = 0; i < TRA; i++)TraNum[i] = i;
			for (i = 0; i < TRA; i++) {
				in = (rand() / (RAND_MAX + 1.0)) * (TRA - i) + i;
				temp = TraNum[i];
				TraNum[i] = TraNum[in];
				TraNum[in] = temp;
			}
		}
		// �w�K���[���ݒ�
		if (LEARNRULE == 1) {// cosine decay
			if (BATCHLEARN == 1) {// �o�b�`�w�K
				if (Epoc < WARMUP) Eta = BETA * WARMUPRATE;
				else Eta = 0.5 * (1 + cos(((Epoc - WARMUP) * M_PI) / (EPOC))) * BETA;
			}
			else {// �ʏ�
				if (Epoc < WARMUP) Eta = ETA * WARMUPRATE;
				else Eta = 0.5 * (1 + cos(((Epoc - WARMUP) * M_PI) / (EPOC))) * ETA;
			}
		}
		// �w�K
		if (BATCHLEARN) {
			for (count = 0, d = 0; d < TRA - BS; d += BS, SOIPCount++) {
				/* ���t�f�[�^���� */
				for (b = 0; b < BS; b++) {// �o�b�`��
					/* ��������(���̓f�[�^�ԍ�) */
					if (RANDMODE == 1)in = TraNum[d + b];
					else in = (rand() / (RAND_MAX + 1.0)) * TRA;
					/* ���̓f�[�^�쐬 */
					if (NETMODEL == 0) {
						Input(TraData[in], BOut[0][b]);
					}
					else if (NETMODEL == 1) {// SkipCNN
						for (k = 0, i = 0; i < ICH; i++)
							for (h = 0; h < H; h++)for (w = 0; w < W; w++, k++)
								BInOut[i][b][h][w] = TraData[in][k] / (double)MAXV;// / MAXV;
					}
					else if (NETMODEL == 2 || NETMODEL == 3) {// SkipSMP
						for (k = 0, i = 0; i < ICH; i++)
							for (h = 0; h < H; h++)for (w = 0; w < W; w++, k++)
								BInOut[i][b][h][w] = TraData[in][k] / (double)MAXV;// / MAXV;
					}
					/* ���t�f�[�^�쐬 */
					if (LS == 1) {// ���x��������
						for (k = 0; k < K; k++) BTk[b][k] = LSEPS / (K - 1);
						BTk[b][TraClass[in]] = 1.0 - LSEPS;//Hot One Vecter
					}
					else {
						for (k = 0; k < K; k++) BTk[b][k] = 0;
						BTk[b][TraClass[in]] = 1.0;//Hot One Vecter
					}
				}
				/* �I�v�V���� + MLP�̓��͑w�쐬 */
				if (NETMODEL == 1) {// SkipCNN
					BSkipForward(SkipChN, CBias, CWoi, BCOut, BOut[0], CBNet, CMean, CVar, CGamma, CBeta, CMMean, CMVar,
						BInOut, InBias, InWoi, BSkipOut, SkipBias, SkipWoi);
				}
				else if (NETMODEL == 2 || NETMODEL == 3) {// SkipSMP
					BSMPSkipForward(SkipChN, CBias, CWoi, BCOut, BOut[0], CGamma, CBeta, CMMean, CMVar, BInOut, InBias, InWoi,
						BSkipOut, SkipBias, SkipWoi, Pi, Ri, Wi, NP, CIE, Dis, G, SumCount, CBNet, CMean, CVar);
				}
				/* MLP(�w�K) */
				BForward(NodeN, Bias, Woi, BOut, BNet, Mean, Var, Gamma, Beta, MMean, MVar);
				BBackProp(NodeN, Bias, Woi, BOut, BTk, BDelta, BNet, Mean, Var, Gamma, Beta, Eta);
				/* �I�v�V�����̊w�K */
				if (NETMODEL == 1) {// SkipCNN
					BSkipBackProp(SkipChN, CBias, CWoi, BCOut, BCDelta, BDelta[0], DCBias, DCWoi, Eta, BInOut,
						InBias, InWoi, BSkipOut, BSkipDelta, CBNet, CMean, CVar, CGamma, CBeta, SkipBias, SkipWoi,
						DSkipBias, DSkipWoi, DInBias, DInWoi);
				}
				else if (NETMODEL == 2 || NETMODEL == 3) {// SkipSMP
					BSMPSkipBackProp(SkipChN, CBias, CWoi, BCOut, BCDelta, BDelta[0], DCBias, DCWoi, Eta, BInOut, InBias, InWoi,
						BSkipOut, BSkipDelta, SkipBias, SkipWoi, DSkipBias, DSkipWoi, DInBias, DInWoi,
						Pi, Ri, Wi, NP, CIE, Dis, G, SumCount, DPi, DRi, DWi, WUseCount, PRUseCount,
						CBNet, CMean, CVar, CGamma, CBeta, DPsum, DRsum);
				}
				if (DEBUG_MODE)DebugSum();// Debug
				/* �w�K���x */
				for (b = 0; b < BS; b++) {
					for (maxk = 0, k = 1; k < K; k++)
						if (BOut[L - 1][b][maxk] < BOut[L - 1][b][k]) maxk = k;
					if (BTk[b][maxk] > 0.5)count++;
				}
				/* �U���_�̑���(��Ď�@) */
				if (NETMODEL == 3) {
					if (Epoc == SOIPSTART && d == 0) {// �U���_�����J�n����
						SOIPCount = 0;// ������
						SOIPInit(DPsum, DRsum);
					}
					else if (Epoc >= SOIPSTART && Epoc <= SOIPEND) {// �����A���S���Y���̓K�p�͈�
						if (SOIPCount % BSOIPTIME == 0) {// �Ăяo������
							// SineGrowth���g�p�i�폜�p�j
							if (SINGROWTH == 1)DelW = 0.5 * (1 + sin((((Epoc - SOIPSTART) * TRA + d) - ((SOIPEND - SOIPSTART + 1) * TRA) / 2.0) * M_PI / ((SOIPEND - SOIPSTART + 1) * TRA))) * DELW;// Sine Growth
							// �U���_�̑���
							SOIP(SkipChN, Pi, Ri, Wi, NP, &AllNP, DelC, DPsum, DRsum, Add, DelW, Eta, &delN, &merN, &addN);// �U���_�̑���
							SOIPInit(DPsum, DRsum);// DPsum������������
						}
					}
				}
			}
		}
		else {
			for (count = 0, d = 0; d < TRA; d++) {
				/* ���t�f�[�^���� */
				if (RANDMODE == 1)in = TraNum[d];
				else in = (rand() / (RAND_MAX + 1.0)) * TRA;

				/* �I�v�V���� + MLP�̓��͑w�쐬 */
				if (NETMODEL == 1) {// SkipCNN
					CInput(TraData[in], InOut);
					SkipForward(SkipChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar, InOut, InBias, InWoi, SkipOut, SkipBias, SkipWoi);
				}
				else if (NETMODEL == 2 || NETMODEL == 3) {// SkipSMP
					CInput(TraData[in], InOut);
					SMPSkipForward(SkipChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar, InOut, InBias, InWoi,
						SkipOut, SkipBias, SkipWoi, Pi, Ri, Wi, NP, CIE, Dis, G, SumCount);
				}
				else {// MLP
					Input(TraData[in], Out[0]);
				}
				/* ���t�f�[�^�쐬 */
				if (LS == 1) {// ���x��������
					for (k = 0; k < K; k++) Tk[k] = LSEPS / (K - 1);
					Tk[TraClass[in]] = 1.0 - LSEPS;//Hot One Vecter
				}
				else {
					for (k = 0; k < K; k++) Tk[k] = 0;
					Tk[TraClass[in]] = 1.0;//Hot One Vecter
				}
				/* MLP(�w�K) */
				Forward(NodeN, Bias, Woi, Out, MMean, MVar, Gamma, Beta);
				BackProp(NodeN, Bias, Woi, Out, Tk, Delta, Eta);
				/* �I�v�V�����̊w�K */
				if (NETMODEL == 1) {
					SkipBackProp(SkipChN, CBias, CWoi, COut, CDelta, Delta[0], DCBias, DCWoi, Eta, InOut, InBias, InWoi,
						SkipOut, SkipDelta, SkipBias, SkipWoi, DSkipBias, DSkipWoi, DInBias, DInWoi);// CNN-Skip
				}
				else if (NETMODEL == 2 || NETMODEL == 3) {
					SMPSkipBackProp(SkipChN, CBias, CWoi, COut, CDelta, Delta[0], DCBias, DCWoi, Eta, InOut, InBias, InWoi, SkipOut, SkipDelta, SkipBias, SkipWoi, DSkipBias, DSkipWoi, DInBias, DInWoi,
						Pi, Ri, Wi, NP, CIE, Dis, G, SumCount, DPi, DRi, DWi, WUseCount, PRUseCount, DPsum, DRsum);
				}
				if (DEBUG_MODE)DebugSum();// Debug
				/* ���x�v�Z */
				for (maxk = 0, k = 1; k < K; k++)
					if (Out[L - 1][maxk] < Out[L - 1][k]) maxk = k;
				if (maxk == TraClass[in])count++;
				/* �U���_�̑���(��Ď�@) */
				if (NETMODEL == 3) {
					if (Epoc == SOIPSTART && d == 0) {// �U���_�����J�n����
						SOIPCount = 0;// ������
						SOIPInit(DPsum, DRsum);// DPsum������������
					}
					else if (Epoc >= SOIPSTART && Epoc <= SOIPEND) {// �����A���S���Y���̓K�p�͈�
						if (SOIPCount % SOIPTIME == 0) {// �Ăяo������
							// SineGrowth���g�p�i�폜�p�j
							if (SINGROWTH == 1)DelW = 0.5 * (1 + sin((((Epoc - SOIPSTART) * TRA + d) - ((SOIPEND - SOIPSTART + 1) * TRA) / 2.0) * M_PI / ((SOIPEND - SOIPSTART + 1) * TRA))) * DELW;// Sine Growth
							// �U���_�̑���
							SOIP(SkipChN, Pi, Ri, Wi, NP, &AllNP, DelC, DPsum, DRsum, Add, DelW, Eta, &delN, &merN, &addN);// �U���_�̑���
							SOIPInit(DPsum, DRsum);// DPsum������������
						}
					}
				}
			}
		}
		End = time(NULL);
		AccTime += End - LStart;// �w�K���Ԃ̂݋L��
		STest(delN, merN, addN);// TraAcc,TesAcc�v��
		Display();
	}
}

void STest(int delN, int merN, int addN) {// test���͈ړ����ς𗘗p����@�i�o�b�`�Ȃ��j
	int t, k, in, max, count, lay = L - 1;
	FILE* testfp, * paramDWsumfp, * paramDPsumfp, * paramDRsumfp, * paramWfp, * paramPfp, * paramRfp;

	if (fopen_s(&testfp, SaveFileName, "a")) return;		//file open ���s
	for (TrainErr = 0, t = 0, count = 0; t < TRA; t++) { // �w�K�f�[�^
		in = t;
		/* �I�v�V���� + MLP�̓��͑w�쐬 */
		if (NETMODEL == 1) {// SkipCNN
			CInput(TraData[in], InOut);
			SkipForward(SkipChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar, InOut, InBias, InWoi,
				SkipOut, SkipBias, SkipWoi);
		}
		else if (NETMODEL == 2 || NETMODEL == 3) {// SkipSMP
			CInput(TraData[in], InOut);
			SMPSkipForward(SkipChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar, InOut, InBias, InWoi,
				SkipOut, SkipBias, SkipWoi, Pi, Ri, Wi, NP, CIE, Dis, G, SumCount);
		}
		else {
			Input(TraData[in], Out[0]);
		}
		Forward(NodeN, Bias, Woi, Out, MMean, MVar, Gamma, Beta);
		for (max = 0, k = 1; k < K; k++)
			if (Out[lay][max] < Out[lay][k]) max = k;
		if (TraClass[t] == max) count++;//���x
		for (k = 0; k < K; k++)
			if (k == TraClass[t])TrainErr += (1.0 - Out[lay][k]) * (1.0 - Out[lay][k]);
			else TrainErr += (Out[lay][k]) * (Out[lay][k]);
	}
	TrainAcc = (double)count / (double)(t); TrainErr /= (double)(t);
	for (TestErr = 0, t = 0, count = 0; t < TES; t++) {
		in = t;
		/* �I�v�V���� + MLP�̓��͑w�쐬 */
		if (NETMODEL == 1) {// SkipCNN
			CInput(TesData[in], InOut);
			SkipForward(SkipChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar, InOut, InBias, InWoi,
				SkipOut, SkipBias, SkipWoi);
		}
		else if (NETMODEL == 2 || NETMODEL == 3) {// SkipSMP
			CInput(TesData[in], InOut);
			SMPSkipForward(SkipChN, CBias, CWoi, COut, Out[0], CGamma, CBeta, CMMean, CMVar, InOut, InBias, InWoi,
				SkipOut, SkipBias, SkipWoi, Pi, Ri, Wi, NP, CIE, Dis, G, SumCount);
		}
		else {
			Input(TesData[in], Out[0]);
		}
		Forward(NodeN, Bias, Woi, Out, MMean, MVar, Gamma, Beta);
		for (max = 0, k = 1; k < K; k++)
			if (Out[lay][max] < Out[lay][k]) max = k;
		if (TesClass[t] == max) count++;//���x	
		for (k = 0; k < K; k++)
			if (k == TesClass[t])TestErr += (1.0 - Out[lay][k]) * (1.0 - Out[lay][k]);
			else TestErr += (Out[lay][k]) * (Out[lay][k]);
	}
	TestAcc = (double)count / (double)(t); TestErr /= (double)(t);
	fprintf_s(testfp, "%d,%lf,%lf,%lf,%lf,%d,%d,%lf,%d,%d,%d,%d\n", Epoc, TrainAcc, TestAcc, TrainErr, TestErr, (int)(End - Start), (int)AccTime, Eta, AllNP, delN, merN, addN);

	if (DEBUG_MODE) {
		if (Epoc == EPOC - 1) {
			fprintf_s(testfp, "DP,DR,DW\n");
			int s, c;
			for (s = 0; s < SEG; s++) {
				for (c = 0; c < CONV; c++) {
					fprintf_s(testfp, "%lf,%lf,%lf\n", DebugDPave[s][c], DebugDRave[s][c], DebugDWave[s][c]);
				}
			}
		}
	}
	fclose(testfp);

	// �p�����[�^�`�F�b�N
	if (DEBUG_MODE) {
		int s, c, j, i, m, d;
		double dis;
		if (fopen_s(&paramDWsumfp, SaveParamDWsumFileName, "a")) return;	//file open ���s
		if (fopen_s(&paramDPsumfp, SaveParamDPsumFileName, "a")) return;	//file open ���s
		if (fopen_s(&paramDRsumfp, SaveParamDRsumFileName, "a")) return;	//file open ���s
		if (fopen_s(&paramWfp, SaveParamWFileName, "a")) return;	//file open ���s
		if (fopen_s(&paramPfp, SaveParamPFileName, "a")) return;	//file open ���s
		if (fopen_s(&paramRfp, SaveParamRFileName, "a")) return;	//file open ���s
		// �ω��ʁF�d��
		fprintf_s(paramDWsumfp, "\nEpoc,Seg,Conv,OutCh,InCh\n");
		for (s = 0; s < SEG; s++) {
			for (c = 0; c < CONV; c++)
				for (j = 0; j < SkipChN[s][c + 1]; j++)
					for (i = 0; i < SkipChN[s][c]; i++) {
						fprintf_s(paramDWsumfp, "%d,%d,%d,%d,%d", Epoc, s, c, j, i);
						for (m = 0; m < NP[s][c][i]; m++) {
							fprintf_s(paramDWsumfp, ",%lf", DebugDWsum[s][c][j][i][m]);
						}
						fprintf_s(paramDWsumfp, "\n");
					}
			fprintf_s(paramDWsumfp, "\n");
		}
		// �ω��ʁF���W
		fprintf_s(paramDPsumfp, "\nEpoc,Seg,Conv,InCh\n");
		for (s = 0; s < SEG; s++) {
			for (c = 0; c < CONV; c++)
				for (i = 0; i < SkipChN[s][c]; i++) {
					fprintf_s(paramDPsumfp, "%d,%d,%d,%d", Epoc, s, c, i);
					for (m = 0; m < NP[s][c][i]; m++) {
						dis = sqrt(DebugDPsum[s][c][i][m][0] * DebugDPsum[s][c][i][m][0] + DebugDPsum[s][c][i][m][1] * DebugDPsum[s][c][i][m][1]);// L2����
						fprintf_s(paramDPsumfp, ",%lf", dis);
					}
					fprintf_s(paramDPsumfp, "\n");
				}
			fprintf_s(paramDPsumfp, "\n");
		}
		// �ω��ʁF���a
		fprintf_s(paramDRsumfp, "\nEpoc,Seg,Conv,InCh\n");
		for (s = 0; s < SEG; s++) {
			for (c = 0; c < CONV; c++)
				for (i = 0; i < SkipChN[s][c]; i++) {
					fprintf_s(paramDRsumfp, "%d,%d,%d,%d", Epoc, s, c, i);
					for (m = 0; m < NP[s][c][i]; m++) {
						fprintf_s(paramDRsumfp, ",%lf", DebugDRsum[s][c][i][m]);
					}
					fprintf_s(paramDRsumfp, "\n");
				}
			fprintf_s(paramDRsumfp, "\n");
		}
		// �d��
		fprintf_s(paramWfp, "\nEpoc,Seg,Conv,OutCh,InCh\n");
		for (s = 0; s < SEG; s++) {
			for (c = 0; c < CONV; c++)
				for (j = 0; j < SkipChN[s][c + 1]; j++)
					for (i = 0; i < SkipChN[s][c]; i++) {
						fprintf_s(paramWfp, "%d,%d,%d,%d,%d", Epoc, s, c, j, i);
						for (m = 0; m < NP[s][c][i]; m++) {
							fprintf_s(paramWfp, ",%lf", Wi[s][c][j][i][m]);
						}
						fprintf_s(paramWfp, "\n");
					}
			fprintf_s(paramWfp, "\n");
		}
		// ���W
		fprintf_s(paramPfp, "\nEpoc,Seg,Conv,InCh\n");
		for (s = 0; s < SEG; s++) {
			for (c = 0; c < CONV; c++)
				for (i = 0; i < SkipChN[s][c]; i++) {
					fprintf_s(paramPfp, "%d,%d,%d,%d", Epoc, s, c, i);
					for (m = 0; m < NP[s][c][i]; m++) {
						for (d = 0; d < SD; d++) {
							fprintf_s(paramPfp, ",%lf", Pi[s][c][i][m][d]);
						}
					}
					fprintf_s(paramPfp, "\n");
				}
			fprintf_s(paramPfp, "\n");
		}
		// ���a
		fprintf_s(paramRfp, "\nEpoc,Seg,Conv,InCh\n");
		for (s = 0; s < SEG; s++) {
			for (c = 0; c < CONV; c++)
				for (i = 0; i < SkipChN[s][c]; i++) {
					fprintf_s(paramRfp, "%d,%d,%d,%d", Epoc, s, c, i);
					for (m = 0; m < NP[s][c][i]; m++) {
						fprintf_s(paramRfp, ",%lf", Ri[s][c][i][m]);
					}
					fprintf_s(paramRfp, "\n");
				}
			fprintf_s(paramRfp, "\n");
		}
		fclose(paramDWsumfp);
		fclose(paramDPsumfp);
		fclose(paramDRsumfp);
		fclose(paramWfp);
		fclose(paramPfp);
		fclose(paramRfp);
	}

}

void DebugInit() {
	int s, c, j, i, m, d;
	// �d��
	for (s = 0; s < SEG; s++) {
		for (c = 0; c < CONV; c++)
			for (j = 0; j < SkipChN[s][c + 1]; j++)
				for (i = 0; i < SkipChN[s][c]; i++)
					for (m = 0; m < NP[s][c][i]; m++) {
						DebugDWsum[s][c][j][i][m] = 0;
					}
	}
	// SMP���W�E���a
	for (s = 0; s < SEG; s++) {
		for (c = 0; c < CONV; c++)
			for (i = 0; i < SkipChN[s][c]; i++)
				for (m = 0; m < NP[s][c][i]; m++) {
					DebugDRsum[s][c][i][m] = 0;// ���a�̕ω���
					for (d = 0; d < SD; d++) {
						DebugDPsum[s][c][i][m][d] = 0;// ���W�̕ω���
					}
				}
	}
}

void DebugSum() {
	int s, c, j, i, m, d;
	// �d��
	for (s = 0; s < SEG; s++) {
		for (c = 0; c < CONV; c++)
			for (j = 0; j < SkipChN[s][c + 1]; j++)
				for (i = 0; i < SkipChN[s][c]; i++)
					for (m = 0; m < NP[s][c][i]; m++) {
						if (WUseCount[s][c][j][i][m] != 0) {
							if (DWi[s][c][j][i][m] >= 0)DebugDWsum[s][c][j][i][m] += Eta * DWi[s][c][j][i][m] / WUseCount[s][c][j][i][m];
							else DebugDWsum[s][c][j][i][m] -= Eta * DWi[s][c][j][i][m] / WUseCount[s][c][j][i][m];
						}
					}
	}
	// SMP���W�E���a
	for (s = 0; s < SEG; s++) {
		for (c = 0; c < CONV; c++)
			for (i = 0; i < SkipChN[s][c]; i++)
				for (m = 0; m < NP[s][c][i]; m++) {
					if (PRUseCount[s][c][i][m] != 0) {
						if (DRi[s][c][i][m] >= 0)DebugDRsum[s][c][i][m] += Eta * DRi[s][c][i][m] / PRUseCount[s][c][i][m];	// ���a�̕ω���
						else DebugDRsum[s][c][i][m] -= Eta * DRi[s][c][i][m] / PRUseCount[s][c][i][m];	// ���a�̕ω���

						for (d = 0; d < SD; d++) {
							if (DPi[s][c][i][m][d] >= 0)DebugDPsum[s][c][i][m][d] += Eta * DPi[s][c][i][m][d] / PRUseCount[s][c][i][m];	// ���W�̕ω���
							else DebugDPsum[s][c][i][m][d] -= Eta * DPi[s][c][i][m][d] / PRUseCount[s][c][i][m];	// ���W�̕ω���
						}
					}
				}
	}
	// ������
	for (s = 0; s < SEG; s++) {
		for (c = 0; c < CONV; c++) {
			DebugDPave[s][c] = 0;
			DebugDRave[s][c] = 0;
			DebugDWave[s][c] = 0;
		}
	}
	// �d�݂̕���
	for (s = 0; s < SEG; s++) {
		for (c = 0; c < CONV; c++) {
			for (j = 0; j < SkipChN[s][c + 1]; j++)
				for (i = 0; i < SkipChN[s][c]; i++)
					for (m = 0; m < NP[s][c][i]; m++) {
						DebugDWave[s][c] += DebugDWsum[s][c][j][i][m];
					}
			DebugDWave[s][c] /= (double)(SkipChN[s][c + 1] * SkipChN[s][c] * NP[s][c][i - 1]);
		}
	}
	// ���W�̕��ρA���a�̏d��
	for (s = 0; s < SEG; s++) {
		for (c = 0; c < CONV; c++) {
			for (i = 0; i < SkipChN[s][c]; i++)
				for (m = 0; m < NP[s][c][i]; m++) {
					DebugDPave[s][c] += sqrt(DebugDPsum[s][c][i][m][0] * DebugDPsum[s][c][i][m][0] + DebugDPsum[s][c][i][m][1] * DebugDPsum[s][c][i][m][1]);
					DebugDRave[s][c] += DebugDRsum[s][c][i][m];
				}
			DebugDPave[s][c] /= (double)(SkipChN[s][c] * NP[s][c][i - 1]);
			DebugDRave[s][c] /= (double)(SkipChN[s][c] * NP[s][c][i - 1]);
		}
	}
}