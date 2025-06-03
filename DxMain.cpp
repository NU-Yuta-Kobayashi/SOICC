#include "DxLib.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include "MLP.h"
#include "CNN.h"
#include "SMP.h"

// 色
#define BLACK 0         // 黒
#define BLUE 255        // 青
#define GREEN 65280     // 緑
#define RED 16711680    // 赤
#define WHITE 16777215  // 白
#define PURPLE 14431708	// 紫

// 表示関係
#define FLAMEX 1200	// X軸の描画サイズ
#define FLAMEY 800	// Y軸の描画サイズ
#define DSIPLAY 10	// 描画開始位置(X,Y共通)
#define MENUX 128	// メニューのX座標
#define MENUY 64	// メニューのY座標
#define OPTIONX 30	// メニューのX座標
#define OPTIONY 440	// メニューのY座標
#define MAPCHIP 32	// マップチップサイズ
#define RD H		// 1辺の長さ
#define BOXS 64		// 箱描画サイズ
#define MARGIN 46	// 余白
#define MAXW 5		// 描画可能なセグメント数
#define MAXH 6		// 描画可能なチャンネル数
#define DISPSEG 2	// 描画するセグメント

// 学習関係
#define EPOC 30		// 学習回数
#define TRIALS 3	// 実験回数
#define DEBUG_MODE 0// デバッグモード 0:OFF 1:ON

/* ------------------------------ ↓ 学習データに依存(※MLP.h, CCNN.hも変更) ↓ ------------------------------ */
/* Optdigit */
//#define TRA 3823// 学習データ数
//#define TES 1797// テストデータ数
//#define MAXV 16	// 入力データの最大値
//char TrainFileName[] = "optdigits_tra.csv", TestFileName[] = "optdigits_tes.csv";
//int NodeN[L] = { IND,NMAX,K };
//int SkipChN[SEG][CONV + 1] = {
//	{ CHMAX / 4,CHMAX / 4,CHMAX / 4},
//	{ CHMAX / 2,CHMAX / 2,CHMAX / 2},
//	{ CHMAX,CHMAX,CHMAX}
//};// CNNチャンネル数

/* MNIST */
//#define TRA 60000	// 学習データ
//#define TES 10000	// テストデータ
//#define MAXV 255	// データの最大値
//char TrainFileName[] = "dataset_mnist_train.csv", TestFileName[] = "dataset_mnist_test.csv";
//int NodeN[L] = { IND,160,K };
//int SkipChN[SEG][CONV + 1] = {
//	{ CHMAX / 4,CHMAX / 4,CHMAX / 4},
//	{ CHMAX / 2,CHMAX / 2,CHMAX / 2},
//	{ CHMAX,CHMAX,CHMAX}
//};// CNNチャンネル数

/* Cifar-10 */
#define TRA 500	// 学習データ
#define TES 100	// テストデータ
#define MAXV 255	// データの最大値
char TrainFileName[] = "Dataset\\cifar10_train.csv", TestFileName[] = "Dataset\\cifar10_test.csv";// データファイル名
int NodeN[L] = { IND,400,K }; // MLPチャンネル数
int SkipChN[SEG][CONV + 1] = {
	{ CHMAX / 4,CHMAX / 4,CHMAX / 4},
	{ CHMAX / 2,CHMAX / 2,CHMAX / 2},
	{ CHMAX,CHMAX,CHMAX}
};// CNNチャンネル数
/* ------------------------------ ↑ 学習データに依存(※MLP.h, CCNN.hも変更) ↑ ------------------------------ */

// ファイル出力
char SaveFileName[] = "LogFile\\AccLog_CIFAR10_SkipSMP_1_Time.csv";			// ファイル保存名
char SaveParamDWsumFileName[] = "LogFile\\Parametrer_CIFAR10_SMP_DWsum.csv";// 重みの合計（デバッグ用）
char SaveParamDPsumFileName[] = "LogFile\\Parametrer_CIFAR10_SMP_DPsum.csv";// 座標の合計（デバッグ用）
char SaveParamDRsumFileName[] = "LogFile\\Parametrer_CIFAR10_SMP_DRsum.csv";// 半径の合計（デバッグ用）
char SaveParamWFileName[] = "LogFile\\Parametrer_CIFAR10_SMP_W.csv";// 重み（デバッグ用）
char SaveParamPFileName[] = "LogFile\\Parametrer_CIFAR10_SMP_P.csv";// 座標（デバッグ用）
char SaveParamRFileName[] = "LogFile\\Parametrer_CIFAR10_SMP_R.csv";// 半径（デバッグ用）

// 描画座標
int InCIE[CONV + 1][CHMAX][2], MapCIE[CONV][CHMAX * CHMAX][2];// 入出力の描画座標, 特徴マップの描画座標

// データ関連
unsigned char TraData[TRA][IND], TesData[TES][IND];	// 0～255まで
int TraNum[TRA];									// 入力ID保持
int TraClass[TRA], TesClass[TES];					// ラベル
int In, Epoc, Loop;									// 入力、学習回数
double TrainErr, TestErr, TrainAcc, TestAcc;		// 誤差、正答率
time_t Start, End, LStart, AccTime;					// タイマー

// MLP用
double Out[L][NMAX], Tk[K], Delta[L][NMAX];		// 出力、教師、誤差
double Bias[L][NMAX], Woi[L - 1][NMAX][NMAX];	// NNの重み

// MLP用 バッチ出力
double BOut[L][BS][NMAX], BTk[BS][K], BDelta[L][NMAX][BS];

// MLP用 バッチ正則化
double BNet[L][NMAX][BS], Mean[L][NMAX], Var[L][NMAX];
double Gamma[L][NMAX], Beta[L][NMAX], MMean[L][NMAX], MVar[L][NMAX];

// CNN用
double COut[SEG][CONV + 1][CHMAX][H][W], CDelta[SEG][CONV + 1][CHMAX][H][W];
double CBias[SEG][CONV][CHMAX], CWoi[SEG][CONV][CHMAX][CHMAX][KS][KS];		// CNNの重み
double DCBias[SEG][CONV][CHMAX], DCWoi[SEG][CONV][CHMAX][CHMAX][KS][KS];	// 修正量を保持する変数
double BCOut[SEG][CONV + 1][CHMAX][BS][H][W], BCDelta[SEG][CONV + 1][CHMAX][BS][H][W];

// CNN用 
double CBNet[SEG][CONV + 1][CHMAX][BS][H][W], CMean[SEG][CONV + 1][CHMAX], CVar[SEG][CONV + 1][CHMAX];
double CGamma[SEG][CONV + 1][CHMAX], CBeta[SEG][CONV + 1][CHMAX], CMMean[SEG][CONV + 1][CHMAX], CMVar[SEG][CONV + 1][CHMAX];

// Skip-CNN用
double InOut[ICH][H][W], InBias[CHMAX], InWoi[CHMAX][ICH][IKS][IKS];
double DInBias[CHMAX], DInWoi[CHMAX][ICH][IKS][IKS];
double SkipOut[SEG][CHMAX][H][W], SkipDelta[SEG][CHMAX][H][W];
double SkipBias[SEG][CHMAX], SkipWoi[SEG][CHMAX][CHMAX][KS][KS];
double DSkipBias[SEG][CHMAX], DSkipWoi[SEG][CHMAX][CHMAX][KS][KS];

// Skip-CNN用 バッチ正則化
double BInOut[ICH][BS][H][W], BSkipOut[SEG][CHMAX][BS][H][W], BSkipDelta[SEG][CHMAX][BS][H][W];

// Skip-SMP用
int NP[SEG][CONV][CHMAX];// 各層の誘導点数
double Pi[SEG][CONV][CHMAX][NPMAX][SD], Ri[SEG][CONV][CHMAX][NPMAX], Wi[SEG][CONV][CHMAX][CHMAX][NPMAX];	// 座標・半径・重み
double CIE[SEG][H][W][SD], Dis[SEG][CONV + 1][H][W][CHMAX][NPMAX], G[SEG][CONV + 1][H][W][CHMAX][NPMAX];	// 入力座標・距離・距離関数
double DPi[SEG][CONV][CHMAX][NPMAX][SD], DRi[SEG][CONV][CHMAX][NPMAX], DWi[SEG][CONV][CHMAX][CHMAX][NPMAX];	// 座標の変化量・半径の変化量・重みの変化量
int SumCount[SEG][CONV + 1][H][W][CHMAX], WUseCount[SEG][CONV][CHMAX][CHMAX][NPMAX], PRUseCount[SEG][CONV][CHMAX][NPMAX];// 足し込み回数・重みの使用回数・座標と半径の使用回数


// 提案手法
bool Add[SEG][CONV][CHMAX];					// ノードの追加条件
int DelC[SEG][CONV][CHMAX][NPMAX];			// 削除カウント
double DelW;								// 削除閾値(重み)
double DPsum[SEG][CONV][CHMAX][NPMAX][SD];	// 座標の変化量
double DRsum[SEG][CONV][CHMAX][NPMAX];		// 半径の変化量
int SOIPCount;								// 提案手法実行タイミング
int AllNP;									// 誘導点数の合計

// 学習率(cosine decay)
double Eta;

// デバッグ用変数・関数
double DebugDPsum[SEG][CONV][CHMAX][NPMAX][SD];		// 座標の変化量
double DebugDRsum[SEG][CONV][CHMAX][NPMAX];			// 半径の変化量
double DebugDWsum[SEG][CONV][CHMAX][CHMAX][NPMAX];	// 重みの変化量
double DebugDPave[SEG][CONV];
double DebugDRave[SEG][CONV];
double DebugDWave[SEG][CONV];

// 関数
void Display();											// 描画
void ReadFileData();									// ファイル読み込み
void Initialize();										// 初期化
void Input(unsigned char indata[], double out[]);		// MLP入力
void CInput(unsigned char indata[], double out[][H][W]);// CNN入力
void MLP();												// NN
void STest(int delN, int merN, int addN);				// テスト
void DebugInit();										// 初期化（デバッグ）
void DebugSum();										// 更新量（デバッグ）

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpCmdLine, int nCmdShow) {// メイン処理
	int MouseInput, MouseX, MouseY, k, i;

	/* ----------------- 初期化 ----------------- */
	//srand((unsigned int)time(NULL));
	srand(1);		// シード値
	ReadFileData();	// 学習データ読み込み
	Initialize();	// 初期化

	/* ----------- ＤＸライブラリ設定 ----------- */
	SetGraphMode(1200, 800, 32);		// 画面モードの設定(解像度・カラービット数)
	SetBackgroundColor(255, 255, 255);	// 背景色設定
	ChangeWindowMode(TRUE);				// ウインドウモードに変更(非フルスクリーン)
	SetAlwaysRunFlag(TRUE);				// ウィンドウが非アクティブ時にも実行
	SetMouseDispFlag(TRUE);				// マウスを表示状態にする
	if (DxLib_Init() == -1) return -1;	// ＤＸライブラリ初期化処理(エラー時に終了)

	/* --------------- メイン処理 --------------- */
	Display();
	while (1) {
		if (ProcessMessage() == -1) break;	// エラー時に終了
		MouseInput = GetMouseInput();		// マウスの入力を待つ			
		if ((MouseInput & MOUSE_INPUT_LEFT) != 0) {	// 左ボタン押された
			GetMousePoint(&MouseX, &MouseY);		// マウスの位置を取得
			if (MouseX < MENUX) {					// Menu area click
				if (MouseY < MENUY) break;			// 終了
				else if (MouseY < MENUY * 2) Initialize();//NN初期化
				else if (MouseY < MENUY * 3) { //Input
					In = (rand() / (RAND_MAX + 1.0)) * TRA;
					Input(TraData[In], Out[0]);
					for (k = 0; k < K; k++) Tk[k] = 0;
					Tk[TraClass[In]] = 1.0;//Hot One Vecter
				}
				else if (MouseY < MENUY * 4)Forward(NodeN, Bias, Woi, Out, MMean, MVar, Gamma, Beta); //移動平均あり
				else if (MouseY < MENUY * 5)MLP();
				else if (MouseY < MENUY * 6) {// 実験用
					for (i = 0; i < TRIALS; i++) {
						Initialize();	// 初期化
						MLP();			// MLP
					}
				}
			}
			Display();
		}
		WaitTimer(100);
	}
	DxLib_End();// ＤＸライブラリ使用の終了処理
	return 0;	// ソフトの終了 
}

void Display() {// ディスプレイ表示
	int i, j, k, mx, my, cr, size = BOXS / RD, b, d, c, dispS = DISPSEG, l, dsize = 8, in, out, rate, p, temp, x, y, s, h, w, disY, poiY;
	double max;
	char fileName[50];
	char menu[][20] = { "END", "Init", "Input", "Forward", "Learn","n-Learn" };

	ClearDrawScreen();
	/* メニュー表示 */
	for (i = 0; i < 6; i++)DrawLine(0, MENUY * (i + 1), MENUX, MENUY * (i + 1), BLACK, 2);// 枠表示
	DrawLine(MENUX, 0, MENUX, MENUY * 6, BLACK, 2);

	for (i = 0; i < 6; i++)DrawString(20 + DSIPLAY, i * MENUY + 12 + DSIPLAY, menu[i], BLACK);// メニュー項目	

	/* 学習アルゴリズム表示 */
	if (NETMODEL == 1) DrawString(OPTIONX, OPTIONY, "CNN-Skip", BLACK);
	else if (NETMODEL == 2) DrawString(OPTIONX, OPTIONY, "SMP-Skip", BLACK);
	else if (NETMODEL == 3) DrawString(OPTIONX, OPTIONY, "SMP-Skip-Prop", BLACK);
	else DrawString(OPTIONX, OPTIONY, "MLP", BLACK);
	if (BATCHLEARN)DrawFormatString(OPTIONX, OPTIONY + 20, BLACK, "BatchLearn %d", BS);

	/* 学習情報表示 */
	DrawFormatString(OPTIONX, OPTIONY + 20 * 2, BLACK, "%s", TrainFileName);
	DrawFormatString(OPTIONX, OPTIONY + 20 * 3, BLACK, "Learning Rate %.3lf", Eta);
	DrawFormatString(OPTIONX, OPTIONY + 20 * 4, BLACK, "Epoc %3d   Time %3d", Epoc, End - Start);
	DrawFormatString(OPTIONX, OPTIONY + 20 * 5, BLACK, "TraAcc %.3lf", TrainAcc);
	DrawFormatString(OPTIONX + 120, OPTIONY + 20 * 5, BLACK, "TesAcc %.3lf", TestAcc);
	DrawFormatString(OPTIONX, OPTIONY + 20 * 12, BLACK, "Loop %d", Loop);
	mx = MENUX + 190; my = MENUY;

	/* 全結合層入力データとアルゴリズムの詳細表示 */
	if (NETMODEL == 2 || NETMODEL == 3) {
		// セグメント数, 畳み込み数, 誘導点数
		DrawFormatString(OPTIONX, OPTIONY + 20 * 6, BLACK, "Seg:%d  Conv:%d", SEG, CONV);
		DrawFormatString(OPTIONX, OPTIONY + 20 * 7, BLACK, "NPINIT:%d  NPMAX:%d", NPINIT, NPMAX);
		DrawFormatString(OPTIONX, OPTIONY + 20 * 8, BLACK, "AllNP:%d", AllNP);
		// 半径、分散
		DrawFormatString(OPTIONX, OPTIONY + 20 * 9, BLACK, "Rad:%.3lf  Sig:%.3lf", R, SIGMA);
		// チャンネル数
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
		for (k = 0; k < K; k++) if (Tk[k] > 0) DrawFormatString(mx + 66, my + 50, BLACK, "%d", k);				// 教師データの表示
		for (k = 0; k < K; k++) DrawFormatString(mx, my + size * RD + k * 20, BLACK, "%.2lf", Out[L - 1][k]);	// 出力の表示
	}

	/* 各モードごとの結果表示 */
	if (NETMODEL == 1) {//畳み込み層入力データの情報
		for (my = 300, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {
			if (COut[0][0][0][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
			else {
				/*cr = (int)(COut[0][0][0][i][j] * 255);*/
				//DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
				cr = GetColor((int)(COut[0][0][0][i][j] * 255), (int)(COut[0][0][0][i][j] * 255), (int)(COut[0][0][0][i][j] * 255));
				DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, cr, TRUE);
			}
		}
		for (my = 400, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//1回畳み込み後入力データの情報
			if (COut[0][1][0][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
			else {
				cr = (int)(COut[0][1][0][i][j] * 255);
				DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
			}
		}
		for (my = 500, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//プーリング後入力データの情報
			if (COut[1][0][0][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
			else {
				cr = (int)(COut[1][0][0][i][j] * 255);
				DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
			}
		}
		if (SEG > 2)
			for (my = 600, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//プーリング後入力データの情報
				if (COut[2][0][0][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
				else {
					cr = (int)(COut[2][0][0][i][j] * 255);
					DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
				}
			}
	}
	if ((NETMODEL == 2 || NETMODEL == 3) && !BATCHLEARN) {
		mx += 10; poiY = 14; disY = 6;
		/* 入出力描画座標を計算・描画 */
		for (c = 0; c < CONV + 1; c++) {
			for (i = 0; i < MAXH - 1 && i < SkipChN[dispS][c]; i++) {
				// 座標計算
				InCIE[c][i][0] = mx + (BOXS + MARGIN) * c * 2;// x座標
				InCIE[c][i][1] = (FLAMEY / MAXH / 2 - BOXS / 2) + (FLAMEY / MAXH) * i - disY * i + poiY;// y座標
				// フレーム描画
				DrawBox(InCIE[c][i][0], InCIE[c][i][1], InCIE[c][i][0] + BOXS, InCIE[c][i][1] + BOXS, WHITE, TRUE);
				DrawBox(InCIE[c][i][0], InCIE[c][i][1], InCIE[c][i][0] + BOXS, InCIE[c][i][1] + BOXS, BLACK, FALSE);
				DrawFormatString(InCIE[c][i][0] + 8, InCIE[c][i][1] + BOXS + 4, BLACK, "Ch[%d]", i);
				// 出力描画
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
			if (i < SkipChN[dispS][c]) {// 残りの描画数が描画可能範囲より大きいなら、最後の値を表示する
				DrawLine(InCIE[c][0][0] - 10, InCIE[c][i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS + 2, InCIE[c][0][0] + BOXS + 10, InCIE[c][i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS + 2, PURPLE);
				DrawLine(InCIE[c][0][0] - 10, InCIE[c][i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS - 2, InCIE[c][0][0] + BOXS + 10, InCIE[c][i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS - 2, PURPLE);
				InCIE[c][i][0] = mx + (BOXS + MARGIN) * c * 2;// w座標
				InCIE[c][i][1] = (FLAMEY / MAXH / 2 - BOXS / 2) + (FLAMEY / MAXH) * i - disY * i + poiY;// h座標
				// フレーム描画
				DrawBox(InCIE[c][i][0], InCIE[c][i][1], InCIE[c][i][0] + BOXS, InCIE[c][i][1] + BOXS, WHITE, TRUE);
				DrawBox(InCIE[c][i][0], InCIE[c][i][1], InCIE[c][i][0] + BOXS, InCIE[c][i][1] + BOXS, BLACK, FALSE);
				DrawFormatString(InCIE[c][i][0] + 8, InCIE[c][i][1] + BOXS + 4, BLACK, "Ch[%d]", SkipChN[dispS][c] - 1);
				// 出力描画
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
			// 層番号描画
			DrawFormatString(InCIE[c][0][0] - 4, 14 + poiY, BLACK, "[%d][%d]層", dispS, c);
		}
		/* 特徴マップ描画 */
		for (c = 0; c < CONV; c++) {
			for (j = 0; j < SkipChN[dispS][c + 1]; j++) {
				for (i = 0; i < SkipChN[dispS][c]; i++) {
					if (j * SkipChN[dispS][c] + i >= MAXH - 1)break;
					// 座標計算
					MapCIE[c][j * SkipChN[dispS][c] + i][0] = mx + (BOXS + MARGIN) * c * 2 + (BOXS + MARGIN);// x座標
					MapCIE[c][j * SkipChN[dispS][c] + i][1] = (FLAMEY / MAXH / 2 - BOXS / 2) + (FLAMEY / MAXH) * (j * SkipChN[dispS][c] + i) - disY * i + poiY;// y座標
					// 枠描画
					DrawBox(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1], MapCIE[c][j * SkipChN[dispS][c] + i][0] + BOXS, MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS, WHITE, TRUE);
					DrawBox(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1], MapCIE[c][j * SkipChN[dispS][c] + i][0] + BOXS, MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS, BLACK, FALSE);
					DrawFormatString(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS + 4, BLACK, "W[%d][%d]", j, i);
					// 重み表示線
					DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 8, MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 8, BLACK);				// 重み表示線：横線
					DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 13, MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 2, BLACK);							// 重み表示線：縦線(始点)
					DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 13, MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 2, BLACK);	// 重み表示線：横線(終点)
					// 重みの数値表示(点)
					for (p = 0; p < NP[dispS][c][i]; p++) {
						if (Wi[dispS][c][j][i][p] >= 0) {// 重みが正の点
							cr = 255;
							DrawPixel(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][i][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][i][p][1] * (RD * size - 1), GetColor(cr, 0, 0));// 座標
							DrawCircle(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][i][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][i][p][1] * (RD * size - 1), Ri[dispS][c][i][p] * (RD * size - 1), GetColor(cr, 0, 0), FALSE);// 範囲
							DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Wi[dispS][c][j][i][p] * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 11, MapCIE[c][j * SkipChN[dispS][c] + i][0] + Wi[dispS][c][j][i][p] * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 4, GetColor(cr, 0, 0));// 重み表示線
						}
						else {// 重みが負の点
							cr = 255;
							DrawPixel(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][i][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][i][p][1] * (RD * size - 1), GetColor(0, 0, cr));// 座標
							DrawCircle(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][i][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][i][p][1] * (RD * size - 1), Ri[dispS][c][i][p] * (RD * size - 1), GetColor(0, 0, cr), FALSE);// 範囲
							DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + (-Wi[dispS][c][j][i][p]) * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 11, MapCIE[c][j * SkipChN[dispS][c] + i][0] + (-Wi[dispS][c][j][i][p]) * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 4, GetColor(0, 0, cr));// 重み表示線
						}
					}
				}
				if (j * SkipChN[dispS][c] + i >= MAXH - 1)break;
			}
			if (j * SkipChN[dispS][c] + i < SkipChN[dispS][c] * SkipChN[dispS][c] - 1) {// 残りの描画数が描画可能範囲より大きいなら、最後の値を表示する
				// 分割ライン表示
				DrawLine(MapCIE[c][0][0] - 10, MapCIE[c][j * SkipChN[dispS][c] + i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS + 2, MapCIE[c][0][0] + BOXS + 10, MapCIE[c][j * SkipChN[dispS][c] + i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS + 2, PURPLE);
				DrawLine(MapCIE[c][0][0] - 10, MapCIE[c][j * SkipChN[dispS][c] + i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS - 2, MapCIE[c][0][0] + BOXS + 10, MapCIE[c][j * SkipChN[dispS][c] + i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS - 2, PURPLE);
				// 距離計算
				MapCIE[c][j * SkipChN[dispS][c] + i][0] = mx + (BOXS + MARGIN) * c * 2 + (BOXS + MARGIN);// x座標
				MapCIE[c][j * SkipChN[dispS][c] + i][1] = (FLAMEY / MAXH / 2 - BOXS / 2) + (FLAMEY / MAXH) * (j * SkipChN[dispS][c] + i) - disY * i + poiY;// y座標
				// 枠表示
				DrawBox(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1], MapCIE[c][j * SkipChN[dispS][c] + i][0] + BOXS, MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS, WHITE, TRUE);
				DrawBox(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1], MapCIE[c][j * SkipChN[dispS][c] + i][0] + BOXS, MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS, BLACK, FALSE);
				DrawFormatString(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS + 4, BLACK, "W[%d][%d]", SkipChN[dispS][c + 1] - 1, SkipChN[dispS][c] - 1);
				// 誘導点描画
				DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 8, MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 8, BLACK);				// 重み表示線：横線
				DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 13, MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 2, BLACK);							// 重み表示線：縦線(始点)
				DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 13, MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 2, BLACK);	// 重み表示線：横線(終点)
				for (p = 0; p < NP[dispS][c][i]; p++) {
					if (Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p] >= 0) {// 重みが正の点
						cr = 255;
						DrawPixel(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][1] * (RD * size - 1), GetColor(cr, 0, 0));// 座標
						DrawCircle(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][1] * (RD * size - 1), Ri[dispS][c][SkipChN[dispS][c] - 1][p] * (RD * size - 1), GetColor(cr, 0, 0), FALSE);// 範囲
						DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p] * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 11, MapCIE[c][j * SkipChN[dispS][c] + i][0] + Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p] * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 4, GetColor(cr, 0, 0));// 重み表示線
					}
					else {// 重みが負の点
						cr = 255;
						DrawPixel(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][1] * (RD * size - 1), GetColor(0, 0, cr));// 座標
						DrawCircle(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][1] * (RD * size - 1), Ri[dispS][c][SkipChN[dispS][c] - 1][p] * (RD * size - 1), GetColor(0, 0, cr), FALSE);// 範囲
						DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + (-Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p]) * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 11, MapCIE[c][j * SkipChN[dispS][c] + i][0] + (-Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p]) * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 4, GetColor(0, 0, cr));// 重み表示線
					}

				}
			}
		}
	}

	/* バッチ学習表示 */
	if (BATCHLEARN) {
		if ((NETMODEL == 2 || NETMODEL == 3)) {
			/* 特徴マップ描画 */
			mx += 10; poiY = 14; disY = 6;
			for (c = 0; c < CONV; c++) {
				for (j = 0; j < SkipChN[dispS][c + 1]; j++) {
					for (i = 0; i < SkipChN[dispS][c]; i++) {
						if (j * SkipChN[dispS][c] + i >= MAXH - 1)break;
						// 座標計算
						MapCIE[c][j * SkipChN[dispS][c] + i][0] = mx + (BOXS + MARGIN) * c * 2 + (BOXS + MARGIN);// x座標
						MapCIE[c][j * SkipChN[dispS][c] + i][1] = (FLAMEY / MAXH / 2 - BOXS / 2) + (FLAMEY / MAXH) * (j * SkipChN[dispS][c] + i) - disY * i + poiY;// y座標
						// 枠描画
						DrawBox(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1], MapCIE[c][j * SkipChN[dispS][c] + i][0] + BOXS, MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS, WHITE, TRUE);
						DrawBox(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1], MapCIE[c][j * SkipChN[dispS][c] + i][0] + BOXS, MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS, BLACK, FALSE);
						DrawFormatString(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS + 4, BLACK, "W[%d][%d]", j, i);
						// 重み表示線
						DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 8, MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 8, BLACK);				// 重み表示線：横線
						DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 13, MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 2, BLACK);							// 重み表示線：縦線(始点)
						DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 13, MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 2, BLACK);	// 重み表示線：横線(終点)
						// 重みの数値表示(点)
						for (p = 0; p < NP[dispS][c][i]; p++) {
							if (Wi[dispS][c][j][i][p] >= 0) {// 重みが正の点
								cr = 255;
								DrawPixel(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][i][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][i][p][1] * (RD * size - 1), GetColor(cr, 0, 0));// 座標
								DrawCircle(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][i][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][i][p][1] * (RD * size - 1), Ri[dispS][c][i][p] * (RD * size - 1), GetColor(cr, 0, 0), FALSE);// 範囲
								DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Wi[dispS][c][j][i][p] * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 11, MapCIE[c][j * SkipChN[dispS][c] + i][0] + Wi[dispS][c][j][i][p] * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 4, GetColor(cr, 0, 0));// 重み表示線
							}
							else {// 重みが負の点
								cr = 255;
								DrawPixel(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][i][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][i][p][1] * (RD * size - 1), GetColor(0, 0, cr));// 座標
								DrawCircle(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][i][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][i][p][1] * (RD * size - 1), Ri[dispS][c][i][p] * (RD * size - 1), GetColor(0, 0, cr), FALSE);// 範囲
								DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + (-Wi[dispS][c][j][i][p]) * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 11, MapCIE[c][j * SkipChN[dispS][c] + i][0] + (-Wi[dispS][c][j][i][p]) * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 4, GetColor(0, 0, cr));// 重み表示線
							}
						}
					}
					if (j * SkipChN[dispS][c] + i >= MAXH - 1)break;
				}
				if (j * SkipChN[dispS][c] + i < SkipChN[dispS][c] * SkipChN[dispS][c] - 1) {// 残りの描画数が描画可能範囲より大きいなら、最後の値を表示する
					// 分割ライン表示
					DrawLine(MapCIE[c][0][0] - 10, MapCIE[c][j * SkipChN[dispS][c] + i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS + 2, MapCIE[c][0][0] + BOXS + 10, MapCIE[c][j * SkipChN[dispS][c] + i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS + 2, PURPLE);
					DrawLine(MapCIE[c][0][0] - 10, MapCIE[c][j * SkipChN[dispS][c] + i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS - 2, MapCIE[c][0][0] + BOXS + 10, MapCIE[c][j * SkipChN[dispS][c] + i - 1][1] + (FLAMEY / MAXH / 2 - BOXS / 2) + BOXS - 2, PURPLE);
					// 距離計算
					MapCIE[c][j * SkipChN[dispS][c] + i][0] = mx + (BOXS + MARGIN) * c * 2 + (BOXS + MARGIN);// x座標
					MapCIE[c][j * SkipChN[dispS][c] + i][1] = (FLAMEY / MAXH / 2 - BOXS / 2) + (FLAMEY / MAXH) * (j * SkipChN[dispS][c] + i) - disY * i + poiY;// y座標
					// 枠表示
					DrawBox(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1], MapCIE[c][j * SkipChN[dispS][c] + i][0] + BOXS, MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS, WHITE, TRUE);
					DrawBox(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1], MapCIE[c][j * SkipChN[dispS][c] + i][0] + BOXS, MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS, BLACK, FALSE);
					DrawFormatString(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] + BOXS + 4, BLACK, "W[%d][%d]", SkipChN[dispS][c + 1] - 1, SkipChN[dispS][c] - 1);
					// 誘導点描画
					DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 8, MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 8, BLACK);				// 重み表示線：横線
					DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 13, MapCIE[c][j * SkipChN[dispS][c] + i][0], MapCIE[c][j * SkipChN[dispS][c] + i][1] - 2, BLACK);							// 重み表示線：縦線(始点)
					DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 13, MapCIE[c][j * SkipChN[dispS][c] + i][0] + RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 2, BLACK);	// 重み表示線：横線(終点)
					for (p = 0; p < NP[dispS][c][i]; p++) {
						if (Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p] >= 0) {// 重みが正の点
							cr = 255;
							DrawPixel(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][1] * (RD * size - 1), GetColor(cr, 0, 0));// 座標
							DrawCircle(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][1] * (RD * size - 1), Ri[dispS][c][SkipChN[dispS][c] - 1][p] * (RD * size - 1), GetColor(cr, 0, 0), FALSE);// 範囲
							DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p] * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 11, MapCIE[c][j * SkipChN[dispS][c] + i][0] + Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p] * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 4, GetColor(cr, 0, 0));// 重み表示線
						}
						else {// 重みが負の点
							cr = 255;
							DrawPixel(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][1] * (RD * size - 1), GetColor(0, 0, cr));// 座標
							DrawCircle(MapCIE[c][j * SkipChN[dispS][c] + i][0] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][0] * (RD * size - 1), MapCIE[c][j * SkipChN[dispS][c] + i][1] + Pi[dispS][c][SkipChN[dispS][c] - 1][p][1] * (RD * size - 1), Ri[dispS][c][SkipChN[dispS][c] - 1][p] * (RD * size - 1), GetColor(0, 0, cr), FALSE);// 範囲
							DrawLine(MapCIE[c][j * SkipChN[dispS][c] + i][0] + (-Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p]) * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 11, MapCIE[c][j * SkipChN[dispS][c] + i][0] + (-Wi[dispS][c][SkipChN[dispS][c + 1] - 1][SkipChN[dispS][c] - 1][p]) * RD * size, MapCIE[c][j * SkipChN[dispS][c] + i][1] - 4, GetColor(0, 0, cr));// 重み表示線
						}
					}
				}
			}
		}
		else {
			for (mx = MENUX + 269, b = 0; b < BS && b < 10; b++, mx += 79) {
				for (my = MENUY, i = 0; i < RD; i++) for (j = 0; j < RD && (i * RD + j) < NodeN[0]; j++) {//全結合層入力データの情報
					if (BOut[0][b][i * RD + j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
					else {
						cr = (int)(BOut[0][b][i * RD + j] * 255);
						DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
					}
				}
				for (k = 0; k < K; k++) if (BTk[b][k] > 0) DrawFormatString(mx + 58, my + 50, BLACK, "%d", k);
				for (k = 0; k < K; k++) // 出力の表示
					DrawFormatString(mx, my + size * RD + k * 20, BLACK, "%.2lf", BOut[L - 1][b][k]);
				if (NETMODEL == 10) {//畳み込み層入力データの情報
					for (my = 300, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {
						if (BCOut[0][0][0][b][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
						else {
							cr = (int)(BCOut[0][0][0][b][i][j] * 255);
							DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
						}
					}
					for (my = 360, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//1回畳み込み後入力データの情報
						if (BCOut[0][1][0][b][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
						else {
							cr = (int)(BCOut[0][1][0][b][i][j] * 255);
							DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
						}
					}
					for (my = 420, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//プーリング後入力データの情報
						if (BCOut[1][0][0][b][i][j] == 0) DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, 255, TRUE);
						else {
							cr = (int)(BCOut[1][0][0][b][i][j] * 255);
							DrawBox(mx + j * size, my + i * size, mx + (j + 1) * size, my + (i + 1) * size, GetColor(cr, cr, cr), TRUE);
						}
					}
					if (SEG > 2) {
						for (my = 480, i = 0; i < RD; i++) for (j = 0; j < RD; j++) {//プーリング後入力データの情報
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

void ReadFileData() {// 学習・テストデータ読み込み
	FILE* fp;
	int i, n;

	/* --------- 学習データ読み込み --------- */
	fopen_s(&fp, TrainFileName, "r");
	for (n = 0; n < TRA; n++) {
		for (i = 0; i < IND; i++) {
			fscanf_s(fp, "%d,", &TraData[n][i]);
		}
		fscanf_s(fp, "%d\n", &TraClass[n]);
	}
	fclose(fp);

	/* -------- テストデータ読み込み -------- */
	fopen_s(&fp, TestFileName, "r");
	for (n = 0; n < TES; n++) {
		for (i = 0; i < IND; i++) {
			fscanf_s(fp, "%d,", &TesData[n][i]);
		}
		fscanf_s(fp, "%d\n", &TesClass[n]);
	}
	fclose(fp);
}

void Initialize() {// パラメータの初期化
	int i, j, k, l, s, c, y, x, d, m, och, r = sqrt(NPINIT);
	double p1, p2, he_wp;

	/* ---------- MLP入力層数の初期化 ---------- */
	if (NETMODEL == 1) NodeN[0] = ((H - IKS + 1) / pow(PS, SEG)) * ((W - IKS + 1) / pow(PS, SEG)) * CHMAX;
	else if (NETMODEL == 2 || NETMODEL == 3) NodeN[0] = ((H - IKS + 1) / pow(PS, SEG)) * ((W - IKS + 1) / pow(PS, SEG)) * CHMAX;

	/* --------------- MLP初期化 --------------- */
	for (l = 0; l < L; l++)for (i = 0; i < NodeN[l]; i++) {
		Gamma[l][i] = 1.0; Beta[l][i] = 0;
		MMean[l][i] = 0; MVar[l][i] = 1;
	}
	if (RELU) { //He初期化
		for (l = 0; l < L - 2; l++) {//中間層外側　ReLUする場合もある
			he_wp = 2 * sqrt(6.0 / NodeN[l]);
			for (j = 0; j < NodeN[l + 1]; j++)//出力側
				for (Bias[l][j] = 0, i = 0; i < NodeN[l]; i++) //入力側
					Woi[l][j][i] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp;
		}
	}
	else {
		for (l = 0; l < L - 2; l++) //中間層外側　ReLUする場合もある
			for (j = 0; j < NodeN[l + 1]; j++)//出力側
				for (Bias[l][j] = 0, i = 0; i < NodeN[l]; i++) //入力側
					Woi[l][j][i] = (double)rand() / (RAND_MAX + 1.0) - 0.5; //+-0.5
	}
	for (j = 0; j < NodeN[l + 1]; j++) {//出力側
		for (Bias[l][j] = 0, i = 0; i < NodeN[l]; i++) //入力側
			Woi[l][j][i] = (double)rand() / (RAND_MAX + 1.0) - 0.5;
	}

	/* ---------- Skip-Connection-CNN ---------- */
	if (NETMODEL == 1) {//CNN(Skip) 重み初期化
		for (he_wp = 2 * sqrt(6.0 / (ICH * IKS * IKS)), j = 0; j < SkipChN[0][0]; j++) {
			for (InBias[j] = 0, i = 0; i < ICH; i++)
				for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
					InWoi[j][i][y][x] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He初期化
		}

		for (s = 0; s < SEG - 1; s++) {
			for (he_wp = 2 * sqrt(6.0 / (SkipChN[s][CONV] * KS * KS)), j = 0; j < SkipChN[s + 1][0]; j++)
				for (SkipBias[s][j] = 0, i = 0; i < SkipChN[s][CONV]; i++)
					for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
						SkipWoi[s][j][i][y][x] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He初期化
		}

		// バッチ正規化
		for (s = 0; s < SEG; s++) for (c = 0; c <= CONV; c++)for (j = 0; j < CHMAX; j++) {
			CGamma[s][c][j] = 1.0; CBeta[s][c][j] = 0;
			CMMean[s][c][j] = 0; CMVar[s][c][j] = 1.0;
		}

		// 重み・バイアス
		for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++)
			for (he_wp = 2 * sqrt(6.0 / (SkipChN[s][c] * KS * KS)), j = 0; j < SkipChN[s][c + 1]; j++)
				for (CBias[s][c][j] = 0, i = 0; i < SkipChN[s][c]; i++)
					for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
						CWoi[s][c][j][i][y][x] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He初期化
	}

	/* ---------- Skip-Connection-SMP ---------- */
	if (NETMODEL == 2 || NETMODEL == 3) {
		/* ------ CNN ------ */
		// InConv重み・バイアス
		for (he_wp = 2 * sqrt(6.0 / (ICH * IKS * IKS)), j = 0; j < SkipChN[0][0]; j++) {
			for (InBias[j] = 0, i = 0; i < ICH; i++)
				for (y = 0; y < IKS; y++)for (x = 0; x < IKS; x++)
					InWoi[j][i][y][x] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He初期化
		}
		// SkipOut重み・バイアス
		for (s = 0; s < SEG - 1; s++) {
			for (he_wp = 2 * sqrt(6.0 / (SkipChN[s][CONV] * KS * KS)), j = 0; j < SkipChN[s + 1][0]; j++)
				for (SkipBias[s][j] = 0, i = 0; i < SkipChN[s][CONV]; i++)
					for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
						SkipWoi[s][j][i][y][x] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He初期化
		}
		// バッチ正規化
		for (s = 0; s < SEG; s++) for (c = 0; c <= CONV; c++)for (j = 0; j < CHMAX; j++) {
			CGamma[s][c][j] = 1.0; CBeta[s][c][j] = 0;
			CMMean[s][c][j] = 0; CMVar[s][c][j] = 1;
		}
		// 重み・バイアス
		for (s = 0; s < SEG; s++) for (c = 0; c < CONV; c++) {
			for (he_wp = 2 * sqrt(6.0 / (SkipChN[s][c] * KS * KS)), j = 0; j < SkipChN[s][c + 1]; j++)
				for (CBias[s][c][j] = 0, i = 0; i < SkipChN[s][c]; i++)
					for (y = 0; y < KS; y++)for (x = 0; x < KS; x++)
						CWoi[s][c][j][i][y][x] = ((double)rand() / (RAND_MAX + 1.0) - 0.5) * he_wp; //He初期化
		}
		/* ------ SMP ------ */
		// 誘導点数初期化
		for (AllNP = 0, s = 0; s < SEG; s++) {// セグメント数
			for (c = 0; c < CONV; c++)// 畳み込み数
				for (j = 0; j < SkipChN[s][c]; j++) {// チャンネル
					if (POINTINIT == 2) {// チャンネルごとに誘導点数を変更する
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
		for (s = 0; s < SEG; s++) {// セグメント数
			for (c = 0; c < CONV; c++) {// 畳み込み数
				// 重み・半径・座標初期化
				for (j = 0; j < CHMAX; j++) {// 出力チャンネル
					for (i = 0; i < CHMAX; i++)// 入力チャンネル
						for (m = 0; m < NPMAX; m++) {// 誘導点数
							Wi[s][c][j][i][m] = (double)rand() / (RAND_MAX + 1.0) - 0.5; //+-0.5
						}
				}
				for (i = 0; i < CHMAX; i++) {
					for (m = 0; m < NPMAX; m++) {// 誘導点数
						// 削除カウント初期化
						DelC[s][c][i][m] = 0;
						// 座標初期化
						if (POINTINIT == 1 && r * r == NPINIT) {
							/* 正方形 */
							Pi[s][c][i][m][0] = (1.0 / r) * (m % r) + (1.0 / r) / 2.0;
							Pi[s][c][i][m][1] = (1.0 / r) * (m / r) + (1.0 / r) / 2.0;
						}
						else if (POINTINIT == 2 && r * r == NPINIT && s == 0) {
							/* 正方形 */
							Pi[s][c][i][m][0] = (1.0 / r) * (m % r) + (1.0 / r) / 2.0;
							Pi[s][c][i][m][1] = (1.0 / r) * (m / r) + (1.0 / r) / 2.0;
						}
						else {
							/* ガウス分布 */
							for (d = 0; d < SD; d++) {// 次元数
								while (1) {
									// ガウス分布による確率(Box-Muller法)
									p1 = (double)rand() / (RAND_MAX);// 一様乱数生成
									p2 = (double)rand() / (RAND_MAX);// 一様乱数生成
									//p1 = sqrt(-2.0 * log(p1)) * sin(2 * M_PI * p2);
									p1 = sqrt(-2.0 * log(p1)) * cos(2 * M_PI * p2);// Box-Muller法
									p1 = MU + SIGMA * p1;// 線形変換
									if (p1 > 0 && p1 < 1) {// 生成された確率が範囲内
										Pi[s][c][i][m][d] = p1;
										break;
									}
								}
							}
						}
						// 半径初期化
						Ri[s][c][i][m] = R;
					}
				}
			}
		}
		// 変化量の合計値初期化
		SOIPInit(DPsum, DRsum);
	}
}

void Input(unsigned char indata[], double out[]) {
	int i;
	for (i = 0; i < IND; i++) out[i] = indata[i] / (double)MAXV;// 正規化
}

void CInput(unsigned char indata[], double out[][H][W]) {
	int i, h, w, n;
	for (n = 0, i = 0; i < ICH; i++)
		for (h = 0; h < H; h++)for (w = 0; w < W; w++, n++)
			out[i][h][w] = indata[n] / (double)MAXV;// 正規化
}

void MLP() {
	int d, in, k, count, maxk, b, s, c, j, i, temp, h, w;
	int addN, delN, merN;
	double a;
	FILE* fp;
	fopen_s(&fp, SaveFileName, "a");
	fprintf_s(fp, "\nEpoc,TrainAcc,TestAcc,TrainErr,TestErr,Time,LTime,Eta,Point,Del,Mer,Add\n");
	fclose(fp);

	// 学習率初期化
	if (BATCHLEARN == 1)Eta = BETA;
	else Eta = ETA;
	// 削除条件初期化
	if (SINGROWTH == 1)DelW = 0;
	else DelW = DELW;
	// 時間初期化
	Start = time(NULL);

	/* ---------------------- 学習 ---------------------- */
	if (DEBUG_MODE)DebugInit();// Debug
	for (AccTime = 0, SOIPCount = 0, Epoc = 0; Epoc < EPOC; Epoc++) {
		// 初期化
		LStart = time(NULL);
		delN = 0; merN = 0; addN = 0;
		// 学習データ作成
		if (RANDMODE == 1) {
			for (i = 0; i < TRA; i++)TraNum[i] = i;
			for (i = 0; i < TRA; i++) {
				in = (rand() / (RAND_MAX + 1.0)) * (TRA - i) + i;
				temp = TraNum[i];
				TraNum[i] = TraNum[in];
				TraNum[in] = temp;
			}
		}
		// 学習ルール設定
		if (LEARNRULE == 1) {// cosine decay
			if (BATCHLEARN == 1) {// バッチ学習
				if (Epoc < WARMUP) Eta = BETA * WARMUPRATE;
				else Eta = 0.5 * (1 + cos(((Epoc - WARMUP) * M_PI) / (EPOC))) * BETA;
			}
			else {// 通常
				if (Epoc < WARMUP) Eta = ETA * WARMUPRATE;
				else Eta = 0.5 * (1 + cos(((Epoc - WARMUP) * M_PI) / (EPOC))) * ETA;
			}
		}
		// 学習
		if (BATCHLEARN) {
			for (count = 0, d = 0; d < TRA - BS; d += BS, SOIPCount++) {
				/* 教師データ生成 */
				for (b = 0; b < BS; b++) {// バッチ数
					/* 乱数生成(入力データ番号) */
					if (RANDMODE == 1)in = TraNum[d + b];
					else in = (rand() / (RAND_MAX + 1.0)) * TRA;
					/* 入力データ作成 */
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
					/* 教師データ作成 */
					if (LS == 1) {// ラベル平滑化
						for (k = 0; k < K; k++) BTk[b][k] = LSEPS / (K - 1);
						BTk[b][TraClass[in]] = 1.0 - LSEPS;//Hot One Vecter
					}
					else {
						for (k = 0; k < K; k++) BTk[b][k] = 0;
						BTk[b][TraClass[in]] = 1.0;//Hot One Vecter
					}
				}
				/* オプション + MLPの入力層作成 */
				if (NETMODEL == 1) {// SkipCNN
					BSkipForward(SkipChN, CBias, CWoi, BCOut, BOut[0], CBNet, CMean, CVar, CGamma, CBeta, CMMean, CMVar,
						BInOut, InBias, InWoi, BSkipOut, SkipBias, SkipWoi);
				}
				else if (NETMODEL == 2 || NETMODEL == 3) {// SkipSMP
					BSMPSkipForward(SkipChN, CBias, CWoi, BCOut, BOut[0], CGamma, CBeta, CMMean, CMVar, BInOut, InBias, InWoi,
						BSkipOut, SkipBias, SkipWoi, Pi, Ri, Wi, NP, CIE, Dis, G, SumCount, CBNet, CMean, CVar);
				}
				/* MLP(学習) */
				BForward(NodeN, Bias, Woi, BOut, BNet, Mean, Var, Gamma, Beta, MMean, MVar);
				BBackProp(NodeN, Bias, Woi, BOut, BTk, BDelta, BNet, Mean, Var, Gamma, Beta, Eta);
				/* オプションの学習 */
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
				/* 学習精度 */
				for (b = 0; b < BS; b++) {
					for (maxk = 0, k = 1; k < K; k++)
						if (BOut[L - 1][b][maxk] < BOut[L - 1][b][k]) maxk = k;
					if (BTk[b][maxk] > 0.5)count++;
				}
				/* 誘導点の増減(提案手法) */
				if (NETMODEL == 3) {
					if (Epoc == SOIPSTART && d == 0) {// 誘導点増加開始直後
						SOIPCount = 0;// 初期化
						SOIPInit(DPsum, DRsum);
					}
					else if (Epoc >= SOIPSTART && Epoc <= SOIPEND) {// 増減アルゴリズムの適用範囲
						if (SOIPCount % BSOIPTIME == 0) {// 呼び出す周期
							// SineGrowthを使用（削除用）
							if (SINGROWTH == 1)DelW = 0.5 * (1 + sin((((Epoc - SOIPSTART) * TRA + d) - ((SOIPEND - SOIPSTART + 1) * TRA) / 2.0) * M_PI / ((SOIPEND - SOIPSTART + 1) * TRA))) * DELW;// Sine Growth
							// 誘導点の増減
							SOIP(SkipChN, Pi, Ri, Wi, NP, &AllNP, DelC, DPsum, DRsum, Add, DelW, Eta, &delN, &merN, &addN);// 誘導点の増減
							SOIPInit(DPsum, DRsum);// DPsumを初期化する
						}
					}
				}
			}
		}
		else {
			for (count = 0, d = 0; d < TRA; d++) {
				/* 教師データ生成 */
				if (RANDMODE == 1)in = TraNum[d];
				else in = (rand() / (RAND_MAX + 1.0)) * TRA;

				/* オプション + MLPの入力層作成 */
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
				/* 教師データ作成 */
				if (LS == 1) {// ラベル平滑化
					for (k = 0; k < K; k++) Tk[k] = LSEPS / (K - 1);
					Tk[TraClass[in]] = 1.0 - LSEPS;//Hot One Vecter
				}
				else {
					for (k = 0; k < K; k++) Tk[k] = 0;
					Tk[TraClass[in]] = 1.0;//Hot One Vecter
				}
				/* MLP(学習) */
				Forward(NodeN, Bias, Woi, Out, MMean, MVar, Gamma, Beta);
				BackProp(NodeN, Bias, Woi, Out, Tk, Delta, Eta);
				/* オプションの学習 */
				if (NETMODEL == 1) {
					SkipBackProp(SkipChN, CBias, CWoi, COut, CDelta, Delta[0], DCBias, DCWoi, Eta, InOut, InBias, InWoi,
						SkipOut, SkipDelta, SkipBias, SkipWoi, DSkipBias, DSkipWoi, DInBias, DInWoi);// CNN-Skip
				}
				else if (NETMODEL == 2 || NETMODEL == 3) {
					SMPSkipBackProp(SkipChN, CBias, CWoi, COut, CDelta, Delta[0], DCBias, DCWoi, Eta, InOut, InBias, InWoi, SkipOut, SkipDelta, SkipBias, SkipWoi, DSkipBias, DSkipWoi, DInBias, DInWoi,
						Pi, Ri, Wi, NP, CIE, Dis, G, SumCount, DPi, DRi, DWi, WUseCount, PRUseCount, DPsum, DRsum);
				}
				if (DEBUG_MODE)DebugSum();// Debug
				/* 精度計算 */
				for (maxk = 0, k = 1; k < K; k++)
					if (Out[L - 1][maxk] < Out[L - 1][k]) maxk = k;
				if (maxk == TraClass[in])count++;
				/* 誘導点の増減(提案手法) */
				if (NETMODEL == 3) {
					if (Epoc == SOIPSTART && d == 0) {// 誘導点増加開始直後
						SOIPCount = 0;// 初期化
						SOIPInit(DPsum, DRsum);// DPsumを初期化する
					}
					else if (Epoc >= SOIPSTART && Epoc <= SOIPEND) {// 増減アルゴリズムの適用範囲
						if (SOIPCount % SOIPTIME == 0) {// 呼び出す周期
							// SineGrowthを使用（削除用）
							if (SINGROWTH == 1)DelW = 0.5 * (1 + sin((((Epoc - SOIPSTART) * TRA + d) - ((SOIPEND - SOIPSTART + 1) * TRA) / 2.0) * M_PI / ((SOIPEND - SOIPSTART + 1) * TRA))) * DELW;// Sine Growth
							// 誘導点の増減
							SOIP(SkipChN, Pi, Ri, Wi, NP, &AllNP, DelC, DPsum, DRsum, Add, DelW, Eta, &delN, &merN, &addN);// 誘導点の増減
							SOIPInit(DPsum, DRsum);// DPsumを初期化する
						}
					}
				}
			}
		}
		End = time(NULL);
		AccTime += End - LStart;// 学習時間のみ記憶
		STest(delN, merN, addN);// TraAcc,TesAcc計測
		Display();
	}
}

void STest(int delN, int merN, int addN) {// test時は移動平均を利用する　（バッチなし）
	int t, k, in, max, count, lay = L - 1;
	FILE* testfp, * paramDWsumfp, * paramDPsumfp, * paramDRsumfp, * paramWfp, * paramPfp, * paramRfp;

	if (fopen_s(&testfp, SaveFileName, "a")) return;		//file open 失敗
	for (TrainErr = 0, t = 0, count = 0; t < TRA; t++) { // 学習データ
		in = t;
		/* オプション + MLPの入力層作成 */
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
		if (TraClass[t] == max) count++;//精度
		for (k = 0; k < K; k++)
			if (k == TraClass[t])TrainErr += (1.0 - Out[lay][k]) * (1.0 - Out[lay][k]);
			else TrainErr += (Out[lay][k]) * (Out[lay][k]);
	}
	TrainAcc = (double)count / (double)(t); TrainErr /= (double)(t);
	for (TestErr = 0, t = 0, count = 0; t < TES; t++) {
		in = t;
		/* オプション + MLPの入力層作成 */
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
		if (TesClass[t] == max) count++;//精度	
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

	// パラメータチェック
	if (DEBUG_MODE) {
		int s, c, j, i, m, d;
		double dis;
		if (fopen_s(&paramDWsumfp, SaveParamDWsumFileName, "a")) return;	//file open 失敗
		if (fopen_s(&paramDPsumfp, SaveParamDPsumFileName, "a")) return;	//file open 失敗
		if (fopen_s(&paramDRsumfp, SaveParamDRsumFileName, "a")) return;	//file open 失敗
		if (fopen_s(&paramWfp, SaveParamWFileName, "a")) return;	//file open 失敗
		if (fopen_s(&paramPfp, SaveParamPFileName, "a")) return;	//file open 失敗
		if (fopen_s(&paramRfp, SaveParamRFileName, "a")) return;	//file open 失敗
		// 変化量：重み
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
		// 変化量：座標
		fprintf_s(paramDPsumfp, "\nEpoc,Seg,Conv,InCh\n");
		for (s = 0; s < SEG; s++) {
			for (c = 0; c < CONV; c++)
				for (i = 0; i < SkipChN[s][c]; i++) {
					fprintf_s(paramDPsumfp, "%d,%d,%d,%d", Epoc, s, c, i);
					for (m = 0; m < NP[s][c][i]; m++) {
						dis = sqrt(DebugDPsum[s][c][i][m][0] * DebugDPsum[s][c][i][m][0] + DebugDPsum[s][c][i][m][1] * DebugDPsum[s][c][i][m][1]);// L2距離
						fprintf_s(paramDPsumfp, ",%lf", dis);
					}
					fprintf_s(paramDPsumfp, "\n");
				}
			fprintf_s(paramDPsumfp, "\n");
		}
		// 変化量：半径
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
		// 重み
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
		// 座標
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
		// 半径
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
	// 重み
	for (s = 0; s < SEG; s++) {
		for (c = 0; c < CONV; c++)
			for (j = 0; j < SkipChN[s][c + 1]; j++)
				for (i = 0; i < SkipChN[s][c]; i++)
					for (m = 0; m < NP[s][c][i]; m++) {
						DebugDWsum[s][c][j][i][m] = 0;
					}
	}
	// SMP座標・半径
	for (s = 0; s < SEG; s++) {
		for (c = 0; c < CONV; c++)
			for (i = 0; i < SkipChN[s][c]; i++)
				for (m = 0; m < NP[s][c][i]; m++) {
					DebugDRsum[s][c][i][m] = 0;// 半径の変化量
					for (d = 0; d < SD; d++) {
						DebugDPsum[s][c][i][m][d] = 0;// 座標の変化量
					}
				}
	}
}

void DebugSum() {
	int s, c, j, i, m, d;
	// 重み
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
	// SMP座標・半径
	for (s = 0; s < SEG; s++) {
		for (c = 0; c < CONV; c++)
			for (i = 0; i < SkipChN[s][c]; i++)
				for (m = 0; m < NP[s][c][i]; m++) {
					if (PRUseCount[s][c][i][m] != 0) {
						if (DRi[s][c][i][m] >= 0)DebugDRsum[s][c][i][m] += Eta * DRi[s][c][i][m] / PRUseCount[s][c][i][m];	// 半径の変化量
						else DebugDRsum[s][c][i][m] -= Eta * DRi[s][c][i][m] / PRUseCount[s][c][i][m];	// 半径の変化量

						for (d = 0; d < SD; d++) {
							if (DPi[s][c][i][m][d] >= 0)DebugDPsum[s][c][i][m][d] += Eta * DPi[s][c][i][m][d] / PRUseCount[s][c][i][m];	// 座標の変化量
							else DebugDPsum[s][c][i][m][d] -= Eta * DPi[s][c][i][m][d] / PRUseCount[s][c][i][m];	// 座標の変化量
						}
					}
				}
	}
	// 初期化
	for (s = 0; s < SEG; s++) {
		for (c = 0; c < CONV; c++) {
			DebugDPave[s][c] = 0;
			DebugDRave[s][c] = 0;
			DebugDWave[s][c] = 0;
		}
	}
	// 重みの平均
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
	// 座標の平均、半径の重み
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