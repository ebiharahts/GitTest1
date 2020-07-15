# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.5.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # +
################################################
#  背景差分を用いた物体検知 BackSub_v1.1
#    Copyright © 2020 Retail AI Lab,Inc.
#
#  Magic nuber埋め込みを止め、定数として切り出し
################################################
import cv2
import numpy as np

# (定数) ------------------------------------------------------------
# 短蓄、長蓄の合成比率
ACCUM_RATE_01 = 0.1    # 短期蓄積比率
ACCUM_RATE_001 = 0.02  # 長期蓄積比率
# フィルター最終処理
THRES_VAL_MIN = 100  # 2値化する際の最小値 ～255
# 外れ値除去
GS_VAL_MIN = 0.9  # GS値 = 標準偏差の何倍離れているか？
# フレーム毎の最大値枠にカウントする基準
DETECT_SIZE_MIN = 100  # 最小検出ピクセル数
# フレームをまたぐ最大外枠更新の基準
DETECT_SUMSIZE_MIN = 2000  # 検知する最小のピクセル数
DETECT_AREA_COUNT_MIN = 5      # 検出数5以下はカウントしない
DETECT_AREA_COUNT_MAX = 100    # 検出数100以上だったら最大枠にカウントしない
# 最大枠をリセットする条件(シングルフレーム内)
SF_UPDATE_AREA_MIN = 3  # フレーム内検出領域がこの数より少なくなったら最大枠をリセット
SF_UPDATE_SIZE_MIN = 50  # フレーム内検出領域の最大枠がこのサイズより少なくなったら最大枠をリセット
SF_INVALID_AREA_RATIO = 0.5  # 大き過ぎる領域を除去する基準 全画面の50%越えは無視
# (定数) ------------------------------------------------------------


# 2つの領域間の最小距離を算出 (x1s,x1e,x0s,x1e)
def distance(s0, e0, s1, e1):
    if e0 < s1:
        dist = s1 - e0
    elif e1 < s0:
        dist = s0 - e1
    else:  # 領域位置に重複がある場合は距離ゼロ
        dist = 0
    return dist


# 検出された領域同士の距離を積算
def CheckDist(stats):
    global GS_VAL_MIN
    arct = len(stats)
    if arct < 5:  # 領域数が5未満なら領域除去処理はskip
        stats2 = stats
    else:
        dist = [0]  # 積算距離のリスト
        stats2 = [[0, 0, 0, 0, 0]]  # 外れ値を取り除いた領域リスト
        for i in range(1, arct):
            x, y, w, h, size = stats[i]
            sum = 0
            for j in range(1, arct):
                xt, yt, wt, ht, sizet = stats[j]
                distX = distance(x, x+w, xt, xt+wt)
                distY = distance(y, y+h, yt, yt+ht)
                d1 = int(np.sqrt(distX**2 + distY**2))
                sum += d1
            dist.append(int(sum/(arct-2)))

        # 積算距離の平均と分散を求める
        ave = int(np.mean(dist))
        sd = int(np.std(dist))
#         print('No=', arct, 'ave=', ave, 'S.D.=', sd, 'ave+sd=', ave+sd)

        # 積算距離が平均より小さいものは無条件にgs値=0
        for i in range(1, arct):
            if ave < dist[i]:
                gs = (dist[i]-ave)/sd  # グラブス・スミルノフ棄却検定
            else:
                gs = 0

            # 算出したgs値で外れ値を除去 ⇒ サイズを-1に置換
            x, y, w, h, size = stats[i]
            if GS_VAL_MIN < gs:
                stats2.append([x, y, w, h, -1])  # GS基準に満たない(離れている)ものはサイズを-1に変更
            else:
                stats2.append([x, y, w, h, size])
    return stats2


# カメラ起動
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print('Camera is not ready')
    exit()
print(cap.set(cv2.CAP_PROP_FPS, 2), end='')  # 2fpsにセット

frame = cap.read()[1]
height, width = frame.shape[:2]
print(' x=', width, ' y=', height)
# cv2.imshow('Detected', frame)
# cv2.imwrite('backImg.png', frame)

# 初期化
frmAcc01 = None
frmAcc001 = None
nlbl1 = 0
MMaxX, MMaxY, MMinX, MMinY = 0, 0, width, height
PeakFlag = False
PeakCount = 0

while True:
    frame = cap.read()[1]
#     cv2.imshow('Input', frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # エッジ抽出
    edges = cv2.Canny(gray, 128, 255)
    if frmAcc01 is None:
        frmAcc01 = edges.copy().astype('float')
        frmAcc001 = edges.copy().astype('float')
        continue

    # 1/10, 1/100積分器
    cv2.accumulateWeighted(edges, frmAcc01, ACCUM_RATE_01)
    cv2.accumulateWeighted(edges, frmAcc001, ACCUM_RATE_001)

    # 差分計算
    frm01 = cv2.convertScaleAbs(frmAcc01)
    frm001 = cv2.convertScaleAbs(frmAcc001)
    frmSub = cv2.subtract(frm01, frm001)

#     cv2.imshow('frmAcc01', frm01)
#     cv2.imshow('frmAcc001', frm001)
#     cv2.imshow('frmSub', frmSub)

    # 単体ノイズ除去
    median = cv2.medianBlur(frmSub, 3)

    # クロージング処理 線分以外のノイズを除去
    kernel = np.ones((3, 3), np.uint8)
    closing = cv2.morphologyEx(median, cv2.MORPH_CLOSE, kernel)

    # 領域抽出安定化するための領域拡張処理
    dilation = cv2.dilate(closing, kernel, iterations=3)

    # ここで2値化
    thres = cv2.threshold(dilation, THRES_VAL_MIN, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow('Thres', thres)

    # 領域抽出
    nlbl0, labeledImg, stats, center = \
        cv2.connectedComponentsWithStats(thres, 8, cv2.CV_32S)

    # 積算距離を用いた外れ値除外処理を実行
    stats = CheckDist(stats)

    # 抽出領域全体を包含する外枠を決定
    MinX = width
    MinY = height
    MaxX, MaxY = 0, 0
    frame2 = frame.copy()
    SUMsize = 0

    for i in range(1, nlbl0):
        x, y, w, h, size = stats[i]
        SUMsize += size  # 検出された領域の合計サイズ
        if size == -1:  # sp値で除去された領域 赤で描画
            frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 0, 255), 1)
#             cv2.putText(frame2, str(i), (x+3, y+10), cv2.FONT_HERSHEY_PLAIN,
#                         0.7, (0, 255, 255), 1, cv2.LINE_AA)

        if DETECT_SIZE_MIN < size:  # 面積100以上は緑で描画、小さ過ぎる領域は処理対象外
            frame2 = cv2.rectangle(frame2, (x, y), (x+w, y+h), (0, 255, 0), 1)
#             cv2.putText(frame2, str(i), (x+3, y+10), cv2.FONT_HERSHEY_PLAIN,
#                         0.7, (0, 255, 255), 1, cv2.LINE_AA)
            if x < MinX:
                MinX = x
            if y < MinY:
                MinY = y
            if MaxX < x+w:
                MaxX = x+w
            if MaxY < y+h:
                MaxY = y+h
            cv2.imshow('Rect', frame2)

    # フレームを跨ぎ、最大サイズの外枠を選択
    rev = 0
    # シングルフレーム最大外枠サイズの面積を計算
    areaSF = (MaxX-MinX)*(MaxY-MinY)

    if MinX < MMinX:
        MMinX = MinX
        rev += 1
    if MinY < MMinY:
        MMinY = MinY
        rev += 1
    if MMaxX < MaxX:
        MMaxX = MaxX
        rev += 1
    if MMaxY < MaxY:
        MMaxY = MaxY
        rev += 1
#     if rev != 0:
#         cv2.imwrite('Rect_'+str(PeakCount)+'.png', frame2)
#         cv2.imwrite('frmAcc01_'+str(PeakCount)+'.png', frm01)
#         cv2.imwrite('frmAcc001_'+str(PeakCount)+'.png', frm001)
#         cv2.imwrite('frmSub_'+str(PeakCount)+'.png', frmSub)
#         cv2.imwrite('Thres_'+str(PeakCount)+'.png', thres)

    # マルチフレーム最大外枠サイズの面積を計算
    areaMF = (MMaxX-MMinX)*(MMaxY-MMinY)

    # フレーム間での最大サイズ外枠を更新
    if (1 <= rev) & (DETECT_SUMSIZE_MIN < SUMsize) \
       & (areaMF < width*height*SF_INVALID_AREA_RATIO) \
       & (DETECT_AREA_COUNT_MIN <= nlbl0) & (nlbl0 < DETECT_AREA_COUNT_MAX):
        frame3 = cv2.rectangle(frame2, (MMinX, MMinY), (MMaxX, MMaxY),
                               (0, 255, 255), 2)
        cv2.imshow('Detected', frame3)
        PeakFlag = True

    # 領域が抽出されなくなったら最大サイズ外枠をリセット
    elif (nlbl0 <= SF_UPDATE_AREA_MIN) or (areaSF < SF_UPDATE_SIZE_MIN):
        MMaxX, MMaxY, MMinX, MMinY = 0, 0, width, height
        if PeakFlag:
            cv2.imwrite('Detect_'+str(PeakCount)+'.png', frame3)
            PeakFlag = False
            PeakCount += 1

    # qが押されたら終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()