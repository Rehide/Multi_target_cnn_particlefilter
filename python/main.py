# -*- coding:utf-8 -*-
import cv2
import csv
import time
import numpy as np
from detecting_particlefilter import *
from tracking_particlefilter import *


# IoU
def IoU(box_d, box_t):
    IOU_THRESHOLD = 0.1
    flag = True
    if box_t != []:
        for box in box_t:
            x_a = max(int(box_d[0]), int(box[0]))
            y_a = max(int(box_d[1]), int(box[1]))
            x_b = min(int(box_d[0] + box_d[2]), int(box[0] + box[2]))
            y_b = min(int(box_d[1] + box_d[3]), int(box[1] + box[3]))
            # 2つのboxが重なっている領域の面積
            inter_area = (x_b - x_a + 1) * (y_b - y_a + 1)
            # それぞれのboxの面積
            box_d_area = (int(box_d[0] + box_d[2]) - int(box_d[0]) + 1) * (int(box_d[1] + box_d[3]) - int(box_d[1]) + 1)
            box_t_area = (int(box[0] + box[2]) - int(box[0]) + 1) * (int(box[1] + box[3]) - int(box[1]) + 1)
            # iouの計算
            iou = inter_area / float(box_d_area + box_t_area - inter_area)
            # iouが閾値を超えたらTrue
            if iou > IOU_THRESHOLD:
                flag = False
    return flag


if __name__ == "__main__":
    # 動画の読み込み
    cap = cv2.VideoCapture("movie/movie.m4v")
    # ウェブカメラ
    # cap = cv2.VideoCapture(0)
    # 動画の書き出し
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, 30,
    (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
     int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    SAMPLEMAX = 200 # パーティクルの数
    # パーティクルが取りうる座標の上限値
    upper_x, upper_y = 640, 480
    lower_x, lower_y = 0, 0
    upper_w, upper_h = 200, 150
    lower_w, lower_h = 32, 32

    # クラスの呼び出し
    t_filter = TrackingParticleFilter(SAMPLEMAX, upper_x, upper_y, lower_x, lower_y, upper_w, upper_h, lower_w, lower_h)
    d_filter = DetectingParticleFilter(SAMPLEMAX, upper_x, upper_y, lower_x, lower_y)
    # 初期化関数の呼び出し
    t_filter.initialize()
    d_filter.initialize()
    # caffeの設定関数の呼び出し
    t_filter.caffe_preparation()
    d_filter.caffe_preparation()
    # カラーリスト
    COLOR_NUM = 10
    color_list = [(0,0,255),(255,0,0),(0,150,255),(0,255,0),(128,64,128),(60,40,222),(128,128,0),(192,128,128),(147,20,255),(64,0,128),(64,64,0)]

    # tracking開始
    while cv2.waitKey(30) < 0:
        start = time.clock()    # 時間の計測開始
        ret, frame = cap.read() # 動画の読み込み
        # 追跡関数の呼び出し(ave:尤度の平均値， x,y,w,h:パーティクルの推定値)
        box_t = []
        if t_filter.pre_id != [] or t_filter.frame_count == 1:
            ave, id = t_filter.filtering(frame)
            for index in range(len(id)):
                box_t.append([t_filter.bx[index][0], t_filter.by[index][0], t_filter.bw[index][0], t_filter.bh[index][0]])
        x, y, w, h, prob  = d_filter.filtering(frame) # 検出関数の呼び出し(x,y,w,h:パーティクルの推定値)
        box_d = [x, y, w, h]                    # リスト作成
        iou_flag = IoU(box_d, box_t)            # IoU関数の呼び出し

        print "id = {}".format(id)
        print "d_filter.flag = {}".format(d_filter.flag)
        print "IoU_flag = {}".format(iou_flag)

        # TrackingとDetectingのboxが重なっていないかチェック
        if d_filter.flag == True:
            if t_filter.flag_count > 5 or t_filter.cluster > 1 or t_filter.pre_id == []:
                del_num = []
                particle_num = []
                # 重なっていなければ、新しい追跡フィルタのbounding boxとして追加
                if iou_flag == True:
                    # 追跡フィルタのbounding boxに新しい追跡フィルタのbounding boxを追加する
                    t_filter.bx = np.vstack([t_filter.bx, np.array([[x]])])
                    t_filter.by = np.vstack([t_filter.by, np.array([[y]])])
                    t_filter.bw = np.vstack([t_filter.bw, np.array([[w]])])
                    t_filter.bh = np.vstack([t_filter.bh, np.array([[h]])])
                    # idが空でない時
                    if t_filter.pre_id != []:
                        new_t_filter_num = SAMPLEMAX // (len(id) + 1)   # 新しい追跡フィルタのbounding boxに移動させるパーティクルの数
                        for i in range(len(id)):
                            del_num.append(new_t_filter_num // len(id)) # 元の追跡フィルタのbounding boxから削除するパーティクルの数のリスト
                        mod = new_t_filter_num % len(id)                # 余りを計算する
                        # 余りがあれば、余りをindex0に追加する
                        if mod != 0:
                            del_num[0] = del_num[0] + mod
                        # 元の追跡フィルタのbounding boxからdel_num分削除する
                        for i in range(len(id)):
                            particle_num.append(len(t_filter.X[i]))
                        if min(particle_num) > del_num[0] + 10:
                            for i in range(len(id)):
                                t_filter.X[i] = t_filter.X[i][:len(t_filter.X[i]) - del_num[i]]
                                t_filter.Y[i] = t_filter.Y[i][:len(t_filter.Y[i]) - del_num[i]]
                                t_filter.W[i] = t_filter.W[i][:len(t_filter.W[i]) - del_num[i]]
                                t_filter.H[i] = t_filter.H[i][:len(t_filter.H[i]) - del_num[i]]
                        else:
                            index = particle_num.index(max(particle_num))
                            t_filter.X[index] = t_filter.X[index][:len(t_filter.X[index]) - new_t_filter_num]
                            t_filter.Y[index] = t_filter.Y[index][:len(t_filter.Y[index]) - new_t_filter_num]
                            t_filter.W[index] = t_filter.W[index][:len(t_filter.W[index]) - new_t_filter_num]
                            t_filter.H[index] = t_filter.H[index][:len(t_filter.H[index]) - new_t_filter_num]
                        # 追跡フィルタのパーティクルに新しい追跡フィルタの情報を追加
                        t_filter.X.append([x] * new_t_filter_num)
                        t_filter.Y.append([y] * new_t_filter_num)
                        t_filter.W.append([w] * new_t_filter_num)
                        t_filter.H.append([h] * new_t_filter_num)
                        # id,ave,pre_id、max_id、pre_pxの更新
                        id.append(t_filter.max_id + 1)
                        ave.append(prob)
                        t_filter.pre_id = id[:]
                        t_filter.max_id += 1
                        t_filter.pre_px = np.vstack([t_filter.pre_px, np.array([[x, y, w, h]])])
                        # 検出用フィルタの初期化
                        d_filter.initialize()
                    # idが空の時
                    else:
                        new_t_filter_num = SAMPLEMAX
                        # 追跡フィルタのパーティクルに新しい追跡フィルタの情報を追加
                        t_filter.X.append([x] * new_t_filter_num)
                        t_filter.Y.append([y] * new_t_filter_num)
                        t_filter.W.append([w] * new_t_filter_num)
                        t_filter.H.append([h] * new_t_filter_num)
                        id.append(t_filter.max_id + 1)
                        ave.append(prob)
                        t_filter.pre_id = id[:]
                        t_filter.max_id += 1
                        t_filter.pre_px = np.array([[x, y, w, h]])
                        d_filter.initialize()
                # 重なっていたら、検出フィルタを初期化させる
                else:
                    d_filter.initialize()
        else:
            if iou_flag == False:
                d_filter.initialize()

        # 追跡フィルタの描画
        if id != []:
            for num in range(len(id)):
            # 尤度の平均値が閾値を超えたら描画
                if ave[num] >= 0.5:
                    # bounding boxの描画
                    cv2.rectangle(frame, (int(t_filter.bx[num][0]), int(t_filter.by[num][0])),
                                (int(t_filter.bx[num][0] + t_filter.bw[num][0]), int(t_filter.by[num][0] + t_filter.bh[num][0])), color_list[id[num] % COLOR_NUM], 2)
                    # パーティクルの描画
                    # for i in range(len(t_filter.X[num])):
                        # cv2.circle(frame, (int(t_filter.X[num][i] + (t_filter.W[num][i] / 2)), int(t_filter.Y[num][i] + (t_filter.H[num][i] / 2))), 2, color_list[id[num] % COLOR_NUM], -1)
                    # idの描画
                    if int(t_filter.by[num][0]) < 20:
                        cv2.rectangle(frame, (int(t_filter.bx[num][0]), int(t_filter.by[num][0])),
                                    (int(t_filter.bx[num][0]) + 40, int(t_filter.by[num][0]) + 20), color_list[id[num] % COLOR_NUM], -1)
                        cv2.putText(frame, str(id[num]), (int(t_filter.bx[num][0]) + 3, int(t_filter.by[num][0]) + 12), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255))
                    else:
                        cv2.rectangle(frame, (int(t_filter.bx[num][0]), int(t_filter.by[num][0]) - 20),
                                    (int(t_filter.bx[num][0]) + 40, int(t_filter.by[num][0])), color_list[id[num] % COLOR_NUM], -1)
                        cv2.putText(frame, str(id[num]), (int(t_filter.bx[num][0]) + 3, int(t_filter.by[num][0]) - 3), cv2.FONT_HERSHEY_DUPLEX, 0.7, (255,255,255))

            # 各クラスタの各状態を一つのリストに統合
            temp_x, temp_y, temp_w, temp_h = [], [], [], []
            for cluster in range(len(id)):
                temp_x.extend(list(t_filter.X[cluster]))
                temp_y.extend(list(t_filter.Y[cluster]))
                temp_w.extend(list(t_filter.W[cluster]))
                temp_h.extend(list(t_filter.H[cluster]))
            t_filter.X = np.array(temp_x)
            t_filter.Y = np.array(temp_y)
            t_filter.W = np.array(temp_w)
            t_filter.H = np.array(temp_h)
        get_image_time = int((time.clock() - start) * 1000)  # 時間の計測終了
        cv2.putText(frame, str(1000/get_image_time) + "fps", (10,30), 2, 1, (0,255,0)) # fpsを描画
        cv2.imshow("frame", frame) # フレームを表示
        t_filter.frame_count += 1  # カウント
        out.write(frame)           # 動画の保存
        if cv2.waitKey(30) & 0xFF == 27:break
        print "--------------------------------"

    # リソースの破棄
    cap.release()
    out.release()
    cv2.destroyAllWindows()
