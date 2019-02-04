# -*- coding:utf-8 -*-
import cv2
import numpy as np
import skimage
import caffe
from caffe.proto import caffe_pb2
caffe.set_mode_gpu()
from scipy import signal
import pyclustering
from pyclustering.cluster import xmeans
from munkres import Munkres
from scipy.spatial import distance
import copy
import itertools


# 追跡用パーティクルフィルタ
class TrackingParticleFilter:
    # パーティクルの定義
    def __init__(self, SAMPLEMAX, upper_x, upper_y, lower_x, lower_y, upper_w, upper_h, lower_w, lower_h):
        # パーティクルの数
        self.SAMPLEMAX = SAMPLEMAX
        # パーティクルが取りうる座標の上限値
        self.upper_x, self.upper_y = upper_x, upper_y
        self.lower_x, self.lower_y = lower_x, lower_y
        self.upper_w, self.upper_h = upper_w, upper_h
        self.lower_w, self.lower_h = lower_w, lower_h
        # カウンタ
        self.frame_count = 1
        self.flag_count = 0
        # 過去情報
        self.pre_px = np.zeros((1, 4))
        self.pre_id = [0]
        self.pre_dict_ave = {}
        # idの最大値
        self.max_id = 0

    # パーティクルの初期化関数
    def initialize(self):
        self.W = np.random.random(self.SAMPLEMAX) * (self.upper_w - self.lower_w) + self.lower_w
        self.H = np.random.random(self.SAMPLEMAX) * (self.upper_h - self.lower_h) + self.lower_h
        self.X = np.random.random(self.SAMPLEMAX) * (self.upper_x - self.lower_x) + (self.lower_x - self.W/2)
        self.Y = np.random.random(self.SAMPLEMAX) * (self.upper_y - self.lower_y) + (self.lower_y - self.H/2)

    # 状態遷移関数
    def modeling(self):
        # ランダムウォーク
        self.X += np.random.random(self.SAMPLEMAX) * 30 - 15
        self.Y += np.random.random(self.SAMPLEMAX) * 30 - 15
        self.W += np.random.random(self.SAMPLEMAX) * 20 - 10
        self.H += np.random.random(self.SAMPLEMAX) * 20 - 10
        # 上限、下限を超えないよう設定（必要）
        for i in range(self.SAMPLEMAX):
            if (self.X[i] + self.W[i]) > self.upper_x:
                self.W[i] = self.upper_x - self.X[i]
            if (self.Y[i] + self.H[i]) > self.upper_y:
                self.H[i] = self.upper_y - self.Y[i]
            if self.X[i] > self.upper_x - self.lower_w:
                self.X[i] = self.upper_x - self.lower_w
                self.W[i] = self.lower_w - 1
            if self.Y[i] > self.upper_y - self.lower_h:
                self.Y[i] = self.upper_y - self.lower_h
                self.H[i] = self.lower_h - 1
            if self.W[i] > self.upper_w: self.W[i] = self.upper_w
            if self.H[i] > self.upper_h: self.H[i] = self.upper_h
            if self.X[i] < self.lower_x: self.X[i] = self.lower_x
            if self.Y[i] < self.lower_y: self.Y[i] = self.lower_y
            if self.W[i] < self.lower_w: self.W[i] = self.lower_w
            if self.H[i] < self.lower_h: self.H[i] = self.lower_h

    # caffeの設定
    def caffe_preparation(self):
        mean_blob = caffe_pb2.BlobProto()
        with open('../model/crow_mean.binaryproto') as f:
            mean_blob.ParseFromString(f.read())
        mean_array = np.asarray(
        mean_blob.data,
        dtype = np.float32).reshape(
        	(mean_blob.channels,
        	mean_blob.height,
        	mean_blob.width))
        self.classifier = caffe.Classifier(
            '../model/crow.prototxt',
            '../model/crow_iter_100000.caffemodel',
            mean=mean_array,
            raw_scale=255)

    # 尤度関数
    def calcLikelihood(self, image):
        intensity = []     # 対象物体クラスの尤度を格納する配列
        intensity_all = [] # すべてのクラスの尤度を格納する配列
        # パーティクルごとで尤度を取得
        for i in range(self.SAMPLEMAX):
            # パーティクルの領域画像を取得
            y, x, w, h = self.Y[i], self.X[i], self.W[i], self.H[i]
            roi = image[y:y+h, x:x+w]
            # 領域画像をcaffeのフォーマットに合わせる
            img = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
            img = skimage.img_as_float(img).astype(np.float32)
            # caffe識別器から対象物体である確率を取得
            predictions = self.classifier.predict([img], oversample=False)
            # 対象物体クラスの確率を尤度として配列に保存
            intensity.append(predictions[0][int(0)])
            intensity_all.append(predictions[0])
        return intensity, intensity_all

    # x-means
    def x_means(self, intensity, intensity_all):
        x_data = np.empty((0,5))
        x_center = np.empty((0,2))
        # クラスタリングのデータ
        for i in range(self.SAMPLEMAX):
            x_data = np.append(x_data, np.array([[self.X[i], self.Y[i], self.W[i], self.H[i], intensity[i]]]), axis=0)
            x_center = np.append(x_center, np.array([[int(self.X[i] + self.W[i]/2), int(self.Y[i] + self.H[i]/2)]]), axis=0)
        init_center = pyclustering.cluster.xmeans.kmeans_plusplus_initializer(x_center, 1).initialize() # クラスタリングの初期値
        # クラスタリング実行
        xm = pyclustering.cluster.xmeans.xmeans(x_center, init_center, ccore=True)
        xm.process()
        # 結果の取得
        clusters = xm.get_clusters()
        # クラスタの数
        self.cluster = len(clusters)
        self.x_kmeans = [''] * self.cluster
        self.ave_all = [''] * self.cluster
        for index, class_number in enumerate(clusters):
            self.x_kmeans[index] = []
            self.ave_all[index] = []
            self.x_kmeans[index].append(x_data[class_number[:]])
            self.ave_all[index].append(np.array(intensity_all)[class_number[:]])

    # 正規化関数
    def normalize(self):
        # 重みの計算
        self.weights = [''] * self.cluster
        for i in range(self.cluster):
            self.weights[i] = self.x_kmeans[i][0][:,4] / np.sum(self.x_kmeans[i][0][:,4])

    # リサンプリング関数
    def resampling(self):
        self.sample = [''] * self.cluster
        for i in range(self.cluster):
            self.sample[i] = []
            index = np.arange(len(self.weights[i]))
            # 重みが大きいパーティクルの引数が確率的に多く選択される
            for num in range(len(self.weights[i])):
                idx = np.random.choice(index, p = self.weights[i])
                self.sample[i].append(idx)

    # ハンガリアン法
    def hungarian(self):
        p1 = np.array(self.px)
        p2 = self.pre_px
        # 次元数
        dimN_p1 = len(p1)
        dimN_p2 = len(p2)
        print "p1:{}".format(dimN_p1)
        print "p2:{}".format(dimN_p2)

        if dimN_p1 <= dimN_p2:
            sum_list = []
            index_list = []
            comb = np.array(list(itertools.combinations(p2, dimN_p1))) # 組み合わせ
            for num, v in enumerate(comb):
                dist_M = distance.cdist(p1, v, metric='euclidean')     # ユークリッド距離の計算
                copy_dist_M = dist_M.copy() # コピー
                m = Munkres()               # ハンガリアン法のインスタンス生成
                index = m.compute(dist_M)   # ハンガリアン法の関数の実行
                for i in range(len(index)):
                    index_list.append(index[i][1])
                sum = 0 # コストの和
                for num, index in enumerate(index):
                    sum += copy_dist_M[num][index[1]]      # コストを足す
                sum_list.append(sum)                       # sumをリストに格納
            min_comb_index = sum_list.index(min(sum_list)) # コストが一番小さい組み合わせのindexを取得
            index_list = index_list[min_comb_index * dimN_p1 : min_comb_index * dimN_p1 + dimN_p1]

            # comb[min_comb_index]とp2で重複していない要素のindexを取得
            new_pre_id = []
            for i, comb in enumerate(comb[min_comb_index]):
                same_index = np.where(p2 == comb)
                count = np.bincount(same_index[0])
                mode = np.argmax(count)
                new_pre_id.append(self.pre_id[mode])
            self.pre_id = new_pre_id

        elif dimN_p1 > dimN_p2:
            sum_list = []
            index_list = []
            comb = np.array(list(itertools.combinations(p1, dimN_p2))) # 組み合わせ
            for num, v in enumerate(comb):
                dist_M = distance.cdist(v, p2, metric='euclidean')     # ユークリッド距離の計算
                copy_dist_M = dist_M.copy() # コピー
                m = Munkres()               # ハンガリアン法のインスタンス生成
                index = m.compute(dist_M)   # ハンガリアン法の関数の実行
                for i in range(len(index)):
                    index_list.append(index[i][1])
                sum = 0 # コストの和
                for num, ind in enumerate(index):
                    sum += copy_dist_M[num][ind[1]]        # コストを足す
                sum_list.append(sum)                       # sumをリストに格納
            min_comb_index = sum_list.index(min(sum_list)) # コストが一番小さい組み合わせのindexを取得
            index_list = index_list[min_comb_index * dimN_p2 : min_comb_index * dimN_p2 + dimN_p2]

            # 並び替える
            same_index_list = [""] * dimN_p1
            for i, comb in enumerate(comb[min_comb_index]):
                same_index = np.where(p1 == comb)
                count = np.bincount(same_index[0])
                mode = np.argmax(count)
                same_index_list[mode] = index_list[i]
            index_list = [1000 if i == "" else i for i in same_index_list]

        num = 1
        for i, index in enumerate(index_list):
            # index_listとpre_idの対応付け
            if index < dimN_p2:
                index_list[i] = self.pre_id[index]
            # index_listのindexがpre_idに存在しない場合、新しいクラスタが誕生したと判断し、新しいidを付与する
            else:
                # idの重複を防ぐため
                index_list[i] = self.max_id + num
                num += 1
        # max_idの更新
        if max(index_list) > self.max_id:
            self.max_id = max(index_list)
        return index_list

    # フレームアウト関数
    def frameout(self, id):
        class_num = 10  # 識別器の出力クラス数
        copy_id = id[:] # idのコピー
        average = [''] * self.cluster
        # 各クラスタの尤度の平均値を求める
        for i in range(self.cluster):
            sum_row = np.sum(self.ave_all[i][0], axis = 0)
            average[i] = sum_row / len(self.ave_all[i][0])
        new_average = [''] * len(average)
        # averageを辞書型に変換
        dict_ave = {}
        for i, ave in enumerate(average):
            dict_ave[id[i]] = ave
        if self.pre_dict_ave != {}:
            intersection_keys = set(dict_ave.keys()) & set(self.pre_dict_ave.keys()) # 共通keyの取得
            # ベイズ更新
            failure_id = []
            index_failure_id = []
            for keys in list(intersection_keys):
                bayes_denominator = 0
                bayes_numerator = self.pre_dict_ave[keys][0] * dict_ave[keys][0]
                for i in range(class_num):
                    bayes_denominator += self.pre_dict_ave[keys][i] * dict_ave[keys][i]
                bayes_ave = float(bayes_numerator) / bayes_denominator
                dict_ave[keys][0] = bayes_ave
                if bayes_ave < 0.0001:
                    failure_id.append(keys)
                    index_failure_id.append(id.index(keys))
                    print "{} : {}".format(keys, dict_ave[keys][0])
                else:
                    print "{} : {}".format(keys, dict_ave[keys][0])
                new_average[id.index(keys)] = dict_ave[keys][0] # dict_aveの要素の入れ替え

            print "failure_id = {}".format(failure_id)

            if failure_id != []:
                # idのリストから閾値が下回ったクラスタのidを削除
                id = sorted(list(set(id) ^ set(failure_id)), key = id.index)
                # 削除した結果、閾値を上回ったクラスタが一つもなかった場合
                if id == []:
                    self.pre_id = []
                    self.pre_px = []
                    self.X = []
                    self.Y = []
                    self.W = []
                    self.H = []
                # 削除した結果、閾値を上回ったクラスタが一つ以上あった場合、その各クラスタに削除されたクラスタのパーティクルを割り振る
                else:
                    # new_averageから閾値を下回ったクラスタの確率を削除
                    dellist = lambda items, indexes: [item for index, item in enumerate(items) if index not in indexes]
                    dellist(new_average, index_failure_id)
                    # 辞書型から閾値を下回ったクラスタの確率を削除
                    for fail_id in failure_id:
                        dict_ave.pop(fail_id)
                    # 閾値が下回ったクラスタのパーティクルの個数を数える
                    sum_fail_particle = 0
                    for index_fail_id in index_failure_id:
                        sum_fail_particle += len(self.X[index_fail_id])

                    # 削除されたパーティクルが閾値を上回ったクラスタ数よりも多い場合
                    if sum_fail_particle > len(id):
                        # パーティクルを各クラスタに均等に割り振れる場合
                        if sum_fail_particle % len(id) == 0:
                            for i in id:
                                index = copy_id.index(i)
                                for j in range(sum_fail_particle / len(id)):
                                    self.X[index] = np.append(self.X[index], self.bx[index][0])
                                    self.Y[index] = np.append(self.Y[index], self.by[index][0])
                                    self.W[index] = np.append(self.W[index], self.bw[index][0])
                                    self.H[index] = np.append(self.H[index], self.bh[index][0])
                        # パーティクルを各クラスタに均等に割り振れない場合
                        else:
                            for i in id:
                                index = copy_id.index(i)
                                for j in range(sum_fail_particle // len(id)):
                                    self.X[index] = np.append(self.X[index], self.bx[index][0])
                                    self.Y[index] = np.append(self.Y[index], self.by[index][0])
                                    self.W[index] = np.append(self.W[index], self.bw[index][0])
                                    self.H[index] = np.append(self.H[index], self.bh[index][0])
                            index = copy_id.index(id[0])
                            for i in range(sum_fail_particle % len(id)):
                                self.X[index] = np.append(self.X[index], self.bx[index][0])
                                self.Y[index] = np.append(self.Y[index], self.by[index][0])
                                self.W[index] = np.append(self.W[index], self.bw[index][0])
                                self.H[index] = np.append(self.H[index], self.bh[index][0])
                    # 削除されたパーティクルが閾値を上回ったクラスタ数よりも少ない場合
                    else:
                        index = copy_id.index(id[0])
                        for i in range(sum_fail_particle):
                            self.X[index] = np.append(self.X[index], self.bx[index][0])
                            self.Y[index] = np.append(self.Y[index], self.by[index][0])
                            self.W[index] = np.append(self.W[index], self.bw[index][0])
                            self.H[index] = np.append(self.H[index], self.bh[index][0])

                    # 閾値が下回ったクラスタのパーティクルを削除
                    self.X = list(np.delete(self.X, index_failure_id, 0))
                    self.Y = list(np.delete(self.Y, index_failure_id, 0))
                    self.W = list(np.delete(self.W, index_failure_id, 0))
                    self.H = list(np.delete(self.H, index_failure_id, 0))
                    # 閾値が下回ったクラスタのbounding boxを削除
                    self.bx = np.delete(self.bx, index_failure_id, 0)
                    self.by = np.delete(self.by, index_failure_id, 0)
                    self.bw = np.delete(self.bw, index_failure_id, 0)
                    self.bh = np.delete(self.bh, index_failure_id, 0)
                    self.pre_px = np.delete(self.pre_px, index_failure_id, 0)
                    # idの保存
                    self.pre_id = id[:]

        if self.flag_count > 5 or self.cluster > 1:
            self.pre_dict_ave = dict_ave.copy()
        return new_average, id

    # 追跡関数
    def filtering(self, image):
        # 各関数の呼び出し
        self.modeling()
        intensity, intensity_all = self.calcLikelihood(image)
        self.x_means(intensity, intensity_all)
        self.normalize()
        # リストの用意
        self.X = [''] * self.cluster
        self.Y = [''] * self.cluster
        self.W = [''] * self.cluster
        self.H = [''] * self.cluster
        self.bx = np.zeros((self.cluster, 1))
        self.by = np.zeros((self.cluster, 1))
        self.bw = np.zeros((self.cluster, 1))
        self.bh = np.zeros((self.cluster, 1))
        self.px = [''] * self.cluster
        # リサンプリング
        self.resampling()
        for i in range(self.cluster):
            self.X[i] = self.x_kmeans[i][0][:,0][self.sample[i]]
            self.Y[i] = self.x_kmeans[i][0][:,1][self.sample[i]]
            self.W[i] = self.x_kmeans[i][0][:,2][self.sample[i]]
            self.H[i] = self.x_kmeans[i][0][:,3][self.sample[i]]
        # 対象推定
        for i in range(self.cluster):
            for j in range(len(self.X[i])):
                self.bx[i][0] += float(self.X[i][j]) * float(self.weights[i][j])
            for j in range(len(self.Y[i])):
                self.by[i][0] += float(self.Y[i][j]) * float(self.weights[i][j])
            for j in range(len(self.W[i])):
                self.bw[i][0] += float(self.W[i][j]) * float(self.weights[i][j])
            for j in range(len(self.H[i])):
                self.bh[i][0] += float(self.H[i][j]) * float(self.weights[i][j])
            self.px[i] = [self.bx[i][0], self.by[i][0], self.bw[i][0], self.bh[i][0]] # 各クラスタのバウンディングボックスのx,y,w,h
        # ハンガリアン法
        if self.frame_count > 1 and self.cluster > 1:
            id = self.hungarian() # ハンガリアン関数の呼び出し
            self.pre_px = np.array([x[:] for x in self.px]) # コピー self.px → self.pre_px
            self.pre_id = id[:]   # コピー id → self.pre_id
            self.flag_count = 0
        # クラスタが一つの場合
        elif self.frame_count > 1 and self.cluster == 1:
            self.flag_count += 1
            p1 = np.array(self.px)
            p2 = self.pre_px
            print "p1:{}".format(len(p1))
            print "p2:{}".format(len(p2))
            distance = []
            # ユークリッド距離を比較し、一番ユークリッド距離が小さいidとする
            for px in p2:
                distance.append(np.linalg.norm(p1[0] - px))
            id = [self.pre_id[distance.index(min(distance))]]
            # flag_countが5以上になったら、追跡対象が一匹になったと判断し、idを保存する
            if self.flag_count > 5:
                self.pre_px = np.array([x[:] for x in self.px])
                self.pre_id = id[:]
        # 1フレーム目
        else:
            id = range(self.cluster)
        # フレームアウト
        average, id = self.frameout(id)
        return average, id
