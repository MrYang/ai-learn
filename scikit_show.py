import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np

from sklearn import neighbors, cluster, datasets, tree, linear_model, svm
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

iris_test = [[3.1, 2.6, 1.0, 0.2], [6.1, 4.6, 4.5, 1.8]]


# K近邻, 在训练数据集中找到实例最邻近的K个实例
# 精度高, 对异常值不敏感, 计算复杂度高, 空间复杂度高, 适用于数值型和标称型
def knn():
    iris = datasets.load_iris()
    X, y = iris.data, iris.target

    # 将数据集切分为训练集和测试集,82开, random_state为随机种子
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=25)
    clf = neighbors.KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)

    score = clf.score(X_test, y_test)
    print(score)

    # 将数据集分为10折, 每次用9折训练, 1折测试, 有10份结果, 得到10次的平均值
    result = cross_val_score(clf, X, y, cv=10)
    print(result, result.mean())

    y_pred = clf.predict(np.array(iris_test))
    print(y_pred)

    y_proba = clf.predict_proba(np.array(iris_test))
    print(y_proba)


# 在数据量小的情况下仍然有效,可以处理多类别问题, 对于输入数据的准备方式较为敏感, 适用于标称型数据
def gnb():
    iris = datasets.load_iris()
    g = GaussianNB()
    g.fit(iris.data, iris.target)
    y_pred = g.predict(iris.data)
    print(iris.data.shape[0], (iris.target != y_pred).sum())

    y_pred = g.predict(np.array([[5.0, 3.1, 1.5, 0.4]]))
    print(y_pred)


# 各分类算法比较
def cls_compare():
    h = .02

    names = ['Nearest Neighbors', 'Linear SVM', 'RBF SVM', 'Gaussian Process', 'Decision Tree',
             'Random Forest', 'Neural Net', 'AdaBoost', 'Naive Bayes', 'QDA']

    classifiers = [
        neighbors.KNeighborsClassifier(n_neighbors=3),  # K近邻
        svm.SVC(kernel='linear', C=0.025),  # 支持向量机, 线性函数
        svm.SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),  # 高斯过程分类
        tree.DecisionTreeClassifier(max_depth=5),  # 决策树
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),  # 随机森林
        MLPClassifier(alpha=1, max_iter=1000),  # 多层感知机分类(人工神经网络)
        AdaBoostClassifier(),
        GaussianNB(),  # 高斯分布朴素贝叶斯
        QuadraticDiscriminantAnalysis(),
    ]

    # 生成随机分类数据集
    X, y = datasets.make_classification(n_samples=100, n_features=2, n_redundant=0, n_informative=2,
                                        random_state=1, n_clusters_per_class=1)
    rng = np.random.RandomState(2)
    X += 2 * rng.uniform(size=X.shape)
    linearly_separable = (X, y)

    # 生成月亮形状以及环形形状数据
    dataset = [datasets.make_moons(n_samples=100, noise=0.3, random_state=0),
               datasets.make_circles(n_samples=100, noise=0.2, factor=0.5, random_state=1),
               linearly_separable]

    # 10列3行图表
    plt.figure(figsize=(30, 10))
    i = 1
    for ds_cnt, ds in enumerate(dataset):
        X, y = ds
        # 去均值与方差归一化处理, 针对每一个特征维度去做
        X = StandardScaler().fit_transform(X)
        # 数据集切分为训练集与测试集, 64分
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

        x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
        y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        ax = plt.subplot(len(dataset), len(classifiers) + 1, i)
        if ds_cnt == 0:
            ax.set_title('input data')

        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolor='k')
        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        # 设置无数字坐标
        ax.set_xticks(())
        ax.set_yticks(())
        i += 1

        for name, clf in zip(names, classifiers):
            ax = plt.subplot(len(dataset), len(classifiers) + 1, i)
            clf.fit(X_train, y_train)
            score = clf.score(X_test, y_test)

            if hasattr(clf, 'decision_function'):
                Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
            else:
                Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, alpha=.8)
            ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors='k')
            ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors='k')

            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xticks(())
            ax.set_yticks(())
            if ds_cnt == 0:
                ax.set_title(name)

            ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'), size=15, horizontalalignment='right')
            i += 1

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    pass
    # knn()
    # gnb()
    cls_compare()
