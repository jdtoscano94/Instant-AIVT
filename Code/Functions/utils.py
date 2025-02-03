import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
#from tensorflow import keras
#from tensorflow.keras import callbacks, layers
import matplotlib as mpl
from sklearn.decomposition import PCA
from matplotlib import cm

nu = 0.01


def normalize_weights(weights, origin):
    return [
        w * np.linalg.norm(wc) / np.linalg.norm(w)
        for w, wc in zip(weights, origin)
    ]


class RandomCoordinates(object):
    def __init__(self, origin):
        self.origin_ = origin
        self.v0_ = normalize_weights(
            [np.random.normal(size=w.shape) for w in origin], origin
        )
        self.v1_ = normalize_weights(
            [np.random.normal(size=w.shape) for w in origin], origin
        )

    def __call__(self, a, b):
        return [
            a * w0 + b * w1 + wc
            for w0, w1, wc in zip(self.v0_, self.v1_, self.origin_)
        ]


class LossSurface(object):
    def __init__(self, model, inputs, outputs, X_f_train_f):
        self.a_grid_ = None
        self.b_grid_ = None
        self.loss_grid_ = None
        self.model_ = model
        self.inputs_ = inputs
        self.outputs_ = outputs
        self.xf = X_f_train_f
        self.no_of_interior_points = 8000
        self.loss_fn = tf.keras.losses.MeanSquaredError()

    def residual_loss(self, ):
        with tf.GradientTape() as tape:
            tape.watch(self.xf)
            with tf.GradientTape() as tape2:
                u_predicted = self.model_(self.xf)
            grad = tape2.gradient(u_predicted, self.xf)
            du_dx = grad[:, 0]
            du_dt = grad[:, 1]
        j = tape.gradient(grad, self.xf)
        d2u_dx2 = j[:, 0]

        u_predicted = tf.cast(u_predicted, dtype=tf.float64)
        du_dx = tf.reshape(du_dx, [self.no_of_interior_points + 456, 1])
        d2u_dx2 = tf.reshape(d2u_dx2, [self.no_of_interior_points + 456, 1])
        du_dt = tf.reshape(du_dt, [self.no_of_interior_points + 456, 1])
        f = du_dt + u_predicted * du_dx - (nu / 3.14 * d2u_dx2)
        f = tf.math.reduce_mean(tf.math.square(f))
        return f

    def loss_total(self, ):
        y_pred = self.model_(self.inputs_)
        loss_data = self.loss_fn(y_pred, self.outputs_)
        loss_res = self.residual_loss()
        loss = tf.reduce_mean(tf.square(loss_data)) + tf.reduce_mean(tf.square(loss_res))
        return loss

    def compile(self, range_val, points, coords):
        a_grid = tf.linspace(-1.0, 1.0, num=points) ** 3 * range_val
        b_grid = tf.linspace(-1.0, 1.0, num=points) ** 3 * range_val
        loss_grid = np.empty([len(a_grid), len(b_grid)])
        for i, a in enumerate(a_grid):
            for j, b in enumerate(b_grid):
                self.model_.set_weights(coords(a, b))
                loss = self.loss_total()
                loss_grid[j, i] = loss
        print(loss_grid)
        self.model_.set_weights(coords.origin_)
        self.a_grid_ = a_grid
        self.b_grid_ = b_grid
        self.loss_grid_ = loss_grid

    def plot(self, range_val=1.0, points=24, levels=70, ax=None, **kwargs):
        xs = self.a_grid_
        ys = self.b_grid_
        zs = self.loss_grid_
        # if ax is None:
        #     #ax = plt.figure().add_subplot(projection='3d')
        #     _, ax = plt.subplots(**kwargs)
        #     ax.set_title("Loss Surface With trajectories")
        #     ax.set_aspect("equal")
        #
        # # Set Levels
        # min_loss = zs.min()
        # max_loss = zs.max()
        # levels = tf.exp(
        #     tf.linspace(
        #         tf.math.log(min_loss), tf.math.log(max_loss), num=levels
        #     )
        # )
        # # Create Contour Plot
        # CS = ax.contour(
        #     xs,
        #     ys,
        #     zs,
        #     levels=levels,
        #     cmap="magma",
        #     linewidths=0.75,
        #     norm=mpl.colors.LogNorm(vmin=min_loss, vmax=max_loss * 2.0),
        # )
        #
        # ax.clabel(CS, inline=True, fontsize=8, fmt="%1.2f")
        # # surf = ax.plot_surface(xs, ys, zs, cmap=cm.coolwarm,
        # #                        linewidth=0, antialiased=False)

        return ax, xs, ys, zs


def vectorize_weights_(weights):
    vec = [w.flatten() for w in weights]
    vec = np.hstack(vec)
    return vec


def vectorize_weight_list_(weight_list):
    vec_list = []
    for weights in weight_list:
        vec_list.append(vectorize_weights_(weights))
    weight_matrix = np.column_stack(vec_list)
    return weight_matrix


def shape_weight_matrix_like_(weight_matrix, example):
    weight_vecs = np.hsplit(weight_matrix, weight_matrix.shape[1])
    sizes = [v.size for v in example]
    shapes = [v.shape for v in example]
    weight_list = []
    for net_weights in weight_vecs:
        vs = np.split(net_weights, np.cumsum(sizes))[:-1]
        vs = [v.reshape(s) for v, s in zip(vs, shapes)]
        weight_list.append(vs)
    return weight_list


def get_path_components_(training_path, n_components=2):
    weight_matrix = vectorize_weight_list_(training_path)
    pca = PCA(n_components=2, whiten=True)
    components = pca.fit_transform(weight_matrix)
    example = training_path[0]
    weight_list = shape_weight_matrix_like_(components, example)
    return pca, weight_list


class PCACoordinates(object):
    def __init__(self, training_path):
        self.v1_ = None
        self.v0_ = None
        self.origin_ = None
        origin = training_path[-1]
        self.pca_, self.components = get_path_components_(training_path)
        self.set_origin(origin)

    def __call__(self, a, b):
        return [
            a * w0 + b * w1 + wc
            for w0, w1, wc in zip(self.v0_, self.v1_, self.origin_)
        ]

    def set_origin(self, origin, renorm=True):
        self.origin_ = origin
        if renorm:
            self.v0_ = normalize_weights(self.components[0], origin)
            self.v1_ = normalize_weights(self.components[1], origin)


def weights_to_coordinates(coords, training_path):
    components = [coords.v0_, coords.v1_]
    comp_matrix = vectorize_weight_list_(components)
    comp_matrix_i = np.linalg.pinv(comp_matrix)
    w_c = vectorize_weights_(training_path[-1])
    coord_path = np.array(
        [
            comp_matrix_i @ (vectorize_weights_(weights) - w_c)
            for weights in training_path
        ]
    )
    return coord_path


def plot_training_path(coords, training_path, ax=None, end=None, **kwargs):
    path = weights_to_coordinates(coords, training_path)
    if ax is None:
        fig, ax = plt.subplots(**kwargs)
    colors = range(path.shape[0])
    end = path.shape[0] if end is None else end
    norm = plt.Normalize(0, end)
    ax.scatter(
        path[:, 0], path[:, 1], s=4, c=colors, cmap="cividis", norm=norm,
    )
    return ax, path, colors

def get_splope_bias_from_points(r1,r2):
    x1,y1=r1
    x2,y2=r2
    m=(y2-y1)/(x2-x1)
    b=y2-m*x2
    return m,b