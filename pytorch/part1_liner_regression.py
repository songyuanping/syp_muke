import numpy as np
from tensorflow.keras.models import  Model
import torch


def compute_error_for_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (w * x + b)) ** 2
    return float(totalError) / len(points)


def step_gradient(b_current, w_current, points, learning_rate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((w_current * x) + b_current))
        w_gradient += -(2 / N) * x * (y - ((w_current * x) + b_current))
    new_b = b_current - learning_rate * b_gradient
    new_w = w_current - learning_rate * w_gradient
    return [new_w, new_b]


def train(iteration, w, b, points_num):
    points = []
    for i in range(0, points_num):
        eps = torch.randn(1)*11
        e = eps*0.001
        points.append([eps, eps* w + b + e])
    w_current = 0
    b_current = 0
    learning_rate = 0.001
    for j in range(0, iteration):
        [w_current, b_current] = step_gradient(b_current, w_current, np.array(points), learning_rate)
        loss = compute_error_for_points(b_current, w_current, np.array(points))
        if j%100==0:
            print('w:',w_current,'b:',b_current,'loss: ',loss)
    print('w:', w_current, 'b:', b_current, 'loss: ', loss)


if __name__ == '__main__':
    train(2000, 1.477, 0.089, 500)
