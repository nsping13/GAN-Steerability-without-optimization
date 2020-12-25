import numpy as np
import math
import functools
import torch
import torch.nn as nn
from torch.nn import init
import torch.optim as optim
import torch.nn.functional as F

def extract_directions_user_pre_described(W, b, desired_effect):
    # input:
        # W: FC Weight matrix
        # b: FC bias vector
        # desired_effect: shift_y_up,shift_y_down,shift_x_right,shift_x_left,zoom_in,zoom_out
    # output:
        # direction: q

    dimw = W.shape[0]
    if desired_effect == "shift_y_up":
        P = torch.zeros((dimw, dimw)).cuda()
        T = torch.eye(dimw - 4).cuda()
        P[0:dimw - 4, 4:] = T  # shift up
        D = torch.zeros((dimw, dimw)).cuda()
        T = torch.zeros((dimw)).cuda()
        ix = torch.arange(0, dimw, 16)
        for ixy in range(ix.shape[0]):
            T[ix[ixy]:ix[ixy] + 12] = 1.0
        D.as_strided([dimw], [dimw + 1]).copy_(T)
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))

    if desired_effect == "shift_y_down":
        P = torch.zeros((dimw, dimw)).cuda()
        T = torch.eye(dimw - 4).cuda()
        P[4:, 0:dimw - 4] = T  # shift up
        D = torch.zeros((dimw, dimw)).cuda()
        T = torch.zeros((dimw)).cuda()
        ix = torch.arange(4, dimw, 16)
        for ixy in range(ix.shape[0]):
            T[ix[ixy]:ix[ixy] + 12] = 1.0
        D.as_strided([dimw], [dimw + 1]).copy_(T)
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))

    if desired_effect == "shift_x_right":
        P = torch.zeros((dimw, dimw)).cuda()
        T = (torch.tensor([0, 1, 1, 1])).repeat(6144).cuda()
        P.as_strided([dimw], [dimw + 1]).copy_(T)
        P = torch.roll(P, -1, [1])
        P = P.T
        D = torch.zeros((dimw, dimw)).cuda()
        T = (torch.tensor([1,1,1,0])).repeat(6144).cuda()
        D.as_strided([dimw], [dimw + 1]).copy_(T)
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))

    if desired_effect == "shift_x_left":
        P = torch.zeros((dimw, dimw)).cuda()
        T = (torch.tensor([1, 1, 1, 0])).repeat(6144).cuda()
        P.as_strided([dimw], [dimw + 1]).copy_(T)
        P = torch.roll(P, 1, [1])
        P = P.T
        D = torch.zeros((dimw, dimw)).cuda()
        T = (torch.tensor([0,1,1,1])).repeat(6144).cuda()
        D.as_strided([dimw], [dimw + 1]).copy_(T)
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))

    if desired_effect == "zoom_in":
        P = torch.zeros((dimw, dimw))
        T = torch.tensor([0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0]).repeat(1536)
        P.as_strided([dimw], [dimw + 1]).copy_(T)
        D = torch.eye(dimw)
        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))

    if desired_effect == "zoom_out":
        P = torch.eye(dimw)
        D = torch.zeros((dimw, dimw))
        ix = torch.arange(5, dimw, 16)
        for ixy in range(ix.shape[0]):
            D[ix[ixy],ix[ixy]] = 1.0
            D[ix[ixy]- 1,ix[ixy]] = 1.0
            D[ix[ixy] - 4,ix[ixy]] = 1.0
            D[ix[ixy] - 5,ix[ixy]] = 1.0

        ix = torch.arange(6, dimw, 16)
        for ixy in range(ix.shape[0]):
            D[ix[ixy],ix[ixy]] = 1.0
            D[ix[ixy] - 4,ix[ixy]] = 1.0
            D[ix[ixy] - 3,ix[ixy]] = 1.0
            D[ix[ixy] + 1,ix[ixy]] = 1.0

        ix = torch.arange(9, dimw, 16)
        for ixy in range(ix.shape[0]):
            D[ix[ixy],ix[ixy]] = 1.0
            D[ix[ixy]- 1,ix[ixy]] = 1.0
            D[ix[ixy] + 3, ix[ixy]] = 1.0
            D[ix[ixy] + 4, ix[ixy]] = 1.0

        ix = torch.arange(10, dimw, 16)
        for ixy in range(ix.shape[0]):
            D[ix[ixy],ix[ixy]] = 1.0
            D[ix[ixy] + 1,ix[ixy]] = 1.0
            D[ix[ixy] + 5, ix[ixy]] = 1.0
            D[ix[ixy] + 4, ix[ixy]] = 1.0


        q = (torch.inverse(((W.T.matmul(D.T)).matmul(D)).matmul(W))).matmul(W.T).matmul(D.T).matmul(P - D).matmul((b))


    return q

def extract_principal_directions(W, order = 0):
    # input:
        # W: FC Weight matrix
        # principal direction order:
    # output:
        # direction: v_direction

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    start.record()
    u, s, v = torch.svd(W)
    end.record()
    torch.cuda.synchronize()
    print("svd", start.elapsed_time(end))
    v_direction =  v[:, order]

    return v_direction

def great_circle_walk(z_start, v, step):
    # input:
        # z_start vector [20]
        # v direction [20]

    Pv = v.unsqueeze(0).T.matmul(v.unsqueeze(0))
    Pv_ort = torch.eye(20).cuda() - Pv
    pvzzo = Pv_ort.matmul(z_start).unsqueeze(0)
    z_norm = z_start.norm(2)
    tetain = torch.sign(z_start.matmul(v.unsqueeze(0).T))
    theta_zero = tetain * np.arccos((pvzzo.norm(2) / z_norm).detach().cpu().numpy())
    z_new = z_norm * ((torch.cos((theta_zero + step))) * (pvzzo / pvzzo.norm(2)) + (torch.sin(theta_zero + step)) * v)

    print(torch.sin(theta_zero + step))

    return z_new



def small_circle_walk(z_start, v, v_ref, step):
    # input:
        # z_start vector [20]
        # v direction [20]
        # v_ref: reference direction
    vv = torch.zeros((v.shape[0], 2)).cuda()
    vv[:, 0] = v
    vv[:, 1] = v_ref
    Pv = vv.matmul(vv.T)
    Pv_ort = torch.eye(v.shape[0]).cuda() - Pv
    Pvz = (Pv.matmul(z_start)).T.matmul(Pv.matmul(z_start))
    pvzzo = Pv_ort.matmul(z_start).unsqueeze(0)
    # theta = step
    pv_ref = v_ref.matmul(v_ref.T)
    pv_refzo = pv_ref.matmul(z_start).unsqueeze(0)

    tetain = torch.sign((Pv.matmul(z_start)).matmul(v.unsqueeze(0).T))
    theta_zero = tetain * np.arccos((pv_refzo.norm(2) / torch.sqrt(Pvz)).detach().cpu().numpy())
    z_new = pvzzo + torch.sqrt(Pvz) * (torch.cos(theta_zero+step) * v_ref + torch.sin(theta_zero+step) * v)


    return z_new


def linear_walk(z_start, v, step):
    # input:
        # z_start vector [20]
        # v direction [20]
    return z_start+step*v
