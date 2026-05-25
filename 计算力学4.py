import numpy as np

# ========== 1. 三维杆单元刚度矩阵计算 ==========
def truss3d_element_stiffness(x1, x2, E, A):
    """
    输入：节点1坐标、节点2坐标、弹性模量E、截面积A
    输出：单元长度、方向余弦、整体坐标系刚度矩阵
    """
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)
    dx = x2 - x1

    # 单元长度
    L = np.linalg.norm(dx)
    if L < 1e-12:
        raise ValueError("节点重合，单元退化！")

    # 方向余弦
    cx = dx[0] / L
    cy = dx[1] / L
    cz = dx[2] / L
    direction_cosines = [cx, cy, cz]

    # 坐标变换矩阵
    T = np.array([[-cx, -cy, -cz, cx, cy, cz]])

    # 整体刚度矩阵
    Ke = (E * A / L) * T.T @ T

    return L, direction_cosines, Ke


# ========== 2. 单元应变、应力、轴力计算（已修复类型报错） ==========
def truss3d_element_stress(x1, x2, E, A, de):
    """
    输入：坐标、材料参数、节点位移de = [u1,v1,w1,u2,v2,w2]
    输出：应变ε、应力σ、轴力N（原生标量，无numpy类型报错）
    """
    x1 = np.array(x1, dtype=float)
    x2 = np.array(x2, dtype=float)
    de = np.array(de, dtype=float)
    dx = x2 - x1

    L = np.linalg.norm(dx)
    if L < 1e-12:
        raise ValueError("节点重合！")

    cx = dx[0] / L
    cy = dx[1] / L
    cz = dx[2] / L

    # 纯标量计算
    epsilon = (cx*(de[3]-de[0]) + cy*(de[4]-de[1]) + cz*(de[5]-de[2])) / L
    sigma = E * epsilon
    N = sigma * A

    return epsilon, sigma, N


# ========== 算例1：沿X轴杆单元 ==========
print("========== 算例1：沿x轴杆单元 ==========")
x1 = [0, 0, 0]
x2 = [2, 0, 0]
E = 200e9    # 弹性模量 Pa
A = 1.0e-4   # 截面积 m²
de = [0, 0, 0, 1e-3, 0, 0]  # 节点位移

L1, c1, Ke1 = truss3d_element_stiffness(x1, x2, E, A)
eps1, sig1, N1 = truss3d_element_stress(x1, x2, E, A, de)

print(f"长度 L = {L1:.2f} m")
print(f"方向余弦 c = {c1}")
print(f"应变 ε = {eps1:.2e}")
print(f"应力 σ = {sig1:.2e} Pa")
print(f"轴力 N = {N1:.2e} N")
print("单元刚度矩阵 Ke：")
print(Ke1)


# ========== 算例2：空间任意方向杆 ==========
print("\n========== 算例2：空间任意方向杆 ==========")
x1 = [0, 0, 0]
x2 = [1, 2, 2]
E = 210e9
A = 2.0e-4
de = [0, 0, 0, 1e-3, 2e-3, 2e-3]  # 节点位移

L2, c2, Ke2 = truss3d_element_stiffness(x1, x2, E, A)
eps2, sig2, N2 = truss3d_element_stress(x1, x2, E, A, de)

print(f"长度 L = {L2:.2f} m")
print(f"方向余弦 c = {[round(i,3) for i in c2]}")
print(f"应变 ε = {eps2:.2e}")
print(f"应力 σ = {sig2:.2e} Pa")
print(f"轴力 N = {N2:.2e} N")
print("单元刚度矩阵 Ke：")
print(Ke2)


# ========== 刚度矩阵性质验证 ==========
print("\n========== 刚度矩阵性质验证 ==========")
print(" Ke 是否对称：", np.allclose(Ke2, Ke2.T))
eig_vals = np.linalg.eigvalsh(Ke2)
print(" Ke 是否半正定（无负特征值）：", np.all(eig_vals >= -1e-9))