import numpy as np

# ==============================
# 1. 单元刚度矩阵
# ==============================
# 一维杆单元刚度
def element_stiffness_1d(node1, node2, E, A):
    L = abs(node2 - node1)
    k = E * A / L
    return np.array([[k, -k],
                     [-k, k]])

# 二维桁架单元刚度
def element_stiffness_2d(x1, y1, x2, y2, E, A):
    dx = x2 - x1
    dy = y2 - y1
    L = np.hypot(dx, dy)
    c = dx / L
    s = dy / L
    k = E * A / L
    return k * np.array([
        [c*c,  c*s, -c*c, -c*s],
        [c*s,  s*s, -c*s, -s*s],
        [-c*c, -c*s, c*c,  c*s],
        [-c*s, -s*s, c*s,  s*s]
    ])

# ==============================
# 2. 自由度映射LM 单独分开，彻底解决维度冲突
# ==============================
# 一维专用LM
def build_LM_1d(IEN):
    LM = []
    for n1, n2 in IEN:
        LM.append([n1-1, n2-1])
    return np.array(LM).T

# 二维专用LM
def build_LM_2d(IEN):
    LM = []
    for n1, n2 in IEN:
        dofs = [2*(n1-1), 2*(n1-1)+1,
                2*(n2-1), 2*(n2-1)+1]
        LM.append(dofs)
    return np.array(LM).T

# ==============================
# 3. 单独组装函数 彻底杜绝越界
# ==============================
# 一维组装
def assemble_1d(K, Ke, LM_e):
    i0, i1 = int(LM_e[0]), int(LM_e[1])
    K[i0,i0] += Ke[0,0]
    K[i0,i1] += Ke[0,1]
    K[i1,i0] += Ke[1,0]
    K[i1,i1] += Ke[1,1]

# 二维组装
def assemble_2d(K, Ke, LM_e):
    for i in range(4):
        for j in range(4):
            I = int(LM_e[i])
            J = int(LM_e[j])
            K[I, J] += Ke[i, j]

# ==============================
# 4. 边界条件求解 缩减法
# ==============================
def solve_reduced(K, f, fixed_dofs, fixed_vals):
    n = K.shape[0]
    free_dofs = [i for i in range(n) if i not in fixed_dofs]
    K_FF = K[np.ix_(free_dofs, free_dofs)]
    K_EF = K[np.ix_(fixed_dofs, free_dofs)]
    f_F = f[free_dofs]
    d_E = np.array(fixed_vals)
    # 求解未知位移
    d_F = np.linalg.solve(K_FF, f_F - K_EF.T @ d_E)
    # 组装完整位移
    d = np.zeros(n)
    d[fixed_dofs] = d_E
    d[free_dofs] = d_F
    # 计算支座反力
    reaction = K @ d
    return d, reaction

# ==============================
# 5. 应力轴力后处理
# ==============================
# 一维单元应力
def compute_stress_1d(x1, x2, E, d_e):
    L = abs(x2 - x1)
    return E / L * (-d_e[0] + d_e[1])

# 二维单元应力
def compute_stress_2d(x1, y1, x2, y2, E, d_e):
    dx = x2 - x1
    dy = y2 - y1
    L = np.hypot(dx, dy)
    c = dx / L
    s = dy / L
    return E / L * (-c*d_e[0] - s*d_e[1] + c*d_e[2] + s*d_e[3])

# ======================================================
# 算例1：一维两杆单元
# ======================================================
print("======= 算例1：一维杆结构 =======\n")
E1, A1 = 100, 1
E2, A2 = 200, 1
node_x = [0, 1, 2]
IEN_1d = [[1,2], [2,3]]

neq_1d = 3
K1 = np.zeros((neq_1d, neq_1d))
LM1 = build_LM_1d(IEN_1d)

# 单元刚度+组装
ke1 = element_stiffness_1d(0,1,E1,A1)
ke2 = element_stiffness_1d(1,2,E2,A2)
assemble_1d(K1, ke1, LM1[:,0])
assemble_1d(K1, ke2, LM1[:,1])

print("总体刚度矩阵：")
print(K1.round(2))

# 荷载+边界
f1 = np.zeros(neq_1d)
f1[2] = 10
fixed = [0]
fixed_val = [0]

d1, react1 = solve_reduced(K1, f1, fixed, fixed_val)
print(f"\n节点位移 d = {d1.round(4)}")
print(f"支座反力 = {react1[fixed].round(4)}")

# 单元应力
s1 = compute_stress_1d(0,1,E1, d1[LM1[:,0]])
s2 = compute_stress_1d(1,2,E2, d1[LM1[:,1]])
print(f"单元1应力 = {s1.round(4)}")
print(f"单元2应力 = {s2.round(4)}")

# ======================================================
# 算例2：二维桁架
# ======================================================
print("\n\n======= 算例2：二维桁架结构 =======\n")
E = 1.0
A = 1.0
# 节点坐标
nodes = np.array([[1, 0],   # 节点1
                  [0, 0],   # 节点2
                  [1, 1]])  # 节点3
IEN_2d = [[1,3], [2,3]]

neq_2d = 3 * 2
K2 = np.zeros((neq_2d, neq_2d))
LM2 = build_LM_2d(IEN_2d)

# 单元1 1-3
x1,y1 = nodes[0]
x2,y2 = nodes[2]
ke_a = element_stiffness_2d(x1,y1,x2,y2,E,A)
assemble_2d(K2, ke_a, LM2[:,0])

# 单元2 2-3
x1,y1 = nodes[1]
x2,y2 = nodes[2]
ke_b = element_stiffness_2d(x1,y1,x2,y2,E,A)
assemble_2d(K2, ke_b, LM2[:,1])

print("总体刚度矩阵对称性：", np.allclose(K2, K2.T))

# 荷载边界
f2 = np.zeros(neq_2d)
# 节点3 Y向施加10向下荷载
f2[4+1] = -10
# 1、2节点全固定
fixed_dof = [0,1, 2,3]
fixed_value = [0,0, 0,0]

d2, react2 = solve_reduced(K2, f2, fixed_dof, fixed_value)
print(f"\n节点位移：\n{d2.round(6)}")

# 单元应力计算
de1 = d2[LM2[:,0]]
stress1 = compute_stress_2d(nodes[0,0],nodes[0,1],nodes[2,0],nodes[2,1],E,de1)
de2 = d2[LM2[:,1]]
stress2 = compute_stress_2d(nodes[1,0],nodes[1,1],nodes[2,0],nodes[2,1],E,de2)

print(f"\n单元1轴应力 = {stress1.round(6)}")
print(f"单元2轴应力 = {stress2.round(6)}")