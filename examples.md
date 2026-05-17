 ````python

data.cst_edges = torch.tensor([
    [0, 0, 0, 0, 0,  1, 1, 1, 1],
    [0, 1, 2, 3, 4,  3, 4, 5, 6]
], dtype=torch.long)

data.num_val = 7

data.LE = torch.tensor([
    1, 0, 0, 0, 1,   # C1 → valores [0,1,2,3,4]
    1, 0, 0, 1       # C2 → valores [3,4,5,6]
], dtype=torch.long)
# data.LE        = [1, 0, 0, 0, 1, 1, 0, 0, 1]

data.var_idx = torch.tensor([0, 0, 0,  1, 1,  2, 2], dtype=torch.long)
data.num_var = 3  # número de variables (X, Y, Z)

#En ANYCSP():

self.hidden_dim = 128

self.h_val_init = torch.nn.Parameter(
    torch.normal(0.0, 1.0, (1, self.hidden_dim), dtype=torch.float32)
)

## En Forward:

assignment = torch.zeros((7, 1))
assignment[[1, 3, 6]] = 1.0

tensor([
    [0.],  # X=1
    [1.],  # X=2 ← asignado
    [0.],  # X=3
    [1.],  # Y=1 ← asignado
    [0.],  # Y=2
    [0.],  # Z=1
    [1.]   # Z=2 ← asignado
])

# assignment = Lv(v)

#tile replica la fila única de self.h_val_init 7 veces
# h_val.shape = (7, 128)
h_val = self.h_val_init.tile(data.num_val, 1)   

h_val = tensor([
    [ 0.12, -0.88,  ...,  0.67],  # valor 0: X=1
    [ 0.12, -0.88,  ...,  0.67],  # valor 1: X=2
    [ 0.12, -0.88,  ...,  0.67],  # valor 2: X=3
    [ 0.12, -0.88,  ...,  0.67],  # valor 3: Y=1
    [ 0.12, -0.88,  ...,  0.67],  # valor 4: Y=2
    [ 0.12, -0.88,  ...,  0.67],  # valor 5: Z=1
    [ 0.12, -0.88,  ...,  0.67],  # valor 6: Z=2
])

En self.val2cst -> r_cst, x_val = self.val2cst(data, h_val, assignment)

x_val = torch.cat([h_val, assign.view(-1, 1).half()], dim=1) #x_val.shape = (7, 129)

# h_val → (7, 128)
# assignment.view(-1, 1).half() → (7, 1)
# dim=1 indica que la concatenación es horizontal (por columnas)

#.view(-1, 1) significa:
# “Reorganiza los elementos en tantas filas como sea necesario (-1 infiere automáticamente el número)”
# “Usa exactamente 1 columna”
# Por defecto, los tensores float son float32 .half() los convierte en float16

x_val = tensor([
    [ 0.12, -0.88, ..., 0.67, 0.0],  # 0: X=1 → no asignado
    [ 0.12, -0.88, ..., 0.67, 1.0],  # 1: X=2 → asignado
    [ 0.12, -0.88, ..., 0.67, 0.0],  # 2: X=3
    [ 0.12, -0.88, ..., 0.67, 1.0],  # 3: Y=1 → asignado
    [ 0.12, -0.88, ..., 0.67, 0.0],  # 4: Y=2
    [ 0.12, -0.88, ..., 0.67, 0.0],  # 5: Z=1
    [ 0.12, -0.88, ..., 0.67, 1.0],  # 6: Z=2 → asignado
], dtype=torch.float16)

x_val = self.val_enc(x_val) # Linear(129→128) → ReLU → Linear(128→128) → LayerNorm 
# x_val.shape = (7, 128)  

x_val = tensor([
  [ 0.21, -0.42, ...,  0.88],  # valor 0: X=1
  [ 0.35, -0.36, ...,  0.92],  # valor 1: X=2
  [ 0.19, -0.51, ...,  0.80],  # valor 2: X=3
  [ 0.50, -0.33, ...,  0.75],  # valor 3: Y=1
  [ 0.28, -0.48, ...,  0.85],  # valor 4: Y=2
  [ 0.33, -0.38, ...,  0.91],  # valor 5: Z=1
  [ 0.61, -0.44, ...,  0.95],  # valor 6: Z=2
])


m_val = self.val_send(x_val) # Linear(128→256) → LayerNorm(256)
# m_val.shape = (7, 256)

m_val = m_val.view(2 * data.num_val, self.hidden_dim)
# m_val.shape = (14, 128)

m_val = tensor([
    [...],  # m(v0, 0)
    [...],  # m(v0, 1)
    [...],  # m(v1, 0)
    [...],  # m(v1, 1)
    [...],  # m(v2, 0)
    [...],  # m(v2, 1)
    [...],  # m(v3, 0)
    [...],  # m(v3, 1)
    [...],  # m(v4, 0)
    [...],  # m(v4, 1)
    [...],  # m(v5, 0)
    [...],  # m(v5, 1)
    [...],  # m(v6, 0)
    [...],  # m(v6, 1)
])  # shape (14, 128)

# Para cada valor v, generamos dos mensajes distintos: uno si LE(C, v) = 0 y otro si LE(C, v) = 1.
# El primero está en la fila 2*v, el segundo en la fila 2*v + 1.

# data.cst_edges[0] = [0, 0, 0, 0, 0, 1, 1, 1, 1]
# data.cst_edges[1] = [0, 1, 2, 3, 4,  3, 4, 5, 6]
# data.LE        = [1, 0, 0, 0, 1,   1, 0, 0, 1]

out_idx = 2 * data.cst_edges[1] + data.LE
# equivale a:
# 2*[0,1,2,3,4, 3,4,5,6] + [1,0,0,0,1, 1,0,0,1]
# = [1,2,4,6,9,  7,8,10,13]

in_idx = data.cst_edges[0]
# = [0,0,0,0,0, 1,1,1,1]  ← constraint indices (C1=0, C2=1) ← a qué constraint se envía cada mensaje

r_cst = aggregate(m_val, out_idx, in_idx, data.num_cst, self.aggr)
# shape (2, 128)

r_cst = tensor([
    [...],  # constraint C1 aggregate message (suma de 5 vectores de 128 dim)
    [...],  # constraint C2 aggregate message (suma de 4 vectores de 128 dim)
])  # shape (2, 128)

# r_cst[0] = m_val[1] + m_val[2] + m_val[4] + m_val[6] + m_val[9]   ← entradas para C1
# r_cst[1] = m_val[7] + m_val[8] + m_val[10] + m_val[13]           ← entradas para C2

# scatter(msg[out_idx], in_idx, dim=0, dim_size=dim_size, reduce=aggr)
# Toma los mensajes msg[out_idx] (i.e., selecciona las filas correspondientes en m_val).
# Agrupa (suma, promedio, etc. según aggr) según los índices en in_idx, a lo largo del dim=0.
# Produce un tensor de tamaño (dim_size, hidden_dim), que en este caso es (data.num_cst, 128)

En self.cst2val -> y_val = self.cst2val(data, x_val, r_cst)

m_cst = self.cst_send(r_cst)
# self.cst_send: Linear(128 → 128) → ReLU → Linear(128 → 256) → LayerNorm(256)
# Resultado: m_cst.shape = (2, 256)

m_cst = m_cst.view(2 * data.num_cst, self.hidden_dim)
# m_cst.shape = (4, 128)
# Para cada restricción (2 en total), genera dos mensajes:
m_cst = tensor([
    [...],  # m_cst[0] = mensaje de C1 con LE=0
    [...],  # m_cst[1] = mensaje de C1 con LE=1
    [...],  # m_cst[2] = mensaje de C2 con LE=0
    [...],  # m_cst[3] = mensaje de C2 con LE=1
])  # shape (4, 128)


out_idx = 2 * data.cst_edges[0] + data.LE
# data.cst_edges[0] = [0, 0, 0, 0, 0, 1, 1, 1, 1]
# data.cst_edges[1] = [0, 1, 2, 3, 4,  3, 4, 5, 6]
# data.LE           = [1, 0, 0, 0, 1, 1, 0, 0, 1]
# out_idx = [1, 0, 0, 0, 1, 3, 2, 2, 3]  ← índices para seleccionar mensajes desde m_cst

in_idx = data.cst_edges[1]
# in_idx = [0, 1, 2, 3, 4, 3, 4, 5, 6]  ← a qué valor v se envía cada mensaje

r_val = aggregate(m_cst, out_idx, in_idx, data.num_val, self.aggr)
# Toma los mensajes m_cst[out_idx], los suma y los deja en r_val[in_idx]

# out_idx = [1, 0, 0, 0, 1, 3, 2, 2, 3]
# in_idx = [0, 1, 2, 3, 4, 3, 4, 5, 6]
r_val = tensor([
    [...],  # v=0 ← recibe m_cst[1]       ← C1 con LE=1
    [...],  # v=1 ← recibe m_cst[0]       ← C1 con LE=0
    [...],  # v=2 ← recibe m_cst[0]       ← C1 con LE=0
    [...],  # v=3 ← m_cst[0] + m_cst[3]   ← C1 con LE=0 + C2 con LE=1
    [...],  # v=4 ← m_cst[1] + m_cst[2]   ← C1 con LE=1 + C2 con LE=0
    [...],  # v=5 ← m_cst[2]              ← C2 con LE=0
    [...],  # v=6 ← m_cst[3]              ← C2 con LE=1
])  # shape (7, 128)

y_val = self.val_rec(x_val + r_val) + x_val

# x_val (7, 128) + r_val (7, 128) → tensor shape (7, 128)
# self.val_rec: Linear(128 → 128) → ReLU → Linear(128 → 128) → LayerNorm(128)
# self.val_rec(...)(7, 128) + r_val (7, 128) → tensor shape (7, 128)

y_val = tensor([
    [...],  # valor 0: X=1
    [...],  # valor 1: X=2
    [...],  # valor 2: X=3
    [...],  # valor 3: Y=1
    [...],  # valor 4: Y=2
    [...],  # valor 5: Z=1
    [...],  # valor 6: Z=2
])  # shape (7, 128)

En self.val2val -> z_val = self.val2val(data, y_val) 

# data.var_idx = torch.tensor([0, 0, 0,  1, 1,  2, 2], dtype=torch.long)
# data.num_var = 3  # número de variables (X, Y, Z)

z_var = scatter(y_val, data.var_idx, dim=0, dim_size=data.num_var, reduce=self.aggr)

z_var = tensor([
  [...],  # z_var[0] = y_val[0] + y_val[1] + y_val[2]   ← variable X (valores 0,1,2)
  [...],  # z_var[1] = y_val[3] + y_val[4]              ← variable Y (valores 3,4)
  [...],  # z_var[2] = y_val[5] + y_val[6]              ← variable Z (valores 5,6)
])  # shape (3, 128)

# Apply U_X and send result back
z_var = self.var_enc(z_var) # Linear(128→128)->ReLU->Linear(128→128)->LayerNorm(128)

z_var = tensor([
  [...],  # var_enc(z_var[0])  ← info agregada de X ya transformada
  [...],  # var_enc(z_var[1])  ← info agregada de Y ya transformada
  [...],  # var_enc(z_var[2])  ← info agregada de Z ya transformada
])  # shape (3, 128)


y_val += z_var[data.var_idx]

(z_val)
y_val = tensor([
  [...],  # valor 0: X=1    ← y_val[0] (previo) + z_var[0]
  [...],  # valor 1: X=2    ← y_val[1] (previo) + z_var[0]
  [...],  # valor 2: X=3    ← y_val[2] (previo) + z_var[0]
  [...],  # valor 3: Y=1    ← y_val[3] (previo) + z_var[1]
  [...],  # valor 4: Y=2    ← y_val[4] (previo) + z_var[1]
  [...],  # valor 5: Z=1    ← y_val[5] (previo) + z_var[2]
  [...],  # valor 6: Z=2    ← y_val[6] (previo) + z_var[2]
])  # shape (7, 128)


```
