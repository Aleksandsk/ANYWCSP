import torch
from torch_geometric.utils import degree
from torch_scatter import scatter_sum, scatter_min, scatter_max, segment_max_csr


class Constraint_Data_Base:

    def __init__(self, csp_data, cst_edges, cst_var_edges, batch=None):
        self.cst_edges = cst_edges
        self.cst_var_edges = cst_var_edges

        self.LE = None
        self.num_cst = cst_edges[0].max().numpy() + 1
        self.num_edges = cst_edges.shape[1]
        self.edge_cost = torch.zeros((self.num_edges, 1), dtype=torch.float32)

        self.cst_deg = degree(cst_edges[0], dtype=torch.int64)
        self.cst_arity = degree(cst_var_edges[0], dtype=torch.int64)
        self.val_deg = degree(cst_edges[1], dtype=torch.int64, num_nodes=csp_data.num_val)

        self.batch = batch if batch is not None else torch.zeros((self.num_cst,), dtype=torch.int64)
        self.batch_size = int(self.batch.max().numpy())

        self.device = 'cpu'

    def to(self, device):
        self.device = device
        self.cst_edges = self.cst_edges.to(device)
        self.cst_var_edges = self.cst_var_edges.to(device)
        self.batch = self.batch.to(device)
        self.cst_deg = self.cst_deg.to(device)
        self.cst_arity = self.cst_arity.to(device)
        self.val_deg = self.val_deg.to(device)
        self.edge_cost = self.edge_cost.to(device)

    def update_LE_(self, **kwargs):
        raise NotImplementedError

    def is_sat(self, assignment):
        raise NotImplementedError


class Constraint_Data(Constraint_Data_Base):

    def __init__(self, csp_data, cst_idx, tup_idx, val_idx, cst_type, cst_edges=None, cst_var_edges=None, lit_edge_map=None, batch=None):
        self.cst_idx = cst_idx
        self.tup_idx = tup_idx
        self.val_idx = val_idx
        self.cst_type = cst_type

        if cst_var_edges is None:
            cst_var_edges = torch.unique(torch.stack([cst_idx[tup_idx], csp_data.var_idx[self.val_idx]], dim=0), dim=1)
        if cst_edges is None or lit_edge_map is None:
            cst_var_dom_size = csp_data.domain_size[cst_var_edges[1]]
            cst_edges = torch.repeat_interleave(cst_var_edges, cst_var_dom_size, dim=1)
            cst_edges[1] = csp_data.var_off[cst_edges[1]]

            cst_val_off = torch.zeros_like(cst_var_dom_size)
            cst_val_off[1:] += torch.cumsum(cst_var_dom_size[:-1], dim=0)
            cst_val_shift = torch.arange(cst_edges.shape[1])
            cst_val_shift -= torch.repeat_interleave(cst_val_off, cst_var_dom_size, dim=0)
            cst_edges[1] += cst_val_shift

            cst_edges = torch.cat([
                cst_edges,
                torch.stack([cst_idx[tup_idx], self.val_idx], dim=0)
            ], dim=1)
            cst_edges, lit_edge_map = torch.unique(cst_edges, dim=1, return_inverse=True)
            lit_edge_map = lit_edge_map[cst_edges.shape[1]:].contiguous()

        super(Constraint_Data, self).__init__(
                csp_data=csp_data,
                cst_edges=cst_edges,
                cst_var_edges=cst_var_edges,
                batch=batch,
        )

        self.num_tup = tup_idx.max().numpy() + 1
        self.tup_deg = degree(tup_idx, num_nodes=self.num_tup, dtype=torch.uint8).view(-1, 1)
        self.cst_num_tup = degree(cst_idx, num_nodes=self.num_cst, dtype=torch.float32).view(-1, 1)

        self.lit_edge_map = lit_edge_map
        self.cst_neg_mask = self.cst_type.bool()
        self.neg_edge_mask = self.cst_neg_mask[self.cst_edges[0]]
        self.edge_comp_thresh = (self.cst_arity.int()-1)[self.cst_edges[0]].view(-1, 1)

    def to(self, device):
        super(Constraint_Data, self).to(device)
        self.cst_idx = self.cst_idx.to(device)
        self.val_idx = self.val_idx.to(device)
        self.tup_idx = self.tup_idx.to(device)
        self.tup_deg = self.tup_deg.to(device)
        self.cst_num_tup = self.cst_num_tup.to(device)
        self.lit_edge_map = self.lit_edge_map.to(device)
        self.cst_type = self.cst_type.to(device)
        self.cst_neg_mask = self.cst_neg_mask.to(device)
        self.neg_edge_mask = self.neg_edge_mask.to(device)
        self.edge_comp_thresh = self.edge_comp_thresh.to(device)

    @staticmethod
    def collate(batch_list, merged_csp_data):
        cst_idx, tup_idx, val_idx, cst_type, batch_idx = [], [], [], [], []
        cst_off, tup_off, edge_off = 0, 0, 0
        cst_val_edges, cst_var_edges, lit_edge_map = [], [], []
        for cst_data, var_off, val_off, i in batch_list:
            cst_idx.append(cst_data.cst_idx + cst_off)
            tup_idx.append(cst_data.tup_idx + tup_off)
            val_idx.append(cst_data.val_idx + val_off)

            cur_cst_val_edges = cst_data.cst_edges.clone()
            cur_cst_val_edges[0] += cst_off
            cur_cst_val_edges[1] += val_off
            cst_val_edges.append(cur_cst_val_edges)

            cur_cst_var_edges = cst_data.cst_var_edges.clone()
            cur_cst_var_edges[0] += cst_off
            cur_cst_var_edges[1] += var_off
            cst_var_edges.append(cur_cst_var_edges)

            lit_edge_map.append(cst_data.lit_edge_map + edge_off)

            cst_type.append(cst_data.cst_type)
            batch_idx.append(cst_data.batch + i)

            cst_off += cst_data.num_cst
            tup_off += cst_data.num_tup
            edge_off += cst_data.num_edges

        cst_idx = torch.cat(cst_idx, dim=0)
        tup_idx = torch.cat(tup_idx, dim=0)
        val_idx = torch.cat(val_idx, dim=0)
        cst_type = torch.cat(cst_type, dim=0)
        batch_idx = torch.cat(batch_idx, dim=0)
        cst_val_edges = torch.cat(cst_val_edges, dim=1)
        cst_var_edges = torch.cat(cst_var_edges, dim=1)
        lit_edge_map = torch.cat(lit_edge_map, dim=0)

        batch_cst_data = Constraint_Data(
            csp_data=merged_csp_data,
            cst_idx=cst_idx,
            tup_idx=tup_idx,
            val_idx=val_idx,
            cst_type=cst_type,
            batch=batch_idx,
            cst_edges=cst_val_edges,
            cst_var_edges=cst_var_edges,
            lit_edge_map=lit_edge_map,
        )
        return batch_cst_data

    def update_LE_(self, assignment, tup_sum, **kwargs):
        num_sat_neigh = scatter_max(tup_sum[self.tup_idx], self.lit_edge_map, dim=0, dim_size=self.num_edges)[0]
        num_sat_neigh = num_sat_neigh.int()
        num_sat_neigh -= assignment[self.cst_edges[1]]

        comp_mask = num_sat_neigh != self.edge_comp_thresh
        comp_mask[self.neg_edge_mask] = ~comp_mask[self.neg_edge_mask]
        self.LE = comp_mask.long()

    def is_sat(self, assignment, update_val_comp=False):
        assignment = assignment.byte()
        tup_sum = scatter_sum(assignment[self.val_idx], self.tup_idx, dim=0, dim_size=self.num_tup)
        tup_sat = self.tup_deg - tup_sum
        cst_sat = scatter_min(tup_sat, self.cst_idx, dim=0, dim_size=self.num_cst)[0]
        cst_sat = cst_sat == 0
        cst_sat[self.cst_neg_mask] = ~cst_sat[self.cst_neg_mask]

        if update_val_comp:
            self.update_LE_(assignment, tup_sum)

        return cst_sat.float()


class Constraint_Data_All_Diff(Constraint_Data_Base):

    def __init__(self, csp_data, cst_idx, var_idx, cst_edges=None, cst_var_edges=None, count_edge_map=None, batch=None):
        self.cst_idx = cst_idx
        self.var_idx = var_idx

        cst_var_dom_size = csp_data.domain_size[var_idx]
        self.cst_dom_size = scatter_max(cst_var_dom_size, cst_idx, dim=0)[0]
        self.num_val = self.cst_dom_size.sum().cpu().numpy()
        self.cst_ptr = torch.zeros((self.cst_dom_size.shape[0]+1,), dtype=torch.int64)
        self.cst_ptr[1:] = torch.cumsum(self.cst_dom_size, dim=0)

        if cst_var_edges is None:
            cst_var_edges = torch.stack([cst_idx, var_idx], dim=0)
        if cst_edges is None:
            cst_edges = torch.repeat_interleave(cst_var_edges, cst_var_dom_size, dim=1)
            cst_edges[1] = csp_data.var_off[cst_edges[1]]

            cst_val_off = torch.zeros_like(cst_var_dom_size)
            cst_val_off[1:] += torch.cumsum(cst_var_dom_size[:-1], dim=0)
            cst_val_shift = torch.arange(cst_edges.shape[1])
            cst_val_shift -= torch.repeat_interleave(cst_val_off, cst_var_dom_size, dim=0)
            cst_edges[1] += cst_val_shift

        super(Constraint_Data_All_Diff, self).__init__(
            csp_data=csp_data,
            cst_edges=cst_edges,
            cst_var_edges=cst_var_edges,
            batch=batch,
        )

        self.count_edge_map = self.cst_ptr[cst_edges[0]] + csp_data.dom_idx[cst_edges[1]]
        self.val_var_idx = csp_data.var_idx
        self.var_off = csp_data.var_off

    def to(self, device):
        super(Constraint_Data_All_Diff, self).to(device)
        self.cst_idx = self.cst_idx.to(device)
        self.var_idx = self.var_idx.to(device)
        self.val_var_idx = self.val_var_idx.to(device)
        self.var_off = self.var_off.to(device)
        self.cst_ptr = self.cst_ptr.to(device)
        self.cst_dom_size = self.cst_dom_size.to(device)
        self.count_edge_map = self.count_edge_map.to(device)

    @staticmethod
    def collate(batch_list, merged_csp_data):
        cst_idx, var_idx, batch_idx = [], [], []
        cst_off, edge_off = 0, 0
        cst_val_edges, cst_var_edges, count_edge_map = [], [], []
        for cst_data, var_off, val_off, i in batch_list:
            cst_idx.append(cst_data.cst_idx + cst_off)
            var_idx.append(cst_data.var_idx + var_off)

            cur_cst_val_edges = cst_data.cst_edges.clone()
            cur_cst_val_edges[0] += cst_off
            cur_cst_val_edges[1] += val_off
            cst_val_edges.append(cur_cst_val_edges)

            cur_cst_var_edges = cst_data.cst_var_edges.clone()
            cur_cst_var_edges[0] += cst_off
            cur_cst_var_edges[1] += var_off
            cst_var_edges.append(cur_cst_var_edges)

            count_edge_map.append(cst_data.count_edge_map + edge_off)

            batch_idx.append(cst_data.batch + i)

            cst_off += cst_data.num_cst
            edge_off += cst_data.num_edges

        cst_idx = torch.cat(cst_idx, dim=0)
        var_idx = torch.cat(var_idx, dim=0)
        batch_idx = torch.cat(batch_idx, dim=0)
        cst_val_edges = torch.cat(cst_val_edges, dim=1)
        cst_var_edges = torch.cat(cst_var_edges, dim=1)
        count_edge_map = torch.cat(count_edge_map, dim=0)

        batch_cst_data = Constraint_Data_All_Diff(
            csp_data=merged_csp_data,
            cst_idx=cst_idx,
            var_idx=var_idx,
            batch=batch_idx,
            cst_edges=cst_val_edges,
            cst_var_edges=cst_var_edges,
            count_edge_map=count_edge_map,
        )
        return batch_cst_data

    def update_LE_(self, assignment, val_count, **kwargs):
        val_count = val_count[self.count_edge_map]
        val_count -= assignment[self.cst_edges[1]].int().flatten()
        comp_mask = val_count == 0
        self.LE = 1.0 - comp_mask.float().view(-1, 1)

    def is_sat(self, assignment, update_val_comp=False):
        idx_assignment = scatter_max(assignment, self.val_var_idx, dim=0)[1] - self.var_off.view(-1, 1)
        idx_assignment = idx_assignment[self.var_idx] + self.cst_ptr[self.cst_idx].view(-1, 1)
        value_count = degree(idx_assignment.flatten(), num_nodes=self.num_val, dtype=torch.int32)
        cst_sat = segment_max_csr(value_count, self.cst_ptr)[0]
        cst_sat = cst_sat <= 1

        if update_val_comp:
            self.update_LE_(assignment, value_count)

        return cst_sat.view(-1, assignment.shape[1]).float()


class Constraint_Data_TSP_Edge(Constraint_Data_Base):

    def __init__(
            self,
            csp_data,
            src_var_idx,
            dst_var_idx,
            pair_exists,
            pair_cost,
            cst_edges=None,
            cst_var_edges=None,
            edge_side=None,
            edge_city=None,
            batch=None
    ):
        self.src_var_idx = src_var_idx.long()
        self.dst_var_idx = dst_var_idx.long()
        self.pair_exists = pair_exists.bool()
        self.pair_cost = pair_cost.float()

        num_cst = self.src_var_idx.shape[0]
        if self.pair_exists.dim() == 2:
            self.pair_exists = self.pair_exists.unsqueeze(0).repeat(num_cst, 1, 1)
        if self.pair_cost.dim() == 2:
            self.pair_cost = self.pair_cost.unsqueeze(0).repeat(num_cst, 1, 1)

        if cst_var_edges is None:
            cst_idx = torch.arange(num_cst, dtype=torch.int64)
            cst_var_edges = torch.stack([
                torch.repeat_interleave(cst_idx, 2),
                torch.stack([self.src_var_idx, self.dst_var_idx], dim=1).flatten()
            ], dim=0)

        if cst_edges is None:
            cst_var_dom_size = csp_data.domain_size[cst_var_edges[1]]
            cst_edges = torch.repeat_interleave(cst_var_edges, cst_var_dom_size, dim=1)
            cst_edges[1] = csp_data.var_off[cst_edges[1]]

            cst_val_off = torch.zeros_like(cst_var_dom_size)
            cst_val_off[1:] += torch.cumsum(cst_var_dom_size[:-1], dim=0)
            cst_val_shift = torch.arange(cst_edges.shape[1])
            cst_val_shift -= torch.repeat_interleave(cst_val_off, cst_var_dom_size, dim=0)
            cst_edges[1] += cst_val_shift

        if edge_side is None:
            cst_var_dom_size = csp_data.domain_size[cst_var_edges[1]]
            cst_var_side = torch.arange(cst_var_edges.shape[1], dtype=torch.int64) % 2
            edge_side = torch.repeat_interleave(cst_var_side, cst_var_dom_size)
        if edge_city is None:
            edge_city = csp_data.dom_idx[cst_edges[1]]

        super(Constraint_Data_TSP_Edge, self).__init__(
            csp_data=csp_data,
            cst_edges=cst_edges,
            cst_var_edges=cst_var_edges,
            batch=batch,
        )

        self.edge_side = edge_side.long()
        self.edge_city = edge_city.long()
        self.val_var_idx = csp_data.var_idx
        self.var_off = csp_data.var_off

    def to(self, device):
        super(Constraint_Data_TSP_Edge, self).to(device)
        self.src_var_idx = self.src_var_idx.to(device)
        self.dst_var_idx = self.dst_var_idx.to(device)
        self.pair_exists = self.pair_exists.to(device)
        self.pair_cost = self.pair_cost.to(device)
        self.edge_side = self.edge_side.to(device)
        self.edge_city = self.edge_city.to(device)
        self.val_var_idx = self.val_var_idx.to(device)
        self.var_off = self.var_off.to(device)

    @staticmethod
    def _pad_square(matrix, size):
        if matrix.shape[1] == size and matrix.shape[2] == size:
            return matrix
        padded = torch.zeros(
            (matrix.shape[0], size, size),
            dtype=matrix.dtype,
            device=matrix.device
        )
        padded[:, :matrix.shape[1], :matrix.shape[2]] = matrix
        return padded

    @staticmethod
    def collate(batch_list, merged_csp_data):
        src_var_idx, dst_var_idx, pair_exists, pair_cost, batch_idx = [], [], [], [], []
        cst_val_edges, cst_var_edges, edge_side, edge_city = [], [], [], []
        cst_off = 0
        for cst_data, var_off, val_off, i in batch_list:
            src_var_idx.append(cst_data.src_var_idx + var_off)
            dst_var_idx.append(cst_data.dst_var_idx + var_off)
            pair_exists.append(Constraint_Data_TSP_Edge._pad_square(cst_data.pair_exists, merged_csp_data.max_dom))
            pair_cost.append(Constraint_Data_TSP_Edge._pad_square(cst_data.pair_cost, merged_csp_data.max_dom))

            cur_cst_val_edges = cst_data.cst_edges.clone()
            cur_cst_val_edges[0] += cst_off
            cur_cst_val_edges[1] += val_off
            cst_val_edges.append(cur_cst_val_edges)

            cur_cst_var_edges = cst_data.cst_var_edges.clone()
            cur_cst_var_edges[0] += cst_off
            cur_cst_var_edges[1] += var_off
            cst_var_edges.append(cur_cst_var_edges)

            edge_side.append(cst_data.edge_side)
            edge_city.append(cst_data.edge_city)
            batch_idx.append(cst_data.batch + i)

            cst_off += cst_data.num_cst

        batch_cst_data = Constraint_Data_TSP_Edge(
            csp_data=merged_csp_data,
            src_var_idx=torch.cat(src_var_idx, dim=0),
            dst_var_idx=torch.cat(dst_var_idx, dim=0),
            pair_exists=torch.cat(pair_exists, dim=0),
            pair_cost=torch.cat(pair_cost, dim=0),
            batch=torch.cat(batch_idx, dim=0),
            cst_edges=torch.cat(cst_val_edges, dim=1),
            cst_var_edges=torch.cat(cst_var_edges, dim=1),
            edge_side=torch.cat(edge_side, dim=0),
            edge_city=torch.cat(edge_city, dim=0),
        )
        return batch_cst_data

    def _value_idx(self, assignment):
        value_idx = scatter_max(assignment, self.val_var_idx, dim=0)[1]
        value_idx = value_idx - self.var_off.view(-1, 1)
        return value_idx.long()

    def update_LE_(self, assignment, value_idx=None, **kwargs):
        if value_idx is None:
            value_idx = self._value_idx(assignment)

        src_city = value_idx[self.src_var_idx]
        dst_city = value_idx[self.dst_var_idx]

        edge_cst = self.cst_edges[0].long().view(-1, 1)
        cand_city = self.edge_city.view(-1, 1)
        is_src_side = (self.edge_side == 0).view(-1, 1)

        other_city = torch.where(is_src_side, dst_city[edge_cst.flatten()], src_city[edge_cst.flatten()])
        from_city = torch.where(is_src_side, cand_city.expand_as(other_city), other_city)
        to_city = torch.where(is_src_side, other_city, cand_city.expand_as(other_city))

        self.LE = self.pair_exists[edge_cst, from_city, to_city].long()
        self.edge_cost = self.pair_cost[edge_cst, from_city, to_city].float()

    def is_sat(self, assignment, update_val_comp=False):
        value_idx = self._value_idx(assignment)
        src_city = value_idx[self.src_var_idx]
        dst_city = value_idx[self.dst_var_idx]
        cst_idx = torch.arange(self.num_cst, device=assignment.device).view(-1, 1)

        cst_sat = self.pair_exists[cst_idx, src_city, dst_city]
        if update_val_comp:
            self.update_LE_(assignment, value_idx=value_idx)

        return cst_sat.float()


class Constraint_Data_Linear(Constraint_Data_Base):

    def __init__(self, csp_data, cst_idx, var_idx, coeffs, b, comp_op, cst_edges=None, cst_var_edges=None, cst_var_dom_size=None, batch=None):
        self.cst_idx = cst_idx
        self.var_idx = var_idx
        self.coeffs = coeffs
        self.comp_op = comp_op
        self.b = b

        if cst_var_edges is None:
            cst_var_edges = torch.stack([cst_idx, var_idx], dim=0)
        if cst_edges is None:
            cst_var_dom_size = csp_data.domain_size[cst_var_edges[1]]
            cst_edges = torch.repeat_interleave(cst_var_edges, cst_var_dom_size, dim=1)
            cst_edges[1] = csp_data.var_off[cst_edges[1]]

            cst_val_off = torch.zeros_like(cst_var_dom_size)
            cst_val_off[1:] += torch.cumsum(cst_var_dom_size[:-1], dim=0)
            cst_val_shift = torch.arange(cst_edges.shape[1])
            cst_val_shift -= torch.repeat_interleave(cst_val_off, cst_var_dom_size, dim=0)
            cst_edges[1] += cst_val_shift

        super(Constraint_Data_Linear, self).__init__(
            csp_data=csp_data,
            cst_edges=cst_edges,
            cst_var_edges=cst_var_edges,
            batch=batch
        )

        self.cst_var_dom_size = cst_var_dom_size
        self.b_expanded = b[cst_edges[0]]
        self.comp_op_expanded = comp_op[cst_edges[0]]
        self.scaled_val = torch.repeat_interleave(coeffs.int(), self.cst_var_dom_size) * csp_data.domain[cst_edges[1]].int()

    def to(self, device):
        super(Constraint_Data_Linear, self).to(device)
        self.cst_idx = self.cst_idx.to(device)
        self.var_idx = self.var_idx.to(device)
        self.b = self.b.to(device)
        self.coeffs = self.coeffs.to(device)
        self.comp_op = self.comp_op.to(device)
        self.cst_var_dom_size = self.cst_var_dom_size.to(device)
        self.b_expanded = self.b_expanded.to(device)
        self.comp_op_expanded = self.comp_op_expanded.to(device)
        self.scaled_val = self.scaled_val.to(device)

    @staticmethod
    def collate(batch_list, merged_csp_data):
        cst_idx, var_idx, coeffs, b, comp_op, batch_idx = [], [], [], [], [], []
        cst_off, edge_off = 0, 0
        cst_val_edges, cst_var_edges, cst_var_dom_size = [], [], []
        for cst_data, var_off, val_off, i in batch_list:
            cst_idx.append(cst_data.cst_idx + cst_off)
            var_idx.append(cst_data.var_idx + var_off)
            coeffs.append(cst_data.coeffs)
            b.append(cst_data.b)
            comp_op.append(cst_data.comp_op)

            cur_cst_val_edges = cst_data.cst_edges.clone()
            cur_cst_val_edges[0] += cst_off
            cur_cst_val_edges[1] += val_off
            cst_val_edges.append(cur_cst_val_edges)

            cur_cst_var_edges = cst_data.cst_var_edges.clone()
            cur_cst_var_edges[0] += cst_off
            cur_cst_var_edges[1] += var_off
            cst_var_edges.append(cur_cst_var_edges)

            cst_var_dom_size.append(cst_data.cst_var_dom_size)

            batch_idx.append(cst_data.batch + i)

            cst_off += cst_data.num_cst
            edge_off += cst_data.num_edges

        cst_idx = torch.cat(cst_idx, dim=0)
        var_idx = torch.cat(var_idx, dim=0)
        coeffs = torch.cat(coeffs, dim=0)
        b = torch.cat(b, dim=0)
        comp_op = torch.cat(comp_op, dim=0)
        batch_idx = torch.cat(batch_idx, dim=0)
        cst_val_edges = torch.cat(cst_val_edges, dim=1)
        cst_var_edges = torch.cat(cst_var_edges, dim=1)
        cst_var_dom_size = torch.cat(cst_var_dom_size, dim=0)

        batch_cst_data = Constraint_Data_Linear(
            csp_data=merged_csp_data,
            cst_idx=cst_idx,
            var_idx=var_idx,
            coeffs=coeffs,
            b=b,
            comp_op=comp_op,
            batch=batch_idx,
            cst_edges=cst_val_edges,
            cst_var_edges=cst_var_edges,
            cst_var_dom_size=cst_var_dom_size,
        )
        return batch_cst_data

    @staticmethod
    def comp_idx(comp):
        if comp == 'eq':
            return 0
        elif comp == 'ne':
            return 1
        elif comp == 'le':
            return 2
        elif comp == 'ge':
            return 3
        else:
            raise ValueError(f'Operator {comp} not supported for linear constraints!')

    def compare_(self, x, b, op):
        y = torch.stack([
            x == b,
            x != b,
            x <= b,
            x >= b,
        ], dim=1)
        y = y[torch.arange(y.shape[0], device=x.device), op]
        return y

    def update_LE_(self, assignment, lin_comb, val_sel, **kwargs):
        var_lin_comb = lin_comb[self.cst_var_edges[0]] - val_sel
        val_lin_comb = torch.repeat_interleave(var_lin_comb, self.cst_var_dom_size) + self.scaled_val
        #comp_mask = self.compare_(val_lin_comb, self.b_expanded, self.comp_op_expanded)

        diff = val_lin_comb - self.b_expanded
        cost = torch.stack([
            torch.abs(diff),
            (diff == 0).float(),
            torch.relu(diff),
            ], dim=1
        )
        cost = cost[torch.arange(diff.shape[0], device=diff.device), self.comp_op_expanded]
        self.LE = cost.float().view(-1, 1)

    def is_sat(self, assignment, update_val_comp=False):
        assignment = assignment.int()
        val_sel = self.scaled_val[assignment[self.cst_edges[1]].flatten().bool()]
        lin_comb = scatter_sum(val_sel, self.cst_var_edges[0], dim=0)
        cst_sat = self.compare_(lin_comb, self.b, self.comp_op)

        if update_val_comp:
            self.update_LE_(assignment, lin_comb, val_sel)

        return cst_sat.view(-1, assignment.shape[1]).float()
