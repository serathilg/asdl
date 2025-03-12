import os
from typing import Tuple, Iterable
from operator import iadd

import numpy as np
import torch
from torch import Tensor
from .utils import cholesky_inv, psd_damping, smw_inv, target_cond_to_damping
from .vector import ParamVector

__all__ = [
    'matrix_to_tril',
    'tril_to_matrix',
    'get_n_cols_by_tril',
    'SymMatrix',
    'Kron',
    'KFE',
    'Diag',
    'UnitWise'
]

_default_damping = 1e-5


def matrix_to_tril(mat: torch.Tensor):
    """
    Convert matrix (2D array)
    to lower triangular of it (1D array, row direction)

    Example:
      [[1, x, x],
       [2, 3, x], -> [1, 2, 3, 4, 5, 6]
       [4, 5, 6]]
    """
    if mat.ndim != 2:
        raise ValueError(f'mat.ndim has to be 2. Got {mat.ndim}.')
    tril_indices = torch.tril_indices(*mat.shape)
    return mat[tril_indices[0], tril_indices[1]]


def tril_to_matrix(tril: torch.Tensor):
    """
    Convert lower triangular of matrix (1D array)
    to full symmetric matrix (2D array)

    Example:
                            [[1, 2, 4],
      [1, 2, 3, 4, 5, 6] ->  [2, 3, 5],
                             [4, 5, 6]]
    """
    if tril.ndim != 1:
        raise ValueError(f'tril.ndim has to be 1. Got {tril.ndim}.')
    n_cols = get_n_cols_by_tril(tril)
    rst = torch.zeros(n_cols, n_cols, device=tril.device, dtype=tril.dtype)
    tril_indices = torch.tril_indices(n_cols, n_cols)
    rst[tril_indices[0], tril_indices[1]] = tril
    rst = rst + rst.T - torch.diag(torch.diag(rst))
    return rst


def get_n_cols_by_tril(tril: torch.Tensor):
    """
    Get number of columns of original matrix
    by lower triangular (tril) of it.

    ncols^2 + ncols = 2 * tril.numel()
    """
    if tril.ndim != 1:
        raise ValueError(f'tril.ndim has to be 1. Got {tril.ndim}.')
    numel = tril.numel()
    return int(np.sqrt(2 * numel + 0.25) - 0.5)


def symeig(A: torch.Tensor, upper=True):
    return torch.linalg.eigvalsh(A, UPLO='U' if upper else 'L')


def _save_as_numpy(path, tensor):
    dirname = os.path.dirname(path)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    np.save(path, tensor.cpu().numpy().astype('float32'))


def _load_from_numpy(path, device='cpu'):
    data = np.load(path)
    return torch.from_numpy(data).to(device)


def is_all_none(xs: Iterable):
    return all(x is None for x in xs)


class SymMatrix:
    def __init__(self, data=None, inv=None, kron=None, kfe=None, diag=None, unit=None,
                 kron_A=None, kron_B=None, kron_A_inv=None, kron_B_inv=None,
                 kfe_A=None, kfe_B=None, kfe_scale=None,
                 unit_data=None, unit_inv=None,
                 diag_weight=None, diag_bias=None, diag_weight_inv=None, diag_bias_inv=None):
        self.data: Tensor = data
        self.inv: Tensor = inv
        if not is_all_none([kron_A, kron_B, kron_A_inv, kron_B_inv]):
            self.kron = Kron(kron_A, kron_B, kron_A_inv, kron_B_inv)
        else:
            self.kron: Kron = kron
        if not is_all_none([kfe_A, kfe_B, kfe_scale]):
            self.kfe = KFE(kfe_A, kfe_B, kfe_scale)
        else:
            self.kfe: KFE = kfe
        if not is_all_none([unit_data, unit_inv]):
            self.unit = UnitWise(unit_data, unit_inv)
        else:
            self.unit: UnitWise = unit
        if not is_all_none([diag_weight, diag_bias, diag_weight_inv, diag_bias_inv]):
            self.diag = Diag(diag_weight, diag_bias, diag_weight_inv, diag_bias_inv)
        else:
            self.diag: Diag = diag

    def __repr__(self):
        text = ''
        if self.has_data:
            text += f'data {tuple(self.data.shape)}, '
        if self.has_kron:
            if self.kron.has_A:
                text += f'kron.A {tuple(self.kron.A.shape)}, '
            if self.kron.has_B:
                text += f'kron.B {tuple(self.kron.B.shape)}, '
        if self.has_kfe:
            if self.kfe.has_Ua:
                text += f'kfe.Ua {tuple(self.kfe.Ua.shape)}'
            if self.kfe.has_Ub:
                text += f'kfe.Ub {tuple(self.kfe.Ub.shape)}'
            if self.kfe.has_scale:
                for i in range(len(self.kfe.scale)):
                    text += f'kfe.scale{i} {tuple(self.kfe.scale[i].shape)}'
        if self.has_unit and self.unit.has_data:
            text += f'unit {tuple(self.unit.data.shape)}'
        if self.has_diag:
            if self.diag.has_weight:
                text += f'diag.weight {tuple(self.diag.weight.shape)}, '
            if self.diag.has_bias:
                text += f'diag.bias {tuple(self.diag.bias.shape)}, '
        if len(text) > 0:
            text = text[:-2]
        text = 'SymMatrix(' + text + ')'
        return text

    @property
    def has_data(self):
        return self.data is not None

    @property
    def has_inv(self):
        return self.inv is not None

    @property
    def has_kron(self):
        return self.kron is not None

    @property
    def has_kfe(self):
        return self.kfe is not None

    @property
    def has_diag(self):
        return self.diag is not None

    @property
    def has_unit(self):
        return self.unit is not None

    def __add__(self, other):
        # NOTE: inv will not be preserved
        values = {}
        for attr in ['data', 'kron', 'kfe', 'diag', 'unit']:
            self_value = getattr(self, attr)
            other_value = getattr(other, attr)
            if other_value is not None:
                if self_value is not None:
                    value = self_value + other_value
                else:
                    value = other_value
            else:
                value = self_value
            values[attr] = value

        return SymMatrix(**values)

    def __iadd__(self, other):
        for attr in ['data', 'kron', 'kfe', 'diag', 'unit']:
            self_value = getattr(self, attr)
            other_value = getattr(other, attr)
            if other_value is not None:
                if self_value is not None:
                    iadd(self_value, other_value)
                else:
                    setattr(self, attr, other_value)
        return self

    def mul_(self, value):
        if self.has_data:
            self.data.mul_(value)
        if self.has_kron:
            self.kron.mul_(value)
        if self.has_kfe:
            self.kfe.mul_(value)
        if self.has_diag:
            self.diag.mul_(value)
        if self.has_unit:
            self.unit.mul_(value)
        return self

    def eigenvalues(self):
        if not self.has_data:
            raise ValueError('data do not exist.')
        eig = symeig(self.data)
        return torch.sort(eig, descending=True)[0]

    def top_eigenvalue(self):
        if not self.has_data:
            raise ValueError('data do not exist.')
        eig = symeig(self.data)
        return eig.max().item()

    def trace(self):
        if not self.has_data:
            raise ValueError('data do not exist.')
        return torch.diag(self.data).sum().item()

    def save(self, root, relative_dir):
        relative_paths = {}
        if self.has_data:
            tril = matrix_to_tril(self.data)
            relative_path = os.path.join(relative_dir, 'tril.npy')
            absolute_path = os.path.join(root, relative_path)
            _save_as_numpy(absolute_path, tril)
            relative_paths['tril'] = relative_path
        if self.has_kron:
            rst = self.kron.save(root, relative_dir)
            relative_paths['kron'] = rst
        if self.has_diag:
            rst = self.diag.save(root, relative_dir)
            relative_paths['diag'] = rst
        if self.has_unit:
            rst = self.unit.save(root, relative_dir)
            relative_paths['unit_wise'] = rst

        return relative_paths

    def load(self, path=None, kron_path=None, diag_path=None, unit_path=None, device='cpu'):
        if path:
            tril = _load_from_numpy(path, device)
            self.data = tril_to_matrix(tril)
        if kron_path is not None:
            if not self.has_kron:
                self.kron = Kron(A=None, B=None)
            self.kron.load(
                A_path=kron_path['A_tril'],
                B_path=kron_path['B_tril'],
                device=device
            )
        if diag_path is not None:
            if not self.has_diag:
                self.diag = Diag()
            self.diag.load(
                w_path=diag_path.get('weight', None),
                b_path=diag_path.get('bias', None),
                device=device
            )
        if unit_path is not None:
            if not self.has_unit:
                self.unit = UnitWise()
            self.unit.load(path=unit_path, device=device)

    def to_vector(self):
        vec = []
        if self.has_data:
            vec.append(self.data)
        if self.has_kron:
            vec.extend(self.kron.data)
        if self.has_diag:
            vec.extend(self.diag.data)
        if self.has_unit:
            vec.extend(self.unit.data)

        vec = [v.flatten() for v in vec]
        return vec

    def to_matrices(self, vec, pointer):
        def unflatten(mat, p):
            numel = mat.numel()
            mat.copy_(vec[p:p + numel].view_as(mat))
            p += numel
            return p

        if self.has_data:
            pointer = unflatten(self.data, pointer)
        if self.has_kron:
            pointer = self.kron.to_matrices(unflatten, pointer)
        if self.has_diag:
            pointer = self.diag.to_matrices(unflatten, pointer)
        if self.has_unit:
            pointer = self.unit.to_matrices(unflatten, pointer)

        return pointer

    def update_inv(self, damping=_default_damping, replace=False):
        if self.has_data and not torch.all(self.data == 0):
            damping = psd_damping(self.data, damping)
            self.inv = cholesky_inv(self.data, damping)
            if replace:
                del self.data
                self.data = None
        if self.has_kron:
            self.kron.update_inv(damping, replace=replace)
        if self.has_diag:
            self.diag.update_inv(damping, replace=replace)
        if self.has_unit:
            self.unit.update_inv(damping, replace=replace)

    def mvp(self, vectors: ParamVector = None,
            vec_weight: torch.Tensor = None, vec_bias: torch.Tensor = None,
            use_inv=False, inplace=False):
        mat = self.inv if use_inv else self.data

        # full
        if vectors is not None:
            v = vectors.get_flatten_vector()
            mat_v = torch.mv(mat, v)
            rst = ParamVector(vectors.params(), mat_v)
            if inplace:
                for v1, v2 in zip(vectors.values(), rst.values()):
                    v1.copy_(v2)
            return rst

        # layer-wise
        if vec_weight is None and vec_bias is None:
            raise ValueError('Either vec_weight or vec_bias has to be set.')
        vecs = []
        if vec_weight is not None:
            vecs.append(vec_weight.flatten())
        if vec_bias is not None:
            vecs.append(vec_bias.flatten())
        vec1d = torch.cat(vecs)
        mvp1d = torch.mv(mat, vec1d)
        if vec_weight is not None:
            if vec_bias is not None:
                w_numel = vec_weight.numel()
                mvp_w = mvp1d[:w_numel].view_as(vec_weight)
                mvp_b = mvp1d[w_numel:]
                if inplace:
                    vec_weight.copy_(mvp_w)
                    vec_bias.copy_(mvp_b)
                return mvp_w, mvp_b
            mvp_w = mvp1d.view_as(vec_weight)
            if inplace:
                vec_weight.copy_(mvp_w)
            return [mvp_w]
        else:
            mvp_b = mvp1d.view_as(vec_bias)
            if inplace:
                vec_bias.copy_(mvp_b)
            return [mvp_b]


class Kron:
    def __init__(self, A, B, A_inv=None, B_inv=None):
        self.A = A
        self.B = B
        self.A_inv = A_inv
        self.B_inv = B_inv
        self._A_dim = self._B_dim = None

    def __add__(self, other):
        # NOTE: inv will not be preserved
        if not other.has_data:
            return self
        if other.has_A:
            A = self.A.add(other.A) if self.has_A else other.A
        else:
            A = self.A
        if other.has_B:
            B = self.B.add(other.B) if self.has_B else other.B
        else:
            B = self.B
        return Kron(A, B)

    def __iadd__(self, other):
        if not other.has_data:
            return self
        if other.has_A:
            if self.has_A:
                self.A.add_(other.A)
            else:
                self.A = other.A
        if other.has_B:
            if self.has_B:
                self.B.add_(other.B)
            else:
                self.B = other.B
        return self

    @property
    def data(self):
        return [self.A, self.B]

    @property
    def has_data(self):
        return self.has_A or self.has_B

    @property
    def has_A(self):
        return self.A is not None

    @property
    def has_B(self):
        return self.B is not None

    @property
    def has_inv(self):
        return self.A_inv is not None and self.B_inv is not None

    @property
    def A_dim(self):
        if self._A_dim is None:
            if self.A is not None:
                self._A_dim = self.A.shape[-1]
            elif self.A_inv is not None:
                self._A_dim = self.A_inv.shape[-1]
            else:
                raise ValueError('A nor A_inv has not been set.')
        return self._A_dim

    @property
    def B_dim(self):
        if self._B_dim is None:
            if self.B is not None:
                self._B_dim = self.B.shape[-1]
            elif self.B_inv is not None:
                self._B_dim = self.B_inv.shape[-1]
            else:
                raise ValueError('B nor B_inv has not been set.')
        return self._B_dim

    @property
    def A_is_square(self):
        return self.A.shape[0] == self.A.shape[1]

    @property
    def B_is_square(self):
        return self.B.shape[0] == self.B.shape[1]

    def mul_(self, value):
        if self.has_A:
            self.A.mul_(value)
        if self.has_B:
            self.B.mul_(value)
        return self

    def eigenvalues(self):
        eig_A = symeig(self.A)
        eig_B = symeig(self.B)
        eig = torch.ger(eig_A, eig_B).flatten()
        return torch.sort(eig, descending=True)[0]

    def top_eigenvalue(self):
        eig_A = symeig(self.A)
        eig_B = symeig(self.B)
        return (eig_A.max() * eig_B.max()).item()

    def trace(self):
        trace_A = torch.diag(self.A).sum().item()
        trace_B = torch.diag(self.B).sum().item()
        return trace_A * trace_B

    def save(self, root, relative_dir):
        relative_paths = {}
        for name in ['A', 'B']:
            mat = getattr(self, name, None)
            if mat is None:
                continue
            tril = matrix_to_tril(mat)
            tril_name = f'{name}_tril'
            relative_path = os.path.join(
                relative_dir, 'kron', f'{tril_name}.npy'
            )
            absolute_path = os.path.join(root, relative_path)
            _save_as_numpy(absolute_path, tril)
            relative_paths[tril_name] = relative_path

        return relative_paths

    def load(self, A_path, B_path, device):
        A_tril = _load_from_numpy(A_path, device)
        self.A = tril_to_matrix(A_tril)
        B_tril = _load_from_numpy(B_path, device)
        self.B = tril_to_matrix(B_tril)

    def to_matrices(self, unflatten, pointer):
        pointer = unflatten(self.A, pointer)
        pointer = unflatten(self.B, pointer)
        return pointer

    def update_inv(self, damping=_default_damping, calc_A_inv=True, calc_B_inv=True, eps=1e-7, replace=False):
        if not self.has_data:
            raise ValueError('data do not exist.')
        if damping < -1:
            # consider damping as target maximum condition number
            target_cond = -damping
            damping = self._kron_prod_damping(target_cond)
        damping_A = damping_B = damping
        if self.has_A and self.has_B:
            A_eig_mean = (self.A.trace() if self.A_is_square else torch.sum(self.A ** 2)) / self.A_dim
            B_eig_mean = (self.B.trace() if self.B_is_square else torch.sum(self.B ** 2)) / self.B_dim
            pi = torch.sqrt(A_eig_mean / B_eig_mean)
            if pi != 0 and pi != float('inf'):
                r = damping**0.5
                damping_A = max(r * pi, eps)
                damping_B = max(r / pi, eps)


        if calc_A_inv:
            if not self.has_A:
                raise ValueError('A does not exist.')
            if not torch.all(self.A == 0):
                if self.A_is_square:
                    self.A_inv = cholesky_inv(self.A, damping_A)
                else:
                    self.A_inv = smw_inv(self.A, damping_A)
                if replace:
                    del self.A
                    self.A = None
        if calc_B_inv:
            if not self.has_B:
                raise ValueError('B does not exist.')
            if not torch.all(self.B == 0):
                if self.B_is_square:
                    self.B_inv = cholesky_inv(self.B, damping_B)
                else:
                    self.B_inv = smw_inv(self.B, damping_B)
                if replace:
                    del self.B
                    self.B = None

    def mvp(self, vec_weight, vec_bias=None, use_inv=False, inplace=False):
        mat_A = self.A_inv if use_inv else self.A
        mat_B = self.B_inv if use_inv else self.B
        vec_weight_2d = vec_weight.view(self.B_dim, -1)
        mvp_w = mat_B.mm(vec_weight_2d).mm(mat_A).view_as(vec_weight)
        if inplace:
            vec_weight.copy_(mvp_w)
        if vec_bias is not None:
            mvp_b = mat_B.mv(vec_bias)
            if inplace:
                vec_bias.copy_(mvp_b)
            return mvp_w, mvp_b
        return mvp_w

    def _kron_prod_damping(
        self,
        target_cond: float,
        max_iters=32,
        eigv_tol=1e-3,
        eigvalsh_faster: int = 256,
    ):
        # Eigenvalues of Kronecker product are the products of factors' eigenvalues.
        # Factors A and B are real symmetric, so non-negative real eigenvalues.
        # Find largest eigenvalue as product of largest eigvals of A and B.
        # Could do following:
        # # Eigvalsh are in ascending order, so first is smallest and last largest.
        # eigv_largest = (
        #     torch.linalg.eigvalsh(self.A)[-1]
        #     * torch.linalg.eigvalsh(self.B)[-1]
        # )
        # but torch.linalg.eigvalsh is O(n^3).
        # Lanczos iteration, even naive python implementation, is faster for larger
        # matrices (~ n > 256), although eigvalsh very optizized.

        assert isinstance(self.A, torch.Tensor)
        assert isinstance(self.B, torch.Tensor)

        device = self.A.device
        dtype = self.A.dtype

        def paige80_lanczos(matrix, ritz_every=4):
            # Paige - Accuracy and effectiveness of the Lanczos algorithm for the
            # symmetric eigenproblem
            # subscript i in array[i-1], offset beta to start at 0 as well, rename v->q
            n = matrix.size()[0]
            max_k = min(max_iters, n)
            alpha = torch.empty(size=(max_k,), device=device, dtype=dtype)
            # beta[n-1] will be zero but less ifs this way
            beta = torch.zeros(size=(max_k,), device=device, dtype=dtype)
            # only need two vecs, one always holds a q, the other u->w->q->u...
            temp = torch.empty(size=(2, n), device=device, dtype=dtype)
            q = temp[0]
            uwq = temp[1].zero_()
            # q[0] = torch.randn_like(matrix[0])
            q.normal_()
            # 2.1, 2.2
            # q[0] /= torch.linalg.norm(q[0])
            q.div_(torch.linalg.norm(q))
            # 2.3
            # u = matrix @ q[0]
            uwq.addmv_(matrix, q)
            # change loop start after 2.6 to interrupt if Parlett bound low enough
            # 2.4
            # alpha[0] = q[0].dot(u)
            alpha[0] = torch.dot(q, uwq)
            # 2.5
            # w = u - alpha[0] * q[0]
            # dont need u anymore, so inplace w from u
            uwq.sub_(q, alpha=alpha[0])
            # 2.6
            # beta[0] = torch.linalg.norm(w)
            torch.linalg.vector_norm(uwq, out=beta[0])
            # Krylov subspace only rank 1 but q was random, so almost certain
            # nonzero component of all basisvectors of A. -> A has rank 1
            done = float(beta[0]) < 1e-7
            eigv_largest = None
            for k in range(0, max_k - 1):
                # Invariant: computed T_(k+1) fully, now compute alpha[k+1], beta[k+1]
                done_k = k + 1
                if done_k % ritz_every == 0 or done_k == max_k or done:
                    # TODO: ideally we would have a fast (compiled) bisection algo based
                    # on Sturm Sequence Sign changes (or laguerre?) and the tridiagonal
                    # determinant continuant recursion to find the extremal Ritz values;
                    # and a tridiagonal matrix solver to find the Ritz vector.
                    # T_k should be small, so eigh should be reasonably fast
                    T_k = torch.zeros(size=(done_k, done_k), device=device, dtype=dtype)
                    T_k.diagonal().copy_(alpha[:done_k])
                    T_k.diagonal(offset=-1).copy_(beta[: done_k - 1])
                    T_k.diagonal(offset=1).copy_(beta[: done_k - 1])
                    ritzvals, ritzvecs = torch.linalg.eigh(T_k)
                    # check error bound of Ritz values
                    # see 10.1.4 Golub, van Loan MC 4th or bound 4.2 of "On Estimating
                    # the Largest Eigenvalue With the Lanczos Algorithm", Parlett et al.
                    # Technically, this only guarantees that *a* eigval is close, not
                    # that the smallest or largest is close.
                    ritz_largest = ritzvals[-1]
                    # last COLUMN of ritzvecs is largest eigenvector
                    bound_largest = (beta[done_k - 1] * ritzvecs[-1, -1]).abs()
                    tol_largest = (bound_largest / ritz_largest).abs()
                    eigv_largest = ritz_largest
                    done |= float(tol_largest) <= eigv_tol
                if done:
                    break

                # 2.7
                # q[k+1] = w / beta[k]
                # new q from w, switch variable names now (q is q[k+1] then)
                uwq.div_(beta[k])
                q, uwq = uwq, q
                # 2.8
                # u = matrix @ q[k+1] - beta[k] * q[k]
                uwq.mul_(-beta[k]).addmv_(matrix, q)
                # next Paige's loop iter due to changed loop split
                # 2.4
                # alpha[k+1] = q[k+1].dot(u)
                # torch.dot(q_kp1, u, out=alpha[k+1])
                alpha[k + 1] = torch.dot(q, uwq)
                # 2.5 (this Paige version apparently doesnt need reortho)
                # w = u - alpha[k+1] * q[k+1]
                # dont need u anymore, so inplace w from u
                # uwq.sub_(q, alpha=alpha[k+1])
                uwq.sub_(alpha[k + 1] * q)
                # 2.6
                # beta[k+1] = torch.linalg.norm(w)
                torch.linalg.vector_norm(uwq, out=beta[k + 1])
                if float(beta[k + 1]) < 1e-7:
                    # Krylov subspace only rank k+2 but q was random, so almost certain
                    # nonzero component of all basisvectors of A. -> A has rank k+2
                    # (if k = n - 2 then full rank or max_k - 2 then done anyway)
                    done = True
                # Invariant: computed alpha[k+1], beta[k+1] for T_(k+2) and bound
            return eigv_largest

        # TODO: improve lanczos, below are other implementations of lanczos,
        # TODO: block and single with reorthogonalization
        # TODO: block tends to be slower as few iters needed due to good seperation and
        # TODO: some compiled/jit of python seems more important. Maybe some cuBLAS,
        # TODO: cuSolver, etc. implementation around that needs just a pytorch wrapper?
        # def random_orthogonal_vectors(dim, num, device, dtype):
        #     # elementwise random normal square matrix -> QR -> rescale Q by sign of
        #     # R diagonal -> Q is Haar measure distributed O(N) matrix
        #     # see arxiv.org/abs/math-ph/0609050
        #     # only need first `num` many columns, so reduced QR on tall
        #     a = torch.randn(size=(dim, num), device=device, dtype=dtype)
        #     q, r = torch.linalg.qr(a, mode="reduced")
        #     q = q * torch.diagonal(r, dim1=-2, dim2=-1).sign().unsqueeze(-2)
        #     return q

        # def block_lanczos(matrix, ritz_every=2, block=2):
        #     # TODO inplace math
        #     # naive Lanczos tridiag (Golub, van Loan MC 4th 10.3.6)
        #     # subscript i in array[i]
        #     n = matrix.size()[0]
        #     max_k = min(max_iters // block, n // block)
        #     diag_blocks = torch.empty(
        #         size=(max_k + 1, block, block), device=device, dtype=dtype
        #     )
        #     off_triangular = torch.zeros(
        #         size=(max_k + 1, block, block), device=device, dtype=dtype
        #     )
        #     Q = torch.empty(size=(max_k + 1, n, block), device=device, dtype=dtype)
        #     R = torch.empty(size=(max_k + 1, n, block), device=device, dtype=dtype)
        #     Q[0] = torch.zeros(size=(n, block), device=device, dtype=dtype)
        #     eigv_largest = None
        #     done = False
        #     # special case k=1 out of loop
        #     # X_1, i.e. Q[1], needs to be block many orthogonal vectors
        #     Q[1] = random_orthogonal_vectors(n, block, device, dtype)
        #     MQ_k = matrix @ Q[1]
        #     diag_blocks[1] = Q[1].mT @ MQ_k
        #     R[1] = MQ_k - Q[1] @ diag_blocks[1]
        #     for k in range(2, max_k + 1):
        #         # Invariant: computed up to diag_blocks[k-1], off_triangular[k-2], R[k-1]
        #         done_k = k - 1
        #         if torch.linalg.matrix_rank(R[done_k]) < block:
        #             # Krylov subspace not full but q was random, so almost
        #             # certain nonzero component of all basisvectors of A. -> A singular
        #             done = True
        #         if done_k % ritz_every == 0 or done or done_k == max_k:
        #             # TODO: ideally we would have a fast (compiled) bisection algo based
        #             # on Sturm Sequence Sign changes and the tridiagonal determinant
        #             # continuant recursion to find the extremal Ritz values; and a
        #             # tridiagonal matrix solver to find the Ritz vector.
        #             # T_k should be small, so eigh should be reasonably fast
        #             T_k = torch.block_diag(*diag_blocks[1 : done_k + 1])
        #             below_block_diag = torch.zeros(
        #                 size=(done_k * block, done_k * block),
        #                 device=device,
        #                 dtype=dtype,
        #             )
        #             below_block_diag[block:, :-block] = torch.block_diag(
        #                 *off_triangular[1:done_k]
        #             )
        #             T_k.add_(below_block_diag)
        #             T_k.add_(below_block_diag.mT)
        #             ritzvals, ritzvecs = torch.linalg.eigh(T_k)
        #             # check error bound of Ritz values
        #             # see 10.1.4 Golub, van Loan MC 4th or bound 4.2 of "On Estimating
        #             # the Largest Eigenvalue With the Lanczos Algorithm", Parlett et al.
        #             ritz_largest = ritzvals[-1]
        #             print(float(ritz_largest))
        #             # based on Parlett norm of diff and MC4 10.3.9
        #             bound_largest = torch.linalg.norm(R[done_k] @ ritzvecs[-block:, -1])
        #             tol_largest = bound_largest / ritz_largest
        #             eigv_largest = ritz_largest
        #             done |= tol_largest <= eigv_tol
        #             if done:
        #                 break
        #         # Continue, compute diag_blocks[k], off_triangular[k-1], R[k]
        #         Q[k], off_triangular[k - 1] = torch.linalg.qr(R[k - 1], mode="reduced")
        #         MQ_k = matrix @ Q[k]
        #         diag_blocks[k] = Q[k].mT @ MQ_k
        #         R[k] = (
        #             MQ_k - Q[k] @ diag_blocks[k] - Q[k - 1] @ off_triangular[k - 1].mT
        #         )
        #         # complete reorthogonalization
        #         # TODO: selective
        #         # TODO: MGS (remove orth to first, then dot and remove to second, ...)
        #         # dot[l][m][n] = Q[l][m].dot(R[k][n])
        #         dot = Q[1 : k + 1].mT @ R[k]
        #         # parallel part for R[k][n] is dot[l][m][n] * Q[l][m], n last dim
        #         R[k] = R[k] - (dot.unsqueeze(2) * Q[1 : k + 1].mT.unsqueeze(3)).sum(
        #             dim=(0, 1)
        #         )
        #     return eigv_largest

        # def naive_lanczos(matrix, ritz_every=4):
        #     # TODO inplace and selective reorth
        #     # naive Lanczos tridiag (Golub, van Loan MC 4th 10.1.1)
        #     # subscript i in array[i]
        #     n = matrix.size()[0]
        #     max_k = min(max_iters, n)
        #     alpha = torch.empty(size=(max_k + 1,), device=device, dtype=dtype)
        #     beta = torch.zeros(size=(max_k + 1,), device=device, dtype=dtype)
        #     q = torch.empty(size=(max_k + 1, n), device=device, dtype=dtype)
        #     r = torch.empty(size=(max_k + 1, n), device=device, dtype=dtype)
        #     q[0] = torch.zeros_like(matrix[0])
        #     q[1] = torch.randn_like(matrix[0])
        #     q[1] /= torch.linalg.norm(q[1])
        #     r[0] = q[1]
        #     eigv_largest = eigv_smallest = None
        #     done = False
        #     for k in range(1, max_k + 1):
        #         Mq_k = matrix @ q[k]
        #         alpha[k] = q[k].dot(Mq_k)
        #         if 1 == k:
        #             r[k] = Mq_k - alpha[k] * q[k]
        #         else:
        #             r[k] = Mq_k - alpha[k] * q[k] - beta[k - 1] * q[k - 1]
        #         # complete reorthogonalization
        #         # TODO: selective only
        #         # CGS
        #         r[k] = r[k] - ((q[1 : k + 1] @ r[k, :, None]) * q[1 : k + 1]).sum(dim=0)
        #         # # MGS
        #         # for i in range(1,k+1):
        #         #     r[k] = r[k] - (q[i] @ r[k]) * q[i]
        #         beta[k] = torch.linalg.norm(r[k])
        #         if beta[k] < 1e-7 and k < n:
        #             # Krylov subspace only rank k but q was random, so almost certain
        #             # nonzero component of all basisvectors of A. -> A has rank k < n
        #             done = True
        #         if k < max_k:
        #             q[k + 1] = r[k] / beta[k]
        #         if k % ritz_every == 0 or k == max_k or done:
        #             # TODO: ideally we would have a fast (compiled) bisection algo based
        #             # on Sturm Sequence Sign changes (or laguerre?) and the tridiagonal
        #             # determinant continuant recursion to find the extremal Ritz values;
        #             # and a tridiagonal matrix solver to find the Ritz vector.
        #             # T_k should be small, so eigh should be reasonably fast
        #             T_k = torch.zeros(size=(k, k), device=device, dtype=dtype)
        #             T_k.diagonal().copy_(alpha[1 : k + 1])
        #             T_k.diagonal(offset=-1).copy_(beta[1:k])
        #             T_k.diagonal(offset=1).copy_(beta[1:k])
        #             ritzvals, ritzvecs = torch.linalg.eigh(T_k)
        #             # check error bound of Ritz values
        #             # see 10.1.4 Golub, van Loan MC 4th or bound 4.2 of "On Estimating
        #             # the Largest Eigenvalue With the Lanczos Algorithm", Parlett et al.
        #             # Technically, this only guarantees that *a* eigval is close, not
        #             # that the smallest or largest is close.
        #             ritz_largest = ritzvals[-1]
        #             print(float(ritz_largest))
        #             # k-1 th COLUMN of ritzvecs is largest eigenvector (s_:k of MC4)
        #             bound_largest = (beta[k] * ritzvecs[k - 1, k - 1]).abs()
        #             tol_largest = (bound_largest / ritz_largest).abs()
        #             eigv_largest = ritz_largest
        #             done |= tol_largest <= eigv_tol

        #         if done:
        #             print(f"naive: {k}")
        #             break
        #     return eigv_largest

        def fast_case(matrix):
            if matrix.shape[-1] == 1:
                return matrix[0, 0]
            if matrix.shape[-1] <= eigvalsh_faster:
                # "small" matrix so O(n^3) eigendecomp is faster (optimized, compiled)
                # Eigvalsh in ascending order, so first is smallest and last largest.
                # Near zero eigval might be negative due to numerics, so abs() again but
                # ignore possibly wrong order now.
                try:
                    eigvalsh = torch.linalg.eigvalsh(matrix).abs()
                    if eigvalsh[-1].isfinite():
                        return eigvalsh[-1]
                except torch._C._LinAlgError:
                    # probably singular, fall back to lanczos
                    pass
                return None

        # Largest eigenvalue is well seperated for FIM, so few iterations are needed
        # such that vector implementation ends up faster than block lanczos.
        # Reorthogonalization is also not needed and Paige's version works well enough.

        A_max_eigval = fast_case(self.A)
        if A_max_eigval is None:
            # A_max_eigval = naive_lanczos(self.A)
            # A_max_eigval = block_lanczos(self.A)
            A_max_eigval = paige80_lanczos(self.A)

        B_max_eigval = fast_case(self.B)
        if B_max_eigval is None:
            # B_max_eigval = naive_lanczos(self.B)
            # B_max_eigval = block_lanczos(self.B)
            B_max_eigval = paige80_lanczos(self.B)

        return target_cond_to_damping(
            target=target_cond, max_eigenvalue=A_max_eigval * B_max_eigval
        )


class KFE:
    def __init__(self, Ua: Tensor, Ub: Tensor, scale: Tuple[Tensor]):
        self.Ua: Tensor = Ua
        self.Ub: Tensor = Ub
        self.scale: Tuple[Tensor] = scale

    @property
    def has_Ua(self):
        return self.Ua is not None

    @property
    def has_Ub(self):
        return self.Ub is not None

    @property
    def has_scale(self):
        return self.scale is not None
    
    @property
    def has_inv(self):
        # KFE does not calculate the inverse matrix.
        return False

    def update_inv(self, *args, **kwargs):
        pass

    def __add__(self, other):
        raise NotImplementedError

    def __iadd__(self, other):
        # NOTE: iadd only scale
        if not other.has_scale:
            return self
        if self.has_scale:
            for i in range(len(self.scale)):
                self.scale[i].add_(other.scale[i])
        else:
            self.scale = other.scale
        return self

    def mul_(self, value):
        if self.has_scale:
            for i in range(len(self.scale)):
                self.scale[i].mul_(value)
        return self

    def mvp(self, vec_weight, vec_bias=None, use_inv=False, inplace=False, eps=1.e-7):
        if use_inv:
            raise ValueError('KFE does not calculate the inverse matrix.')
        Ua, Ub = self.Ua, self.Ub
        vec_weight_2d = vec_weight.view(Ub.shape[0], -1)
        vec_weight_2d_kfe = Ub.mm(vec_weight_2d).mm(Ua)
        vec_weight_2d_kfe /= (self.scale[0] + eps)
        mvp_w = Ub.mm(vec_weight_2d_kfe).mm(Ua).view_as(vec_weight)
        if inplace:
            vec_weight.copy_(mvp_w)
        if vec_bias is not None:
            vec_bias_kfe = Ub.mv(vec_bias)
            vec_bias_kfe /= (self.scale[1] + eps)
            mvp_b = Ub.mv(vec_bias_kfe)
            if inplace:
                vec_bias.copy_(mvp_b)
            return mvp_w, mvp_b
        return mvp_w


class UnitWise:
    def __init__(self, data=None, inv=None):
        if isinstance(data, torch.Tensor):
            self.data = data.contiguous()
        else:
            self.data = data
        self.inv = inv

    def __add__(self, other):
        # NOTE: inv will not be preserved
        if not other.has_data:
            return self
        if self.has_data:
            data = self.data.add(other.data)
        else:
            data = other.data
        return UnitWise(data=data)

    def __iadd__(self, other):
        if not other.has_data:
            return self
        if self.has_data:
            self.data.add_(other.data)
        else:
            self.data = other.data
        return self

    @property
    def has_data(self):
        return self.data is not None

    @property
    def has_inv(self):
        return self.inv is not None

    def mul_(self, value):
        if self.has_data:
            self.data.mul_(value)
        return self

    def eigenvalues(self):
        if not self.has_data:
            raise ValueError('data do not exist.')
        eig = [symeig(block) for block in self.data]
        eig = torch.cat(eig)
        return torch.sort(eig, descending=True)[0]

    def top_eigenvalue(self):
        top = max([symeig(block).max().item() for block in self.data])
        return top

    def trace(self):
        trace = sum([torch.trace(block).item() for block in self.data])
        return trace

    def save(self, root, relative_dir):
        relative_path = os.path.join(relative_dir, 'unit_wise.npy')
        absolute_path = os.path.join(root, relative_path)
        _save_as_numpy(absolute_path, self.data)
        return relative_path

    def load(self, path=None, device='cpu'):
        if path:
            self.data = _load_from_numpy(path, device)

    def to_matrices(self, unflatten, pointer):
        if self.has_data:
            pointer = unflatten(self.data, pointer)
        return pointer

    def update_inv(self, damping=_default_damping, replace=False):
        if not self.has_data:
            raise ValueError('data do not exist.')
        data = self.data
        damping = psd_damping(data, damping)
        if not torch.all(data == 0):
            diag = torch.diagonal(data, dim1=1, dim2=2)
            diag += damping
            self.inv = torch.inverse(data)
            diag -= damping
            if replace:
                del self.data
                self.data = None

    def mvp(self, vec_weight, vec_bias, use_inv=False, inplace=False):
        mat = self.inv if use_inv else self.data
        # BatchNormNd and LayerNorm: vec_weight (f,), vec_bias (f,) or None
        # mat: (f, 2, 2) or (f, 1, 1) for bias=False
        # Linear/Conv2d: vec_weight (f_out, f_in), vec_bias (f_out,) or None
        # mat: (f_out, f_in+1, f_in+1) or (f_out, f_in, f_in) for bias=False
        assert mat.ndim == 3 and mat.shape[1] == mat.shape[2]
        assert vec_weight.shape[0] == mat.shape[0]
        if vec_bias is None:
            if vec_weight.ndim == 1 and mat.shape[1] == 1:
                # for BatchNormNd and LayerNorm
                # mat (f, 1, 1), vec_weight (f,)
                mvp_w = mat.squeeze(dim=(1, 2)) * vec_weight  # (f,)
            elif vec_weight.ndim == 2 and mat.shape[1] == vec_weight.shape[1]:
                # mat (f_out, f_in, f_in), vec_weight (f_out, f_in)
                v = vec_weight.unsqueeze(2)  # (f_out, f_in, 1)
                mvp_w = torch.matmul(mat, v).squeeze(2)  # (f, f_in)
            else:
                raise ValueError(
                    f"Unimplemented shapes {vec_weight.shape=}, {mat.shape=} for "
                    "unit-wise."
                )
            if inplace:
                vec_weight.copy_(mvp_w)
            return mvp_w
        else:
            assert vec_weight.shape[0] == vec_bias.shape[0]
            assert vec_bias.ndim == 1
            if vec_weight.ndim == 1 and mat.shape[1] == 2:
                # mat (f, 2, 2), vec_weight (f,), vec_bias (f,)
                v = torch.stack([vec_weight, vec_bias], dim=1)  # (f, 2)
                v = v.unsqueeze(2)  # (f, 2, 1)
                mvp_wb = torch.matmul(mat, v).squeeze(2)  # (f, 2)
                mvp_w = mvp_wb[:, 0]  # (f,)
                mvp_b = mvp_wb[:, 1]  # (f,)
            elif vec_weight.ndim == 2 and mat.shape[1] == vec_weight.shape[1] + 1:
                # mat (f_out, f_in+1, f_in+1), vec_weight (f_out, f_in), vec_bias (f_out,)
                v = torch.hstack(
                    [vec_weight, vec_bias.unsqueeze(dim=1)]
                )  # (f_out, f_in+1)
                v = v.unsqueeze(2)  # (f_out, f_in+1, 1)
                mvp_wb = torch.matmul(mat, v).squeeze(2)  # (f_out, f_in+1)
                mvp_w = mvp_wb[:, :-1]  # (f_out, f_in)
                mvp_b = mvp_wb[:, -1]  # (f_out, )
            else:
                raise ValueError(
                    f"Unimplemented shapes {vec_weight.shape=}, {vec_bias.shape=}, "
                    f"{mat.shape=} for unit-wise."
                )
            if inplace:
                vec_weight.copy_(mvp_w)
                vec_bias.copy_(mvp_b)
            return mvp_w, mvp_b


class Diag:
    def __init__(self, weight=None, bias=None, weight_inv=None, bias_inv=None):
        self.weight = weight
        self.bias = bias
        self.weight_inv = weight_inv
        self.bias_inv = bias_inv

    def __add__(self, other):
        # NOTE: inv will not be preserved
        if other.has_weight:
            if self.has_weight:
                weight = self.weight.add(other.weight)
            else:
                weight = other.weight
        else:
            weight = self.weight
        if other.has_bias:
            if self.has_bias:
                bias = self.bias.add(other.bias)
            else:
                bias = other.bias
        else:
            bias = self.bias
        return Diag(weight=weight, bias=bias)

    def __iadd__(self, other):
        if other.has_weight:
            if self.has_weight:
                self.weight.add_(other.weight)
            else:
                self.weight = other.weight
        if other.has_bias:
            if self.has_bias:
                self.bias.add_(other.bias)
            else:
                self.bias = other.bias
        return self

    @property
    def data(self):
        return [d for d in [self.weight, self.bias] if d is not None]

    @property
    def has_weight(self):
        return self.weight is not None

    @property
    def has_bias(self):
        return self.bias is not None

    @property
    def has_inv(self):
        has_inv = self.weight_inv is not None
        if self.has_bias:
            has_inv = has_inv and self.bias_inv is not None
        return has_inv

    def mul_(self, value):
        if self.has_weight:
            self.weight.mul_(value)
        if self.has_bias:
            self.bias.mul_(value)
        return self

    def eigenvalues(self):
        eig = []
        if self.has_weight:
            eig.append(self.weight.flatten())
        if self.has_bias:
            eig.append(self.bias.flatten())
        eig = torch.cat(eig)
        return torch.sort(eig, descending=True)[0]

    def top_eigenvalue(self):
        top = -1
        if self.has_weight:
            top = max(top, self.weight.max().item())
        if self.has_bias:
            top = max(top, self.bias.max().item())
        return top

    def trace(self):
        trace = 0
        if self.has_weight:
            trace += self.weight.sum().item()
        if self.has_bias:
            trace += self.bias.sum().item()
        return trace

    def save(self, root, relative_dir):
        relative_paths = {}
        for name in ['weight', 'bias']:
            mat = getattr(self, name, None)
            if mat is None:
                continue
            relative_path = os.path.join(relative_dir, 'diag', f'{name}.npy')
            absolute_path = os.path.join(root, relative_path)
            _save_as_numpy(absolute_path, mat)
            relative_paths[name] = relative_path

        return relative_paths

    def load(self, w_path=None, b_path=None, device='cpu'):
        if w_path:
            self.weight = _load_from_numpy(w_path, device)
        if b_path:
            self.bias = _load_from_numpy(b_path, device)

    def to_matrices(self, unflatten, pointer):
        if self.has_weight:
            pointer = unflatten(self.weight, pointer)
        if self.has_bias:
            pointer = unflatten(self.bias, pointer)
        return pointer

    def update_inv(self, damping=_default_damping, replace=False):
        if damping < 0:
            # diagonal treats every parameter independently, so every system has
            # condition number of 1 by definition. Does not make sense to dampen.
            # Additionally, smallest diagonal value overestimates smallest eigenvalue,
            # diag_i = e_i^T A e_i >= min_v v^T A v = lambda_N, so diag approximation
            # of this layer is effectively damped already in some way.
            # Only need to look out for division by zero here (just like Adam does).
            damping = 1e-7
        if self.has_weight:
            if not torch.all(self.weight == 0):
                self.weight_inv = 1 / (self.weight + damping)
                if replace:
                    del self.weight
                    self.weight = None
        if self.has_bias:
            if not torch.all(self.bias == 0):
                self.bias_inv = 1 / (self.bias + damping)
                if replace:
                    del self.bias
                    self.bias = None

    def mvp(self, vec_weight=None, vec_bias=None, use_inv=False, inplace=False):
        if vec_weight is None and vec_bias is None:
            raise ValueError('Either vec_weight or vec_bias has to be set.')
        rst = []
        if vec_weight is not None:
            mat_w = self.weight_inv if use_inv else self.weight
            if inplace:
                mvp_w = vec_weight.mul_(mat_w)
            else:
                mvp_w = vec_weight.mul(mat_w)
            rst.append(mvp_w)
        if vec_bias is not None:
            mat_b = self.bias_inv if use_inv else self.bias
            if inplace:
                mvp_b = vec_bias.mul_(mat_b)
            else:
                mvp_b = vec_bias.mul(mat_b)
            rst.append(mvp_b)
        return rst
