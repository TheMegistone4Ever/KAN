from math import sqrt

from torch import (arange, Tensor, no_grad, rand, linalg, bmm, sort, linspace, concatenate, float32, int64,
                   sum as torch_sum)
from torch.nn import SiLU, Parameter, init, functional, ModuleList

from nns.inn import INN


class KANLinear(INN):
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0,
                 scale_spline=1.0, enable_standalone_scale_spline=True, base_activation=SiLU, grid_eps=0.02,
                 grid_range=(-1, 1), ):
        super(KANLinear, self).__init__(in_features)
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = ((arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]).expand(in_features,
                                                                                                 -1).contiguous())
        self.register_buffer("grid", grid)

        self.base_weight = Parameter(Tensor(out_features, in_features))
        self.spline_weight = Parameter(Tensor(out_features, in_features, grid_size + spline_order))
        if enable_standalone_scale_spline:
            self.spline_scaler = Parameter(Tensor(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps

        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.base_weight, a=sqrt(5) * self.scale_base)
        with no_grad():
            noise = ((rand(self.grid_size + 1, self.in_features,
                           self.out_features) - 1 / 2) * self.scale_noise / self.grid_size)
            self.spline_weight.data.copy_(
                (self.scale_spline if not self.enable_standalone_scale_spline else 1.0) * self.curve2coeff(
                    self.grid.T[self.spline_order: -self.spline_order], noise, ))
            if self.enable_standalone_scale_spline:
                # init.constant_(self.spline_scaler, self.scale_spline)
                init.kaiming_uniform_(self.spline_scaler, a=sqrt(5) * self.scale_spline)

    def b_splines(self, x: Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: Tensor = self.grid  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = ((x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]) + (
                    (grid[:, k + 1:] - x) / (grid[:, k + 1:] - grid[:, 1:(-k)]) * bases[:, :, 1:])

        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order,)
        return bases.contiguous()

    def curve2coeff(self, X: Tensor, Y: Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            X (Tensor): Input tensor of shape (batch_size, in_features).
            Y (Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert X.dim() == 2 and X.size(1) == self.in_features
        assert Y.size() == (X.size(0), self.in_features, self.out_features)

        A = self.b_splines(X).transpose(0, 1)  # (in_features, batch_size, grid_size + spline_order)
        B = Y.transpose(0, 1)  # (in_features, batch_size, out_features)
        solution = linalg.lstsq(A, B).solution  # (in_features, grid_size + spline_order, out_features)
        result = solution.permute(2, 0, 1)  # (out_features, in_features, grid_size + spline_order)

        assert result.size() == (self.out_features, self.in_features, self.grid_size + self.spline_order,)
        return result.contiguous()

    @property
    def scaled_spline_weight(self):
        return self.spline_weight * (self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0)

    def forward(self, x: Tensor, update_grid=False):
        assert x.dim() == 2 and x.size(1) == self.in_features

        base_output = functional.linear(self.base_activation(x), self.base_weight)
        spline_output = functional.linear(self.b_splines(x).view(x.size(0), -1),
                                          self.scaled_spline_weight.view(self.out_features, -1), )
        return base_output + spline_output

    @no_grad()
    def update_grid(self, x: Tensor, margin=0.01):
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x)  # (batch, in, coeff)
        splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        unreduced_spline_output = bmm(splines, orig_coeff)  # (in, batch, out)
        unreduced_spline_output = unreduced_spline_output.permute(1, 0, 2)  # (batch, in, out)

        # sort each channel individually to collect data distribution
        x_sorted = sort(x, dim=0)[0]
        grid_adaptive = x_sorted[linspace(0, batch - 1, self.grid_size + 1, dtype=int64, device=x.device)]

        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
                arange(self.grid_size + 1, dtype=float32, device=x.device).unsqueeze(1) * uniform_step +
                x_sorted[0] - margin)

        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = concatenate(
            [grid[:1] - uniform_step * arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1), grid,
             grid[-1:] + uniform_step * arange(1, self.spline_order + 1, device=x.device).unsqueeze(1), ],
            dim=0, )

        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a foolish simulation of the original L1 regularization as stated in the
        paper. Since the original one requires computing absolute and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the linear function if we want a memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors' implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch_sum(p * p.log())
        return (regularize_activation * regularization_loss_activation +
                regularize_entropy * regularization_loss_entropy)


class KAN(INN):
    def __init__(self, layers_hidden, grid_size=5, spline_order=3, scale_noise=0.1, scale_base=1.0, scale_spline=1.0,
                 base_activation=SiLU, grid_eps=0.02, grid_range=(-1, 1), ):
        super(KAN, self).__init__(layers_hidden[0])
        self.grid_size = grid_size
        self.spline_order = spline_order

        self.layers = ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(KANLinear(in_features, out_features, grid_size=grid_size, spline_order=spline_order,
                                         scale_noise=scale_noise, scale_base=scale_base, scale_spline=scale_spline,
                                         base_activation=base_activation, grid_eps=grid_eps, grid_range=grid_range, ))

    def forward(self, x: Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(layer.regularization_loss(regularize_activation, regularize_entropy) for layer in self.layers)
