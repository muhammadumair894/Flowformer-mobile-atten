import torch
import torch.nn as nn
import torch.nn.functional as F

class MobileAttention(nn.Module):
    """
    Mobile Attention with head competing mechanism, adapted to match
    the Flow_Attention API:
      - __init__(dim, num_heads=12)
      - forward(x) where x: (B, N, C)
    """
    def __init__(self, dim, num_heads=12, drop_out=0.05, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.eps = eps

        # Instead of separate query/key/value_projection, 
        # we use a single qkv projection for compatibility with Flow_Attention
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        
        # Output projection, matching Flow_Attention's "proj"
        self.proj = nn.Linear(dim, dim)

        # Optional dropout on the output
        self.dropout = nn.Dropout(drop_out)
        
    def kernel_method(self, x: torch.Tensor) -> torch.Tensor:
        """Non-negative projection via sigmoid."""
        return torch.sigmoid(x)

    def dot_product(self, q, k, v):
        """
        We replicate the "mobile" style dot product from the original code
        but adapted for the dimension usage in this setup.
        """
        # In the original code, it used:
        #   kv = torch.einsum("nhld,nhlm->nhdm", k, v)
        #   qkv = torch.einsum("nhld,nhdm->nhlm", q, kv)
        # We'll keep it if we want a 2-step approach. However, we can also 
        # rely on matrix multiplication. For clarity, let's keep the einsum form.
        kv = torch.einsum("nhld,nhlm->nhdm", k, v)
        qkv = torch.einsum("nhld,nhdm->nhlm", q, kv)
        return qkv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x shape: (B, N, C)
        We must:
          1) Generate q, k, v from x
          2) Perform "mobile attention" logic 
             (sink_incoming, source_outgoing, etc.)
          3) Return the output in shape (B, N, C)
        """
        B, N, C = x.shape
        # 1. QKV
        qkv = self.qkv(x)  # (B, N, 3*C)
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads)
        # permute to separate Q, K, V: shape -> (3, B, num_heads, N, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: (B, num_heads, N, head_dim)

        # 2. Non-negative projection
        q = self.kernel_method(q)
        k = self.kernel_method(k)

        # Let’s define some helper variables for clarity:
        # B_ = B, H_ = self.num_heads, L_ = N, D_ = C // self.num_heads

        # 3. Mobile-Attention logic

        # (a) sink_incoming:
        #    From original code, shaped as (B, n_heads, L)
        #    We do a sum over last dimension. But we must replicate the “(queries + eps)*(keys.sum(dim=2) + eps)”
        #    carefully. We'll sum over the dimension representing N in k (dim=2).
        #    The original used:
        #      sink_incoming = 1.0 / torch.sum((queries + eps) * (keys.sum(dim=1)+eps), dim=-1)
        #    but note that we have shapes (B, H, N, D). We'll adapt carefully.
        #
        #    In "Flow_Attention", for normalizing, it uses the sum across the 'N' dimension. We'll do similarly.
        #    We'll match the approach used in the original Mobile_Attention code with repeats.

        eps = self.eps
        # sum of keys over the "N" dimension => shape (B, H, D)
        k_sum = k.sum(dim=2)  # shape (B, H, D)
        # we need to broadcast for multiplication with q: (B,H,N,D)
        # so let's do something like: (k_sum[:, :, None, :] + eps)
        # to get shape (B,H,1,D)
        
        sink_incoming = 1.0 / torch.sum(
            (q + eps) * (k_sum[:, :, None, :] + eps),
            dim=-1  # summation over D dimension
        )  # shape: (B, H, N)

        # (b) source_outgoing:
        #    sum of queries over the "N" dimension => shape (B, H, D)
        q_sum = q.sum(dim=2)
        source_outgoing = 1.0 / torch.sum(
            (k + eps) * (q_sum[:, :, None, :] + eps),
            dim=-1  # sum over D
        )  # (B, H, N)

        # (c) conserved_sink
        #    from the original:
        #      conserved_sink = torch.sum((queries+eps)*((keys*source_outgoing[:, :, :, None]).sum(dim=1)+eps),dim=-1)
        #    but we have dimension order: (B,H,N,D).
        #    We'll do something similar with a sum over N dimension in k after multiplying by source_outgoing.

        # keys * source_outgoing => shape (B, H, N, D)
        k_so = k * source_outgoing[:, :, :, None]  
        k_so_sum = k_so.sum(dim=2, keepdim=True)  # (B, H, 1, D)
        
        # (queries+eps)*(k_so_sum+eps) -> sum over D => shape (B,H,N)
        conserved_sink = torch.sum((q + eps) * (k_so_sum + eps), dim=-1) + eps

        # (d) conserved_source
        #    similarly, we do queries * sink_incoming => shape (B, H, N, D)
        q_si = q * sink_incoming[:, :, :, None]
        q_si_sum = q_si.sum(dim=2, keepdim=True)  # (B, H, 1, D)
        
        conserved_source = torch.sum((k + eps) * (q_si_sum + eps), dim=-1) + eps
        # clamp for stability:
        conserved_source = torch.clamp(conserved_source, min=-1.0, max=1.0)

        # (e) Competition & Allocation
        #     sink_allocation => shape (B,H,N)
        #        = sigmoid(conserved_sink * (float(q.shape[2])/float(k.shape[2])))
        #        i.e. scale by ratio of N in q vs. N in k. Usually same N though?
        ratio = float(q.shape[2]) / float(k.shape[2])  # typically 1 if same N
        sink_allocation = torch.sigmoid(conserved_sink * ratio)

        # source_competition => shape (B,H,N)
        #   = softmax(conserved_source, dim=-1) * float(k.shape[2])
        #   i.e. multiply by the number of tokens in k
        source_competition = F.softmax(conserved_source, dim=-1) * float(k.shape[2])

        # (f) Dot product
        #   In the original, x = dot_product(queries * sink_incoming, keys, values * source_competition)*sink_allocation
        #   We'll replicate that but with shape (B,H,N,D).
        #   queries * sink_incoming => shape (B,H,N,D)
        q_norm = q * sink_incoming[:, :, :, None]
        v_comp = v * source_competition[:, :, :, None]

        # Perform the "mobile-style" dot product, i.e. qkv = ...
        x_mobil = self.dot_product(q_norm, k, v_comp)
        # shape => (B,H,N,D)

        # multiply by sink_allocation => shape => (B,H,N,D)
        x_mobil = x_mobil * sink_allocation[:, :, :, None]

        # (g) Rearrange back to (B,N,C)
        x_mobil = x_mobil.transpose(1, 2).reshape(B, N, C)

        # (h) Final projection + dropout
        x_mobil = self.proj(x_mobil)  # (B, N, C)
        x_mobil = self.dropout(x_mobil)

        return x_mobil

