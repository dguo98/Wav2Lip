import torch
import torch.nn as nn
import copy
import math
import numpy as np
from IPython import embed
import torch.nn.functional as F


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many 
    other models.
    """
    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        
    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        return self.decode(self.encode(src, src_mask), src_mask,
                            tgt, tgt_mask)
    
    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)
    
    def decode(self, memory, src_mask, tgt, tgt_mask, gen=True):
        out = self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)
        if gen:
            out = self.generator(out)
        return out

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, mask):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))

class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Decoder(nn.Module):
    "Generic N layer decoder with masking."
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)
        
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)
 
    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        # TODO(demi): mask == 0? should it be mask == 1
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        
        # 1) Do all the linear projections in batch from d_model => h x d_k 
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        
        # 2) Apply attention on all the projected vectors in batch. 
        x, self.attn = attention(query, key, value, mask=mask, 
                                 dropout=self.dropout)
        
        # 3) "Concat" using a view and apply a final linear. 
        x = x.transpose(1, 2).contiguous() \
             .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Linear(vocab, d_model)  # NB(demi): continuous -> continuous
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    "Implement the PE function."
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model, requires_grad=False)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        fac = -(math.log(10000.0)*1.0 / d_model)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * fac)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class Generator(nn.Module):
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)
        # TODO(demi): tie weights with target embeddings

    def forward(self, x):
        return self.proj(x)  # continuous values


class Transformer(nn.Module):
    def __init__(self, args, input_dim=512, output_dim=512*18):
        super(Transformer, self).__init__()

        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        c = copy.deepcopy
        attn = MultiHeadedAttention(args.h, args.d_model)
        ff = PositionwiseFeedForward(args.d_model, args.d_ff, args.dropout)
        position = PositionalEncoding(args.d_model, args.dropout)
        
        self.model = EncoderDecoder(
            Encoder(EncoderLayer(args.d_model, c(attn), c(ff), args.dropout), args.N),
            Decoder(DecoderLayer(args.d_model, c(attn), c(attn), c(ff), args.dropout), args.N),
            nn.Sequential(Embeddings(args.d_model, input_dim), c(position)),
            nn.Sequential(Embeddings(args.d_model, output_dim), c(position)),
            Generator(args.d_model, output_dim))
        for p in self.model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform(p)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.model(src, tgt, src_mask, tgt_mask)

    def encode(self, src, src_mask):
        return self.model.encode(src, src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask, gen=True):
        return self.model.decode(memory, src_mask, tgt, tgt_mask, gen=gen)
    
    def generate(self, out):
        return self.model.generator(out)
    
    def parameters(self):
        return self.model.parameters()

class LinearMapper(torch.nn.Module):
    def __init__(self, args, input_dim=512, output_dim=512*18):
        super(LinearMapper, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.model = nn.Linear(input_dim, output_dim)

    def forward(self, src, tgt, src_mask, tgt_mask):
        return self.model(src)

    def encode(self, src, src_mask):
        return self.model(src)

    def decode(self, memory, src_mask, tgt, tgt_mask, gen=True):
        return memory
    
    def generate(self, out):
        return out

class Conv1D(nn.Module):
    def __init__(self, cin, cout, kernel_size, stride, padding, residual=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.conv_block = nn.Sequential(
                            nn.Conv1d(cin, cout, kernel_size, stride, padding),
                            nn.BatchNorm1d(cout)
                            )
        self.act = nn.ReLU()
        self.residual = residual

    def forward(self, x):
        out = self.conv_block(x)
        if self.residual:
            out += x
        return self.act(out)


class AutoRegConvMapper(torch.nn.Module):
    def __init__(self, args, input_dim=512, output_dim=512*18):
        super(AutoRegConvMapper, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.seq_len = args.seq_len
        self.nlayer = args.nlayer
        self.hidden_dim = args.hidden_dim

        self.layers = nn.ModuleList()

        # weight tying, conversion
        self.conversion_weight = nn.Parameter(torch.randn(self.hidden_dim, self.output_dim))
        
        self.layers.append(Conv1D(self.input_dim+self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1))
        
        tmp_len = self.seq_len
        # NB(demi): are channels size too large
        while (tmp_len > 1):
            assert tmp_len % 2 == 0
            self.layers.append(Conv1D(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=2, padding=1))
            for i in range(self.nlayer):
                self.layers.append(Conv1D(self.hidden_dim, self.hidden_dim, kernel_size=3, stride=1, padding=1, residual=True))
            tmp_len = tmp_len // 2

    def forward(self, src, tgt, src_mask, tgt_mask):
        # tgt is prev target
        bsz, seqlen, _ = src.shape
        assert tgt.shape == (bsz, seqlen, self.output_dim)
        
        # convert
        tgt = F.linear(tgt.reshape(-1, self.output_dim), self.conversion_weight).reshape(bsz, seqlen, self.hidden_dim)
        inp = torch.cat([src, tgt], dim=2)
        assert inp.shape == (bsz, seqlen, self.input_dim+self.hidden_dim)
        x = inp.permute((0, 2, 1))  # [bsz, C, L]

        for layer in self.layers:
            x = layer(x)
        assert x.shape == (bsz, self.hidden_dim, 1)
        #print("self.conversion_weight.t() shape=", self.conversion_weight.t().shape)
        x = F.linear(x.reshape(bsz, self.hidden_dim), self.conversion_weight.t())
        return x.reshape(bsz, 1, self.output_dim)

    def encode(self, src, src_mask):
        return src

    def decode(self, memory, src_mask, tgt, tgt_mask, gen=True):
        src = memory
        return self.forward(src, tgt, src_mask, tgt_mask)

    def generate(self, out):
        return out


class AutoRegMLPMapper(torch.nn.Module):
    def __init__(self, args, input_dim=512, output_dim=512*18):
        super(AutoRegMLPMapper, self).__init__()
        self.args = args
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = args.hidden_dim
        self.nlayer = args.nlayer
        self.residual = args.mlp_residual == 1
        
        self.dropout = nn.Dropout(args.mlp_dropout)
        self.layers = nn.ModuleList()
        if self.nlayer == 1:
            self.layers.append(nn.Linear(self.input_dim+self.output_dim, self.output_dim))
        else:
            self.layers.append(nn.Sequential(
                nn.Linear(self.input_dim+self.output_dim, self.hidden_dim),
                nn.ReLU()))
            for i in range(self.nlayer-2):
                self.layers.append(nn.Sequential(
                    nn.Linear(self.hidden_dim, self.hidden_dim),
                    nn.ReLU()))
            self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))


    def forward(self, src, tgt, src_mask, tgt_mask):
        # src: [bsz, 1, input_dim]
        # tgt=prev_tgt: [bsz, 1, output_dim]
        bsz, seq_len = src.shape[:2]
        assert seq_len == 1
        assert torch.sum(1-src_mask).item() == 0 and torch.sum(1-tgt_mask).item() == 0

        inp = torch.cat([src, tgt], dim=2)
        inp = inp.reshape(bsz, self.input_dim+self.output_dim)
        
        x = inp
        for i, layer in enumerate(self.layers):
            if i == 0 or i == len(self.layers)-1 or (not self.residual):
                x = self.dropout(layer(x))
            else:
                x = x + self.dropout(layer(x)) 

        return x.reshape(bsz, 1, self.output_dim)
        

    def encode(self, src, src_mask):
        return src

    def decode(self, memory, src_mask, tgt, tgt_mask, gen=True):
        src = memory
        return self.forward(src, tgt, src_mask, tgt_mask)

    def generate(self, out):
        return out


