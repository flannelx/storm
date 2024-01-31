#![allow(unused)]

use std::collections::HashMap;

use regex::bytes::Regex;
use serde_json::Value;
use storm::nn::*;
use storm::prelude::*;

pub struct AttnBlock {
    pub norm: GroupNorm,
    pub q: Conv2d,
    pub k: Conv2d,
    pub v: Conv2d,
    pub proj_out: Conv2d,
}

impl AttnBlock {
    fn new(in_channel: usize) -> Self {
        Self {
            norm: GroupNorm::new(32, in_channel, None, None),
            q: Conv2d::default(in_channel, in_channel, 1),
            k: Conv2d::default(in_channel, in_channel, 1),
            v: Conv2d::default(in_channel, in_channel, 1),
            proj_out: Conv2d::default(in_channel, in_channel, 1),
        }
    }

    fn call(&self, x: &Tensor) -> Tensor {
        let h_ = self.norm.call(x);
        let q = self.q.call(&h_);
        let k = self.k.call(&h_);
        let v = self.v.call(&h_);
        let [b, c, h, w] = q.shape().dims[..] else {
            panic!()
        };
        let q = q.reshape([b, c, h * w]).transpose(1, 2);
        let k = k.reshape([b, c, h * w]).transpose(1, 2);
        let v = v.reshape([b, c, h * w]).transpose(1, 2);
        let h_ = Tensor::scaled_dot_product_attention(&q, &k, &v, None, None, None)
            .transpose(1, 2)
            .reshape([b, c, h, w]);
        x + &self.proj_out.call(&h_)
    }
}

pub struct ResnetBlock {
    pub norm1: GroupNorm,
    pub conv1: Conv2d,
    pub norm2: GroupNorm,
    pub conv2: Conv2d,
    pub nin_shortcut: Option<Conv2d>,
}

impl ResnetBlock {
    fn new(in_channels: usize, out_channels: usize) -> Self {
        Self {
            norm1: GroupNorm::new(32, in_channels, None, None),
            conv1: Conv2d::new(in_channels, out_channels, 3, None, [1], None, None, None),
            norm2: GroupNorm::new(32, in_channels, None, None),
            conv2: Conv2d::new(in_channels, out_channels, 3, None, [1], None, None, None),
            nin_shortcut: if in_channels != out_channels {
                Some(Conv2d::default(in_channels, out_channels, 1))
            } else {
                None
            },
        }
    }

    fn call(&self, x: &Tensor) -> Tensor {
        let mut h = self.conv1.call(&self.norm1.call(x).swish());
        h = self.conv2.call(&self.norm2.call(x).swish());
        if let Some(nin) = &self.nin_shortcut {
            nin.call(x) + h
        } else {
            h
        }
    }
}

pub struct Mid {
    pub block_1: ResnetBlock,
    pub attn_1: AttnBlock,
    pub block_2: ResnetBlock,
}

impl Mid {
    pub fn new(block_in: usize) -> Self {
        Self {
            block_1: ResnetBlock::new(block_in, block_in),
            attn_1: AttnBlock::new(block_in),
            block_2: ResnetBlock::new(block_in, block_in),
        }
    }

    pub fn call(&self, x: &Tensor) -> Tensor {
        let mut ret = self.block_1.call(x);
        ret = self.attn_1.call(&ret);
        ret = self.block_2.call(&ret);
        ret
    }
}

pub struct Decoder {
    pub conv_in: Conv2d,
    pub mid: Mid,
    pub up: Vec<(Vec<ResnetBlock>, Vec<Conv2d>)>,
    pub norm_out: GroupNorm,
    pub conv_out: Conv2d,
}

impl Decoder {
    pub fn new() -> Self {
        let sz = [(128, 256), (256, 512), (512, 512), (512, 512)];
        let mut arr = vec![];
        for (i, s) in sz.iter().enumerate() {
            let block = vec![
                ResnetBlock::new(s.1, s.0),
                ResnetBlock::new(s.0, s.0),
                ResnetBlock::new(s.0, s.0),
            ];
            let mut upsample = vec![];
            if i != 0 {
                upsample = vec![Conv2d::new(s.0, s.0, 3, None, [1], None, None, None)];
            }
            arr.push((block, upsample))
        }
        Self {
            conv_in: Conv2d::new(4, 512, 3, None, [1], None, None, None),
            mid: Mid::new(512),
            up: arr,
            norm_out: GroupNorm::new(32, 128, None, None),
            conv_out: Conv2d::new(128, 3, 3, None, [1], None, None, None),
        }
    }

    pub fn call(&self, x: &Tensor) -> Tensor {
        let mut x = self.conv_in.call(x);
        x = self.mid.call(&x);
        for (block, upsample) in self.up.iter().rev() {
            for b in block {
                x = b.call(&x);
            }
            if upsample.len() > 0 {
                let [bs, c, py, px] = x.shape().dims[..] else {
                    panic!()
                };
                x = x
                    .reshape([bs, c, py, 1, px, 1])
                    .expand([bs, c, py, 2, px, 2])
                    .reshape([bs, c, py * 2, px * 2]);
                x = upsample[0].call(&x);
            }
            x.realize();
        }
        self.conv_out.call(&self.norm_out.call(&x).swish())
    }
}

pub struct Encoder {
    pub conv_in: Conv2d,
    pub mid: Mid,
    pub down: Vec<(Vec<ResnetBlock>, Vec<Conv2d>)>,
    pub norm_out: GroupNorm,
    pub conv_out: Conv2d,
}

impl Encoder {
    pub fn new() -> Self {
        let sz = [(128, 128), (128, 256), (256, 512), (512, 512)];
        let conv_in = Conv2d::new(3, 128, 3, None, [1], None, None, None);
        let mut arr = vec![];
        for (i, s) in sz.iter().enumerate() {
            let block = vec![ResnetBlock::new(s.0, s.1), ResnetBlock::new(s.1, s.1)];
            let mut downsample = vec![];
            if i != 3 {
                downsample = vec![Conv2d::new(
                    s.1,
                    s.1,
                    3,
                    Some(2),
                    [0, 1, 0, 1],
                    None,
                    None,
                    None,
                )];
            }
            arr.push((block, downsample))
        }
        Self {
            conv_in,
            mid: Mid::new(512),
            down: arr,
            norm_out: GroupNorm::new(32, 512, None, None),
            conv_out: Conv2d::new(512, 8, 3, None, [1], None, None, None),
        }
    }

    pub fn call(&self, x: &Tensor) -> Tensor {
        let mut x = self.conv_in.call(x);
        for (block, downsample) in self.down.iter().rev() {
            for b in block {
                x = b.call(&x);
            }
            if downsample.len() > 0 {
                x = downsample[0].call(&x);
            }
            x.realize();
        }
        x = self.mid.call(&x);
        self.conv_out.call(&self.norm_out.call(&x).swish())
    }
}

pub struct AutoencoderKL {
    encoder: Encoder,
    decoder: Decoder,
    quant_conv: Conv2d,
    post_quant_conv: Conv2d,
}

impl AutoencoderKL {
    fn new() -> Self {
        Self {
            encoder: Encoder::new(),
            decoder: Decoder::new(),
            quant_conv: Conv2d::default(8, 8, 1),
            post_quant_conv: Conv2d::default(4, 4, 1),
        }
    }

    fn call(&self, x: &Tensor) -> Tensor {
        let mut latent = self.encoder.call(x);
        latent = self.quant_conv.call(&latent);
        latent = latent.shrink([(0, latent.shape()[0] as usize), (0, 4)]);
        latent = self.post_quant_conv.call(&latent);
        self.decoder.call(&latent)
    }
}

pub struct ResBlock {
    in_gn: GroupNorm,
    in_conv: Conv2d,
    emb_lin: Linear,
    out_gn: GroupNorm,
    out_conv2d: Conv2d,
    skip_con: Option<Conv2d>,
}

impl ResBlock {
    fn new(channels: usize, emb_channels: usize, out_channels: usize) -> Self {
        Self {
            in_gn: GroupNorm::new(32, channels, None, None),
            in_conv: Conv2d::new(channels, out_channels, 3, None, [1], None, None, None),
            emb_lin: Linear::new(emb_channels, out_channels, None),
            out_gn: GroupNorm::new(32, out_channels, None, None),
            out_conv2d: Conv2d::new(out_channels, out_channels, 3, None, [1], None, None, None),
            skip_con: if channels != out_channels {
                Some(Conv2d::default(channels, out_channels, 1))
            } else {
                None
            },
        }
    }

    fn call(&self, x: &Tensor, em: &Tensor) -> Tensor {
        let mut h = self.in_gn.call(x).silu();
        h = self.in_conv.call(&h);

        let emb_out = self.emb_lin.call(&em.silu());
        h = h + emb_out.reshape(vec![emb_out.shape().dims, vec![1, 1]].concat());
        h = self.out_gn.call(&h).silu();
        h = self.out_conv2d.call(&h);
        if let Some(sk) = &self.skip_con {
            sk.call(x) + h
        } else {
            h
        }
    }
}

pub struct CrossAttention {
    to_q: Linear,
    to_k: Linear,
    to_v: Linear,
    num_heads: usize,
    head_size: usize,
    to_out: Vec<Linear>,
}

impl CrossAttention {
    fn new(query_dim: usize, context_dim: usize, n_heads: usize, d_head: usize) -> Self {
        Self {
            to_q: Linear::new(query_dim, n_heads * d_head, None),
            to_k: Linear::new(context_dim, n_heads * d_head, None),
            to_v: Linear::new(context_dim, n_heads * d_head, None),
            num_heads: n_heads,
            head_size: d_head,
            to_out: vec![Linear::new(n_heads * d_head, query_dim, None)],
        }
    }

    fn call(&self, x: &Tensor, context: Option<&Tensor>) -> Tensor {
        let context = context.unwrap_or(x);
        let (mut q, mut k, mut v) = (
            self.to_q.call(x),
            self.to_k.call(context),
            self.to_v.call(context),
        );
        (q, k, v) = (
            q.reshape([
                x.shape()[0],
                -1,
                self.num_heads as isize,
                self.head_size as isize,
            ])
            .transpose(1, 2),
            k.reshape([
                x.shape()[0],
                -1,
                self.num_heads as isize,
                self.head_size as isize,
            ])
            .transpose(1, 2),
            v.reshape([
                x.shape()[0],
                -1,
                self.num_heads as isize,
                self.head_size as isize,
            ])
            .transpose(1, 2),
        );
        let attention =
            Tensor::scaled_dot_product_attention(&q, &k, &v, None, None, None).transpose(1, 2);
        let mut h_ =
            attention.reshape([x.shape()[0], -1, (self.num_heads * self.head_size) as isize]);
        for l in self.to_out.iter() {
            h_ = l.call(&h_);
        }
        h_
    }
}

pub struct GEGLU {
    proj: Linear,
    dim_out: usize,
}

impl GEGLU {
    fn new(dim_in: usize, dim_out: usize) -> Self {
        Self {
            proj: Linear::new(dim_in, dim_out * 2, None),
            dim_out,
        }
    }

    fn call(&self, x: &Tensor) -> Tensor {
        let [ref x, ref gate] = self.proj.call(x).chunk(2, Some(-1))[..] else {
            panic!("{:?}", self.proj.call(x).chunk(2, Some(-1)).len())
        };
        x * &gate.gelu()
    }
}

pub struct FeedForward {
    geglu: GEGLU,
    lin: Linear,
}

impl FeedForward {
    fn new(dim: usize, mult: Option<usize>) -> Self {
        let mult = mult.unwrap_or(4);
        Self {
            geglu: GEGLU::new(dim, dim * mult),
            lin: Linear::new(dim * mult, dim, None),
        }
    }

    fn call(&self, x: &Tensor) -> Tensor {
        self.lin.call(&self.geglu.call(x))
    }
}

pub struct BasicTransformerBlock {
    attn1: CrossAttention,
    ff: FeedForward,
    attn2: CrossAttention,
    norm1: LayerNorm,
    norm2: LayerNorm,
    norm3: LayerNorm,
}

impl BasicTransformerBlock {
    fn new(dim: usize, context_dim: usize, n_heads: usize, d_head: usize) -> Self {
        Self {
            attn1: CrossAttention::new(dim, dim, n_heads, d_head),
            ff: FeedForward::new(dim, None),
            attn2: CrossAttention::new(dim, context_dim, n_heads, d_head),
            norm1: LayerNorm::new([dim], None, None),
            norm2: LayerNorm::new([dim], None, None),
            norm3: LayerNorm::new([dim], None, None),
        }
    }

    fn call(&self, x: &Tensor, context: Option<&Tensor>) -> Tensor {
        let mut x = &self.attn1.call(&self.norm1.call(x), None) + x;
        x = self.attn2.call(&self.norm2.call(&x), context) + x;
        x = self.ff.call(&self.norm3.call(&x)) + x;
        x
    }
}

pub struct SpatialTransformer {
    norm: GroupNorm,
    proj_in: Conv2d,
    transformer_block: Vec<BasicTransformerBlock>,
    proj_out: Conv2d,
}

impl SpatialTransformer {
    fn new(channels: usize, context_dim: usize, n_heads: usize, d_head: usize) -> Self {
        Self {
            norm: GroupNorm::new(32, channels, None, None),
            proj_in: Conv2d::default(channels, n_heads * d_head, 1),
            transformer_block: vec![BasicTransformerBlock::new(
                channels,
                context_dim,
                n_heads,
                d_head,
            )],
            proj_out: Conv2d::default(n_heads * d_head, channels, 1),
        }
    }

    fn call(&self, x: &Tensor, context: Option<&Tensor>) -> Tensor {
        let [b, c, h, w] = x.shape().dims[..] else {
            panic!()
        };
        let x_in = x;
        let mut x = self.norm.call(&x);
        x = self.proj_in.call(&x);
        x = x.reshape([b, c, h * w]).permute([0, 2, 1]);
        for b in self.transformer_block.iter() {
            x = b.call(&x, context);
        }
        x = x.permute([0, 2, 1]).reshape([b, c, h, w]);
        self.proj_out.call(&x) + x_in
    }
}

pub struct Downsample {
    op: Conv2d,
}

impl Downsample {
    fn new(channels: usize) -> Self {
        Self {
            op: Conv2d::new(channels, channels, 3, Some(2), [1], None, None, None),
        }
    }

    fn call(&self, x: &Tensor) -> Tensor {
        self.op.call(x)
    }
}

pub struct Upsample {
    conv: Conv2d,
}

impl Upsample {
    fn new(channels: usize) -> Self {
        Self {
            conv: Conv2d::new(channels, channels, 3, None, [1], None, None, None),
        }
    }

    fn call(&self, x: &Tensor) -> Tensor {
        let [bs, c, py, px] = x.shape().dims[..] else {
            panic!()
        };
        let x = x
            .reshape([bs, c, py, 1, px, 1])
            .expand([bs, c, py, 2, px, 2])
            .reshape([bs, c, py * 2, px * 2])
            .realize();
        self.conv.call(&x)
    }
}

pub fn timestep_embedding(timesteps: &Tensor, dim: usize, max_period: Option<usize>) -> Tensor {
    let half = dim as f32 / 2.;
    let max_period = max_period.unwrap_or(10000);
    let freqs = (-(max_period as f32).ln() * Tensor::arange(half) / half).exp();
    let args = timesteps * &freqs;
    Tensor::cat(&args.cos(), &[args.sin()], None).reshape([1, -1])
}

pub struct UNetModel {
    // Time embedded
    time_emb_lin1: Linear,
    time_emb_lin2: Linear,

    input_blocks: Vec<Vec<UnetComponent>>,

    // middle_block
    mid_res1: ResBlock,
    mid_spat1: SpatialTransformer,
    mid_res2: ResBlock,

    output_blocks: Vec<Vec<UnetComponent>>,

    // Out
    out_gn: GroupNorm,
    out_conv: Conv2d,
}

pub enum UnetComponent {
    Conv2d(Conv2d),
    ResBlock(ResBlock),
    SpatialTransformer(SpatialTransformer),
    GroupNorm(GroupNorm),
    Upsample(Upsample),
    Downsample(Downsample),
}
macro_rules! impl_unet_comp {
    ($t: tt) => {
        impl From<$t> for UnetComponent {
            fn from(value: $t) -> Self {
                UnetComponent::$t(value)
            }
        }
    };
}

impl_unet_comp!(Conv2d);
impl_unet_comp!(ResBlock);
impl_unet_comp!(SpatialTransformer);
impl_unet_comp!(GroupNorm);
impl_unet_comp!(Upsample);
impl_unet_comp!(Downsample);

impl UNetModel {
    #[rustfmt::skip]
    fn new() -> Self {
        Self {
            time_emb_lin1: Linear::new(320, 1280, None),
            time_emb_lin2: Linear::new(1280, 1280, None),

            input_blocks: vec![
                vec![Conv2d::new(4, 320, 3, None, [1], None, None, None).into()],
                vec![ResBlock::new(320, 1280, 320).into(), SpatialTransformer::new(320, 768, 8, 40).into()],
                vec![ResBlock::new(320, 1280, 320).into(), SpatialTransformer::new(320, 768, 8, 40).into()],
                vec![Downsample::new(320).into()],
                vec![ResBlock::new(320, 1280, 640).into(), SpatialTransformer::new(640, 768, 8, 80).into()],
                vec![ResBlock::new(640, 1280, 640).into(), SpatialTransformer::new(640, 768, 8, 80).into()],
                vec![Downsample::new(640).into()],
                vec![ResBlock::new(640, 1280, 1280).into(), SpatialTransformer::new(1280, 768, 8, 160).into()],
                vec![ResBlock::new(1280, 1280, 1280).into(), SpatialTransformer::new(1280, 768, 8, 160).into()],
                vec![Downsample::new(1280).into()],
                vec![ResBlock::new(1280, 1280, 1280).into()],
                vec![ResBlock::new(1280, 1280, 1280).into()]
            ],

            mid_res1: ResBlock::new(1280, 1280, 1280),
            mid_spat1: SpatialTransformer::new(1280, 768, 8, 160),
            mid_res2: ResBlock::new(1280, 1280, 1280),

            output_blocks: vec![
                vec![ResBlock::new(2560, 1280, 1280).into()],
                vec![ResBlock::new(2560, 1280, 1280).into()],
                vec![ResBlock::new(2560, 1280, 1280).into(), Upsample::new(1280).into()],
                vec![ResBlock::new(2560, 1280, 1280).into(), SpatialTransformer::new(1280, 768, 8, 160).into()],
                vec![ResBlock::new(2560, 1280, 1280).into(), SpatialTransformer::new(1280, 768, 8, 160).into()],
                vec![ResBlock::new(1920, 1280, 1280).into(), SpatialTransformer::new(1280, 768, 8, 160).into(), Upsample::new(1280).into()],
                vec![ResBlock::new(1920, 1280, 640).into(), SpatialTransformer::new(640, 768, 8, 80).into()],
                vec![ResBlock::new(1280, 1280, 640).into(), SpatialTransformer::new(640, 768, 8, 80).into()],
                vec![ResBlock::new(960, 1280, 640).into(), SpatialTransformer::new(640, 768, 8, 80).into(), Upsample::new(640).into()],
                vec![ResBlock::new(960, 1280, 320).into(), SpatialTransformer::new(320, 768, 8, 40).into()],
                vec![ResBlock::new(640, 1280, 320).into(), SpatialTransformer::new(320, 768, 8, 40).into()],
                vec![ResBlock::new(640, 1280, 320).into(), SpatialTransformer::new(320, 768, 8, 40).into()],
            ],

            out_gn: GroupNorm::new(32, 320, None, None),
            out_conv: Conv2d::new(320, 4, 3, None, [1], None, None, None),
        }
    }

    fn call(&self, x: &Tensor, timesteps: &Tensor, context: Option<&Tensor>) -> Tensor {
        // time emb
        let mut x = x.clone();
        let t_emb = timestep_embedding(timesteps, 320, None);
        let emb = self
            .time_emb_lin2
            .call(&self.time_emb_lin1.call(&t_emb).silu());
        let mut save_inputs = vec![];

        // input block
        for (i, block) in self.input_blocks.iter().enumerate() {
            println!("input block {i}");
            for b in block.iter() {
                match b {
                    UnetComponent::Conv2d(bb) => {
                        //println!("Conv2d");
                        x = bb.call(&x);
                    }
                    UnetComponent::ResBlock(bb) => {
                        //println!("ResBlock");
                        x = bb.call(&x, &emb);
                    }
                    UnetComponent::SpatialTransformer(bb) => {
                        //println!("SpatialTransformer");
                        x = bb.call(&x, context.clone());
                    }
                    UnetComponent::Downsample(bb) => {
                        //println!("Downsample");
                        x = bb.call(&x);
                    }
                    _ => panic!(),
                }
                //println!("{}", x.nd());
            }
            x.realize();
            save_inputs.push(x.clone());
        }

        // mid
        x = self.mid_res1.call(&x, &emb);
        x = self.mid_spat1.call(&x, context.clone());
        x = self.mid_res2.call(&x, &emb);

        for (i, block) in self.output_blocks.iter().enumerate() {
            println!("output block {i}");
            x = x.cat(&[save_inputs.pop().unwrap()], Some(1)).realize();
            for b in block.iter() {
                match b {
                    UnetComponent::ResBlock(bb) => {
                        //println!("ResBlock");
                        x = bb.call(&x, &emb);
                    }
                    UnetComponent::SpatialTransformer(bb) => {
                        //println!("SpatialTransformer");
                        x = bb.call(&x, context.clone());
                    }
                    UnetComponent::Upsample(bb) => {
                        //println!("Upsample");
                        x = bb.call(&x);
                    }
                    _ => panic!(),
                }
                //println!("{}", x.nd());
            }
            x.realize();
        }

        // out
        x = self.out_gn.call(&x);
        x = x.silu();
        x = self.out_conv.call(&x);

        x
    }
}

pub struct CLIPMLP {
    fc1: Linear,
    fc2: Linear,
}

impl CLIPMLP {
    fn new() -> Self {
        Self {
            fc1: Linear::new(768, 3072, None),
            fc2: Linear::new(3072, 768, None),
        }
    }

    fn call(&self, hidden_states: &Tensor) -> Tensor {
        let mut hidden_states = self.fc1.call(&hidden_states);
        hidden_states = hidden_states.quick_gelu();
        hidden_states = self.fc2.call(&hidden_states);
        hidden_states
    }
}

pub struct CLIPAttention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    k_proj: Linear,
    v_proj: Linear,
    q_proj: Linear,
    out_proj: Linear,
}

impl CLIPAttention {
    fn new() -> Self {
        let embed_dim = 768;
        let num_heads = 12;
        let head_dim = embed_dim / num_heads;

        Self {
            embed_dim,
            num_heads,
            head_dim,
            k_proj: Linear::new(embed_dim, embed_dim, None),
            v_proj: Linear::new(embed_dim, embed_dim, None),
            q_proj: Linear::new(embed_dim, embed_dim, None),
            out_proj: Linear::new(embed_dim, embed_dim, None),
        }
    }

    fn call(&self, hidden_states: &Tensor, causal_attention_mask: Option<Tensor>) -> Tensor {
        let [bsz, tgt_len, embed_dim] = hidden_states.shape().dims[..] else {
            panic!()
        };
        let shape = vec![
            bsz,
            tgt_len,
            self.num_heads as isize,
            self.head_dim as isize,
        ];
        let q = self
            .q_proj
            .call(&hidden_states)
            .reshape(shape.as_ref())
            .transpose(1, 2);
        let k = self
            .k_proj
            .call(&hidden_states)
            .reshape(shape.as_ref())
            .transpose(1, 2);
        let v = self
            .v_proj
            .call(&hidden_states)
            .reshape(shape.as_ref())
            .transpose(1, 2);
        let attn_output =
            Tensor::scaled_dot_product_attention(&q, &k, &v, causal_attention_mask, None, None);
        self.out_proj.call(
            &attn_output
                .transpose(1, 2)
                .reshape([bsz, tgt_len, embed_dim]),
        )
    }
}

pub struct CLIPEncoderLayer {
    self_attn: CLIPAttention,
    layr_norm1: LayerNorm,
    mlp: CLIPMLP,
    layr_norm2: LayerNorm,
}

impl CLIPEncoderLayer {
    fn new() -> Self {
        Self {
            self_attn: CLIPAttention::new(),
            layr_norm1: LayerNorm::new([768], None, None),
            mlp: CLIPMLP::new(),
            layr_norm2: LayerNorm::new([768], None, None),
        }
    }

    fn call(&self, hidden_states: &Tensor, causal_attention_mask: Option<Tensor>) -> Tensor {
        let mut residual = hidden_states.clone();
        let mut hidden_states = self.layr_norm1.call(hidden_states);
        hidden_states = self.self_attn.call(&hidden_states, causal_attention_mask);
        hidden_states = residual + &hidden_states;

        residual = hidden_states.clone();
        hidden_states = self.layr_norm2.call(&hidden_states);
        hidden_states = self.mlp.call(&hidden_states);
        hidden_states = residual + hidden_states;

        hidden_states
    }
}

pub struct CLIPEncoder {
    layers: Vec<CLIPEncoderLayer>,
}

impl CLIPEncoder {
    fn new() -> Self {
        Self {
            layers: v![CLIPEncoderLayer::new(), for _ in 0..12],
        }
    }

    fn call(&self, hidden_states: &Tensor, causal_attention_mask: Option<Tensor>) -> Tensor {
        let mut hidden_states = hidden_states.clone();
        for (i, l) in self.layers.iter().enumerate() {
            hidden_states = l.call(&hidden_states, causal_attention_mask.clone());
        }
        hidden_states
    }
}

pub struct CLIPTextEmbeddings {
    token_embedding: Embedding,
    position_embedding: Embedding,
}

impl CLIPTextEmbeddings {
    fn new() -> Self {
        Self {
            token_embedding: Embedding::new(49408, 768),
            position_embedding: Embedding::new(77, 768),
        }
    }

    fn call(&self, input_ids: &Tensor, position_ids: &Tensor) -> Tensor {
        self.token_embedding.call(input_ids) + self.position_embedding.call(&position_ids)
    }
}

pub struct CLIPTextTransformer {
    embeddings: CLIPTextEmbeddings,
    encoder: CLIPEncoder,
    final_layer_norm: LayerNorm,
}

impl CLIPTextTransformer {
    fn new() -> Self {
        Self {
            embeddings: CLIPTextEmbeddings::new(),
            encoder: CLIPEncoder::new(),
            final_layer_norm: LayerNorm::new([768], None, None),
        }
    }

    fn call(&self, input_ids: &Tensor) -> Tensor {
        let mut x = self.embeddings.call(
            input_ids,
            &Tensor::arange(input_ids.shape()[1] as f32).reshape([1, -1]),
        );
        x = self.encoder.call(
            &x,
            Some(Tensor::full([1, 1, 77, 77], f32::NEG_INFINITY).triu(Some(1))),
        );
        self.final_layer_norm.call(&x)
    }
}

pub struct Tokenizer {
    vocab: Value,
    merge: HashMap<Vec<String>, u64>,
    eos_token: u64,
    bos_token: u64,
    pad_token: u64,
    max_length: u64,
    pat: Regex,
}

fn get_pairs(s: &[String]) -> Vec<Vec<String>> {
    let a = s.iter();
    let mut b = s.iter();
    b.next();
    a.zip(b)
        .into_iter()
        .map(|(s1, s2)| vec![s1.clone(), s2.clone()])
        .collect()
}

impl Tokenizer {
    fn new() -> Self {
        let vocab: Value = serde_json::from_str(include_str!("../tokenizer_vocab.json")).unwrap();
        let merge = include_str!("../tokenizer_merges.txt")
            .split("\n")
            .into_iter()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();
        let merge = HashMap::from_iter(
            merge[1..merge.len() - 1]
                .into_iter()
                .enumerate()
                .map(|(i, s)| {
                    (
                        s.split_whitespace()
                            .into_iter()
                            .map(|s| s.to_string())
                            .collect::<Vec<String>>(),
                        i as u64,
                    )
                })
                .collect::<Vec<(Vec<String>, u64)>>(),
        );
        Self {
            eos_token: vocab["<|endoftext|>"].as_u64().unwrap(),
            bos_token: vocab["<|startoftext|>"].as_u64().unwrap(),
            pad_token: vocab["<|endoftext|>"].as_u64().unwrap(),
            max_length: 77,
            vocab,
            merge,
            pat: Regex::new(r"(?i)<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+").unwrap(),
        }
    }

    fn bpe(&self, texts: Vec<String>) -> Vec<String> {
        let mut words = texts;
        *words.last_mut().unwrap() += "</w>";
        while words.len() > 1 {
            let valid_pairs =
                v![pair, for pair in get_pairs(&words), if self.merge.contains_key(&pair)];
            if valid_pairs.len() == 0 {
                break;
            }

            let bigram = valid_pairs.iter().min_by_key(|&p| self.merge[p]).unwrap();
            let [ref first, ref second] = bigram[..] else {
                panic!()
            };
            let mut new_words = vec![];
            for word in words.iter() {
                if word == second && !new_words.is_empty() && new_words.last().unwrap() == first {
                    *new_words.last_mut().unwrap() = first.clone() + second;
                } else {
                    new_words.push(word.to_string());
                }
            }
            words = new_words;
        }
        words
    }

    fn encode(&self, s: &str) -> Vec<u64> {
        let mut text = String::from_utf8(
            Regex::new(r"\s+")
                .unwrap()
                .replace_all(s.as_bytes(), b" ")
                .to_vec(),
        )
        .unwrap();
        text = text.trim().to_lowercase();
        let mut tokens = vec![self.bos_token];
        for chunk in self.pat.captures_iter(s.as_bytes()) {
            let chunk = chunk
                .iter()
                .map(|s| String::from_utf8(s.unwrap().as_bytes().to_vec()).unwrap())
                .collect();
            tokens.extend(v![self.vocab[word].as_u64().unwrap(),for word in self.bpe(chunk)])
        }
        tokens.push(self.eos_token);
        if tokens.len() <= self.max_length as usize {
            for _ in 0..self.max_length as usize - tokens.len() {
                tokens.push(self.eos_token);
            }
            return tokens;
        }
        tokens = tokens[..self.max_length as usize].to_vec();
        let token_len = tokens.len();
        let pad_len = self.max_length as usize - token_len;
        tokens.extend(vec![self.pad_token; pad_len]);
        tokens
    }
}

pub struct StableDiffusion {
    alphas_comprod: Tensor,
    model: UNetModel,
    first_stage_model: AutoencoderKL,
    cond_stage_model: CLIPTextTransformer,
}

impl StableDiffusion {
    fn new() -> Self {
        Self {
            alphas_comprod: Tensor::empty([1000]),
            model: UNetModel::new(),
            first_stage_model: AutoencoderKL::new(),
            cond_stage_model: CLIPTextTransformer::new(),
        }
    }

    fn get_x_prev_and_pred_x0(
        &self,
        x: &Tensor,
        e_t: &Tensor,
        a_t: &Tensor,
        a_prev: &Tensor,
    ) -> (Tensor, Tensor) {
        let temp = 1.;
        let sigma_t = 0.;
        let sqrt_one_minus_at = (1.0 - a_t).sqrt();

        let pred_x0 = (x - &(sqrt_one_minus_at * e_t)) / a_t.sqrt();
        let dir_xt = (1. - a_prev - sigma_t.powf(2.)).sqrt() * e_t;
        let x_prev = a_prev.sqrt() * &pred_x0 + dir_xt;
        (x_prev, pred_x0)
    }

    fn get_model_output(
        &self,
        unconditional_context: &Tensor,
        context: &Tensor,
        latent: &Tensor,
        timestep: &Tensor,
        unconditional_guidance_scale: &Tensor,
    ) -> Tensor {
        let ctx = unconditional_context.cat(&[context.clone()], Some(0));
        let latent = self.model.call(
            &latent.expand(vec![vec![2], latent.shape().dims[1..].to_vec()].concat()),
            timestep,
            Some(&ctx),
        );

        let uncond_latent = latent.shrink([(0, 1), (0, 2), (0, 2), (0, 2)]);
        let latent = latent.shrink([(1, 2), (0, 2), (0, 2), (0, 2)]);
        let e_t = &uncond_latent + &(unconditional_guidance_scale * &(&latent - &uncond_latent));
        e_t
    }

    fn decode(&self, x: &Tensor) -> Tensor {
        let mut x = self
            .first_stage_model
            .post_quant_conv
            .call(&(1. / 0.18215 * x));
        x = self.first_stage_model.decoder.call(&x);

        x = (x + 1.) / 2.;
        x = x.reshape([3, 512, 512]).permute([1, 2, 0]).clip(0., 1.) * 255.;
        x
    }

    fn call(
        &self,
        uncond_context: &Tensor,
        context: &Tensor,
        latent: &Tensor,
        timestep: &Tensor,
        alphas: &Tensor,
        alphas_prev: &Tensor,
        guidence: &Tensor,
    ) -> Tensor {
        let e_t = self.get_model_output(uncond_context, context, latent, timestep, guidence);
        let (x_prev, _) = self.get_x_prev_and_pred_x0(latent, &e_t, alphas, alphas_prev);
        x_prev.realize()
    }
}

fn main() {
    let model = StableDiffusion::new();
    let tokenizer = Tokenizer::new();
    let tokens = tokenizer.encode("draw a cat");
    let prompt =
        Tensor::from(tokens.iter().map(|&e| e as f32).collect::<Vec<f32>>()).reshape([1, -1]);
    let context = model.cond_stage_model.call(&prompt);
    println!("{}", context.nd());
}
