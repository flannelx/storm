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
            norm2: GroupNorm::new(32, out_channels, None, None),
            conv2: Conv2d::new(out_channels, out_channels, 3, None, [1], None, None, None),
            nin_shortcut: if in_channels != out_channels {
                Some(Conv2d::default(in_channels, out_channels, 1))
            } else {
                None
            },
        }
    }

    fn call(&self, x: &Tensor) -> Tensor {
        let mut h = self.conv1.call(&self.norm1.call(x).swish());
        h = self.conv2.call(&self.norm2.call(&h).swish());
        if let Some(nin) = &self.nin_shortcut {
            nin.call(x) + h
        } else {
            x + &h
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
            println!("\ndecode {}", x.shape());
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
    out_conv: Conv2d,
    conv_shortcut: Option<Conv2d>,
}

impl ResBlock {
    fn new(channels: usize, emb_channels: usize, out_channels: usize) -> Self {
        Self {
            in_gn: GroupNorm::new(32, channels, None, None),
            in_conv: Conv2d::new(channels, out_channels, 3, None, [1], None, None, None),
            emb_lin: Linear::new(emb_channels, out_channels, None),
            out_gn: GroupNorm::new(32, out_channels, None, None),
            out_conv: Conv2d::new(out_channels, out_channels, 3, None, [1], None, None, None),
            conv_shortcut: if channels != out_channels {
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
        h = self.out_conv.call(&h);
        if let Some(sk) = &self.conv_shortcut {
            sk.call(x) + h
        } else {
            x + &h
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
    conv: Conv2d,
}

impl Downsample {
    fn new(channels: usize) -> Self {
        Self {
            conv: Conv2d::new(channels, channels, 3, Some(2), [1], None, None, None),
        }
    }

    fn call(&self, x: &Tensor) -> Tensor {
        self.conv.call(x)
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
            .reshape([bs, c, py * 2, px * 2]);
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
    time_embedding: Vec<Linear>,
    input_blocks: Vec<Vec<UnetComponent>>,
    mid_blocks: Vec<UnetComponent>,
    output_blocks: Vec<Vec<UnetComponent>>,
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
            time_embedding: vec![
            Linear::new(320, 1280, None),
            Linear::new(1280, 1280, None)],

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

            mid_blocks: vec![
            ResBlock::new(1280, 1280, 1280).into(),
            SpatialTransformer::new(1280, 768, 8, 160).into(),
            ResBlock::new(1280, 1280, 1280).into()
            ],

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
        let emb = self.time_embedding[1].call(&self.time_embedding[0].call(&t_emb).silu());
        let mut save_inputs = vec![];

        // input block
        for (i, block) in self.input_blocks.iter().enumerate() {
            // println!("input block {i}");
            for b in block.iter() {
                match b {
                    UnetComponent::Conv2d(bb) => {
                        // println!("Conv2d");
                        x = bb.call(&x);
                    }
                    UnetComponent::ResBlock(bb) => {
                        // println!("ResBlock");
                        x = bb.call(&x, &emb);
                    }
                    UnetComponent::SpatialTransformer(bb) => {
                        // println!("SpatialTransformer");
                        x = bb.call(&x, context.clone());
                    }
                    UnetComponent::Downsample(bb) => {
                        // println!("Downsample");
                        x = bb.call(&x);
                    }
                    _ => panic!(),
                }
            }
            x.realize();
            // println!("realized");
            save_inputs.push(x.clone());
        }

        for (i, block) in self.mid_blocks.iter().enumerate() {
            match block {
                UnetComponent::ResBlock(bb) => {
                    // println!("ResBlock");
                    x = bb.call(&x, &emb);
                }
                UnetComponent::SpatialTransformer(bb) => {
                    // println!("SpatialTransformer");
                    x = bb.call(&x, context.clone());
                }
                _ => panic!(),
            }
            x.realize();
        }

        for (i, block) in self.output_blocks.iter().enumerate() {
            // println!("output block {i}");
            x = x.cat(&[save_inputs.pop().unwrap()], Some(1)).realize();
            for b in block.iter() {
                match b {
                    UnetComponent::Conv2d(bb) => {
                        // println!("Conv2d");
                        x = bb.call(&x);
                    }
                    UnetComponent::ResBlock(bb) => {
                        // println!("ResBlock");
                        x = bb.call(&x, &emb);
                    }
                    UnetComponent::SpatialTransformer(bb) => {
                        // println!("SpatialTransformer");
                        x = bb.call(&x, context.clone());
                    }
                    UnetComponent::Upsample(bb) => {
                        // println!("Upsample");
                        x = bb.call(&x);
                    }
                    _ => panic!(),
                }
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
    layer_norm1: LayerNorm,
    mlp: CLIPMLP,
    layer_norm2: LayerNorm,
}

impl CLIPEncoderLayer {
    fn new() -> Self {
        Self {
            self_attn: CLIPAttention::new(),
            layer_norm1: LayerNorm::new([768], None, None),
            mlp: CLIPMLP::new(),
            layer_norm2: LayerNorm::new([768], None, None),
        }
    }

    fn call(&self, hidden_states: &Tensor, causal_attention_mask: Option<Tensor>) -> Tensor {
        let mut residual = hidden_states.clone();
        //println!("1 residual {}", residual.nd());
        let mut hidden_states = self.layer_norm1.call(hidden_states);
        //println!("1 hidden_states {}", hidden_states.nd());
        hidden_states = self.self_attn.call(&hidden_states, causal_attention_mask);
        //println!("2 hidden_states {}", hidden_states.nd());
        hidden_states = residual + &hidden_states;
        //println!("3 hidden_states {}", hidden_states.nd());

        residual = hidden_states.clone();
        //println!("2 residual {}", residual.nd());
        hidden_states = self.layer_norm2.call(&hidden_states);
        //println!("1 hidden_states {}", hidden_states.nd());
        hidden_states = self.mlp.call(&hidden_states);
        //println!("2 hidden_states {}", hidden_states.nd());
        hidden_states = residual + hidden_states;
        //println!("3 hidden_states {}", hidden_states.nd());

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
        let token_emb = self.token_embedding.call(input_ids);
        let pos_emb = self.position_embedding.call(&position_ids);
        // println!("token_emb\n{:?}\n{:?}", token_emb.nd(), token_emb.buffer.st);
        // println!("pos_emb\n{:?}\n{:?}", pos_emb.nd(), pos_emb.buffer.st);
        token_emb + pos_emb
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
        let vocab: Value = serde_json::from_str(&std::fs::read_to_string("tokenizer_vocab.json").unwrap()).unwrap();
        let merge = std::fs::read_to_string("tokenizer_merges.txt").unwrap()
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
    alphas_comprod: Vec<f32>,
    model: UNetModel,
    first_stage_model: AutoencoderKL,
    cond_stage_model: Option<CLIPTextTransformer>,
}

impl StableDiffusion {
    fn new() -> Self {
        Self {
            alphas_comprod: vec![],
            model: UNetModel::new(),
            first_stage_model: AutoencoderKL::new(),
            cond_stage_model: Some(CLIPTextTransformer::new()),
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
        let latents = self.model.call(
            &latent.expand(vec![vec![2], latent.shape().dims[1..].to_vec()].concat()),
            timestep,
            Some(&ctx),
        );

        let shape = latents
            .shape()
            .dims
            .into_iter()
            .map(|s| s as usize)
            .collect::<Vec<usize>>();
        let uncond_latent = latents.shrink([(0, 1), (0, shape[1]), (0, shape[2]), (0, shape[3])]);
        let latent = latents.shrink([(1, 2), (0, shape[1]), (0, shape[2]), (0, shape[3])]);
        let e_t = &uncond_latent + &(unconditional_guidance_scale * &(&latent - &uncond_latent));
        e_t
    }

    fn decode(&self, x: &Tensor) -> Tensor {
        let mut x = self
            .first_stage_model
            .post_quant_conv
            .call(&(1. / 0.18215 * x));
        x = self.first_stage_model.decoder.call(&x);

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

use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::fs::File;

#[rustfmt::skip]
fn load_text_model(mut text_model: &mut CLIPTextTransformer) {
    let filename = "model.safetensors";
    let file = File::open(filename).unwrap();
    let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
    let tensors = SafeTensors::deserialize(&buffer).unwrap();
    text_model.embeddings.position_embedding.weight.assign_device_buffer(tt(&format!("text_model.embeddings.position_embedding.weight"), &tensors));
    text_model.embeddings.token_embedding.weight.assign_device_buffer(tt(&format!("text_model.embeddings.token_embedding.weight"), &tensors));
    for (i, l) in text_model.encoder.layers.iter_mut().enumerate() {
        l.self_attn.q_proj.weights.assign_device_buffer(tt(&format!("text_model.encoder.layers.{i}.self_attn.q_proj.weight"), &tensors));
        l.self_attn.q_proj.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("text_model.encoder.layers.{i}.self_attn.q_proj.bias"), &tensors));
        l.self_attn.k_proj.weights.assign_device_buffer(tt(&format!("text_model.encoder.layers.{i}.self_attn.k_proj.weight"), &tensors));
        l.self_attn.k_proj.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("text_model.encoder.layers.{i}.self_attn.k_proj.bias"), &tensors));
        l.self_attn.v_proj.weights.assign_device_buffer(tt(&format!("text_model.encoder.layers.{i}.self_attn.v_proj.weight"), &tensors));
        l.self_attn.v_proj.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("text_model.encoder.layers.{i}.self_attn.v_proj.bias"), &tensors));
        l.self_attn.out_proj.weights.assign_device_buffer(tt(&format!("text_model.encoder.layers.{i}.self_attn.out_proj.weight"), &tensors));
        l.self_attn.out_proj.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("text_model.encoder.layers.{i}.self_attn.out_proj.bias"), &tensors));
        l.layer_norm1.weights.as_mut().unwrap().assign_device_buffer(tt(&format!("text_model.encoder.layers.{i}.layer_norm1.weight"), &tensors));
        l.layer_norm1.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("text_model.encoder.layers.{i}.layer_norm1.bias"), &tensors));
        l.mlp.fc1.weights.assign_device_buffer(tt(&format!("text_model.encoder.layers.{i}.mlp.fc1.weight"), &tensors));
        l.mlp.fc1.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("text_model.encoder.layers.{i}.mlp.fc1.bias"), &tensors));
        l.mlp.fc2.weights.assign_device_buffer(tt(&format!("text_model.encoder.layers.{i}.mlp.fc2.weight"), &tensors));
        l.mlp.fc2.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("text_model.encoder.layers.{i}.mlp.fc2.bias"), &tensors));
        l.layer_norm2.weights.as_mut().unwrap().assign_device_buffer(tt(&format!("text_model.encoder.layers.{i}.layer_norm2.weight"), &tensors));
        l.layer_norm2.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("text_model.encoder.layers.{i}.layer_norm2.bias"), &tensors));
    }

    text_model.final_layer_norm.weights.as_mut().unwrap().assign_device_buffer(tt(&format!("text_model.final_layer_norm.weight"), &tensors));
    text_model.final_layer_norm.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("text_model.final_layer_norm.bias"), &tensors));
}

#[rustfmt::skip]
fn load_unet(mut unet: &mut UNetModel) {
    let filename = "diffusion_pytorch_model.safetensors";
    let file = File::open(filename).unwrap();
    let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
    let tensors = SafeTensors::deserialize(&buffer).unwrap();
    for (i, l) in unet.time_embedding.iter_mut().enumerate() {
        let i = i + 1;
        l.weights.assign_device_buffer(tt(&format!("time_embedding.linear_{i}.weight"), &tensors));
        l.bias.as_mut().unwrap().assign_device_buffer(tt(
            &format!("time_embedding.linear_{i}.bias"),
            &tensors,
        ));
    }
    let down_blocks = v![y, for y in x.iter_mut(), for x in unet.input_blocks.iter_mut()];
    load_blocks("down_blocks", down_blocks, &tensors);
    let mid_blocks = v![x, for x in unet.mid_blocks.iter_mut()];
    load_blocks("mid_block", mid_blocks, &tensors);
    let up_blocks = v![y, for y in x.iter_mut(), for x in unet.output_blocks.iter_mut()];
    load_blocks("up_blocks", up_blocks, &tensors);

    unet.out_conv.weights.assign_device_buffer(tt(&format!("conv_out.weight"), &tensors));
    unet.out_conv.bias = Some(tt(&format!("conv_out.bias"), &tensors));

    unet.out_gn.weights.as_mut().unwrap().assign_device_buffer(tt(&format!("conv_norm_out.weight"), &tensors));
    unet.out_gn.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("conv_norm_out.bias"), &tensors));
}

fn tt(name: &str, tensors: &SafeTensors) -> Tensor {
    println!("loading {name}");
    let ret = Tensor::from_bytes(tensors.tensor(name).unwrap().data());
    //println!("data {:?}", ret.nd());
    ret
}

#[rustfmt::skip]
fn load_blocks(block_name: &str, blocks: Vec<&mut UnetComponent>, tensors: &SafeTensors) {
    let mut i = 0;
    let mut atn = 0;
    let mut res = 0;
    for b in blocks {
        let block = if block_name == "mid_block" {
            block_name.to_string()
        } else {
            format!("{block_name}.{i}")
        };
        match b {
            UnetComponent::Conv2d(bb) => {
                bb.weights.assign_device_buffer(tt(&format!("conv_in.weight"), &tensors));
                bb.bias = Some(tt(&format!("conv_in.bias"), &tensors));
            }
            UnetComponent::ResBlock(bb) => {
                bb.in_gn.weights.as_mut().unwrap().assign_device_buffer(tt(&format!("{block}.resnets.{res}.norm1.weight"), &tensors));
                bb.in_gn.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("{block}.resnets.{res}.norm1.bias"), &tensors));
                bb.in_conv.weights.assign_device_buffer(tt(&format!("{block}.resnets.{res}.conv1.weight"), &tensors));
                bb.in_conv.bias = Some(tt(&format!("{block}.resnets.{res}.conv1.bias"), &tensors));

                bb.emb_lin.weights.assign_device_buffer(tt(&format!("{block}.resnets.{res}.time_emb_proj.weight"), &tensors));
                bb.emb_lin.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("{block}.resnets.{res}.time_emb_proj.bias"), &tensors));

                bb.out_gn.weights.as_mut().unwrap().assign_device_buffer(tt(&format!("{block}.resnets.{res}.norm2.weight"), &tensors));
                bb.out_gn.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("{block}.resnets.{res}.norm2.bias"), &tensors));
                bb.out_conv.weights.assign_device_buffer(tt(&format!("{block}.resnets.{res}.conv2.weight"), &tensors));
                bb.out_conv.bias = Some(tt(&format!("{block}.resnets.{res}.conv2.bias"), &tensors));

                if bb.conv_shortcut.is_some() {
                    bb.conv_shortcut.as_mut().unwrap().weights.assign_device_buffer(tt(&format!("{block}.resnets.{res}.conv_shortcut.weight"), &tensors));
                    bb.conv_shortcut.as_mut().unwrap().bias = Some(tt(&format!("{block}.resnets.{res}.conv_shortcut.bias"), &tensors));
                }
                res += 1;
            }
            UnetComponent::SpatialTransformer(bb) => {
                bb.norm.weights.as_mut().unwrap().assign_device_buffer(tt(&format!("{block}.attentions.{atn}.norm.weight"), &tensors));
                bb.norm.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("{block}.attentions.{atn}.norm.bias"), &tensors));
                bb.proj_in.weights.assign_device_buffer(tt(&format!("{block}.attentions.{atn}.proj_in.weight"), &tensors));
                bb.proj_in.bias = Some(tt(&format!("{block}.attentions.{atn}.proj_in.bias"), &tensors));
                bb.proj_out.weights.assign_device_buffer(tt(&format!("{block}.attentions.{atn}.proj_out.weight"), &tensors));
                bb.proj_out.bias = Some(tt(&format!("{block}.attentions.{atn}.proj_out.bias"), &tensors));

                bb.transformer_block[0].attn1.to_q.weights.assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.attn1.to_q.weight"), &tensors));
                bb.transformer_block[0].attn1.to_q.bias = None;
                bb.transformer_block[0].attn1.to_k.weights.assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.attn1.to_k.weight"), &tensors));
                bb.transformer_block[0].attn1.to_k.bias = None;
                bb.transformer_block[0].attn1.to_v.weights.assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.attn1.to_v.weight"), &tensors));
                bb.transformer_block[0].attn1.to_v.bias = None;
                bb.transformer_block[0].attn1.to_out[0].weights.assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.attn1.to_out.0.weight"), &tensors));
                bb.transformer_block[0].attn1.to_out[0].bias.as_mut().unwrap().assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.attn1.to_out.0.bias"), &tensors));


                bb.transformer_block[0].attn2.to_q.weights.assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.attn2.to_q.weight"), &tensors));
                bb.transformer_block[0].attn2.to_q.bias = None;
                bb.transformer_block[0].attn2.to_k.weights.assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.attn2.to_k.weight"), &tensors));
                bb.transformer_block[0].attn2.to_k.bias = None;
                bb.transformer_block[0].attn2.to_v.weights.assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.attn2.to_v.weight"), &tensors));
                bb.transformer_block[0].attn2.to_v.bias = None;
                bb.transformer_block[0].attn2.to_out[0].weights.assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.attn2.to_out.0.weight"), &tensors));
                bb.transformer_block[0].attn2.to_out[0].bias.as_mut().unwrap().assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.attn2.to_out.0.bias"), &tensors));

                bb.transformer_block[0].ff.geglu.proj.weights.assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.ff.net.0.proj.weight"), &tensors));
                bb.transformer_block[0].ff.geglu.proj.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.ff.net.0.proj.bias"), &tensors));
                bb.transformer_block[0].ff.lin.weights.assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.ff.net.2.weight"), &tensors));
                bb.transformer_block[0].ff.lin.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.ff.net.2.bias"), &tensors));

                bb.transformer_block[0].norm1.weights.as_mut().unwrap().assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.norm1.weight"), &tensors));
                bb.transformer_block[0].norm1.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.norm1.bias"), &tensors));
                bb.transformer_block[0].norm2.weights.as_mut().unwrap().assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.norm2.weight"), &tensors));
                bb.transformer_block[0].norm2.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.norm2.bias"), &tensors));
                bb.transformer_block[0].norm3.weights.as_mut().unwrap().assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.norm3.weight"), &tensors));
                bb.transformer_block[0].norm3.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("{block}.attentions.{atn}.transformer_blocks.0.norm3.bias"), &tensors));
                atn += 1;
            }
            UnetComponent::Downsample(bb) => {
                bb.conv.weights.assign_device_buffer(tt(&format!("{block}.downsamplers.0.conv.weight"), &tensors));
                bb.conv.bias = Some(tt(&format!("{block}.downsamplers.0.conv.bias"), &tensors));
                i += 1;
                atn = 0;
                res = 0;
            }
            UnetComponent::Upsample(bb) => {
                bb.conv.weights.assign_device_buffer(tt(&format!("{block}.upsamplers.0.conv.weight"), &tensors));
                bb.conv.bias = Some(tt(&format!("{block}.upsamplers.0.conv.bias"), &tensors));
                i += 1;
                atn = 0;
                res = 0;
            }
            _ => panic!(),
        }
    }
}

#[rustfmt::skip]
fn load_vae(mut model: &mut AutoencoderKL)  {
    let filename = "vae.safetensors";
    let file = File::open(filename).unwrap();
    let buffer = unsafe { MmapOptions::new().map(&file).unwrap() };
    let tensors = SafeTensors::deserialize(&buffer).unwrap();
    fn load_resnet(name: &str, res: &mut ResnetBlock, tensors: &SafeTensors) {
        res.conv1.weights.assign_device_buffer(tt(&format!("{name}.conv1.weight"), &tensors));
        res.conv1.bias = Some(tt(&format!("{name}.conv1.bias"), &tensors));
        res.conv2.weights.assign_device_buffer(tt(&format!("{name}.conv2.weight"), &tensors));
        res.conv2.bias = Some(tt(&format!("{name}.conv2.bias"), &tensors));
        res.norm1.weights.as_mut().unwrap().assign_device_buffer(tt(&format!("{name}.norm1.weight"), &tensors));
        res.norm1.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("{name}.norm1.bias"), &tensors));
        res.norm2.weights.as_mut().unwrap().assign_device_buffer(tt(&format!("{name}.norm2.weight"), &tensors));
        res.norm2.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("{name}.norm2.bias"), &tensors));
        if let Some(short_cut) = res.nin_shortcut.as_mut() {
            short_cut.weights.assign_device_buffer(tt(&format!("{name}.conv_shortcut.weight"), &tensors));
            short_cut.bias = Some(tt(&format!("{name}.conv_shortcut.bias"), &tensors));
        }
    }

    let x = "encoder";
    let sampler = "down";

    model.encoder.conv_in.weights.assign_device_buffer(tt(&format!("{x}.conv_in.weight"), &tensors));
    model.encoder.conv_in.bias = Some(tt(&format!("{x}.conv_in.bias"), &tensors));

    // Mid
    load_resnet(&format!("{x}.mid_block.resnets.0"), &mut model.encoder.mid.block_1, &tensors);
    load_resnet(&format!("{x}.mid_block.resnets.1"), &mut model.encoder.mid.block_2, &tensors);
    model.encoder.mid.attn_1.q.weights.assign_device_buffer(tt(&format!("{x}.mid_block.attentions.0.query.weight"), &tensors));
    model.encoder.mid.attn_1.q.bias = Some(tt(&format!("{x}.mid_block.attentions.0.query.bias"), &tensors));
    model.encoder.mid.attn_1.k.weights.assign_device_buffer(tt(&format!("{x}.mid_block.attentions.0.key.weight"), &tensors));
    model.encoder.mid.attn_1.k.bias = Some(tt(&format!("{x}.mid_block.attentions.0.key.bias"), &tensors));
    model.encoder.mid.attn_1.v.weights.assign_device_buffer(tt(&format!("{x}.mid_block.attentions.0.value.weight"), &tensors));
    model.encoder.mid.attn_1.v.bias = Some(tt(&format!("{x}.mid_block.attentions.0.value.bias"), &tensors));
    model.encoder.mid.attn_1.proj_out.weights.assign_device_buffer(tt(&format!("{x}.mid_block.attentions.0.proj_attn.weight"), &tensors));
    model.encoder.mid.attn_1.proj_out.bias = Some(tt(&format!("{x}.mid_block.attentions.0.proj_attn.bias"), &tensors));
    model.encoder.mid.attn_1.norm.weights.as_mut().unwrap().assign_device_buffer(tt(&format!("{x}.mid_block.attentions.0.group_norm.weight"), &tensors));
    model.encoder.mid.attn_1.norm.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("{x}.mid_block.attentions.0.group_norm.bias"), &tensors));

    // Down
    for (i, block) in model.encoder.down.iter_mut().enumerate() {
        for (ri, res) in block.0.iter_mut().enumerate() {
            let name = format!("{x}.{sampler}_blocks.{i}.resnets.{ri}");
            load_resnet(&name, res, &tensors);
        }
        for (ci, conv) in block.1.iter_mut().enumerate() {
            conv.weights.assign_device_buffer(tt(&format!("{x}.{sampler}_blocks.{i}.{sampler}samplers.{ci}.conv.weight"), &tensors));
            conv.bias = Some(tt(&format!("{x}.{sampler}_blocks.{i}.{sampler}samplers.{ci}.conv.bias"), &tensors));
        }
    }

    model.encoder.norm_out.weights.as_mut().unwrap().assign_device_buffer(tt(&format!("{x}.conv_norm_out.weight"), &tensors));
    model.encoder.norm_out.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("{x}.conv_norm_out.bias"), &tensors));

    model.encoder.conv_out.weights.assign_device_buffer(tt(&format!("{x}.conv_out.weight"), &tensors));
    model.encoder.conv_out.bias = Some(tt(&format!("{x}.conv_out.bias"), &tensors));


    let x = "decoder";
    let sampler = "up";

    model.decoder.conv_in.weights.assign_device_buffer(tt(&format!("{x}.conv_in.weight"), &tensors));
    model.decoder.conv_in.bias = Some(tt(&format!("{x}.conv_in.bias"), &tensors));

    // Mid
    load_resnet(&format!("{x}.mid_block.resnets.0"), &mut model.decoder.mid.block_1, &tensors);
    load_resnet(&format!("{x}.mid_block.resnets.1"), &mut model.decoder.mid.block_2, &tensors);
    model.decoder.mid.attn_1.q.weights.assign_device_buffer(tt(&format!("{x}.mid_block.attentions.0.query.weight"), &tensors));
    model.decoder.mid.attn_1.q.bias = Some(tt(&format!("{x}.mid_block.attentions.0.query.bias"), &tensors));
    model.decoder.mid.attn_1.k.weights.assign_device_buffer(tt(&format!("{x}.mid_block.attentions.0.key.weight"), &tensors));
    model.decoder.mid.attn_1.k.bias = Some(tt(&format!("{x}.mid_block.attentions.0.key.bias"), &tensors));
    model.decoder.mid.attn_1.v.weights.assign_device_buffer(tt(&format!("{x}.mid_block.attentions.0.value.weight"), &tensors));
    model.decoder.mid.attn_1.v.bias = Some(tt(&format!("{x}.mid_block.attentions.0.value.bias"), &tensors));
    model.decoder.mid.attn_1.proj_out.weights.assign_device_buffer(tt(&format!("{x}.mid_block.attentions.0.proj_attn.weight"), &tensors));
    model.decoder.mid.attn_1.proj_out.bias = Some(tt(&format!("{x}.mid_block.attentions.0.proj_attn.bias"), &tensors));
    model.decoder.mid.attn_1.norm.weights.as_mut().unwrap().assign_device_buffer(tt(&format!("{x}.mid_block.attentions.0.group_norm.weight"), &tensors));
    model.decoder.mid.attn_1.norm.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("{x}.mid_block.attentions.0.group_norm.bias"), &tensors));

    // Up
    for (i, block) in model.decoder.up.iter_mut().rev().enumerate() {
        for (ri, res) in block.0.iter_mut().enumerate() {
            let name = format!("{x}.{sampler}_blocks.{i}.resnets.{ri}");
            load_resnet(&name, res, &tensors);
        }
        for (ci, conv) in block.1.iter_mut().enumerate() {
            conv.weights.assign_device_buffer(tt(&format!("{x}.{sampler}_blocks.{i}.{sampler}samplers.{ci}.conv.weight"), &tensors));
            conv.bias = Some(tt(&format!("{x}.{sampler}_blocks.{i}.{sampler}samplers.{ci}.conv.bias"), &tensors));
        }
    }

    model.decoder.norm_out.weights.as_mut().unwrap().assign_device_buffer(tt(&format!("{x}.conv_norm_out.weight"), &tensors));
    model.decoder.norm_out.bias.as_mut().unwrap().assign_device_buffer(tt(&format!("{x}.conv_norm_out.bias"), &tensors));

    model.decoder.conv_out.weights.assign_device_buffer(tt(&format!("{x}.conv_out.weight"), &tensors));
    model.decoder.conv_out.bias = Some(tt(&format!("{x}.conv_out.bias"), &tensors));


    model.quant_conv.weights.assign_device_buffer(tt(&format!("quant_conv.weight"), &tensors));
    model.quant_conv.bias = Some(tt(&format!("quant_conv.bias"), &tensors));
    model.post_quant_conv.weights.assign_device_buffer(tt(&format!("post_quant_conv.weight"), &tensors));
    model.post_quant_conv.bias = Some(tt(&format!("post_quant_conv.bias"), &tensors));
}

/// No idea where this came from, tinygrad example this is load into the empty tensor during torch
/// load
fn alpha_cumprod() -> Vec<f32> {
    vec![
        0.99915, 0.998296, 0.9974381, 0.9965762, 0.99571025, 0.9948404, 0.9939665, 0.9930887,
        0.9922069, 0.9913211, 0.9904313, 0.98953754, 0.9886398, 0.9877381, 0.9868324, 0.98592263,
        0.98500896, 0.9840913, 0.9831696, 0.982244, 0.98131436, 0.9803808, 0.97944313, 0.97850156,
        0.977556, 0.9766064, 0.97565293, 0.9746954, 0.9737339, 0.9727684, 0.97179896, 0.97082555,
        0.96984816, 0.96886677, 0.9678814, 0.96689206, 0.96589875, 0.9649015, 0.96390027,
        0.9628951, 0.9618859, 0.96087277, 0.95985574, 0.95883465, 0.9578097, 0.95678073,
        0.95574784, 0.954711, 0.95367026, 0.9526256, 0.9515769, 0.95052433, 0.94946784, 0.94840735,
        0.947343, 0.94627476, 0.9452025, 0.9441264, 0.9430464, 0.9419625, 0.9408747, 0.939783,
        0.9386874, 0.93758786, 0.9364845, 0.93537724, 0.9342661, 0.9331511, 0.9320323, 0.9309096,
        0.929783, 0.9286526, 0.9275183, 0.9263802, 0.92523825, 0.92409253, 0.92294294, 0.9217895,
        0.92063236, 0.9194713, 0.9183065, 0.9171379, 0.91596556, 0.9147894, 0.9136095, 0.91242576,
        0.9112383, 0.9100471, 0.9088522, 0.9076535, 0.9064511, 0.90524495, 0.9040351, 0.90282154,
        0.9016043, 0.90038335, 0.8991587, 0.8979304, 0.8966984, 0.89546275, 0.89422345, 0.8929805,
        0.89173394, 0.89048374, 0.88922995, 0.8879725, 0.8867115, 0.88544685, 0.88417864,
        0.88290685, 0.8816315, 0.88035256, 0.8790701, 0.87778413, 0.8764946, 0.8752016, 0.873905,
        0.87260497, 0.8713014, 0.8699944, 0.86868393, 0.86737, 0.8660526, 0.8647318, 0.86340755,
        0.8620799, 0.8607488, 0.85941434, 0.8580765, 0.8567353, 0.8553907, 0.8540428, 0.85269153,
        0.85133696, 0.84997904, 0.84861785, 0.8472533, 0.8458856, 0.8445145, 0.84314024,
        0.84176266, 0.8403819, 0.8389979, 0.8376107, 0.8362203, 0.83482677, 0.83343, 0.8320301,
        0.8306271, 0.8292209, 0.82781166, 0.82639927, 0.8249838, 0.82356524, 0.8221436, 0.82071894,
        0.81929123, 0.81786054, 0.8164268, 0.8149901, 0.8135504, 0.81210774, 0.81066215, 0.8092136,
        0.8077621, 0.80630773, 0.80485046, 0.8033903, 0.80192727, 0.8004614, 0.79899275,
        0.79752123, 0.7960469, 0.7945698, 0.7930899, 0.79160726, 0.7901219, 0.7886338, 0.787143,
        0.7856495, 0.7841533, 0.78265446, 0.78115296, 0.7796488, 0.77814204, 0.7766327, 0.7751208,
        0.7736063, 0.77208924, 0.7705697, 0.7690476, 0.767523, 0.7659959, 0.7644664, 0.76293445,
        0.7614, 0.7598632, 0.75832397, 0.75678235, 0.75523835, 0.75369203, 0.7521434, 0.75059247,
        0.7490392, 0.7474837, 0.7459259, 0.7443659, 0.74280363, 0.7412392, 0.7396726, 0.7381038,
        0.73653287, 0.7349598, 0.7333846, 0.73180735, 0.730228, 0.7286466, 0.7270631, 0.7254777,
        0.72389024, 0.72230077, 0.7207094, 0.71911603, 0.7175208, 0.7159236, 0.71432453, 0.7127236,
        0.71112084, 0.7095162, 0.7079098, 0.7063016, 0.70469165, 0.70307994, 0.7014665, 0.69985133,
        0.6982345, 0.696616, 0.6949958, 0.69337404, 0.69175065, 0.69012564, 0.6884991, 0.68687093,
        0.6852413, 0.68361014, 0.6819775, 0.6803434, 0.67870784, 0.6770708, 0.6754324, 0.6737926,
        0.67215145, 0.670509, 0.66886514, 0.66722, 0.6655736, 0.66392595, 0.662277, 0.6606269,
        0.65897554, 0.657323, 0.65566933, 0.6540145, 0.6523586, 0.6507016, 0.6490435, 0.64738435,
        0.6457241, 0.64406294, 0.6424008, 0.64073765, 0.63907355, 0.63740855, 0.6357426, 0.6340758,
        0.6324082, 0.6307397, 0.6290704, 0.6274003, 0.6257294, 0.62405777, 0.6223854, 0.62071234,
        0.6190386, 0.61736417, 0.6156891, 0.61401343, 0.6123372, 0.6106603, 0.6089829, 0.607305,
        0.6056265, 0.6039476, 0.60226816, 0.6005883, 0.598908, 0.59722733, 0.5955463, 0.59386486,
        0.5921831, 0.59050107, 0.5888187, 0.5871361, 0.5854532, 0.5837701, 0.5820868, 0.5804033,
        0.5787197, 0.5770359, 0.575352, 0.57366806, 0.571984, 0.5702999, 0.5686158, 0.56693166,
        0.56524754, 0.5635635, 0.5618795, 0.56019557, 0.5585118, 0.5568281, 0.55514455, 0.5534612,
        0.551778, 0.5500951, 0.5484124, 0.54673, 0.5450478, 0.54336596, 0.54168445, 0.54000324,
        0.53832245, 0.5366421, 0.53496206, 0.5332825, 0.53160346, 0.5299248, 0.52824676, 0.5265692,
        0.52489215, 0.5232157, 0.5215398, 0.51986456, 0.51818997, 0.51651603, 0.51484275,
        0.5131702, 0.5114983, 0.5098272, 0.50815684, 0.5064873, 0.50481856, 0.50315064, 0.50148356,
        0.4998174, 0.4981521, 0.49648774, 0.49482432, 0.49316183, 0.49150035, 0.48983985,
        0.4881804, 0.486522, 0.48486462, 0.4832084, 0.48155323, 0.4798992, 0.47824633, 0.47659463,
        0.4749441, 0.47329482, 0.4716468, 0.47, 0.46835446, 0.46671024, 0.46506736, 0.4634258,
        0.46178558, 0.46014675, 0.45850933, 0.45687333, 0.45523876, 0.45360568, 0.45197406,
        0.45034397, 0.44871536, 0.44708833, 0.44546285, 0.44383895, 0.44221666, 0.440596,
        0.43897697, 0.43735963, 0.43574396, 0.43412998, 0.43251774, 0.43090722, 0.4292985,
        0.42769152, 0.42608637, 0.42448303, 0.4228815, 0.42128187, 0.4196841, 0.41808826,
        0.4164943, 0.4149023, 0.41331223, 0.41172415, 0.41013804, 0.40855396, 0.4069719, 0.4053919,
        0.40381396, 0.4022381, 0.40066436, 0.39909273, 0.39752322, 0.3959559, 0.39439073,
        0.39282778, 0.39126703, 0.3897085, 0.3881522, 0.3865982, 0.38504648, 0.38349706,
        0.38194993, 0.38040516, 0.37886274, 0.37732267, 0.375785, 0.37424973, 0.37271687,
        0.37118647, 0.36965853, 0.36813304, 0.36661002, 0.36508954, 0.36357155, 0.3620561,
        0.36054322, 0.3590329, 0.35752517, 0.35602003, 0.35451752, 0.35301763, 0.3515204,
        0.3500258, 0.3485339, 0.3470447, 0.34555823, 0.34407446, 0.34259343, 0.34111515,
        0.33963963, 0.33816692, 0.336697, 0.3352299, 0.33376563, 0.3323042, 0.33084565, 0.32938993,
        0.32793713, 0.3264872, 0.32504022, 0.32359615, 0.32215503, 0.32071686, 0.31928164,
        0.31784943, 0.3164202, 0.314994, 0.3135708, 0.31215066, 0.31073356, 0.3093195, 0.30790854,
        0.30650064, 0.30509588, 0.30369422, 0.30229566, 0.30090025, 0.299508, 0.2981189,
        0.29673296, 0.29535022, 0.2939707, 0.29259437, 0.29122123, 0.28985137, 0.28848472,
        0.28712133, 0.2857612, 0.28440437, 0.2830508, 0.28170055, 0.2803536, 0.27900997,
        0.27766964, 0.27633268, 0.27499905, 0.2736688, 0.27234194, 0.27101842, 0.2696983,
        0.26838157, 0.26706827, 0.26575837, 0.26445192, 0.26314887, 0.2618493, 0.26055318,
        0.2592605, 0.25797132, 0.2566856, 0.2554034, 0.25412467, 0.25284946, 0.25157773, 0.2503096,
        0.24904492, 0.24778382, 0.24652626, 0.24527225, 0.2440218, 0.24277493, 0.24153163,
        0.24029191, 0.23905578, 0.23782326, 0.23659433, 0.23536903, 0.23414734, 0.23292927,
        0.23171483, 0.23050404, 0.22929688, 0.22809339, 0.22689353, 0.22569734, 0.22450483,
        0.22331597, 0.2221308, 0.22094932, 0.21977153, 0.21859743, 0.21742703, 0.21626033,
        0.21509734, 0.21393807, 0.21278252, 0.21163069, 0.21048258, 0.20933822, 0.20819758,
        0.2070607, 0.20592754, 0.20479813, 0.20367248, 0.20255059, 0.20143245, 0.20031808,
        0.19920748, 0.19810064, 0.19699757, 0.19589828, 0.19480278, 0.19371104, 0.1926231,
        0.19153893, 0.19045855, 0.18938197, 0.18830918, 0.18724018, 0.18617497, 0.18511358,
        0.18405597, 0.18300217, 0.18195218, 0.18090598, 0.1798636, 0.17882504, 0.17779027,
        0.1767593, 0.17573217, 0.17470883, 0.1736893, 0.1726736, 0.1716617, 0.17065361, 0.16964935,
        0.1686489, 0.16765225, 0.16665943, 0.16567042, 0.16468522, 0.16370384, 0.16272627,
        0.16175252, 0.16078258, 0.15981644, 0.15885411, 0.1578956, 0.15694089, 0.15599, 0.15504292,
        0.15409963, 0.15316014, 0.15222447, 0.15129258, 0.1503645, 0.14944021, 0.14851972,
        0.14760303, 0.14669013, 0.14578101, 0.14487568, 0.14397413, 0.14307636, 0.14218238,
        0.14129217, 0.14040573, 0.13952307, 0.13864417, 0.13776903, 0.13689767, 0.13603005,
        0.13516618, 0.13430607, 0.13344972, 0.1325971, 0.13174823, 0.1309031, 0.13006169,
        0.12922402, 0.12839006, 0.12755983, 0.12673332, 0.12591052, 0.12509143, 0.12427604,
        0.12346435, 0.12265636, 0.12185206, 0.12105144, 0.1202545, 0.11946124, 0.11867165,
        0.11788572, 0.11710346, 0.11632485, 0.11554988, 0.11477857, 0.11401089, 0.11324684,
        0.11248643, 0.11172963, 0.11097645, 0.11022688, 0.10948092, 0.10873855, 0.10799977,
        0.10726459, 0.10653298, 0.10580494, 0.10508047, 0.10435956, 0.1036422, 0.10292839,
        0.10221813, 0.1015114, 0.10080819, 0.1001085, 0.09941233, 0.09871966, 0.0980305,
        0.09734483, 0.09666264, 0.09598393, 0.09530868, 0.09463691, 0.09396859, 0.09330372,
        0.09264228, 0.09198428, 0.09132971, 0.09067855, 0.0900308, 0.08938646, 0.0887455,
        0.08810794, 0.08747375, 0.08684293, 0.08621547, 0.08559138, 0.08497062, 0.08435319,
        0.0837391, 0.08312833, 0.08252087, 0.08191671, 0.08131585, 0.08071827, 0.08012398,
        0.07953294, 0.07894517, 0.07836065, 0.07777938, 0.07720133, 0.07662651, 0.07605491,
        0.07548651, 0.07492131, 0.0743593, 0.07380046, 0.0732448, 0.07269229, 0.07214294,
        0.07159673, 0.07105365, 0.0705137, 0.06997685, 0.06944311, 0.06891247, 0.06838491,
        0.06786042, 0.06733901, 0.06682064, 0.06630533, 0.06579305, 0.0652838, 0.06477757,
        0.06427433, 0.0637741, 0.06327686, 0.06278259, 0.06229129, 0.06180295, 0.06131756,
        0.0608351, 0.06035557, 0.05987896, 0.05940525, 0.05893444, 0.05846652, 0.05800147,
        0.0575393, 0.05707997, 0.05662349, 0.05616985, 0.05571903, 0.05527103, 0.05482582,
        0.05438342, 0.05394379, 0.05350694, 0.05307286, 0.05264152, 0.05221293, 0.05178706,
        0.05136392, 0.05094349, 0.05052575, 0.05011071, 0.04969834, 0.04928865, 0.0488816,
        0.04847721, 0.04807544, 0.04767631, 0.04727979, 0.04688587, 0.04649454, 0.0461058,
        0.04571963, 0.04533602, 0.04495496, 0.04457644, 0.04420045, 0.04382697, 0.043456,
        0.04308753, 0.04272155, 0.04235804, 0.04199699, 0.0416384, 0.04128224, 0.04092852,
        0.04057723, 0.04022833, 0.03988184, 0.03953774, 0.03919602, 0.03885666, 0.03851966,
        0.038185, 0.03785268, 0.03752268, 0.037195, 0.03686962, 0.03654652, 0.03622571, 0.03590717,
        0.03559089, 0.03527685, 0.03496506, 0.03465549, 0.03434813, 0.03404298, 0.03374003,
        0.03343925, 0.03314065, 0.03284422, 0.03254993, 0.03225778, 0.03196777, 0.03167988,
        0.03139409, 0.0311104, 0.0308288, 0.03054927, 0.03027181, 0.02999641, 0.02972305,
        0.02945173, 0.02918243, 0.02891514, 0.02864986, 0.02838656, 0.02812525, 0.02786591,
        0.02760853, 0.0273531, 0.02709961, 0.02684805, 0.02659841, 0.02635068, 0.02610484,
        0.02586089, 0.02561882, 0.02537862, 0.02514027, 0.02490377, 0.0246691, 0.02443626,
        0.02420524, 0.02397602, 0.02374859, 0.02352295, 0.02329909, 0.02307699, 0.02285664,
        0.02263804, 0.02242117, 0.02220603, 0.0219926, 0.02178088, 0.02157084, 0.0213625,
        0.02115583, 0.02095082, 0.02074747, 0.02054576, 0.02034568, 0.02014724, 0.0199504,
        0.01975518, 0.01956154, 0.0193695, 0.01917903, 0.01899013, 0.01880278, 0.01861698,
        0.01843272, 0.01824999, 0.01806878, 0.01788907, 0.01771087, 0.01753416, 0.01735893,
        0.01718517, 0.01701287, 0.01684203, 0.01667263, 0.01650466, 0.01633812, 0.016173,
        0.01600928, 0.01584696, 0.01568603, 0.01552648, 0.0153683, 0.01521149, 0.01505602,
        0.0149019, 0.01474911, 0.01459765, 0.01444751, 0.01429868, 0.01415114, 0.0140049,
        0.01385994, 0.01371625, 0.01357382, 0.01343266, 0.01329274, 0.01315405, 0.0130166,
        0.01288038, 0.01274536, 0.01261155, 0.01247894, 0.01234751, 0.01221727, 0.0120882,
        0.01196029, 0.01183354, 0.01170793, 0.01158346, 0.01146013, 0.01133791, 0.01121681,
        0.01109682, 0.01097793, 0.01086013, 0.01074341, 0.01062776, 0.01051319, 0.01039967,
        0.0102872, 0.01017578, 0.0100654, 0.00995604, 0.0098477, 0.00974038, 0.00963406,
        0.00952875, 0.00942442, 0.00932108, 0.00921871, 0.00911731, 0.00901687, 0.00891739,
        0.00881885, 0.00872126, 0.00862459, 0.00852885, 0.00843403, 0.00834012, 0.00824711,
        0.008155, 0.00806378, 0.00797345, 0.00788399, 0.0077954, 0.00770767, 0.0076208, 0.00753477,
        0.00744959, 0.00736524, 0.00728173, 0.00719903, 0.00711715, 0.00703608, 0.00695581,
        0.00687634, 0.00679766, 0.00671976, 0.00664264, 0.00656629, 0.0064907, 0.00641587,
        0.00634179, 0.00626846, 0.00619587, 0.00612401, 0.00605288, 0.00598247, 0.00591277,
        0.00584378, 0.0057755, 0.00570791, 0.00564102, 0.00557481, 0.00550928, 0.00544443,
        0.00538024, 0.00531672, 0.00525385, 0.00519164, 0.00513007, 0.00506914, 0.00500884,
        0.00494918, 0.00489014, 0.00483171, 0.0047739, 0.0047167, 0.0046601,
    ]
}

use show_image::{create_window, ImageInfo, ImageView};

#[show_image::main]
fn main() {
    let mut model = StableDiffusion::new();
    model.alphas_comprod = alpha_cumprod();
    load_text_model(model.cond_stage_model.as_mut().unwrap());
    load_vae(&mut model.first_stage_model);
    load_unet(&mut model.model);
    let tokenizer = Tokenizer::new();
    let tokens = tokenizer.encode("astronaut on the moon");
    let prompt =
        Tensor::from(tokens.iter().map(|&e| e as f32).collect::<Vec<f32>>()).reshape([1, -1]);
    let context = model
        .cond_stage_model
        .as_ref()
        .unwrap()
        .call(&prompt)
        .realize();
    // println!("{:?}", prompt.nd());
    // println!("context\n{:?}\n", context.nd());
    // panic!();
    // println!("token_embedding  \n{:?}", model.cond_stage_model.as_ref().unwrap().embeddings.token_embedding.weight.nd());
    // println!("vae decode norm1 \n{:?}", model.first_stage_model.decoder.up[0].0[0].norm1.weights.as_ref().unwrap().nd());

    let tokenizer = Tokenizer::new();
    let tokens = tokenizer.encode("");
    let prompt =
        Tensor::from(tokens.iter().map(|&e| e as f32).collect::<Vec<f32>>()).reshape([1, -1]);
    let uncon_context = model
        .cond_stage_model
        .as_ref()
        .unwrap()
        .call(&prompt)
        .realize();
    model.cond_stage_model = None;

    let steps = 5;
    let guidance = 7.5;
    let timesteps = (1..1000).step_by(1000 / steps);
    let alphas = v![model.alphas_comprod[i], for i in timesteps.clone()];
    let mut alphas_prev = v![model.alphas_comprod[i], for i in timesteps.clone().rev().skip(1)];
    alphas_prev.push(1.0);
    alphas_prev.reverse();

    let (w, h) = (64, 64);
    let mut latent = Tensor::randn([1, 4, w, h]);

    for (i, (index, timestep)) in timesteps.enumerate().rev().enumerate() {
        println!("step {}", i+1);
        let tid = Tensor::_const(index as f32);
        // println!("uncon_context\n{}", uncon_context.nd());
        // println!("context\n{}", context.nd());
        // println!("latent\n{}", latent.nd());
        // println!("timestep\n{}", timestep);
        // println!("alpha\n{}", alphas[index]);
        // println!("alpha_prev\n{}", alphas_prev[index]);
        // println!("guidence\n{}", guidance);
        latent = model.call(
            &uncon_context,
            &context,
            &latent,
            &Tensor::_const(timestep),
            &Tensor::_const(alphas[index]),
            &Tensor::_const(alphas_prev[index]),
            &Tensor::_const(guidance),
        ).realize();
        // println!("{}", latent.nd());
    }

    let mut x = model.decode(&latent);

    x = (x + 1.) / 2.;
    x = x.reshape([3, w * 8, h * 8]).permute([1, 2, 0]).clip(0., 1.) * 255.;
    println!("out shape {}", x.shape());
    let image_data = x.to_vec().into_iter().map(|s| s as u8).collect::<Vec<u8>>();

    let image = ImageView::new(ImageInfo::rgb8(w * 8, h * 8), &image_data);
    let window = create_window("image", Default::default()).unwrap();
    window.set_image("image-001", image).unwrap();
    loop {}
}
