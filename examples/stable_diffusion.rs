#![allow(unused)]

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
            .reshape([bs, c, py * 2, px * 2]);
        self.conv.call(&x)
    }
}

pub fn timestep_embedding(timesteps: usize, dim: usize, max_period: Option<usize>) -> Tensor {
    let half = dim as f32 / 2.;
    let max_period = max_period.unwrap_or(10000);
    let freqs = (-(max_period as f32).ln() * Tensor::arange(half) / half).exp();
    let args = timesteps as f32 * freqs;
    Tensor::cat(&args.cos(), &[args.sin()], None).reshape([1, -1])
}

pub struct UNetModel {
    // Time embedded
    time_emb_lin1: Linear,
    time_emb_lin2: Linear,

    input_blocks: Vec<Vec<UnetComponent>>,
    output_blocks: Vec<Vec<UnetComponent>>,

    // middle_block
    mid_res1: ResBlock,
    mid_spat1: SpatialTransformer,
    mid_res2: ResBlock,

    // output block
    outb_res1: ResBlock,
    outb_res2: ResBlock,

    outb_res3: ResBlock,
    outb_up1: Upsample,

    outb_res4: ResBlock,
    outb_spat1: SpatialTransformer,

    outb_res5: ResBlock,
    outb_spat2: SpatialTransformer,

    outb_res6: ResBlock,
    outb_spat3: SpatialTransformer,
    outb_up2: Upsample,

    outb_res7: ResBlock,
    outb_spat4: SpatialTransformer,

    outb_res8: ResBlock,
    outb_spat5: SpatialTransformer,

    outb_res9: ResBlock,
    outb_spat6: SpatialTransformer,
    outb_up3: Upsample,

    outb_res10: ResBlock,
    outb_spat7: SpatialTransformer,

    outb_res11: ResBlock,
    outb_spat8: SpatialTransformer,

    outb_res12: ResBlock,
    outb_spat9: SpatialTransformer,

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

            outb_res1: ResBlock::new(2560, 1280, 1280),
            outb_res2: ResBlock::new(2560, 1280, 1280),
            outb_res3: ResBlock::new(2560, 1280, 1280), outb_up1: Upsample::new(1280),
            outb_res4: ResBlock::new(2560, 1280, 1280), outb_spat1: SpatialTransformer::new(1280, 768, 8, 160),
            outb_res5: ResBlock::new(2560, 1280, 1280), outb_spat2: SpatialTransformer::new(1280, 768, 8, 160),
            outb_res6: ResBlock::new(1920, 1280, 1280), outb_spat3: SpatialTransformer::new(1280, 768, 8, 160), outb_up2: Upsample::new(1280),
            outb_res7: ResBlock::new(1920, 1280, 640), outb_spat4: SpatialTransformer::new(640, 768, 8, 80),
            outb_res8: ResBlock::new(1280, 1280, 640), outb_spat5: SpatialTransformer::new(640, 768, 8, 80),
            outb_res9: ResBlock::new(960, 1280, 640), outb_spat6: SpatialTransformer::new(640, 768, 8, 80), outb_up3: Upsample::new(640),
            outb_res10: ResBlock::new(960, 1280, 320), outb_spat7: SpatialTransformer::new(320, 768, 8, 40),
            outb_res11: ResBlock::new(640, 1280, 320), outb_spat8: SpatialTransformer::new(320, 768, 8, 40),
            outb_res12: ResBlock::new(640, 1280, 320), outb_spat9: SpatialTransformer::new(320, 768, 8, 40),

            out_gn: GroupNorm::new(32, 320, None, None),
            out_conv: Conv2d::new(320, 4, 3, None, [1], None, None, None),
        }
    }

    fn call(&self, x: &Tensor, timesteps: usize, context: Option<&Tensor>) -> Tensor {
        // time emb
        let mut x = x.clone();
        let t_emb = timestep_embedding(timesteps, 320, None);
        let emb = self
            .time_emb_lin2
            .call(&self.time_emb_lin1.call(&t_emb).silu());
        let mut save_inputs = vec![];

        // input block
        for (i, block) in self.input_blocks.iter().enumerate() {
            print!("input block {i}");
            for b in block.iter() {
                match b {
                    UnetComponent::Conv2d(bb) => x = bb.call(&x),
                    UnetComponent::ResBlock(bb) => x = bb.call(&x, &emb),
                    UnetComponent::SpatialTransformer(bb) => x = bb.call(&x, context.clone()),
                    UnetComponent::Downsample(bb) => x = bb.call(&x),
                    _ => panic!(),
                }
            }
            x.realize();
            save_inputs.push(x.clone());
        }

        // mid
        x = self.mid_res1.call(&x, &emb);
        x = self.mid_spat1.call(&x, context.clone());
        x = self.mid_res2.call(&x, &emb);

        for (i, block) in self.output_blocks.iter().enumerate() {
            print!("output block {i}");
            x = x.cat(&[save_inputs.pop().unwrap()], Some(1)).realize();
            for b in block.iter() {
                match b {
                    UnetComponent::ResBlock(bb) => x = bb.call(&x, &emb),
                    UnetComponent::SpatialTransformer(bb) => x = bb.call(&x, context.clone()),
                    UnetComponent::Upsample(bb) => x = bb.call(&x),
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

fn main() {
    let a = UNetModel::new();
    let ctx = Tensor::randn([2, 77, 768]);
    let out = a.call(&Tensor::randn([2, 4, 64, 64]), 801, Some(&ctx));
    println!("{:?}", out.realize())
}
