use rand::seq::SliceRandom;
use rand::thread_rng;
use sdl2::pixels::Color;
use sdl2::rect::Rect;
use storm::prelude::*;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;

const SCALE: u32 = 8;

struct AE {
    l1: Tensor,
    l2: Tensor,
    l3: Tensor,
}

impl AE {
    pub fn new() -> Self {
        Self {
            l1: Tensor::uniform([2, 7]),
            l2: Tensor::uniform([7, 2]),
            l3: Tensor::uniform([2, 1]),
        }
    }

    pub fn forward(&self, x: &Tensor) -> Tensor {
        let mut x = x.matmul(&self.l1).sigmoid();
        x = x.matmul(&self.l2).sigmoid();
        x = x.matmul(&self.l3).sigmoid();
        x
    }

    // pub fn save(&self, path: &str) -> Result<(), safetensors::SafeTensorError> {
    //     Tensor::to_safetensor(
    //         &[("l1", &self.l1), ("l2", &self.l2), ("l3", &self.l3)],
    //         path,
    //     )?;
    //     Ok(())
    // }
    //
    // pub fn load(&mut self, path: &str) -> Result<(), safetensors::SafeTensorError> {
    //     self.l1.from_safetensor("l1", path)?;
    //     self.l2.from_safetensor("l2", path)?;
    //     self.l3.from_safetensor("l3", path)?;
    //     Ok(())
    // }
}

fn main() -> Result<(), String> {
    let model_path = "./models/autoencoder.safetensors";
    let mut model = AE::new();
    let train = true;
    let load = false;
    // if load {
    //     model.load(model_path).unwrap();
    // }
    let mut optim = adam(&[&mut model.l1, &mut model.l2, &mut model.l3], 0.005);
    let batch_size = 1;
    let (mut img_batched, _, _, _) = fetch_mnist(batch_size, false);
    let sdl_context = sdl2::init()?;
    let video_subsystem = sdl_context.video()?;
    let window = video_subsystem
        .window("mnist", 28 * SCALE * 4, 28 * SCALE * 4)
        .position_centered()
        .build()
        .map_err(|e| e.to_string())?;
    let mut canvas = window
        .into_canvas()
        .software()
        .build()
        .map_err(|e| e.to_string())?;
    canvas.set_draw_color(Color::RGB(0, 0, 0));
    canvas.present();
    let mut i = 0;
    let mut latent_value = 0.0f32;
    'mainloop: loop {
        if train {
            if i >= img_batched.len() {
                let mut rng = thread_rng();
                img_batched.shuffle(&mut rng);
                i = 0;
            }
            canvas.clear();
            optim.zero_grad();
            let img = &img_batched[10057];
            let mut input = vec![];
            let mut y = vec![];
            for r in 0..28 {
                for c in 0..28 {
                    let v = img[r * 28 + c] / 255.;
                    input.extend(vec![r as f32 / 27., c as f32 / 27.]);
                    y.push(v);
                }
            }
            let x: Tensor = Tensor::from(&*input).reshape([28 * 28, 2]);
            let x = x.detach();
            let y: Tensor = Tensor::from(&*y).reshape([28 * 28, 1]);
            let y = y.detach();
            // let y: Tensor<Cpu> =
            //     Tensor::from_shape(&*img_batched[10057], [batch_size, 28 * 28]) / 255.0;
            // let y = y.detach();
            let out = model.forward(&x);
            let out_vec = out.to_vec();
            let out_2 = &y - &out;
            let mut loss: Tensor = (&out_2 * &out_2).sum_all() / (28 * 28);
            loss.backward();
            optim.step();
            for r in 0..28 {
                for c in 0..28 {
                    let p = (out_vec[r * 28 + c] * 255.0) as u8;
                    let t = (img[r * 28 + c]) as u8;
                    //print!("{p:<5}");
                    canvas.set_draw_color(Color {
                        r: p,
                        g: p,
                        b: p,
                        a: p,
                    });
                    canvas.fill_rect(Rect::new(
                        (c + 28) as i32 * SCALE as i32,
                        r as i32 * SCALE as i32,
                        SCALE,
                        SCALE,
                    ))?;
                    canvas.set_draw_color(Color {
                        r: t,
                        g: t,
                        b: t,
                        a: t,
                    });
                    canvas.fill_rect(Rect::new(
                        c as i32 * SCALE as i32,
                        r as i32 * SCALE as i32,
                        SCALE,
                        SCALE,
                    ))?;
                }
                //println!();
            }

            // upcale
            let mut input = vec![];
            for r in 0..28 * 2 {
                for c in 0..28 * 2 {
                    input.extend(vec![r as f32 / 55., c as f32 / 55.]);
                }
            }
            let out = model.forward(&Tensor::from(&*input).reshape([28 * 28 * 4, 2]));
            let out_vec = out.to_vec();
            for r in 0..28 * 2 {
                for c in 0..28 * 2 {
                    let p = (out_vec[r * 28 * 2 + c] * 255.0) as u8;
                    canvas.set_draw_color(Color {
                        r: p,
                        g: p,
                        b: p,
                        a: p,
                    });
                    canvas.fill_rect(Rect::new(
                        (c) as i32 * SCALE as i32,
                        (r + 28) as i32 * SCALE as i32,
                        SCALE,
                        SCALE,
                    ))?;
                }
            }
            println!("cost: {}", loss.to_vec()[0]);
        } else {
            let mut input = vec![];
            let upscale = 4;
            for r in 0..28 * upscale {
                for c in 0..28 * upscale {
                    input.extend(vec![
                        r as f32 / (upscale * 28 - 1) as f32,
                        c as f32 / (upscale * 28 - 1) as f32,
                    ]);
                }
            }
            let out = model.forward(&Tensor::from(
                &*input).reshape(
                [28 * 28 * upscale * upscale, 2],
            ));
            let out_vec = out.to_vec();
            for r in 0..28 * upscale {
                for c in 0..28 * upscale {
                    let p = (out_vec[r * 28 * upscale + c] * 255.0) as u8;
                    canvas.set_draw_color(Color {
                        r: p,
                        g: p,
                        b: p,
                        a: p,
                    });
                    canvas.fill_rect(Rect::new(
                        (c) as i32 * SCALE as i32,
                        (r) as i32 * SCALE as i32,
                        SCALE,
                        SCALE,
                    ))?;
                }
            }
        }
        canvas.present();
        i += 1;
        for event in sdl_context.event_pump()?.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Option::Some(Keycode::Escape),
                    ..
                } => {
                    //model.save(model_path).unwrap();
                    break 'mainloop;
                }
                Event::KeyDown {
                    keycode: Option::Some(Keycode::M),
                    ..
                } => {
                    latent_value += 1.;
                }
                Event::KeyDown {
                    keycode: Option::Some(Keycode::N),
                    ..
                } => {
                    latent_value -= 1.;
                }
                _ => {}
            }
        }
    }

    Ok(())
}

fn fetch_mnist(
    batch_size: usize,
    shuffle: bool,
) -> (
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
    Vec<Vec<f32>>,
) {
    use num_traits::FromPrimitive;
    use mnist::Mnist;
    let mnist = Mnist::from_download().expect("mnist download failed");
    let Mnist {
        train_images,
        train_labels,
        test_images,
        test_labels,
    } = mnist;
    let mut rng = thread_rng();

    // batching train
    let mut shuffle_idx: Vec<usize> = (0..60000).collect();
    if shuffle {
        shuffle_idx.shuffle(&mut rng);
    }
    let mut train_img_batched: Vec<Vec<f32>> = Vec::with_capacity(60000 * 28 * 28);
    let mut train_lbl_batched: Vec<Vec<f32>> = Vec::with_capacity(60000 * 10);
    let mut tain_img_in_one_batch = Vec::with_capacity(batch_size);
    let mut train_lbl_in_one_batch = Vec::with_capacity(batch_size);
    for i in 0..60000 {
        for ii in 0..28 * 28 {
            tain_img_in_one_batch
                .push(f32::from_u8(train_images[(shuffle_idx[i] * (28 * 28)) + ii]).unwrap());
        }
        train_lbl_in_one_batch.push(f32::from_u8(train_labels[shuffle_idx[i]]).unwrap());
        if (i + 1) % batch_size == 0 {
            train_img_batched.push(tain_img_in_one_batch.drain(..).collect::<Vec<f32>>());
            train_lbl_batched.push(train_lbl_in_one_batch.drain(..).collect::<Vec<f32>>());
        }
    }

    // batching test
    let mut shuffle_idx: Vec<usize> = (0..10000).collect();
    if shuffle {
        shuffle_idx.shuffle(&mut rng);
    }
    let mut test_img_batched: Vec<Vec<f32>> = Vec::with_capacity(10000 * 28 * 28);
    let mut test_lbl_batched: Vec<Vec<f32>> = Vec::with_capacity(10000 * 10);
    let mut test_img_in_one_batch = Vec::with_capacity(batch_size);
    let mut test_lbl_in_one_batch = Vec::with_capacity(batch_size);
    for i in 0..10000 {
        for ii in 0..28 * 28 {
            test_img_in_one_batch
                .push(f32::from_u8(test_images[(shuffle_idx[i] * (28 * 28)) + ii]).unwrap());
        }
        test_lbl_in_one_batch.push(f32::from_u8(test_labels[shuffle_idx[i]]).unwrap());
        if (i + 1) % batch_size == 0 {
            test_img_batched.push(test_img_in_one_batch.drain(..).collect::<Vec<f32>>());
            test_lbl_batched.push(test_lbl_in_one_batch.drain(..).collect::<Vec<f32>>());
        }
    }
    (
        train_img_batched,
        train_lbl_batched,
        test_img_batched,
        test_lbl_batched,
    )
}
