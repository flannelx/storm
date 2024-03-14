use wgpu::{BindGroupLayoutEntry, ShaderModuleDescriptor};

use crate::{
    ops::{Dtype, Op, DEVICE},
    prelude::*,
    renderer::cstyle::LanguageOpts,
};
use std::{collections::HashMap, num::NonZeroU32, sync::Arc};

use super::{Buffer, Device};

#[derive(Debug)]
pub struct WGPUBuffer {
    buffer: *mut wgpu::Buffer,
    dtype: Dtype,
}

impl Buffer for WGPUBuffer {
    fn device(&self) -> String {
        "WGPU".into()
    }

    fn ptr(&self) -> *mut core::ffi::c_void {
        self.buffer as _
    }

    fn dtype(&self) -> crate::ops::Dtype {
        self.dtype.clone()
    }

    fn bytesize(&self) -> usize {
        unsafe { (*self.buffer).size() as _ }
    }

    fn to_cpu(&self) -> Vec<u8> {
        let mut dst = vec![0u8; self.bytesize()];
        let ptr = dst.as_mut_ptr() as *mut u8;
        DEVICE.copyout(self, ptr);
        DEVICE.synchronize();
        dst
    }
}

impl Drop for WGPUBuffer {
    fn drop(&mut self) {
        ALLOCTOR.0.free(&*self);
    }
}

pub struct DeviceWrapper {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

#[derive(Debug)]
pub struct WGPUDevice {
    device: *mut DeviceWrapper,
}
unsafe impl Send for WGPUDevice {}
unsafe impl Sync for WGPUDevice {}

impl Drop for WGPUDevice {
    fn drop(&mut self) {
        unsafe { self.device.drop_in_place() }
    }
}

async fn new() -> anyhow::Result<WGPUDevice> {
    let instance = wgpu::Instance::default();
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions::default())
        .await
        .unwrap();
    let mut limits = wgpu::Limits::default();
    limits.max_buffer_size = 1 << 30;
    limits.max_storage_buffer_binding_size = 1 << 30;
    limits.max_storage_buffers_per_shader_stage = 100;
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                features: wgpu::Features::empty(),
                limits,
            },
            None,
        )
        .await
        .unwrap();
    Ok(WGPUDevice {
        device: Box::into_raw(Box::new(DeviceWrapper { device, queue })),
    })
}

impl WGPUDevice {
    pub fn new() -> anyhow::Result<Arc<dyn Device>> {
        Ok(Arc::new(pollster::block_on(new())?))
    }
}

impl Device for WGPUDevice {
    fn device_ptr(&self) -> *mut std::ffi::c_void {
        self.device as _
    }

    fn name(&self) -> String {
        "WGPU".into()
    }

    fn _alloc(
        &self,
        size: usize,
        dtype: crate::ops::Dtype,
    ) -> anyhow::Result<std::sync::Arc<dyn super::Buffer>> {
        unsafe {
            let device = &*(self.device as *mut wgpu::Device);
            Ok(Arc::new(WGPUBuffer {
                buffer: Box::into_raw(Box::new(device.create_buffer(&wgpu::BufferDescriptor {
                    label: None,
                    size: (size * dtype.size) as _,
                    usage: wgpu::BufferUsages::COPY_SRC
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::STORAGE,
                    mapped_at_creation: false,
                }))),
                dtype,
            }))
        }
    }

    fn buf_from_mem_ptr(
        &self,
        size: usize,
        dtype: crate::ops::Dtype,
        mem: *mut std::ffi::c_void,
    ) -> std::sync::Arc<dyn super::Buffer> {
        unsafe {
            Arc::new(WGPUBuffer {
                buffer: mem as *mut _,
                dtype,
            })
        }
    }

    fn build(&self, name: &str, program: &str) -> std::sync::Arc<dyn super::Program> {
        unsafe {
            let device = &*(self.device as *mut wgpu::Device);
            Arc::new(WGPUProgram {
                name: name.into(),
                program: device.create_shader_module(ShaderModuleDescriptor {
                    label: None,
                    source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(program)),
                }),
            })
        }
    }

    fn copyout(&self, src: &dyn super::Buffer, dst: *mut u8) {
        unsafe {
            let wrapper = &*(self.device as *mut DeviceWrapper);
            let staging_buf = wrapper.device.create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size: src.bytesize() as _,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let mut encoder = wrapper
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            unsafe {
                encoder.copy_buffer_to_buffer(
                    &*(src.ptr() as *mut wgpu::Buffer),
                    0,
                    &staging_buf,
                    0,
                    src.bytesize() as _,
                );
                wrapper.queue.submit(Some(encoder.finish()));
                staging_buf
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, |result| {
                        result.unwrap();
                    });
                self.synchronize();
                let r = staging_buf.slice(..).get_mapped_range();
                let mut ret = r.get(..).unwrap().to_vec();
                drop(r);
                staging_buf.unmap();
                std::ptr::copy_nonoverlapping(ret.as_mut_ptr(), dst, src.bytesize());
            }
        }
    }

    fn copyin(&self, src: Vec<u8>, dst: &dyn super::Buffer) {
        unsafe {
            (*self.device)
                .queue
                .write_buffer(&(*(dst.ptr() as *mut _)), 0, &src)
        }
    }

    fn synchronize(&self) {
        unsafe {
            (*(self.device as *mut wgpu::Device)).poll(wgpu::Maintain::Wait);
        }
    }

    fn renderer(&self) -> std::sync::Arc<dyn crate::renderer::cstyle::Renderer> {
        Arc::new(WGSLRenderer::default())
    }

    fn free(&self, ptr: *mut std::ffi::c_void) {
        unsafe {
            (*(ptr as *mut wgpu::Buffer)).destroy();
            ptr.drop_in_place();
        }
    }
}

#[derive(Debug)]
pub struct WGSLRenderer {
    opts: Arc<LanguageOpts>,
}

impl Default for WGSLRenderer {
    fn default() -> Self {
        Self {
            opts: Arc::new(LanguageOpts {
                size_prefix: "let".into(),
                barrier: "workgroupBarrier();".into(),
                generic_var_prefix: Some("var".into()),
                external_local_bufs: true,
                code_for_workitem: HashMap::from([
                    (
                        "g".into(),
                        (0..3)
                            .map(|i| format!("i32(global_idx.{})", ('x' as u8 + i) as char))
                            .collect(),
                    ),
                    (
                        "l".into(),
                        (0..3)
                            .map(|i| format!("i32(local_idx.{})", ('x' as u8 + i) as char))
                            .collect(),
                    ),
                ]),
                uses_vload: true,
                type_map: HashMap::from([
                    (float32, "f32".into()),
                    (float16, "f16".into()),
                    (int32, "i32".into()),
                    (uint32, "u32".into()),
                ]),
                ..Default::default()
            }),
        }
    }
}

impl Op for WGSLRenderer {
    fn cmplt(&self, a: &str, b: &str) -> String {
        format!("f32({a}<{b})")
    }

    fn mulacc(&self, a: &str, b: &str, c: &str) -> String {
        format!("fma({a},{b},{c})")
    }

    fn _where(&self, a: &str, b: &str, c: &str) -> String {
        format!("select({c},{b},bool({a}))")
    }

    fn sub(&self, a: &str, b: &str) -> String {
        format!("({a}-({b}))")
    }
}

impl crate::renderer::cstyle::Renderer for WGSLRenderer {
    fn lang_opts(&self) -> Arc<LanguageOpts> {
        self.opts.clone()
    }

    fn render_local(&self, name: &str, size: usize, dtype: Dtype) -> String {
        format!("var<workgroup> {name}: array<{},{size}>;", dtype.type_name)
    }

    fn render_const(
        &self,
        x: crate::renderer::cstyle::FloatInt,
        var_dtype: dtype::Dtype,
    ) -> String {
        let val = if var_dtype.is_float() {
            if x.float.is_nan() {
                "nan()".to_string()
            } else if x.float.is_infinite() {
                if x.float < 0.0 {
                    "-inf(1.0)".to_string()
                } else {
                    "inf(1.0)".to_string()
                }
            } else {
                if x.float < 0.0 {
                    format!("({:?}f)", x.float)
                } else {
                    format!("{:?}f", x.float)
                }
            }
        } else {
            if x.int < 0 {
                "(".to_string() + &x.int.to_string() + &")"
            } else {
                x.int.to_string()
            }
        };
        val
    }

    fn render_if(&self, cond: &str) -> String {
        format!("if (bool({cond})) {{")
    }

    fn render_conditional(&self, cond: &str, x: &str, y: &str) -> String {
        format!("select({y},{x},bool({cond}))")
    }

    fn render_for(&self, expr: &str, min: &str, max: &str) -> String {
        format!("for (var {expr} = {min}; {expr} < {max}; {expr}++) {{")
    }

    fn render_cast(&self, x: &[&str], var_dtype: dtype::Dtype) -> String {
        let lang = self.lang_opts();
        // if bitcast {
        //     format!("bitcast<{}>({x[0]})", self.type_map[var_dtype])
        // } else {
            format!("{}({})", lang.type_map[&var_dtype], x[0])
        //}
    }

    fn render_kernel(
        &self,
        function_name: &str,
        kernel: &[String],
        bufs: &[(String, dtype::Dtype)],
        local_size: &[usize],
        uops: &[crate::codegen::linearizer::UOp],
        prekernel: &[String],
    ) -> String {
        let local_size = if local_size.len() > 0 {
            v![*i, for i in local_size.iter().rev()]
        } else {
            vec![1]
        };
        let bind = 0..bufs.len();
        let mut prg = "fn nan() -> f32 { let bits = 0xffffffffu; return bitcast<f32>(bits); }\nfn inf(a: f32) -> f32 { return a/0.0; }\n".to_string();
        prg += &prekernel.join("\n");
        for (i, (name, dtype)) in bufs.iter().enumerate() {
            prg += &format!(
                "@group(0) @binding({i}) var<storage, read_write> {name}: array<{}>;\n",
                dtype.type_name
            );
        }
        prg += &format!("\n@compute\n@workgroup_size({})\nfn {function_name}(@builtin(workgroup_id) global_idx: vec3<u32>, @builtin(local_invocation_id) local_idx: vec3<u32>) {{\n", v![x.to_string(), for x in local_size].join(","));
        prg += &kernel.join("\n");
        prg += "\n}";
        prg
    }
}

#[derive(Debug)]
pub struct WGPUProgram {
    name: String,
    program: wgpu::ShaderModule,
}

impl Program for WGPUProgram {
    fn run(
        &self,
        bufs: &[Arc<dyn Buffer>],
        global_size: &[usize],
        local_size: Option<&[usize]>,
        args: &[isize],
        extra: &[String],
    ) {
        unsafe {
            let wrapper = &*(DEVICE.device_ptr() as *mut DeviceWrapper);
            let bind_group_layout =
                wrapper
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: None,
                        entries: &v![BindGroupLayoutEntry {
                    binding: i as u32,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }, for i in 0..bufs.len()],
                    });

            let bind_group = wrapper
                .device
                .create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &bind_group_layout,
                    entries: &v![wgpu::BindGroupEntry {
                    binding: i as u32,
                    resource: (*(b.ptr() as *mut wgpu::Buffer)).as_entire_binding()
                }, for (i, b) in bufs.iter().enumerate()],
                });
            let pipeline_layout =
                wrapper
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: None,
                        bind_group_layouts: &[&bind_group_layout],
                        push_constant_ranges: &[],
                    });
            let compute_pipeline =
                wrapper
                    .device
                    .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                        label: None,
                        layout: Some(&pipeline_layout),
                        module: &self.program,
                        entry_point: &self.name,
                    });
            let mut encoder = wrapper
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: None,
                    timestamp_writes: None,
                });
                cpass.set_pipeline(&compute_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.dispatch_workgroups(
                    global_size[0] as _,
                    global_size[1] as _,
                    global_size[2] as _,
                );
            }
            wrapper.queue.submit(Some(encoder.finish()));
        }
    }
}
