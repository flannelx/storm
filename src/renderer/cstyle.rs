use crate::arg::Arg;
use crate::codegen::kernel::Buffers;
use crate::{
    codegen::linearizer::{UOp, UOps},
    dtype,
    lazy::LazyBuffer,
    ops::{Binary, Op, OpType},
};
use std::{collections::HashMap, fmt::Display};

#[derive(Debug, Default, Clone)]
pub struct CstyleLanguage {
    pub size_prefix: String,
    pub generic_var_prefix: Option<String>,
    pub kernel_prefix: String,
    pub buffer_prefix: String,
    pub buffer_suffix: String,
    pub smem_prefix: String,
    pub smem_align: String,
    pub arg_int_prefix: String,
    pub barrier: String,
    pub gid: Vec<String>,
    pub lid: Vec<String>,
    pub global_max: Vec<isize>,
    pub local_max: Vec<isize>,
    pub extra_args: Vec<String>,
    pub float4: Option<String>,
    pub half_prekernel: Option<String>,
    pub uses_vload: bool,
    pub external_local_bufs: bool,
    pub uses_ptr_arithmetic: bool,
    pub launch_bounds: bool,
}

impl Op for CstyleLanguage { }

pub struct FloatInt {
    pub float: f64,
    pub int: isize,
}

impl CstyleLanguage {
    pub fn render_cast(&self, x: &[&str], var_dtype: dtype::Dtype) -> String {
        assert!(x.len() == var_dtype.sz);
        assert!(self.float4.is_some());
        if var_dtype == dtype::_float4 {
            return format!(
                "{}({})",
                self.float4.as_ref().unwrap(),
                x.join(",").to_string()
            );
        }
        if var_dtype == dtype::_float2 {
            return format!(
                "{}({})",
                self.float4.as_ref().unwrap().replace("float4", "float2"),
                x.join(",").to_string()
            );
        }
        if var_dtype == dtype::_int2 {
            return format!(
                "{}({})",
                self.float4.as_ref().unwrap().replace("float4", "int2"),
                x.join(",").to_string()
            );
        }
        unimplemented!("no cast for {}", var_dtype)
    }

    pub fn render_const(&self, x: FloatInt, var_dtype: dtype::Dtype) -> String {
        let val = if var_dtype.is_float() {
            if x.float.is_nan() {
                "NAN".to_string()
            } else if x.float.is_infinite() {
                if x.float < 0.0 {
                    "-INFINITY".to_string()
                } else {
                    "INFINITY".to_string()
                }
            } else {
                x.float.to_string() + "f"
            }
        } else {
            x.int.to_string()
        };
        if var_dtype.sz > 1 {
            val
        } else {
            self.render_cast(&vec![val.as_str(); var_dtype.sz], var_dtype)
        }
    }

    pub fn render_load(
        &self,
        output_dtype: dtype::Dtype,
        buf_name: &str,
        buf_dtype: dtype::Dtype,
        idx: &str,
        local: bool,
    ) -> String {
        if buf_dtype.shape.is_some() {
            assert!(output_dtype == dtype::_float4, "images nust be float4");
            return format!("read_imagef({buf_name}, smp, {idx})");
        }
        if self.uses_vload && buf_dtype == dtype::float16 {
            return format!(
                "vload_half({})",
                if output_dtype.sz == 1 {
                    "".to_string()
                } else {
                    output_dtype.sz.to_string()
                } + &format!("(0, {buf_name}+{idx})").to_string()
            );
        }
        let cast = if output_dtype != buf_dtype {
            format!("({})", output_dtype.c_name)
        } else {
            "".to_string()
        };
        if output_dtype.sz > 1 {
            return format!(
                "{cast}(*(({}{}{}*)({buf_name}+{idx})))",
                if local {
                    &self.smem_prefix
                } else {
                    &self.buffer_prefix
                },
                buf_dtype.c_name,
                output_dtype.sz
            );
        }
        if self.uses_ptr_arithmetic {
            return format!("{cast}(*({buf_name}+{idx}))");
        }
        format!("{cast}({buf_name}[{idx}])")
    }

    pub fn render_local(&self, name: &str, size: usize) -> String {
        self.smem_prefix.clone() + &format!("float {name}[{size}]")
    }

    pub fn render_for(&self, expr: &str, min: impl Display, max: impl Display) -> String {
        format!("for (int {expr} = {min}; {expr} <= {max}; ++{expr}) {{")
    }

    pub fn render_conditional(&self, cond: &str, x: &str, y: &str) -> String {
        format!("({cond})?({x}):{y}")
    }

    pub fn render_kernel(
        &self,
        function_name: &str,
        kernel: &[String],
        bufs: &[Buffers],
        local_size: &[usize],
        _prekernel: &[String],
    ) -> String {
        let tmp = if bufs.iter().any(|b| b.dtype().shape.is_some()) {
            "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
        } else {
            ""
        };
        let mut buftypes = vec![];
        for (i, buffer) in bufs.iter().enumerate() {
            let s = if buffer.dtype().c_name.starts_with("image") {
                format!(
                    "{} image2d_t",
                    if 1 > 0 { "read_only" } else { "write_only" }
                )
            } else {
                if buffer.dtype() == dtype::_arg_int32 {
                    self.arg_int_prefix.to_string()
                } else {
                    (if i > 0 {
                        "const ".to_string()
                    } else {
                        "".to_string()
                    }) + &self.buffer_prefix
                        + buffer.dtype().c_name
                        + "*"
                        + &self.buffer_suffix
                }
            };
            buftypes.push((buffer, s));
        }

        let prod_local_size = local_size.iter().product::<usize>();
        let mut prg = {
            format!(
                "{}void {}{function_name}(",
                self.kernel_prefix,
                if self.launch_bounds {
                    format!("__launch_bounds__ ({prod_local_size}, 1)")
                } else {
                    "".to_string()
                }
            )
        };

        let mut args = buftypes
            .iter()
            .map(|(name, t)| format!("{t} {name:?}"))
            .collect::<Vec<String>>();
        args.extend(self.extra_args.clone());
        prg += &args.join(", ");

        prg += &format!("{}{}{}{}", ") {\n", tmp, kernel.join("\n"), "\n");

        if self.half_prekernel.is_some() && bufs.iter().any(|buffer| buffer.dtype() == dtype::float16) {
            prg = self.half_prekernel.as_ref().unwrap().clone() + "\n" + &prg;
        }

        prg
    }

    pub fn render_store(
        &self,
        buf_name: &str,
        buf_dtype: dtype::Dtype,
        var_name: &str,
        var_dtype: dtype::Dtype,
        idx: &str,
        local: bool,
    ) -> String {
        if buf_dtype.shape.is_some() {
            assert!(var_dtype == dtype::_float4);
            return format!("write_imagef({buf_name}, {idx}, {var_name});");
        }
        if self.uses_vload && buf_dtype == dtype::float16 {
            return format!(
                "vstore_half{}({var_name}, 0, {buf_name}+{idx});",
                if var_dtype.sz == 1 {
                    "".to_string()
                } else {
                    var_dtype.sz.to_string()
                }
            );
        }
        if var_dtype.sz > 1 {
            return format!(
                "*(({}{}{}*)({buf_name}+{idx})) = ({}{}){var_name};",
                if local {
                    &self.smem_prefix
                } else {
                    &self.buffer_prefix
                },
                buf_dtype.c_name,
                var_dtype.sz,
                buf_dtype.c_name,
                var_dtype.sz
            );
        }
        if self.uses_ptr_arithmetic {
            format!("*({buf_name}+{idx}) = {var_name};")
        } else {
            format!("{buf_name}[{idx}] = {var_name};")
        }
    }
}

pub fn uops_to_cstyle(lang: CstyleLanguage, function_name: &str, uops: &[UOp]) -> String {
    let mut local_size: Vec<usize> = vec![];
    let mut kernel: Vec<String> = vec![];
    let mut prekernel: Vec<String> = vec![];
    let mut bufs = vec![];
    let mut depth: usize = 1;
    let kk = |s: &str, kernel: &mut Vec<String>, depth: usize| {
        kernel.push("  ".repeat(depth) + s);
    };
    let mut c: HashMap<&str, usize> = HashMap::new();
    let ssa = |u: &UOp,
               mut prefix: &'static str,
               c: &mut HashMap<&str, usize>,
               r: &mut HashMap<UOp, String>|
     -> String {
        if prefix == "" {
            prefix = "t";
        }
        *c.entry(prefix).or_default() += 1;
        r.insert(u.clone(), format!("{prefix}{}", c[prefix] - 1));
        r[u].clone()
    };
    let mut r: HashMap<UOp, String> = HashMap::new();
    let mut child_count: HashMap<&UOp, usize> = HashMap::new();
    for ru in uops.iter() {
        for v in ru.vin.iter() {
            *child_count.entry(v).or_default() += 1;
        }
    }
    for u in uops.iter() {
        let (uop, dtype, vin, args) = (&u.uop, &u.dtype, &u.vin, u.args.clone());
        match uop {
            UOps::LOOP => {
                *r.get_mut(&u).unwrap() = ssa(u, "ridx", &mut c, &mut r);
                kk(
                    &lang.render_for(&r[&u], &r[&vin[0]], &r[&vin[1]]),
                    &mut kernel,
                    depth,
                );
                depth += 1;
            }
            UOps::BARRIER => kk(&lang.barrier, &mut kernel, depth),
            UOps::END => {
                depth -= 1;
                kk("}", &mut kernel, depth);
            }
            UOps::WMMA => {
                if &args[0] == "METAL" {
                    todo!();
                } else if &args[0] == "HIP" {
                    todo!();
                } else {
                    unimplemented!("WMMA not implemented for {args:?}")
                }
            }
            UOps::ALU => {
                assert!(dtype.is_some());
                let dtype = dtype.as_ref().unwrap();
                #[allow(unused_assignments)] // it is used: r.insert(u.clone(), val);
                let mut val = String::new();
                if matches!(vin[0].uop, UOps::ALU)
                    && vin[0].args == args
                    && (args[0] == OpType::Binary(Binary::Add)
                        || args[0] == OpType::Binary(Binary::Sub)
                        || args[0] == OpType::Binary(Binary::Mul))
                {
                    let a = r[&vin[0]].clone().replace("(", "").replace(")", "");
                    let b = r[&vin[1]].clone();
                    val = match &args[0] {
                        Arg::OpType(op) => lang.call(&op, vec![a, b], None),
                        _ => unreachable!(),
                    }
                } else {
                    let a = r[&vin[0]].clone();
                    let b = r[&vin[1]].clone();
                    let c = if vin.len() > 2 {
                        r[&vin[2]].clone()
                    } else {
                        "".to_string()
                    };
                    val = match &args[0] {
                        Arg::OpType(op) => lang.call(&op, vec![a, b, c], None),
                        _ => unreachable!(),
                    }
                }
                assert!(child_count[&u] != 0);
                if child_count[&u] <= 1 || dtype.is_int() {
                    r.insert(u.clone(), val);
                } else {
                    kk(
                        &format!(
                            "{} {} = {val}",
                            if lang.generic_var_prefix.is_some() {
                                lang.generic_var_prefix.as_ref().unwrap().as_str()
                            } else {
                                dtype.c_name
                            },
                            ssa(u, "alu", &mut c, &mut r),
                        ),
                        &mut kernel,
                        depth,
                    );
                }
            }
            UOps::DEFINE_ACC => {
                assert!(dtype.is_some());
                let dtype = dtype.as_ref().unwrap();
                let s = match &args[0] {
                    Arg::Str(s) => s,
                    _ => panic!(),
                };
                let mut fint = FloatInt { float: 0.0, int: 0 };
                if s.contains(".") {
                    fint.float = s.parse::<f64>().unwrap();
                } else {
                    fint.int = s.parse::<isize>().unwrap();
                }
                kk(
                    &format!(
                        "{} {} = {}",
                        if lang.generic_var_prefix.is_some() {
                            lang.generic_var_prefix.as_ref().unwrap().as_str()
                        } else {
                            dtype.c_name
                        },
                        ssa(u, "acc", &mut c, &mut r),
                        lang.render_const(fint, dtype.clone())
                    ),
                    &mut kernel,
                    depth,
                );
            }
            UOps::SPECIAL => {
                let args = args
                    .iter()
                    .map(|a| match a {
                        Arg::Str(v) => v.clone(),
                        _ => panic!(),
                    })
                    .collect::<Vec<String>>();
                let xid = if args[1].starts_with("g") {
                    &lang.gid
                } else {
                    &lang.lid
                };
                kk(
                    &format!(
                        "{} {} = {}; /* {} */",
                        lang.size_prefix,
                        args[1],
                        xid[args[0].parse::<usize>().unwrap()],
                        args[2]
                    ),
                    &mut kernel,
                    depth,
                );
                if args[1].starts_with("l") {
                    local_size.push(args[2].parse::<usize>().unwrap())
                }
                r.insert(u.clone(), args[1].clone());
            }
            UOps::CONST => {
                // r[u] = lang.render_const(args, dtype) if args >= 0 else f"({lang.render_const(args, dtype)})"
                // Huh?????????????????????????
                assert!(args.len() >= 1);
                let s = match &args[0] {
                    Arg::Str(s) => s,
                    _ => panic!(),
                };
                let mut fint = FloatInt { float: 0.0, int: 0 };
                if s.contains(".") {
                    fint.float = s.parse::<f64>().unwrap();
                } else {
                    fint.int = s.parse::<isize>().unwrap();
                }
                r.insert(
                    u.clone(),
                    lang.render_const(fint, dtype.as_ref().unwrap().clone()),
                );
            }
            UOps::LOAD => {
                assert!(dtype.is_some());
                let mut val = lang.render_load(
                    dtype.as_ref().unwrap().clone(),
                    &r[&vin[0]],
                    vin[0].dtype.as_ref().unwrap().clone(),
                    &r[&vin[1]].replace("(", "").replace(")", ""),
                    matches!(vin[0].uop, UOps::DEFINE_LOCAL),
                );
                if vin.len() > 2 {
                    val = lang.render_conditional(&r[&vin[2]], &val, &r[&vin[3]])
                }
                kk(
                    &format!(
                        "{} {} = {val};",
                        if lang.generic_var_prefix.is_some() {
                            lang.generic_var_prefix.as_ref().unwrap()
                        } else {
                            dtype.as_ref().unwrap().c_name
                        },
                        ssa(u, "val", &mut c, &mut r),
                    ),
                    &mut kernel,
                    depth,
                );
            }
            UOps::STORE => {
                if vin.len() == 2 {
                    kk(
                        &format!("{} = {};", r[&vin[0]], r[&vin[1]]),
                        &mut kernel,
                        depth,
                    );
                } else if vin.len() == 3 {
                    assert!(vin[0].dtype.is_some() && vin[2].dtype.is_some());
                    kk(
                        &lang.render_store(
                            &r[&vin[0]],
                            vin[0].dtype.as_ref().unwrap().clone(),
                            &r[&vin[2]],
                            vin[2].dtype.as_ref().unwrap().clone(),
                            &r[&vin[1]].replace("(", "").replace(")", ""),
                            matches!(vin[0].uop, UOps::DEFINE_LOCAL),
                        ),
                        &mut kernel,
                        depth,
                    );
                }
            }
            UOps::CAST => {
                assert!(dtype.is_some() && dtype.as_ref().unwrap().sz > 1);
                let val = lang.render_cast(
                    &vin.iter().map(|x| r[x].as_str()).collect::<Vec<&str>>(),
                    dtype.as_ref().unwrap().clone(),
                );
                if child_count[u] <= 1 {
                    r.insert(u.clone(), val);
                } else {
                    kk(
                        &format!(
                            "{} {} = {val}",
                            if lang.generic_var_prefix.is_some() {
                                lang.generic_var_prefix.as_ref().unwrap()
                            } else {
                                dtype.as_ref().unwrap().c_name
                            },
                            ssa(u, "cast", &mut c, &mut r)
                        ),
                        &mut kernel,
                        depth,
                    );
                }
            }
            UOps::DEFINE_LOCAL => {
                if lang.external_local_bufs {
                    prekernel.push(lang.render_local(
                        &args[0].to_str(),
                        args[1].to_str().parse::<usize>().unwrap(),
                    ))
                } else {
                    kk(
                        &lang.render_local(
                            &args[0].to_str(),
                            args[1].to_str().parse::<usize>().unwrap(),
                        ),
                        &mut kernel,
                        depth,
                    );
                }
                r.insert(u.clone(), args[0].to_str());
            }
            UOps::DEFINE_GLOBAL => {
                bufs = args.iter().map(|args| args.to_buf()).collect::<Vec<_>>();
                r.insert(u.clone(), args[0].to_str());
            }
            UOps::GEP => {
                r.insert(
                    u.clone(),
                    format!(
                        "({}).{}",
                        r[&vin[0]],
                        "xyzw"
                            .chars()
                            .nth(args[0].to_str().parse::<usize>().unwrap())
                            .unwrap()
                    ),
                );
            }
            UOps::PHI => todo!(),
        }
    }
    lang.render_kernel(function_name, &kernel, &bufs, &local_size, &prekernel)
}
