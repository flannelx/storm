use dtype::least_upper_dtype;

use crate::arg::Arg;
use crate::codegen::kernel::Buffers;
use crate::v;
use crate::{
    codegen::linearizer::{UOp, UOps},
    dtype,
    lazy::LazyBuffer,
    ops::{Binary, Op, OpType},
};
use std::sync::Arc;
use std::{collections::HashMap, fmt::Display};

#[derive(Debug, Clone)]
pub struct LanguageOpts {
    pub size_prefix: String,
    pub generic_var_prefix: Option<String>,
    pub kernel_prefix: String,
    pub buffer_prefix: String,
    pub buffer_suffix: String,
    pub smem_prefix: String,
    pub smem_align: String,
    pub arg_int_prefix: String,
    pub barrier: String,
    pub global_max: Vec<isize>,
    pub local_max: Vec<isize>,
    pub extra_args: Vec<String>,
    pub float4: Option<String>,
    pub half_prekernel: Option<String>,
    pub uses_vload: bool,
    pub external_local_bufs: bool,
    pub uses_ptr_arithmetic: bool,
    pub launch_bounds: bool,
    pub code_for_workitem: HashMap<&'static str,Vec<String>>,
    pub type_map: HashMap<String, String>
}

impl Default for LanguageOpts {
    fn default() -> Self {
        Self {
            size_prefix: "int".into(),
            generic_var_prefix: Default::default(),
            kernel_prefix: Default::default(),
            buffer_prefix: Default::default(),
            buffer_suffix: Default::default(),
            smem_prefix: Default::default(),
            smem_align: Default::default(),
            arg_int_prefix: "const int".into(),
            barrier: Default::default(),
            global_max: Default::default(),
            local_max: Default::default(),
            extra_args: Default::default(),
            float4: Default::default(),
            half_prekernel: Default::default(),
            uses_vload: Default::default(),
            external_local_bufs: Default::default(),
            uses_ptr_arithmetic: Default::default(),
            launch_bounds: Default::default(),
            code_for_workitem: Default::default(),
            type_map: Default::default(),
        }
    }
}

impl Op for LanguageOpts {}

pub trait Renderer: 'static + Send + Sync + Op {
    fn lang_opts(&self) -> Arc<LanguageOpts>;

    fn render_cast(&self, x: &[&str], var_dtype: dtype::Dtype) -> String {
        assert!(x.len() == var_dtype.sz);
        assert!(self.lang_opts().float4.is_some());
        if x.len() == 1 {
            return format!("({})({})", var_dtype.c_name, x[0]);
        }
        if var_dtype == dtype::_float4 {
            return format!(
                "{}({})",
                self.lang_opts().float4.as_ref().unwrap(),
                x.join(",").to_string()
            );
        }
        if var_dtype == dtype::_float2 {
            return format!(
                "{}({})",
                self.lang_opts()
                    .float4
                    .as_ref()
                    .unwrap()
                    .replace("float4", "float2"),
                x.join(",").to_string()
            );
        }
        if var_dtype == dtype::_int2 {
            return format!(
                "{}({})",
                self.lang_opts()
                    .float4
                    .as_ref()
                    .unwrap()
                    .replace("float4", "int2"),
                x.join(",").to_string()
            );
        }
        unimplemented!("no cast for {}", var_dtype)
    }
    //
    fn render_const(&self, x: FloatInt, var_dtype: dtype::Dtype) -> String {
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
        // if var_dtype.sz > 1 || var_dtype != dtype::float32 || var_dtype != dtype::int32 {
        //     val
        // } else {
        //     self.render_cast(&vec![val.as_str(); var_dtype.sz], var_dtype)
        // }
    }

    fn render_load(
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
        // if self.lang_opts().uses_vload && buf_dtype == dtype::float16 {
        //     return format!(
        //         "vload_half({})",
        //         if output_dtype.sz == 1 {
        //             "".to_string()
        //         } else {
        //             output_dtype.sz.to_string()
        //         } + &format!("(0, {buf_name}+{idx})").to_string()
        //     );
        // }
        if output_dtype.sz > 1 {
            return format!(
                "*(({}{}{}*)({buf_name}+{idx}))",
                if local {
                    self.lang_opts().smem_prefix.clone()
                } else {
                    self.lang_opts().buffer_prefix.clone()
                },
                buf_dtype.c_name,
                output_dtype.sz
            );
        }
        if self.lang_opts().uses_ptr_arithmetic {
            return format!("*({buf_name}+{idx})");
        }
        let ret = format!("{buf_name}[{idx}]");
        if output_dtype != buf_dtype {
            self.render_cast(&[&ret], output_dtype)
        } else {
            ret
        }
    }

    fn render_local(&self, name: &str, size: usize) -> String {
        self.lang_opts().smem_prefix.clone() + &format!("float {name}[{size}];")
    }

    fn render_for(&self, expr: &str, min: &str, max: &str) -> String {
        format!("for (int {expr} = {min}; {expr} < {max}; {expr}++) {{")
    }

    fn render_conditional(&self, cond: &str, x: &str, y: &str) -> String {
        format!("({cond})?({x}):{y}")
    }

    fn render_kernel(
        &self,
        function_name: &str,
        kernel: &[String],
        bufs: &[(String, dtype::Dtype)],
        local_size: &[usize],
        uops: &[UOp],
        prekernel: &[String],
    ) -> String {
        let tmp = if bufs.iter().any(|(_, dt)| dt.shape.is_some()) {
            "const sampler_t smp = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;\n"
        } else {
            ""
        };
        let mut buftypes = vec![];
        for (i, (name, dtype)) in bufs.iter().enumerate() {
            let s = if dtype.type_name.starts_with("image") {
                format!(
                    "{} image2d_t",
                    if 1 > 0 { "read_only" } else { "write_only" }
                )
            } else {
                if dtype == &dtype::_arg_int32 {
                    self.lang_opts().arg_int_prefix.to_string()
                } else {
                    (if i > 0 {
                        "const ".to_string()
                    } else {
                        "".to_string()
                    }) + &self.lang_opts().buffer_prefix
                        + dtype.c_name
                        + "*"
                        + &self.lang_opts().buffer_suffix
                }
            };
            buftypes.push((name, s));
        }

        let prod_local_size = local_size.iter().product::<usize>();
        let mut prg = {
            format!(
                "{}void {}{function_name}(",
                self.lang_opts().kernel_prefix,
                if self.lang_opts().launch_bounds {
                    format!("__launch_bounds__ ({prod_local_size}, 1)")
                } else {
                    "".to_string()
                }
            )
        };

        let mut args = buftypes
            .iter()
            .map(|(name, t)| format!("{t} {name}"))
            .collect::<Vec<String>>();
        args.extend(self.lang_opts().extra_args.clone());
        prg += &args.join(", ");

        prg += &format!("{}{}{}{}", ") {\n", tmp, kernel.join("\n"), "\n}");

        if self.lang_opts().half_prekernel.is_some()
            && bufs.iter().any(|(_, dtype)| *dtype == dtype::float16)
        {
            prg = self.lang_opts().half_prekernel.as_ref().unwrap().clone() + "\n" + &prg;
        }

        if prekernel.len() > 0 {
            format!("{}\n{}", prekernel.join("\n"), prg)
        } else {
            prg
        }
    }

    fn render_store(
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
        // if self.lang_opts().uses_vload && buf_dtype == dtype::float16 {
        //     return format!(
        //         "vstore_half{}({var_name}, 0, {buf_name}+{idx});",
        //         if var_dtype.sz == 1 {
        //             "".to_string()
        //         } else {
        //             var_dtype.sz.to_string()
        //         }
        //     );
        // }
        if var_dtype.sz > 1 {
            return format!(
                "*(({}{}{}*)({buf_name}+{idx})) = ({}{}){var_name};",
                if local {
                    self.lang_opts().smem_prefix.clone()
                } else {
                    self.lang_opts().buffer_prefix.clone()
                },
                buf_dtype.c_name,
                var_dtype.sz,
                buf_dtype.c_name,
                var_dtype.sz
            );
        }
        if self.lang_opts().uses_ptr_arithmetic {
            format!("*({buf_name}+{idx}) = {var_name};")
        } else {
            format!("{buf_name}[{idx}] = {var_name};")
        }
    }

    fn render_if(&self, cond: &str) -> String {
        format!("if ({cond}) {{")
    }
}

pub struct FloatInt {
    pub float: f64,
    pub int: isize,
}

pub fn uops_to_cstyle(lang: Arc<dyn Renderer>, function_name: &str, uops: &[UOp]) -> String {
    let mut local_size: Vec<usize> = vec![];
    let mut kernel: Vec<String> = vec![];
    let mut prekernel: Vec<String> = vec![];
    let mut bufs = vec![];
    let mut depth: usize = 1;
    let kk = |s: &str, kernel: &mut Vec<String>, depth: usize| {
        //if s == "float alu0 = (gidx0*4)" { panic!() };
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
                kk(
                    &lang.render_for(&ssa(u, "ridx", &mut c, &mut r), &r[&vin[0]], &r[&vin[1]]),
                    &mut kernel,
                    depth,
                );
                depth += 1;
            }
            UOps::BARRIER => kk(&lang.lang_opts().barrier, &mut kernel, depth),
            UOps::END => {
                depth -= 1;
                kk("}", &mut kernel, depth);
            }
            UOps::WMMA => {
                if &args[0].to_str() == "METAL" {
                    todo!();
                } else if &args[0].to_str() == "HIP" {
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
                    let a = strip_parens(&r[&vin[0]]);
                    val = match &args[0] {
                        Arg::OpType(op) => lang.call(
                            &op,
                            vec![vec![a], v![r[&x].clone(), for x in vin[1..].iter()]].concat(),
                            None,
                        ),
                        _ => unreachable!(),
                    }
                } else {
                    val = match &args[0] {
                        Arg::OpType(op) => {
                            lang.call(&op, v![r[&x].clone(), for x in vin.iter()], None)
                        }
                        _ => unreachable!(),
                    };
                    // if args[0] == OpType::Binary(Binary::Cmplt) {
                    //     val = format!("(float){val}");
                    // }
                    //println!(">>{val} {:?}", args[0]);
                }
                assert!(child_count[&u] != 0);
                if child_count[&u] <= 1 && args[0].to_op() != Binary::Max {
                    r.insert(u.clone(), val);
                } else {
                    kk(
                        &format!(
                            "{} {} = {val};",
                            if lang.lang_opts().generic_var_prefix.is_some() {
                                lang.lang_opts()
                                    .generic_var_prefix
                                    .as_ref()
                                    .unwrap()
                                    .to_string()
                            } else {
                                dtype.c_name.to_string()
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
                if dtype.is_float() {
                    fint.float = s.parse::<f64>().unwrap();
                } else {
                    fint.int = s.parse::<isize>().unwrap();
                }
                kk(
                    &format!(
                        "{} {} = {};",
                        if lang.lang_opts().generic_var_prefix.is_some() {
                            lang.lang_opts()
                                .generic_var_prefix
                                .as_ref()
                                .unwrap()
                                .to_string()
                        } else {
                            dtype.c_name.to_string()
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
                        Arg::Usize(u) => u.to_string(),
                        t => panic!("{t:?}"),
                    })
                    .collect::<Vec<String>>();
                kk(
                    &format!(
                        "{} {} = {}; /* {} */",
                        lang.lang_opts().size_prefix,
                        args[1],
                        lang.lang_opts().code_for_workitem[(args[1].as_bytes()[0] as char).to_string().as_str()][args[0].parse::<usize>().unwrap()],
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
                assert!(args.len() >= 1);
                match &args[0] {
                    Arg::Str(s) => {
                        let mut fint = FloatInt { float: 0.0, int: 0 };
                        if dtype.as_ref().unwrap().is_float() {
                            fint.float = s.parse::<f64>().unwrap();
                        } else {
                            fint.int = s.parse::<isize>().unwrap();
                        }
                        r.insert(
                            u.clone(),
                            lang.render_const(fint, dtype.as_ref().unwrap().clone()),
                        );
                    }
                    Arg::Idx(i) => {
                        r.insert(u.clone(), i.to_string());
                    }
                    t => panic!("{t:?}"),
                };
            }
            UOps::LOAD => {
                assert!(dtype.is_some());
                let mut val = lang.render_load(
                    dtype.as_ref().unwrap().clone(),
                    &r[&vin[0]],
                    vin[0].dtype.as_ref().unwrap().clone(),
                    &strip_parens(&r[&vin[1]]),
                    matches!(vin[0].uop, UOps::DEFINE_LOCAL),
                );
                if vin.len() > 3 {
                    val = lang.render_conditional(&r[&vin[2]], &val, &r[&vin[3]])
                }
                kk(
                    &format!(
                        "{} {} = {val};",
                        if lang.lang_opts().generic_var_prefix.is_some() {
                            lang.lang_opts()
                                .generic_var_prefix
                                .as_ref()
                                .unwrap()
                                .to_string()
                        } else {
                            dtype.as_ref().unwrap().c_name.to_string()
                        },
                        ssa(u, "val", &mut c, &mut r),
                    ),
                    &mut kernel,
                    depth,
                );
            }
            UOps::STORE => {
                // if vin.len() == 2 {
                //     kk(
                //         &format!("{} = {};", r[&vin[0]], r[&vin[1]]),
                //         &mut kernel,
                //         depth,
                //     );
                // } else if vin.len() == 3 {
                //     assert!(vin[0].dtype.is_some() && vin[2].dtype.is_some());
                //     kk(
                //         &lang.render_store(
                //             &r[&vin[0]],
                //             vin[0].dtype.as_ref().unwrap().clone(),
                //             &r[&vin[2]],
                //             vin[2].dtype.as_ref().unwrap().clone(),
                //             &r[&vin[1]].replace("(", "").replace(")", ""),
                //             matches!(vin[0].uop, UOps::DEFINE_LOCAL),
                //         ),
                //         &mut kernel,
                //         depth,
                //     );
                // }
                if vin.len() > 3 {
                    kk(&lang.render_if(&r[&vin[3]]), &mut kernel, depth)
                }
                kk(
                    &lang.render_store(
                        &r[&vin[0]],
                        vin[0].dtype.as_ref().unwrap().clone(),
                        &r[&vin[2]],
                        vin[2].dtype.as_ref().unwrap().clone(),
                        &strip_parens(&r[&vin[1]]),
                        matches!(vin[0].uop, UOps::DEFINE_LOCAL),
                    ),
                    &mut kernel,
                    depth,
                );
                if vin.len() > 3 {
                    kk("}", &mut kernel, depth)
                }
            }
            UOps::CAST => {
                //assert!(dtype.is_some() && dtype.as_ref().unwrap().size > 1);
                let val = lang.render_cast(
                    &vin.iter().map(|x| r[x].as_str()).collect::<Vec<&str>>(),
                    dtype.as_ref().unwrap().clone(),
                );
                if child_count[u] <= 1 {
                    r.insert(u.clone(), val);
                } else {
                    kk(
                        &format!(
                            "{} {} = {val};",
                            if lang.lang_opts().generic_var_prefix.is_some() {
                                lang.lang_opts()
                                    .generic_var_prefix
                                    .as_ref()
                                    .unwrap()
                                    .to_string()
                            } else {
                                dtype.as_ref().unwrap().c_name.to_string()
                            },
                            ssa(u, "cast", &mut c, &mut r)
                        ),
                        &mut kernel,
                        depth,
                    );
                }
            }
            UOps::DEFINE_LOCAL => {
                if lang.lang_opts().external_local_bufs {
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
                bufs.extend(
                    args.iter()
                        .map(|args| (args.to_str(), (*dtype).clone().unwrap()))
                        .collect::<Vec<_>>(),
                );
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
            UOps::PHI => {
                kk(
                    &format!("{} = {};", r[&vin[0]], r[&vin[1]]),
                    &mut kernel,
                    depth,
                );
                r.insert(u.clone(), r[&vin[0]].clone());
            }
            UOps::IF => {
                kk(&lang.render_if(&r[&vin[0]]), &mut kernel, depth);
                depth +=1;
            },
        }
    }
    lang.render_kernel(function_name, &kernel, &bufs, &local_size, uops, &prekernel)
}

fn strip_parens(_s: &str) -> String {
    let s = _s.chars().into_iter().collect::<Vec<char>>();
    if s.first() == Some(&'(')
        && s.last() == Some(&')')
        && s[1..s.len() - 1]
            .iter()
            .position(|&s| s == '(')
            .is_some_and(|x| {
                s[1..s.len() - 1]
                    .iter()
                    .position(|&s2| s2 == ')')
                    .is_some_and(|y| x <= y)
            })
    {
        return s[1..s.len() - 1].iter().collect();
    }
    _s.to_string()
}
