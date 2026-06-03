use proc_macro::TokenStream;
use proc_macro2::{Ident, Span, TokenStream as TokenStream2};
use quote::{ToTokens, quote};
use syn::fold::{Fold, fold_type};
use syn::parse::{Parse, ParseStream};
use syn::{
    Error, FnArg, GenericParam, ItemFn, ItemStatic, Pat, PatType, Result, Token, Type,
    parenthesized,
};

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let function = match syn::parse::<ItemFn>(item) {
        Ok(function) => function,
        Err(_) => return compile_error("#[kernel] can only be applied to a function"),
    };

    let attr = match syn::parse::<KernelAttribute>(attr) {
        Ok(attr) => attr,
        Err(err) => return err.to_compile_error().into(),
    };

    expand_kernel(function, attr.monomorphizations)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

#[proc_macro_attribute]
pub fn device_global(_attr: TokenStream, item: TokenStream) -> TokenStream {
    export_static("device_global", item)
}

#[proc_macro_attribute]
pub fn constant(_attr: TokenStream, item: TokenStream) -> TokenStream {
    export_static("constant", item)
}

#[proc_macro_attribute]
pub fn shared(_attr: TokenStream, item: TokenStream) -> TokenStream {
    export_static("shared", item)
}

fn export_static(attribute: &str, item: TokenStream) -> TokenStream {
    let item = match syn::parse::<ItemStatic>(item) {
        Ok(item) => item,
        Err(_) => {
            return compile_error(&format!(
                "#[{attribute}] can only be applied to a static item"
            ));
        }
    };

    expand_static_item(item).into()
}

struct KernelAttribute {
    monomorphizations: Vec<Vec<Type>>,
}

impl Parse for KernelAttribute {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let mut monomorphizations = Vec::new();

        while !input.is_empty() {
            let ident = input.parse::<Ident>().map_err(|_| {
                input.error("unsupported #[kernel] argument; expected monomorphize(...)")
            })?;

            if ident != "monomorphize" {
                return Err(Error::new(
                    ident.span(),
                    format!("unsupported #[kernel] argument `{ident}`; expected monomorphize(...)"),
                ));
            }

            if !input.peek(syn::token::Paren) {
                return Err(Error::new(
                    ident.span(),
                    "expected monomorphize(...) in #[kernel]",
                ));
            }

            let content;
            parenthesized!(content in input);
            let types = content.parse_terminated(Type::parse, Token![,])?;
            if types.is_empty() {
                return Err(Error::new(
                    ident.span(),
                    "monomorphize(...) must include at least one type",
                ));
            }
            monomorphizations.push(types.into_iter().collect());

            if input.is_empty() {
                break;
            }
            if !input.peek(Token![,]) {
                return Err(input.error("unexpected #[kernel] argument tail"));
            }
            input.parse::<Token![,]>()?;
        }

        Ok(Self { monomorphizations })
    }
}

fn expand_kernel(function: ItemFn, monomorphizations: Vec<Vec<Type>>) -> Result<TokenStream2> {
    let function_name = function.sig.ident.to_string();
    let generic_params = generic_type_params(&function)?;

    if generic_params.is_empty() && monomorphizations.is_empty() {
        return Ok(quote! {
            #[unsafe(export_name = #function_name)]
            #function
        });
    }

    if generic_params.is_empty() {
        return Err(Error::new(
            function.sig.ident.span(),
            "#[kernel(monomorphize(...))] requires a generic function",
        ));
    }

    if monomorphizations.is_empty() {
        return Err(Error::new_spanned(
            &function.sig.generics,
            "generic #[kernel] functions require #[kernel(monomorphize(Ty, ...))]",
        ));
    }

    let wrappers = monomorphizations
        .iter()
        .map(|concrete_types| {
            if concrete_types.len() != generic_params.len() {
                return Err(Error::new(
                    function.sig.ident.span(),
                    format!(
                        "kernel `{}` expects {} generic argument(s), but monomorphize(...) supplied {}",
                        function.sig.ident,
                        generic_params.len(),
                        concrete_types.len()
                    ),
                ));
            }
            generate_monomorphized_kernel_wrapper(&function, &generic_params, concrete_types)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(quote! {
        #function
        #(#wrappers)*
    })
}

fn expand_static_item(item: ItemStatic) -> TokenStream2 {
    let export_name = item.ident.to_string();
    quote! {
        #[used]
        #[unsafe(export_name = #export_name)]
        #item
    }
}

fn generic_type_params(function: &ItemFn) -> Result<Vec<Ident>> {
    function
        .sig
        .generics
        .params
        .iter()
        .map(|param| match param {
            GenericParam::Type(param) => Ok(param.ident.clone()),
            GenericParam::Lifetime(param) => Err(unsupported_generic_param(param)),
            GenericParam::Const(param) => Err(unsupported_generic_param(param)),
        })
        .collect()
}

fn unsupported_generic_param(param: &impl ToTokens) -> Error {
    Error::new_spanned(
        param,
        format!(
            "unsupported generic kernel parameter `{}`; only type parameters are supported",
            param.to_token_stream()
        ),
    )
}

fn generate_monomorphized_kernel_wrapper(
    function: &ItemFn,
    generic_params: &[Ident],
    concrete_types: &[Type],
) -> Result<TokenStream2> {
    let export_name = monomorphized_kernel_name(&function.sig.ident.to_string(), concrete_types);
    let wrapper_ident = Ident::new(&export_name, Span::call_site());
    let function_ident = &function.sig.ident;
    let args = monomorphized_args(function, generic_params, concrete_types)?;
    let wrapper_args = args.iter().map(|arg| &arg.wrapper_arg);
    let call_args = args.iter().map(|arg| &arg.binding_name);

    Ok(quote! {
        #[unsafe(export_name = #export_name)]
        pub unsafe extern "C" fn #wrapper_ident(#(#wrapper_args),*) {
            unsafe { #function_ident::<#(#concrete_types),*>(#(#call_args),*) }
        }
    })
}

struct MonomorphizedArg {
    wrapper_arg: PatType,
    binding_name: Ident,
}

fn monomorphized_args(
    function: &ItemFn,
    generic_params: &[Ident],
    concrete_types: &[Type],
) -> Result<Vec<MonomorphizedArg>> {
    function
        .sig
        .inputs
        .iter()
        .map(|arg| match arg {
            FnArg::Receiver(receiver) => Err(Error::new_spanned(
                receiver,
                "unsupported kernel argument receiver",
            )),
            FnArg::Typed(arg) => {
                let binding_name = argument_binding_name(&arg.pat)?;
                let mut wrapper_arg = arg.clone();
                let mut substituter = TypeSubstituter {
                    generic_params,
                    concrete_types,
                };
                wrapper_arg.ty = Box::new(substituter.fold_type((*wrapper_arg.ty).clone()));
                Ok(MonomorphizedArg {
                    wrapper_arg,
                    binding_name,
                })
            }
        })
        .collect()
}

fn argument_binding_name(pattern: &Pat) -> Result<Ident> {
    match pattern {
        Pat::Ident(pattern)
            if pattern.by_ref.is_none() && pattern.subpat.is_none() && pattern.attrs.is_empty() =>
        {
            Ok(pattern.ident.clone())
        }
        _ => Err(Error::new_spanned(
            pattern,
            format!(
                "unsupported kernel argument pattern: {}",
                pattern.to_token_stream()
            ),
        )),
    }
}

struct TypeSubstituter<'a> {
    generic_params: &'a [Ident],
    concrete_types: &'a [Type],
}

impl Fold for TypeSubstituter<'_> {
    fn fold_type(&mut self, ty: Type) -> Type {
        if let Type::Path(path) = &ty {
            if path.qself.is_none() && path.path.segments.len() == 1 {
                let ident = &path.path.segments[0].ident;
                if let Some(index) = self
                    .generic_params
                    .iter()
                    .position(|generic| generic == ident)
                {
                    return self.concrete_types[index].clone();
                }
            }
        }

        fold_type(self, ty)
    }
}

fn monomorphized_kernel_name(base: &str, concrete_types: &[Type]) -> String {
    let suffix = concrete_types
        .iter()
        .map(type_suffix)
        .collect::<Vec<_>>()
        .join("_");
    format!("{base}_{suffix}")
}

fn type_suffix(ty: &Type) -> String {
    sanitize_type_suffix(&ty.to_token_stream().to_string())
}

fn sanitize_type_suffix(ty: &str) -> String {
    let mut out = String::new();
    let mut previous_underscore = false;
    for ch in ty.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch);
            previous_underscore = false;
        } else if !previous_underscore {
            out.push('_');
            previous_underscore = true;
        }
    }
    out.trim_matches('_').to_string()
}

fn compile_error(message: &str) -> TokenStream {
    quote! {
        compile_error!(#message);
    }
    .into()
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;
    use syn::parse_quote;

    fn normalized(tokens: TokenStream2) -> String {
        tokens
            .to_string()
            .chars()
            .filter(|ch| !ch.is_whitespace())
            .collect()
    }

    fn parse_kernel_attr(tokens: TokenStream2) -> Vec<Vec<Type>> {
        syn::parse2::<KernelAttribute>(tokens)
            .expect("kernel attribute should parse")
            .monomorphizations
    }

    #[test]
    fn non_generic_kernel_exports_original_name() {
        let function: ItemFn = parse_quote! {
            pub unsafe fn vector_add(out: *mut f32, lhs: *const f32) {}
        };

        let expanded = expand_kernel(function, Vec::new()).expect("kernel should expand");
        let text = normalized(expanded);

        assert!(text.contains("#[unsafe(export_name=\"vector_add\")]"));
        assert!(text.contains("pubunsafefnvector_add"));
        assert!(!text.contains("extern\"C\"fnvector_add"));
    }

    #[test]
    fn generic_kernel_emits_concrete_extern_wrapper() {
        let function: ItemFn = parse_quote! {
            unsafe fn fill<T>(out: *mut T, value: T, len: usize) {}
        };
        let monomorphizations = parse_kernel_attr(quote! {
            monomorphize(f32), monomorphize(u32)
        });

        let expanded = expand_kernel(function, monomorphizations).expect("kernel should expand");
        let text = normalized(expanded);

        assert!(text.contains("#[unsafe(export_name=\"fill_f32\")]"));
        assert!(text.contains("pubunsafeextern\"C\"fnfill_f32(out:*mutf32,value:f32,len:usize)"));
        assert!(text.contains("unsafe{fill::<f32>(out,value,len)}"));
        assert!(text.contains("#[unsafe(export_name=\"fill_u32\")]"));
    }

    #[test]
    fn nested_generic_argument_types_are_substituted() {
        let function: ItemFn = parse_quote! {
            unsafe fn reduce<T>(out: *mut T, inputs: Option<*const T>) {}
        };
        let monomorphizations = parse_kernel_attr(quote! {
            monomorphize(f32)
        });

        let expanded = expand_kernel(function, monomorphizations).expect("kernel should expand");
        let text = normalized(expanded);

        assert!(text.contains("out:*mutf32"));
        assert!(text.contains("inputs:Option<*constf32>"));
        assert!(text.contains("reduce::<f32>(out,inputs)"));
    }

    #[test]
    fn unsupported_lifetime_generics_are_clear_errors() {
        let function: ItemFn = parse_quote! {
            fn bad<'a>(input: &'a u32) {}
        };

        let err = match expand_kernel(function, Vec::new()) {
            Ok(_) => panic!("kernel should reject lifetime"),
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("unsupported generic kernel parameter")
        );
        assert!(
            err.to_string()
                .contains("only type parameters are supported")
        );
    }

    #[test]
    fn unsupported_const_generics_are_clear_errors() {
        let function: ItemFn = parse_quote! {
            fn bad<const N: usize>(input: [u32; N]) {}
        };

        let err = match expand_kernel(function, Vec::new()) {
            Ok(_) => panic!("kernel should reject const generic"),
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("unsupported generic kernel parameter")
        );
        assert!(
            err.to_string()
                .contains("only type parameters are supported")
        );
    }

    #[test]
    fn static_attrs_export_the_static_name() {
        let item: ItemStatic = parse_quote! {
            pub static mut SCRATCH: u32 = 0;
        };

        let expanded = expand_static_item(item);
        let text = normalized(expanded);

        assert!(text.contains("#[used]"));
        assert!(text.contains("#[unsafe(export_name=\"SCRATCH\")]"));
        assert!(text.contains("pubstaticmutSCRATCH:u32=0;"));
    }

    #[test]
    fn monomorphize_requires_at_least_one_type() {
        let err = match syn::parse2::<KernelAttribute>(quote! {
            monomorphize()
        }) {
            Ok(_) => panic!("empty monomorphize should fail"),
            Err(err) => err,
        };

        assert!(
            err.to_string()
                .contains("monomorphize(...) must include at least one type")
        );
    }
}
