use proc_macro::{TokenStream, TokenTree};

#[proc_macro_attribute]
pub fn kernel(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let Some(name) = find_function_name(item.clone()) else {
        return compile_error("#[kernel] can only be applied to a function");
    };

    let source = item.to_string();
    let exported = format!("#[unsafe(export_name = \"{name}\")]\n{source}");
    exported
        .parse()
        .unwrap_or_else(|_| compile_error("#[kernel] failed to rewrite function"))
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
    let Some(name) = find_static_name(item.clone()) else {
        return compile_error(&format!("#[{attribute}] can only be applied to a static item"));
    };

    let source = item.to_string();
    let exported = format!("#[used]\n#[unsafe(export_name = \"{name}\")]\n{source}");
    exported
        .parse()
        .unwrap_or_else(|_| compile_error(&format!("#[{attribute}] failed to rewrite static")))
}

fn find_function_name(tokens: TokenStream) -> Option<String> {
    let mut saw_fn = false;
    for token in tokens {
        if let TokenTree::Ident(ident) = token {
            if saw_fn {
                return Some(ident.to_string());
            }
            saw_fn = ident.to_string() == "fn";
        }
    }
    None
}

fn find_static_name(tokens: TokenStream) -> Option<String> {
    let mut saw_static = false;
    let mut saw_mut = false;
    for token in tokens {
        if let TokenTree::Ident(ident) = token {
            let ident = ident.to_string();
            if saw_static {
                if ident == "mut" && !saw_mut {
                    saw_mut = true;
                    continue;
                }
                return Some(ident);
            }
            saw_static = ident == "static";
        }
    }
    None
}

fn compile_error(message: &str) -> TokenStream {
    format!("compile_error!({message:?});")
        .parse()
        .expect("compile_error should parse")
}
