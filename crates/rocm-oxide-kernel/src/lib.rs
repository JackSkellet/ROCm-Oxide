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

fn compile_error(message: &str) -> TokenStream {
    format!("compile_error!({message:?});")
        .parse()
        .expect("compile_error should parse")
}
