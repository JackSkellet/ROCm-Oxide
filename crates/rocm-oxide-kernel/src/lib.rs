use proc_macro::{TokenStream, TokenTree};

#[proc_macro_attribute]
pub fn kernel(attr: TokenStream, item: TokenStream) -> TokenStream {
    let Some(name) = find_function_name(item.clone()) else {
        return compile_error("#[kernel] can only be applied to a function");
    };

    let source = item.to_string();
    let monomorphizations = match parse_monomorphizations(attr) {
        Ok(value) => value,
        Err(err) => return compile_error(&err),
    };

    if monomorphizations.is_empty() && !function_has_generic_params(&source, &name) {
        let exported = format!("#[unsafe(export_name = \"{name}\")]\n{source}");
        return exported
            .parse()
            .unwrap_or_else(|_| compile_error("#[kernel] failed to rewrite function"));
    }

    let signature = match parse_function_signature(&source, &name) {
        Ok(value) => value,
        Err(err) => return compile_error(&err),
    };

    if signature.generic_params.is_empty() {
        return compile_error("#[kernel(monomorphize(...))] requires a generic function");
    }

    if monomorphizations.is_empty() {
        return compile_error(
            "generic #[kernel] functions require #[kernel(monomorphize(Ty, ...))]",
        );
    }

    let mut exported = source;
    for concrete_types in monomorphizations {
        if concrete_types.len() != signature.generic_params.len() {
            return compile_error(&format!(
                "kernel `{}` expects {} generic argument(s), but monomorphize(...) supplied {}",
                name,
                signature.generic_params.len(),
                concrete_types.len()
            ));
        }
        match generate_monomorphized_kernel_wrapper(&signature, &concrete_types) {
            Ok(wrapper) => {
                exported.push('\n');
                exported.push_str(&wrapper);
            }
            Err(err) => return compile_error(&err),
        }
    }

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
        return compile_error(&format!(
            "#[{attribute}] can only be applied to a static item"
        ));
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

#[derive(Debug)]
struct FunctionSignature {
    name: String,
    generic_params: Vec<String>,
    args: Vec<FunctionArg>,
}

#[derive(Debug)]
struct FunctionArg {
    source: String,
    binding_name: String,
}

fn parse_function_signature(source: &str, name: &str) -> Result<FunctionSignature, String> {
    let name_pos = source
        .find(name)
        .ok_or_else(|| "#[kernel] failed to parse function name".to_string())?;
    let mut cursor = name_pos + name.len();
    cursor = skip_ws(source, cursor);

    let generic_params = if source[cursor..].starts_with('<') {
        let generic_end = find_matching(source, cursor, '<', '>')?;
        let generic_source = &source[cursor + 1..generic_end];
        cursor = skip_ws(source, generic_end + 1);
        parse_generic_params(generic_source)?
    } else {
        Vec::new()
    };

    let args_start = source[cursor..]
        .find('(')
        .ok_or_else(|| "#[kernel] failed to parse function arguments".to_string())?
        + cursor;
    let args_end = find_matching(source, args_start, '(', ')')?;
    let args = split_top_level(&source[args_start + 1..args_end], ',')
        .into_iter()
        .filter(|arg| !arg.trim().is_empty())
        .map(|arg| {
            let (pattern, _) = arg
                .split_once(':')
                .ok_or_else(|| format!("malformed kernel argument: {arg}"))?;
            let binding_name = argument_binding_name(pattern)
                .ok_or_else(|| format!("unsupported kernel argument pattern: {pattern}"))?;
            Ok(FunctionArg {
                source: arg.trim().to_string(),
                binding_name,
            })
        })
        .collect::<Result<Vec<_>, String>>()?;

    Ok(FunctionSignature {
        name: name.to_string(),
        generic_params,
        args,
    })
}

fn function_has_generic_params(source: &str, name: &str) -> bool {
    source.find(name).is_some_and(|name_pos| {
        let cursor = skip_ws(source, name_pos + name.len());
        source[cursor..].starts_with('<')
    })
}

fn parse_generic_params(source: &str) -> Result<Vec<String>, String> {
    split_top_level(source, ',')
        .into_iter()
        .filter(|param| !param.trim().is_empty())
        .map(|param| {
            let trimmed = param.trim();
            if trimmed.starts_with('\'') || trimmed.starts_with("const ") {
                return Err(format!(
                    "unsupported generic kernel parameter `{trimmed}`; only type parameters are supported"
                ));
            }
            let name = trimmed
                .split(|ch: char| ch == ':' || ch == '=' || ch.is_whitespace())
                .next()
                .unwrap_or("")
                .trim();
            if is_identifier(name) {
                Ok(name.to_string())
            } else {
                Err(format!("unsupported generic kernel parameter `{trimmed}`"))
            }
        })
        .collect()
}

fn parse_monomorphizations(attr: TokenStream) -> Result<Vec<Vec<String>>, String> {
    let source = attr.to_string();
    let mut rest = source.trim();
    if rest.is_empty() {
        return Ok(Vec::new());
    }

    let mut monomorphizations = Vec::new();
    while !rest.is_empty() {
        let Some(after_keyword) = rest.strip_prefix("monomorphize") else {
            return Err(format!(
                "unsupported #[kernel] argument `{rest}`; expected monomorphize(...)"
            ));
        };
        let after_keyword = after_keyword.trim_start();
        if !after_keyword.starts_with('(') {
            return Err("expected monomorphize(...) in #[kernel]".to_string());
        }
        let close = find_matching(after_keyword, 0, '(', ')')?;
        let concrete_types = split_top_level(&after_keyword[1..close], ',')
            .into_iter()
            .map(|ty| ty.trim().to_string())
            .filter(|ty| !ty.is_empty())
            .collect::<Vec<_>>();
        if concrete_types.is_empty() {
            return Err("monomorphize(...) must include at least one type".to_string());
        }
        monomorphizations.push(concrete_types);
        rest = after_keyword[close + 1..].trim_start();
        if let Some(next) = rest.strip_prefix(',') {
            rest = next.trim_start();
        } else if !rest.is_empty() {
            return Err(format!("unexpected #[kernel] argument tail `{rest}`"));
        }
    }
    Ok(monomorphizations)
}

fn generate_monomorphized_kernel_wrapper(
    signature: &FunctionSignature,
    concrete_types: &[String],
) -> Result<String, String> {
    let export_name = monomorphized_kernel_name(&signature.name, concrete_types);
    let wrapper_args = signature
        .args
        .iter()
        .map(|arg| substitute_generic_types(&arg.source, &signature.generic_params, concrete_types))
        .collect::<Vec<_>>()
        .join(", ");
    let call_args = signature
        .args
        .iter()
        .map(|arg| arg.binding_name.as_str())
        .collect::<Vec<_>>()
        .join(", ");
    let type_args = concrete_types.join(", ");
    Ok(format!(
        "#[unsafe(export_name = \"{export_name}\")]\n\
         pub unsafe extern \"C\" fn {export_name}({wrapper_args}) {{\n    \
         unsafe {{ {}::<{type_args}>({call_args}) }}\n\
         }}\n",
        signature.name
    ))
}

fn monomorphized_kernel_name(base: &str, concrete_types: &[String]) -> String {
    let suffix = concrete_types
        .iter()
        .map(|ty| sanitize_type_suffix(ty))
        .collect::<Vec<_>>()
        .join("_");
    format!("{base}_{suffix}")
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

fn substitute_generic_types(
    source: &str,
    generic_params: &[String],
    concrete_types: &[String],
) -> String {
    let mut output = String::new();
    let mut chars = source.char_indices().peekable();
    while let Some((start, ch)) = chars.next() {
        if ch == '_' || ch.is_ascii_alphabetic() {
            let mut end = start + ch.len_utf8();
            while let Some((next_index, next)) = chars.peek().copied() {
                if next == '_' || next.is_ascii_alphanumeric() {
                    chars.next();
                    end = next_index + next.len_utf8();
                } else {
                    break;
                }
            }
            let ident = &source[start..end];
            if let Some(index) = generic_params.iter().position(|param| param == ident) {
                output.push_str(&concrete_types[index]);
            } else {
                output.push_str(ident);
            }
        } else {
            output.push(ch);
        }
    }
    output
}

fn argument_binding_name(pattern: &str) -> Option<String> {
    let trimmed = pattern
        .trim()
        .strip_prefix("mut ")
        .unwrap_or(pattern.trim());
    is_identifier(trimmed).then(|| trimmed.to_string())
}

fn skip_ws(source: &str, mut index: usize) -> usize {
    while source[index..]
        .chars()
        .next()
        .is_some_and(|ch| ch.is_whitespace())
    {
        index += source[index..].chars().next().unwrap().len_utf8();
    }
    index
}

fn find_matching(
    source: &str,
    open_index: usize,
    open: char,
    close: char,
) -> Result<usize, String> {
    let mut depth = 0usize;
    for (index, ch) in source[open_index..].char_indices() {
        let absolute = open_index + index;
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Ok(absolute);
            }
        }
    }
    Err(format!("missing matching `{close}`"))
}

fn split_top_level(source: &str, delimiter: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut paren = 0usize;
    let mut angle = 0usize;
    let mut bracket = 0usize;
    for (index, ch) in source.char_indices() {
        match ch {
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '<' => angle += 1,
            '>' => angle = angle.saturating_sub(1),
            '[' => bracket += 1,
            ']' => bracket = bracket.saturating_sub(1),
            _ if ch == delimiter && paren == 0 && angle == 0 && bracket == 0 => {
                parts.push(&source[start..index]);
                start = index + ch.len_utf8();
            }
            _ => {}
        }
    }
    parts.push(&source[start..]);
    parts
}

fn is_identifier(value: &str) -> bool {
    let mut chars = value.chars();
    let Some(first) = chars.next() else {
        return false;
    };
    (first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
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
