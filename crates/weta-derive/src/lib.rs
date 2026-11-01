//! `#[derive(Module)]`: replaces subclassing `nn.Module`.
//!
//! It implements `weta::nn::Module` for a struct by delegating `forward` to
//! the struct's own inherent `fn forward(&self, x: &Tensor) -> Tensor`, and
//! generates parameter traversal from the fields, so parameter names match
//! PyTorch's (`fc1.weight`, `features.0.bias`).
//!
//! Field attributes:
//! - `#[module(skip)]`: not a submodule or parameter (hyperparameters, etc.).
//! - `#[module(buffer)]`: a `Tensor` that is state but not a parameter, like
//!   BatchNorm's running mean: saved in the state dict, not optimised.
//!
//! Struct attributes:
//! - `#[module(crate = "weta_nn")]`: path to the nn crate, if not `weta::nn`.
//! - `#[module(no_forward)]`: the struct has no single-tensor `forward`.

use proc_macro::TokenStream;
use quote::quote;
use syn::{parse_macro_input, Data, DeriveInput, Fields, Index, LitStr, Meta, Path};

#[proc_macro_derive(Module, attributes(module))]
pub fn derive_module(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    match expand(input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

enum FieldKind {
    Visit,
    Buffer,
    Skip,
}

fn field_kind(attrs: &[syn::Attribute]) -> syn::Result<FieldKind> {
    let mut kind = FieldKind::Visit;
    for attr in attrs.iter().filter(|a| a.path().is_ident("module")) {
        attr.parse_nested_meta(|meta| {
            if meta.path.is_ident("skip") {
                kind = FieldKind::Skip;
                Ok(())
            } else if meta.path.is_ident("buffer") {
                kind = FieldKind::Buffer;
                Ok(())
            } else {
                Err(meta.error("expected `skip` or `buffer`"))
            }
        })?;
    }
    Ok(kind)
}

fn expand(input: DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let mut krate: Path = syn::parse_quote!(::weta::nn);
    let mut has_forward = true;
    for attr in input.attrs.iter().filter(|a| a.path().is_ident("module")) {
        if let Meta::List(_) = &attr.meta {
            attr.parse_nested_meta(|meta| {
                if meta.path.is_ident("crate") {
                    let s: LitStr = meta.value()?.parse()?;
                    krate = s.parse()?;
                    Ok(())
                } else if meta.path.is_ident("no_forward") {
                    has_forward = false;
                    Ok(())
                } else {
                    Err(meta.error("expected `crate = \"...\"` or `no_forward`"))
                }
            })?;
        }
    }

    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(&input.ident, "#[derive(Module)] only supports structs"));
    };

    let mut visits = Vec::new();
    let mut visits_mut = Vec::new();
    let mut trains = Vec::new();
    let members: Vec<(proc_macro2::TokenStream, String, &syn::Field)> = match &data.fields {
        Fields::Named(f) => f
            .named
            .iter()
            .map(|field| {
                let id = field.ident.clone().unwrap();
                let name = id.to_string();
                (quote!(#id), name, field)
            })
            .collect(),
        Fields::Unnamed(f) => f
            .unnamed
            .iter()
            .enumerate()
            .map(|(i, field)| {
                let idx = Index::from(i);
                (quote!(#idx), i.to_string(), field)
            })
            .collect(),
        Fields::Unit => Vec::new(),
    };

    for (member, name, field) in &members {
        match field_kind(&field.attrs)? {
            FieldKind::Skip => {}
            FieldKind::Buffer => {
                visits.push(quote! {
                    f(&#krate::join_name(prefix, #name), &self.#member, #krate::TensorKind::Buffer);
                });
                visits_mut.push(quote! {
                    f(&#krate::join_name(prefix, #name), &mut self.#member, #krate::TensorKind::Buffer);
                });
            }
            FieldKind::Visit => {
                visits.push(quote! {
                    #krate::Visit::visit_tensors(&self.#member, &#krate::join_name(prefix, #name), f);
                });
                visits_mut.push(quote! {
                    #krate::Visit::visit_tensors_mut(&mut self.#member, &#krate::join_name(prefix, #name), f);
                });
                trains.push(quote! {
                    #krate::Visit::set_train_mode(&mut self.#member, training);
                });
            }
        }
    }

    let ident = &input.ident;
    let (impl_g, ty_g, where_c) = input.generics.split_for_impl();
    let tensor = quote!(#krate::Tensor);
    let forward = if has_forward {
        quote! { #ident::forward(self, x) }
    } else {
        let msg = format!("{ident} has no single-tensor forward (it was derived with #[module(no_forward)])");
        quote! { let _ = x; panic!(#msg) }
    };

    Ok(quote! {
        impl #impl_g #krate::Module for #ident #ty_g #where_c {
            fn forward(&self, x: &#tensor) -> #tensor {
                #forward
            }
            fn visit(&self, prefix: &str, f: &mut dyn FnMut(&str, &#tensor, #krate::TensorKind)) {
                #(#visits)*
            }
            fn visit_mut(&mut self, prefix: &str, f: &mut dyn FnMut(&str, &mut #tensor, #krate::TensorKind)) {
                #(#visits_mut)*
            }
            fn set_training(&mut self, training: bool) {
                let _ = training;
                #(#trains)*
            }
        }
    })
}
