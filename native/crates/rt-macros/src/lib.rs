//! # kore-rt-macros — the runtime's entry, task, and authority declarations
//!
//! Resolved through `kore-rt`'s re-export (`kore_rt::main` /
//! `kore_rt::task`, `res/USERLAND.md` §5.4) — service crates never
//! depend on this crate directly.  All three are thin wrappers over landed
//! non-macro spines:
//!
//! - **`#[kore_rt::main]`** (§5.4, the entry contract): on
//!   `async fn main(ctx: Rt) -> !`, emits the `#[no_mangle]`
//!   `__kore_rt_main` symbol crt0 calls, which pins the root future in
//!   the crt0 call frame and hands it to the executor
//!   (`kore_rt::entry::run_main`).  Its optional
//!   `tasks(worker(), ...)` list constructs one bounded, affine startup
//!   set and admits it before the root's first poll through
//!   `kore_rt::entry::run_main_with_tasks`.  The `lang_start`
//!   placeholder: when the `std::sys::kore` port lands, these binaries
//!   migrate to plain `fn main()` with no redesign (§6).
//! - **`#[kore_rt::task]`** (§3.3): on `async fn name(args…)`, emits the
//!   per-task `TaskStorage` static (the embassy pattern — the future is
//!   unnameable, so a type-alias-impl-trait names it) and rewrites the
//!   fn into a spawn wrapper returning the executor's admission token.
//!   The consumer crate opts into `#![feature(type_alias_impl_trait)]`
//!   (the workspace is nightly-pinned).
//!
//! - **`#[derive(kore_rt::Bundle)]`**: makes one service endowment declaration
//!   the source of both its founding [`Profile`](kore_abi::task::Profile) and
//!   its fail-closed boot claims.  Boot roles belong to fields, not token
//!   species: one `SupplyToken` may lawfully be `MEM_SUPPLY`, `DMA_POOL`, or
//!   `FACE_SUPPLY` in different services.
//!
//! Expansion is deliberately mechanical: everything behavioral lives in
//! `kore-rt` (`entry::run_main`, `TaskStorage::spawn`, `service::Bundle`) where
//! the host suite proves it; the macros only name things.

use proc_macro::TokenStream;
use quote::{format_ident, quote};
use syn::{
    Data, DeriveInput, Error, Expr, Field, Fields, FnArg, GenericArgument, Ident, ItemFn, LitInt,
    LitStr, Pat, PathArguments, ReturnType, Token, Type, parenthesized,
    parse::{Parse, ParseStream},
    parse_macro_input,
    punctuated::Punctuated,
    spanned::Spanned,
};

/// Resolve only macro-authored paths; the caller's syntax is never rewritten.
/// Prefer an explicitly named runtime dependency, then the standard facade.
fn dependency_path(package: &str, module: &str) -> proc_macro2::TokenStream {
    use proc_macro_crate::{FoundCrate, crate_name};
    fn root(found: FoundCrate, package: &str) -> proc_macro2::TokenStream {
        // The runtime uses its macros from integration tests too. A package's
        // own public library is available there by its canonical extern name.
        let name = match found {
            FoundCrate::Name(name) => name,
            FoundCrate::Itself => package.replace('-', "_"),
        };
        let ident = format_ident!("{}", name);
        quote!(::#ident)
    }
    if let Ok(found) = crate_name(package) {
        return root(found, package);
    }
    if let Ok(found) = crate_name("kore-std") {
        let facade = root(found, "kore-std");
        let module = format_ident!("{}", module);
        return quote!(#facade::#module);
    }
    // Parser unit tests have no runtime dependency. Preserve the canonical
    // path, which also gives a normal unresolved-crate diagnostic to callers.
    root(FoundCrate::Itself, package)
}

/// Derive a service's static authority bundle from its fields.
///
/// ```ignore
/// #[derive(kore_rt::Bundle)]
/// #[bundle(name = "mm")]
/// struct MmEndowment {
///     #[bundle(role = role::MEM_SUPPLY)]
///     supply: SupplyToken,
///     #[bundle(role = role::MEM_REGION)]
///     region: RegionToken,
/// }
/// ```
///
/// `optional` means founder vacancy is represented as `None`, while a present
/// but unclaimable handle still fails the whole bundle. `capture` records a
/// topology handle without claiming it and therefore does not add it to the
/// capability profile:
///
/// ```ignore
/// #[bundle(role = role::FACE_SUPPLY, optional)]
/// faces: Option<SupplyToken>,
/// #[bundle(role = role::DOORBELL, index = 2, capture, optional)]
/// kick: Option<SlotRef>,
/// ```
#[proc_macro_derive(Bundle, attributes(bundle))]
pub fn derive_bundle(item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as DeriveInput);
    match expand_bundle(input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

#[derive(Default)]
struct BundleArgs {
    name: Option<LitStr>,
}

impl Parse for BundleArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut args = Self::default();
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            if key == "name" {
                if args.name.is_some() {
                    return Err(Error::new(key.span(), "duplicate bundle name"));
                }
                input.parse::<Token![=]>()?;
                args.name = Some(input.parse()?);
            } else {
                return Err(Error::new(key.span(), "expected `name = \"...\"`"));
            }
            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }
        Ok(args)
    }
}

#[derive(Default)]
struct BundleFieldArgs {
    role: Option<Expr>,
    index: Option<Expr>,
    optional: bool,
    capture: bool,
}

impl Parse for BundleFieldArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut args = Self::default();
        while !input.is_empty() {
            let key: Ident = input.parse()?;
            if key == "role" {
                if args.role.is_some() {
                    return Err(Error::new(key.span(), "duplicate field role"));
                }
                input.parse::<Token![=]>()?;
                args.role = Some(input.parse()?);
            } else if key == "index" {
                if args.index.is_some() {
                    return Err(Error::new(key.span(), "duplicate field index"));
                }
                input.parse::<Token![=]>()?;
                args.index = Some(input.parse()?);
            } else if key == "optional" {
                if args.optional {
                    return Err(Error::new(key.span(), "duplicate `optional`"));
                }
                args.optional = true;
            } else if key == "capture" {
                if args.capture {
                    return Err(Error::new(key.span(), "duplicate `capture`"));
                }
                args.capture = true;
            } else {
                return Err(Error::new(
                    key.span(),
                    "expected `role`, `index`, `optional`, or `capture`",
                ));
            }
            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }
        Ok(args)
    }
}

fn bundle_args(input: &DeriveInput) -> Result<BundleArgs, Error> {
    let mut found = None;
    for attr in &input.attrs {
        if attr.path().is_ident("bundle") {
            if found.is_some() {
                return Err(Error::new(
                    attr.span(),
                    "duplicate #[bundle(...)] attribute",
                ));
            }
            found = Some(attr.parse_args::<BundleArgs>()?);
        }
    }
    found.ok_or_else(|| Error::new(input.ident.span(), "missing #[bundle(name = \"...\")]"))
}

fn bundle_field_args(field: &Field) -> Result<BundleFieldArgs, Error> {
    let mut found = None;
    for attr in &field.attrs {
        if attr.path().is_ident("bundle") {
            if found.is_some() {
                return Err(Error::new(
                    attr.span(),
                    "duplicate #[bundle(...)] field attribute",
                ));
            }
            found = Some(attr.parse_args::<BundleFieldArgs>()?);
        }
    }
    let args = found.ok_or_else(|| {
        Error::new(
            field.span(),
            "every named Bundle field needs #[bundle(role = ...)]",
        )
    })?;
    if args.role.is_none() {
        return Err(Error::new(
            field.span(),
            "bundle field is missing `role = ...`",
        ));
    }
    Ok(args)
}

fn option_inner(ty: &Type) -> Option<&Type> {
    let Type::Path(path) = ty else {
        return None;
    };
    let segment = path.path.segments.last()?;
    if segment.ident != "Option" {
        return None;
    }
    let PathArguments::AngleBracketed(args) = &segment.arguments else {
        return None;
    };
    let mut args = args.args.iter();
    let GenericArgument::Type(inner) = args.next()? else {
        return None;
    };
    args.next().is_none().then_some(inner)
}

fn expand_bundle(input: DeriveInput) -> Result<proc_macro2::TokenStream, Error> {
    let runtime = dependency_path("kore-rt", "runtime");
    let abi = dependency_path("kore-abi", "abi");
    let args = bundle_args(&input)?;
    let name = args
        .name
        .ok_or_else(|| Error::new(input.ident.span(), "bundle name cannot be omitted"))?;
    let ident = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let mut profile_roles = Vec::new();
    let mut wiring_roles = Vec::new();
    let mut initializers = Vec::new();
    let mut field_names = Vec::new();

    match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Unit => {}
            Fields::Named(fields) => {
                for field in &fields.named {
                    let field_name = field.ident.as_ref().expect("named field");
                    let field_args = bundle_field_args(field)?;
                    let role = field_args.role.expect("validated above");
                    let index = field_args.index.unwrap_or_else(|| syn::parse_quote!(0u16));
                    let optional_inner = option_inner(&field.ty);
                    if field_args.optional && optional_inner.is_none() {
                        return Err(Error::new(
                            field.ty.span(),
                            "an `optional` bundle field must have type Option<T>",
                        ));
                    }
                    if !field_args.optional && optional_inner.is_some() {
                        return Err(Error::new(
                            field.ty.span(),
                            "an Option<T> bundle field must declare `optional`",
                        ));
                    }
                    let missing = quote! {
                        #runtime::service::BindError::MissingRole {
                            role: #role,
                            index: #index,
                        }
                    };
                    if field_args.capture {
                        let wiring = if field_args.optional {
                            quote! { #runtime::service::WiringSlot::optional(#role, #index) }
                        } else {
                            quote! { #runtime::service::WiringSlot::required(#role, #index) }
                        };
                        wiring_roles.push(wiring);
                        let init = if field_args.optional {
                            quote! { let #field_name = boot.handle(#role, #index); }
                        } else {
                            quote! {
                                let #field_name = boot.handle(#role, #index).ok_or(#missing)?;
                            }
                        };
                        initializers.push(init);
                    } else {
                        let claim_ty = optional_inner.unwrap_or(&field.ty);
                        let bad = quote! {
                            #runtime::service::BindError::BadHandle {
                                role: #role,
                                index: #index,
                            }
                        };
                        let init = if field_args.optional {
                            quote! {
                                let #field_name = match boot.handle(#role, #index) {
                                    ::core::option::Option::Some(slot) =>
                                        ::core::option::Option::Some(
                                            <#claim_ty as #runtime::service::BootClaim>::claim_boot(slot)
                                                .ok_or(#bad)?,
                                        ),
                                    ::core::option::Option::None => ::core::option::Option::None,
                                };
                            }
                        } else {
                            quote! {
                                let #field_name = {
                                    let slot = boot.handle(#role, #index).ok_or(#missing)?;
                                    <#claim_ty as #runtime::service::BootClaim>::claim_boot(slot)
                                        .ok_or(#bad)?
                                };
                            }
                        };
                        initializers.push(init);
                        // An `optional` field declares a coordinate the
                        // machine may not have, so the PROFILE carries
                        // that fact to the founding side rather than the
                        // kernel knowing which coordinates may be absent.
                        // The two halves cannot drift: the same `optional`
                        // that makes `resolve` tolerate a missing handle
                        // is what lets the position stamp vacant.
                        let ctor = if field_args.optional {
                            quote! { indexed_optional }
                        } else {
                            quote! { indexed }
                        };
                        profile_roles.push(quote! {
                            #abi::task::RoleSlot::#ctor(#role, #index)
                        });
                    }
                    field_names.push(field_name);
                }
            }
            Fields::Unnamed(fields) => {
                return Err(Error::new(
                    fields.span(),
                    "Bundle derive supports named-field or unit structs, not tuple structs",
                ));
            }
        },
        _ => {
            return Err(Error::new(
                input.span(),
                "Bundle derive supports structs only",
            ));
        }
    }

    let construct = if field_names.is_empty() {
        quote! { Self }
    } else {
        quote! { Self { #(#field_names),* } }
    };

    Ok(quote! {
        impl #impl_generics #runtime::service::Bundle for #ident #ty_generics #where_clause {
            const PROFILE: #abi::task::Profile = #abi::task::Profile::new(
                #name,
                &[#(#profile_roles),*],
            );

            const WIRING: &'static [#runtime::service::WiringSlot] = &[
                #(#wiring_roles),*
            ];

            fn resolve(
                boot: &#abi::task::Bootstrap<'_>,
            ) -> ::core::result::Result<Self, #runtime::service::BindError> {
                #(#initializers)*
                ::core::result::Result::Ok(#construct)
            }
        }
    })
}

mod kw {
    syn::custom_keyword!(attest);
    syn::custom_keyword!(tasks);
}

#[derive(Default)]
struct MainArgs {
    attest: Option<Expr>,
    tasks: Punctuated<Expr, Token![,]>,
}

impl Parse for MainArgs {
    fn parse(input: ParseStream<'_>) -> syn::Result<Self> {
        let mut args = Self::default();
        while !input.is_empty() {
            if input.peek(kw::attest) {
                if args.attest.is_some() {
                    return Err(input.error("duplicate attest(...) option"));
                }
                input.parse::<kw::attest>()?;
                let content;
                parenthesized!(content in input);
                args.attest = Some(content.parse()?);
                if !content.is_empty() {
                    return Err(content.error("attest(...) takes one rung expression"));
                }
            } else if input.peek(kw::tasks) {
                if !args.tasks.is_empty() {
                    return Err(input.error("duplicate tasks(...) option"));
                }
                input.parse::<kw::tasks>()?;
                let content;
                parenthesized!(content in input);
                args.tasks = content.parse_terminated(Expr::parse, Token![,])?;
                if args.tasks.is_empty() {
                    return Err(content
                        .error("tasks(...) requires at least one #[kore_rt::task] constructor"));
                }
            } else {
                return Err(input.error("expected attest(rung) or tasks(worker(), ...)"));
            }
            if input.is_empty() {
                break;
            }
            input.parse::<Token![,]>()?;
        }
        Ok(args)
    }
}

/// The §5.4 entry attribute — see the crate docs.
///
/// ```ignore
/// #[kore_rt::main]
/// async fn main(ctx: kore_rt::entry::Rt) -> ! {
///     loop { /* the service's root task */ }
/// }
///
/// #[kore_rt::task]
/// async fn watchdog() { /* a declared sibling task */ }
///
/// #[kore_rt::main(tasks(watchdog()))]
/// async fn main_with_watchdog(ctx: kore_rt::entry::Rt) -> ! {
///     loop { /* both tasks are admitted before this first poll */ }
/// }
///
/// // A legacy service may preserve its established NACK decoder rung.
/// #[kore_rt::main(attest(3))]
/// async fn main_with_custom_attestation(ctx: kore_rt::entry::Rt) -> ! {
///     loop {}
/// }
/// ```
#[proc_macro_attribute]
pub fn main(attr: TokenStream, item: TokenStream) -> TokenStream {
    let args = parse_macro_input!(attr as MainArgs);
    let input = parse_macro_input!(item as ItemFn);
    match expand_main(args, input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand_main(args: MainArgs, input: ItemFn) -> Result<proc_macro2::TokenStream, Error> {
    let runtime = dependency_path("kore-rt", "runtime");
    check_main(&input)?;
    let name = &input.sig.ident;
    let attest = args.attest;
    let tasks: Vec<_> = args.tasks.into_iter().collect();
    let task_count = tasks.len();
    let run = match (tasks.is_empty(), attest) {
        (true, None) => quote! { #runtime::entry::run_main(rt, #name) },
        (true, Some(rung)) => {
            quote! { #runtime::entry::run_main_at(rt, #rung, #name) }
        }
        (false, rung) => {
            let startup = quote! {
                || {
                    ::core::result::Result::<
                        [#runtime::executor::SpawnToken; #task_count],
                        #runtime::executor::SpawnRefused,
                    >::Ok([#((#tasks)?),*])
                }
            };
            let run = match rung {
                Some(rung) => quote! {
                    #runtime::entry::run_main_with_tasks_at(rt, #rung, #name, #startup)
                },
                None => quote! {
                    #runtime::entry::run_main_with_tasks(rt, #name, #startup)
                },
            };
            quote! {
                const _: () = assert!(
                    #task_count <= #runtime::executor::MAX_TASKS,
                    "#[kore_rt::main] startup tasks exceed executor::MAX_TASKS",
                );
                #run
            }
        }
    };
    Ok(quote! {
        #input

        /// The entry symbol crt0 calls (`res/USERLAND.md` §5.4) — pins
        /// the root future in this frame, admits any declared startup
        /// tasks, and runs the executor.
        #[unsafe(no_mangle)]
        pub extern "Rust" fn __kore_rt_main(rt: #runtime::entry::Rt) -> ! {
            #run
        }
    })
}

fn check_main(f: &ItemFn) -> Result<(), Error> {
    if f.sig.asyncness.is_none() {
        return Err(Error::new(
            f.sig.span(),
            "#[kore_rt::main] requires an async fn",
        ));
    }
    if f.sig.inputs.len() != 1 {
        return Err(Error::new(
            f.sig.span(),
            "#[kore_rt::main] takes exactly one argument: the entry context (ctx: Rt)",
        ));
    }
    if !f.sig.generics.params.is_empty() {
        return Err(Error::new(
            f.sig.span(),
            "#[kore_rt::main] cannot be generic",
        ));
    }
    if !matches!(
        &f.sig.output,
        ReturnType::Type(_, ty) if matches!(ty.as_ref(), Type::Never(_))
    ) {
        return Err(Error::new(
            f.sig.output.span(),
            "#[kore_rt::main] must return ! (a founded task has no ambient return path)",
        ));
    }
    Ok(())
}

/// The §3.3 task attribute — see the crate docs.
///
/// ```ignore
/// #[kore_rt::task]
/// async fn worker(id: u64) { /* the loop */ }
/// // worker(7)? is now a spawn: Result<SpawnToken, SpawnRefused>
/// ```
///
/// One storage per declaration (the §3.3/§8 bound: spawn capacity is the
/// statically declared task set) — a second spawn while the first
/// instance lives refuses `SpawnRefused`.
#[proc_macro_attribute]
pub fn task(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemFn);
    match expand_task(input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

/// The `(x, y, z)` tile grid of a `#[kore_rt::spmd]` launch —
/// exactly three positive integer literals (`res/SPMD.md` §1/§6).
///
/// Literals by construction, so every bound the geometry implies is a
/// **compile-time** fact: the worker ceiling, the tile-count width, the
/// ctx's grid — none of them a runtime refusal.
struct SpmdDims {
    x: u32,
    y: u32,
    z: u32,
}

impl Parse for SpmdDims {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        let span = input.span();
        let dims: Punctuated<LitInt, Token![,]> = Punctuated::parse_terminated(input)?;
        if dims.len() != 3 {
            return Err(Error::new(
                span,
                "#[kore_rt::spmd] takes exactly three dims: (x, y, z)",
            ));
        }
        let mut vals = [0u32; 3];
        for (slot, lit) in vals.iter_mut().zip(dims.iter()) {
            *slot = lit.base10_parse::<u32>()?;
            if *slot == 0 {
                return Err(Error::new(lit.span(), "a tile grid has no zero dimension"));
            }
        }
        Ok(SpmdDims {
            x: vals[0],
            y: vals[1],
            z: vals[2],
        })
    }
}

/// The §6 spmd attribute — see the crate docs.
///
/// ```ignore
/// #[kore_rt::spmd(1, 1, 16)]
/// fn tile_pass(ctx: &'static SpmdCtx, worker: u32) { /* the kernel */ }
/// // tile_pass() is now the launch:
/// //   Option<SpawnedSpmd<{ min(16, MAX_THREADS - 1) }>>
/// ```
///
/// The dims are the **tile grid** (R1: tiles, not threads); the worker
/// count is `min(x·y·z, MAX_THREADS − 1)`, computed in emitted `const`
/// code against `kore_rt::thread::MAX_THREADS` so the ceiling stays
/// single-sourced in the runtime.  The grid rides the emitted
/// [`SpmdCtx`] (`with_grid`), where the body reads it back
/// (`ctx.tiles()`) instead of re-naming its own launch literals.
#[proc_macro_attribute]
pub fn spmd(attr: TokenStream, item: TokenStream) -> TokenStream {
    let dims = parse_macro_input!(attr as SpmdDims);
    let input = parse_macro_input!(item as ItemFn);
    match expand_spmd(dims, input) {
        Ok(ts) => ts.into(),
        Err(e) => e.to_compile_error().into(),
    }
}

fn expand_spmd(dims: SpmdDims, f: ItemFn) -> Result<proc_macro2::TokenStream, Error> {
    let runtime = dependency_path("kore-rt", "runtime");
    if f.sig.asyncness.is_some() {
        return Err(Error::new(
            f.sig.span(),
            "#[kore_rt::spmd] workers are threads, not futures — the body cannot be async",
        ));
    }
    if !f.sig.generics.params.is_empty() {
        return Err(Error::new(
            f.sig.span(),
            "#[kore_rt::spmd] cannot be generic — the launch storage is a static",
        ));
    }
    if !matches!(f.sig.output, ReturnType::Default) {
        return Err(Error::new(
            f.sig.output.span(),
            "#[kore_rt::spmd] bodies return nothing (a worker's return is exit(0))",
        ));
    }
    // Exactly (ctx, worker), plain identifiers.  The *types* are not
    // checked here: the emitted `spawn_spmd` call coerces the inner fn
    // to `SpmdEntry = fn(&'static SpmdCtx, u32)`, so rustc enforces
    // them with a real type error at the definition site.
    if f.sig.inputs.len() != 2 {
        return Err(Error::new(
            f.sig.span(),
            "#[kore_rt::spmd] bodies take exactly (ctx: &'static SpmdCtx, worker: u32)",
        ));
    }
    for arg in &f.sig.inputs {
        match arg {
            FnArg::Receiver(r) => {
                return Err(Error::new(r.span(), "#[kore_rt::spmd] cannot take self"));
            }
            FnArg::Typed(t) => {
                if !matches!(&*t.pat, Pat::Ident(_)) {
                    return Err(Error::new(
                        t.pat.span(),
                        "#[kore_rt::spmd] arguments must be plain identifiers",
                    ));
                }
            }
        }
    }

    let vis = &f.vis;
    let name = &f.sig.ident;
    let attrs = &f.attrs;
    let (x, y, z) = (dims.x, dims.y, dims.z);
    let inner = format_ident!("__kore_rt_spmd_{}", name);
    let workers = format_ident!("__KORE_RT_SPMD_WORKERS_{}", name);
    let ctx = format_ident!("__KORE_RT_SPMD_CTX_{}", name);

    let mut inner_fn = f.clone();
    inner_fn.sig.ident = inner.clone();
    inner_fn.vis = syn::Visibility::Inherited;
    inner_fn.attrs.clear();

    Ok(quote! {
        #[doc(hidden)]
        #inner_fn

        /// (generated) The worker count: `min(x·y·z, MAX_THREADS − 1)`,
        /// const-computed against the runtime's own ceiling.  Carries
        /// the launcher's visibility because it names the region's width
        /// in the launcher's return type.
        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        #vis const #workers: usize = {
            const TILES: u128 = #x as u128 * #y as u128 * #z as u128;
            assert!(
                TILES <= u32::MAX as u128,
                "#[kore_rt::spmd] tile count exceeds u32"
            );
            let cap = (#runtime::thread::MAX_THREADS - 1) as u128;
            if TILES < cap { TILES as usize } else { cap as usize }
        };

        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        static #ctx: #runtime::spmd::SpmdCtx =
            #runtime::spmd::SpmdCtx::with_grid(#workers as u32, [#x, #y, #z]);

        #(#attrs)*
        /// Launch the grid (the `#[kore_rt::spmd]` wrapper):
        /// found the workers over this launch's own `SpmdCtx` and
        /// release the gate — all-or-nothing, `None` on any refusal
        /// with everything given back.  The body strides
        /// `ctx.tiles()` by `ctx.workers()` from its `worker` index.
        #vis fn #name() -> ::core::option::Option<
            #runtime::spmd::SpawnedSpmd<{ #workers }>,
        > {
            #runtime::spmd::spawn_spmd::<{ #workers }>(&#ctx, #inner)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{
        MainArgs, SpmdDims, check_main, expand_bundle, expand_main, expand_spmd, expand_task,
    };
    use syn::{parse_quote, parse_str};

    #[test]
    fn bundle_refuses_ambiguous_declarations() {
        for bad in [
            parse_quote! {
                struct MissingContainer {
                    #[bundle(role = role::MEM_SUPPLY)]
                    supply: Supply,
                }
            },
            parse_quote! {
                #[bundle(name = "missing-field")]
                struct MissingField { supply: Supply }
            },
            parse_quote! {
                #[bundle(name = "tuple")]
                struct Tuple(#[bundle(role = role::MEM_SUPPLY)] Supply);
            },
            parse_quote! {
                #[bundle(name = "optional-shape")]
                struct OptionalShape {
                    #[bundle(role = role::MEM_SUPPLY, optional)]
                    supply: Supply,
                }
            },
        ] {
            assert!(expand_bundle(bad).is_err());
        }
    }

    #[test]
    fn bundle_expansion_keeps_roles_contextual() {
        let bundle = parse_quote! {
            #[bundle(name = "driver")]
            struct Driver<T> where T: Bound {
                #[bundle(role = role::DMA_POOL)]
                pool: SupplyToken,
                #[bundle(role = role::DOORBELL, index = 2, capture, optional)]
                kick: Option<SlotRef>,
                #[bundle(role = role::DEVICE, optional)]
                device: Option<T>,
            }
        };
        let expanded = expand_bundle(bundle).expect("the declaration lowers");
        syn::parse2::<syn::File>(expanded.clone()).expect("the expansion remains valid Rust");
        let expanded = expanded.to_string();
        assert!(expanded.contains("Profile :: new (\"driver\""));
        // A required endowment declares itself required; an OPTIONAL one
        // carries that fact into the PROFILE, so the founding side reads
        // which positions may stamp vacant instead of knowing the list
        // itself.  The two halves of `optional` cannot drift: the same
        // attribute that makes `resolve` tolerate a missing handle is
        // what emits the optional constructor here.
        assert!(expanded.contains("RoleSlot :: indexed (role :: DMA_POOL"));
        assert!(expanded.contains("RoleSlot :: indexed_optional (role :: DEVICE"));
        assert!(!expanded.contains("RoleSlot :: indexed (role :: DEVICE"));
        // A `capture`d wiring slot is not an endowment at all — it stays
        // out of the role table under either constructor.
        assert!(!expanded.contains("RoleSlot :: indexed (role :: DOORBELL"));
        assert!(!expanded.contains("RoleSlot :: indexed_optional (role :: DOORBELL"));
        assert!(expanded.contains("boot . handle (role :: DOORBELL , 2)"));
        assert!(expanded.contains("WiringSlot :: optional (role :: DOORBELL , 2)"));
        assert!(expanded.contains("T as :: kore_rt :: service :: BootClaim"));
    }

    #[test]
    fn spmd_dims_are_exactly_three_positive_literals() {
        assert!(parse_str::<SpmdDims>("1, 1, 16").is_ok());
        assert!(parse_str::<SpmdDims>("4, 4, 64").is_ok());
        for bad in [
            "",
            "1",
            "1, 1",
            "1, 1, 1, 1",
            "0, 1, 1",
            "1, 0, 1",
            "x, 1, 1",
            "1.5, 1, 1",
        ] {
            assert!(parse_str::<SpmdDims>(bad).is_err(), "accepted {bad:?}");
        }
    }

    #[test]
    fn spmd_refuses_unlowerable_shapes() {
        let dims = || parse_str::<SpmdDims>("1, 1, 4").expect("dims parse");
        for bad in [
            parse_quote! { async fn body(ctx: &'static SpmdCtx, worker: u32) {} },
            parse_quote! { fn body<T>(ctx: &'static SpmdCtx, worker: u32) {} },
            parse_quote! { fn body(ctx: &'static SpmdCtx, worker: u32) -> u32 { worker } },
            parse_quote! { fn body(ctx: &'static SpmdCtx) {} },
            parse_quote! { fn body(&self, worker: u32) {} },
            parse_quote! { fn body((a, b): (u32, u32), worker: u32) { let _ = (a, b, worker); } },
        ] {
            assert!(expand_spmd(dims(), bad).is_err());
        }
        let ok: syn::ItemFn =
            parse_quote! { fn body(ctx: &'static SpmdCtx, worker: u32) { let _ = (ctx, worker); } };
        assert!(expand_spmd(dims(), ok).is_ok());
    }

    #[test]
    fn spmd_expansion_is_a_single_sourced_launch() {
        let dims = parse_str::<SpmdDims>("2, 2, 2").expect("dims parse");
        let body: syn::ItemFn = parse_quote! { pub fn cube(ctx: &'static SpmdCtx, worker: u32) { let _ = (ctx, worker); } };
        let expanded = expand_spmd(dims, body).expect("the region lowers");
        syn::parse2::<syn::File>(expanded.clone()).expect("the expansion remains valid Rust");
        let expanded = expanded.to_string();
        // The ceiling is read from the runtime, never baked into the macro.
        assert!(expanded.contains("MAX_THREADS"));
        // The geometry rides the ctx, workers const-derived from it.
        assert!(expanded.contains("with_grid"));
        assert!(expanded.contains("2u32 as u128 * 2u32 as u128 * 2u32 as u128"));
        // The launcher is the named fn, over the neutral region alias.
        assert!(expanded.contains("pub fn cube"));
        assert!(expanded.contains("SpawnedSpmd"));
        assert!(expanded.contains("spawn_spmd"));
    }

    /// The refusal layer: shapes the attributes cannot lower reject
    /// with a compile error, never a mis-expansion.  (The accepted
    /// shapes' behavior is proven in kore-rt's tests/macros.rs.)
    #[test]
    fn task_refuses_unlowerable_shapes() {
        for bad in [
            parse_quote! { fn not_async() {} },
            parse_quote! { async fn generic<T>(_x: T) {} },
            parse_quote! { async fn returns() -> u32 { 1 } },
            parse_quote! { async fn pattern((a, b): (u32, u32)) { let _ = (a, b); } },
        ] {
            assert!(expand_task(bad).is_err());
        }
        let ok: syn::ItemFn = parse_quote! { async fn fine(x: u32, y: u64) { let _ = (x, y); } };
        assert!(expand_task(ok).is_ok());
    }

    #[test]
    fn main_refuses_unlowerable_shapes() {
        for bad in [
            parse_quote! { fn main(_ctx: Rt) -> ! { loop {} } },
            parse_quote! { async fn main() -> ! { loop {} } },
            parse_quote! { async fn main<T>(_ctx: T) -> ! { loop {} } },
            parse_quote! { async fn main(_ctx: Rt) {} },
            parse_quote! { async fn main(_ctx: Rt) -> u32 { 0 } },
        ] {
            assert!(check_main(&bad).is_err());
        }
        let ok: syn::ItemFn = parse_quote! { async fn main(_ctx: Rt) -> ! { loop {} } };
        assert!(check_main(&ok).is_ok());
    }

    #[test]
    fn main_tasks_are_a_bounded_pre_root_admission_set() {
        let args = parse_str::<MainArgs>("attest(9), tasks(worker(7), watchdog())")
            .expect("the declared task list parses");
        let main: syn::ItemFn = parse_quote! { async fn main(_ctx: Rt) -> ! { loop {} } };
        let expanded = expand_main(args, main).expect("the task-bearing main lowers");
        syn::parse2::<syn::File>(expanded.clone()).expect("the expansion remains valid Rust");
        let expanded = expanded.to_string();
        assert!(expanded.contains("run_main_with_tasks_at"));
        assert!(expanded.contains("9"));
        assert!(expanded.contains("executor :: MAX_TASKS"));
        assert!(expanded.contains("worker (7)"));
        assert!(expanded.contains("watchdog ()"));

        assert!(parse_str::<MainArgs>("tasks()").is_err());
        assert!(parse_str::<MainArgs>("workers(worker())").is_err());
        assert!(parse_str::<MainArgs>("tasks(worker()), extra").is_err());
        assert!(parse_str::<MainArgs>("attest(1), attest(2)").is_err());
        assert!(parse_str::<MainArgs>("tasks(worker()), tasks(watchdog())").is_err());
    }

    #[test]
    fn plain_main_keeps_the_zero_task_spine() {
        let main: syn::ItemFn = parse_quote! { async fn main(_ctx: Rt) -> ! { loop {} } };
        let expanded = expand_main(MainArgs::default(), main)
            .expect("the ordinary main still lowers")
            .to_string();
        assert!(expanded.contains("entry :: run_main"));
        assert!(!expanded.contains("run_main_with_tasks"));
    }

    #[test]
    fn custom_attestation_does_not_require_sibling_tasks() {
        let args = parse_str::<MainArgs>("attest(crate::RUNG)").expect("custom rung parses");
        let main: syn::ItemFn = parse_quote! { async fn main(_ctx: Rt) -> ! { loop {} } };
        let expanded = expand_main(args, main)
            .expect("the custom-rung main lowers")
            .to_string();
        assert!(expanded.contains("entry :: run_main_at"));
        assert!(expanded.contains("crate :: RUNG"));
    }
}

fn expand_task(f: ItemFn) -> Result<proc_macro2::TokenStream, Error> {
    let runtime = dependency_path("kore-rt", "runtime");
    if f.sig.asyncness.is_none() {
        return Err(Error::new(
            f.sig.span(),
            "#[kore_rt::task] requires an async fn",
        ));
    }
    if !f.sig.generics.params.is_empty() {
        return Err(Error::new(
            f.sig.span(),
            "#[kore_rt::task] cannot be generic — the task's storage is a static",
        ));
    }
    if !matches!(f.sig.output, ReturnType::Default) {
        return Err(Error::new(
            f.sig.span(),
            "#[kore_rt::task] tasks return nothing (the executor's Output = () contract)",
        ));
    }
    // Argument names, for the factory forward (plain `ident: Ty` only —
    // pattern arguments would not survive the re-declaration).
    let mut arg_names = Vec::new();
    for arg in &f.sig.inputs {
        match arg {
            FnArg::Receiver(r) => {
                return Err(Error::new(r.span(), "#[kore_rt::task] cannot take self"));
            }
            FnArg::Typed(t) => match &*t.pat {
                Pat::Ident(p) => arg_names.push(p.ident.clone()),
                other => {
                    return Err(Error::new(
                        other.span(),
                        "#[kore_rt::task] arguments must be plain identifiers",
                    ));
                }
            },
        }
    }

    let vis = &f.vis;
    let name = &f.sig.ident;
    let inputs = &f.sig.inputs;
    let attrs = &f.attrs;
    let inner = format_ident!("__kore_rt_task_{}", name);
    let fut_ty = format_ident!("__KoreRtFut_{}", name);
    let factory = format_ident!("__kore_rt_factory_{}", name);
    let storage = format_ident!("__KORE_RT_STORAGE_{}", name);

    let mut inner_fn = f.clone();
    inner_fn.sig.ident = inner.clone();
    inner_fn.vis = syn::Visibility::Inherited;
    inner_fn.attrs.clear();

    Ok(quote! {
        #[doc(hidden)]
        #inner_fn

        #[doc(hidden)]
        #[allow(non_camel_case_types)]
        type #fut_ty = impl ::core::future::Future<Output = ()> + 'static;

        #[doc(hidden)]
        #[define_opaque(#fut_ty)]
        fn #factory(#inputs) -> #fut_ty {
            #inner(#(#arg_names),*)
        }

        #[doc(hidden)]
        #[allow(non_upper_case_globals)]
        static #storage: #runtime::executor::TaskStorage<#fut_ty> =
            #runtime::executor::TaskStorage::new();

        #(#attrs)*
        /// Spawn this task (the `#[kore_rt::task]` wrapper): move the
        /// arguments into the per-task storage static and yield the
        /// admission token for `Executor::spawn`.  Refuses while a
        /// previous instance still occupies the storage (§3.3/§8:
        /// capacity is the declared task set).
        #vis fn #name(#inputs) -> ::core::result::Result<
            #runtime::executor::SpawnToken,
            #runtime::executor::SpawnRefused,
        > {
            #storage.spawn(#factory(#(#arg_names),*))
        }
    })
}
