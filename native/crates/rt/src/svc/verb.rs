//! **The verb seam** — a request verb is a table row, not three edits.
//!
//! `res/INET_STACK.md` §6.1 states the shape a protocol takes here: a
//! contract trait, an engine that consults a table, and selection by table
//! row rather than by `match` or `#[cfg]`.  That section applies it to IP
//! protocols; this module applies the same idea one layer out, to the
//! **service wire**.  A protocol's request vocabulary is a table of [`Verb`]
//! rows, and the encoder, the decoder, the reply correlation, the per-verb
//! grant bound and the serve arm all read that table.  None of them names a
//! verb.
//!
//! ## Why this exists
//!
//! `res/INET_MIGRATION.md` §2.3 measured what the other arrangement costs.
//! Every feature the frame row has grown had to hand-edit the same three
//! places — the encoder's match, the decoder's match, and the service's
//! serve match — so two workstreams adding a verb collide in all three.
//! §2.1 records the collision that was *worse* than a conflict: two sets of
//! constants picked the same numbers, which compiles, and the merged
//! decoder would have read one workstream's record as the other's with
//! every gate green.  §5.9 set the rule that followed: "two is a collision;
//! three is a standards problem".
//!
//! Here a verb is one `impl` block and one name in the row list, and the
//! list's own assertion makes a duplicate code a build error — the same
//! discipline, and the same failure mode, as the refusal-code guard in
//! [`kore_abi::svc`].
//!
//! ## What it costs at run time
//!
//! **No trait object, no function pointer, no table in memory**: the table
//! is a type, every row method is a monomorphized call on a zero-sized
//! type, and the whole list is gone by codegen.  That is the reason it is a
//! type and not rows of `fn` pointers, which would have put an indirect
//! call on every request — control plane, but still the path a flow is
//! minted through.
//!
//! Measured rather than asserted, on `kore-svc-net`'s release asm against
//! the same functions before the migration:
//!
//! - `request_grants`, a seven-row lookup, is four instructions —
//!   `xor eax,eax; cmp byte [rdi],4; sete al; ret`.  The row list collapsed
//!   to the one comparison the hand-written `==` emitted.
//! - `encode_req` is 247 instructions against 250: the seven-row encode
//!   chain is free.
//! - `decode_req` is 408 against 196, and all of the difference is the
//!   re-encode comparison's encoder being **inlined** where it used to be a
//!   call through the GOT.  One fewer indirect call on the decode path, and
//!   a larger function in exchange — LLVM's trade, not the seam's.
//!
//! ## The two faces of one table
//!
//! [`Verb`] is the **wire** half: the code, the codec, the correlation, the
//! grant bound.  [`ServeVerb`] is the **host** half: the arm that answers.
//! They are separate traits for the reason
//! [`SvcProtocol`](super::SvcProtocol) has no host supertrait — a client
//! speaks the wire and hosts nothing — but they are rows of the *same*
//! table, so a verb that is reachable is a verb that is answered.
//!
//! The host half names no host type of its own, not even
//! [`Serve`](crate::service::Serve): what an arm needs besides the host is
//! the protocol's own choice (`C`), because the two rows here already
//! disagree about it — the frame row takes a transferred window and mints a
//! doorbell off its serve handle, and the internet row answers from its own
//! state and a face. Fixing one shape here would have made the second
//! protocol build a second seam.

use core::marker::PhantomData;

/// What a protocol must say about its request record before verbs can
/// register into it.
///
/// Implemented on the protocol's backend-free marker — the same type that
/// carries [`SvcProtocol`](super::SvcProtocol) for a client that hosts
/// nothing — so the table is reachable from both sides of the wire.
pub trait VerbWire {
    /// The request vocabulary rows lift to and lower from.
    type Request;
    /// The reply vocabulary rows are correlated against.
    type Response;
    /// The frozen request record ([`kore_abi::svc`]).  `PartialEq` is
    /// required because the fail-closed decode is a **re-encode
    /// comparison**: the structural lift is checked by encoding the result
    /// and demanding byte equality with what arrived, which is one check
    /// instead of a guard per field and cannot drift from the encoder.
    type ReqWire: PartialEq;

    /// A record with every field zero.
    ///
    /// The encoder's starting point, and what makes the re-encode
    /// comparison complete: a row sets only its own fields, so every
    /// reserved byte, unused word and off-verb field is required to be zero
    /// for free.
    fn blank() -> Self::ReqWire;

    /// The verb discriminant this record carries.
    fn code_of(wire: &Self::ReqWire) -> u8;

    /// Stamp the verb discriminant.  Called by the engine, never by a row —
    /// a row that could stamp its own code could stamp another's.
    fn set_code(wire: &mut Self::ReqWire, code: u8);
}

/// One request verb — a row in a protocol's verb table.
///
/// Adding a verb is this `impl` plus the verb's name in the table's row
/// list.  Nothing else is edited: the encoder, the decoder, the reply
/// correlation and the grant bound all read the table.
pub trait Verb<W: VerbWire> {
    /// The wire discriminant.  Unique and non-zero across the table, which
    /// the table's own assertion enforces at compile time — zero is the
    /// invalid discriminant every fail-closed decoder rejects.
    const CODE: u8;

    /// Grant channels this verb's reply may ride.
    ///
    /// Per verb rather than per protocol, so a face that never reaches a
    /// grant-producing verb wires no grant lane at all — the
    /// [`request_grants`](super::SvcProtocol::request_grants) bound, stated
    /// where the verb is.
    const GRANTS: usize = 0;

    /// Lower `req` into `wire` when it is this verb, and answer whether it
    /// was.
    ///
    /// **A row that answers `false` must not have written anything**: the
    /// engine hands the same record to the next row, and a partial write
    /// would survive into another verb's encoding.  The discriminant is the
    /// engine's to stamp.
    fn encode(req: &W::Request, wire: &mut W::ReqWire) -> bool;

    /// Lift a record whose discriminant is already this verb's.
    ///
    /// Structural only — the engine applies the re-encode comparison, so a
    /// row writes no reserved-zero guard and cannot forget one.
    fn decode(wire: &W::ReqWire) -> Option<W::Request>;

    /// Whether `rep` is an answer to this verb.
    ///
    /// The uncorrelated-pair gate: these rings carry no request identifier,
    /// so a reply that does not answer the verb that was asked is refused
    /// before a client can act on it.
    fn answers(rep: &W::Response) -> bool;
}

/// The handler half of a row: the arm that answers this verb.
///
/// Parameterized by the host because one wire has several — the frame row's
/// host is generic over its backend, and a test double is another.  A
/// protocol whose rows carry no handler simply has no `impl` of this trait;
/// its table's serve face then answers nothing, which is what
/// [`Self::serve`]'s default says.
pub trait ServeVerb<W: VerbWire, H, C = ()> {
    /// Answer `req` when it is this row's verb; `None` when it is another's.
    ///
    /// The default answers nothing, for the protocol whose *synchronous*
    /// harness refuses wholesale rather than per verb — `services/inet`'s
    /// does, because a mint cannot be polled under a no-op waker, and it
    /// says so where it refuses.
    fn serve(_host: &mut H, _cx: &mut C, _req: &W::Request) -> Option<W::Response> {
        None
    }

    /// The same answer, where giving it means reaching another service.
    ///
    /// Mirrors [`SystemService::serve_async`](crate::service::SystemService::serve_async):
    /// the default forwards, so a row that answers from its own state writes
    /// one function and a row that must await writes the other.
    async fn serve_async(host: &mut H, cx: &mut C, req: &W::Request) -> Option<W::Response> {
        Self::serve(host, cx, req)
    }
}

/// The end of a row list.  Claims nothing, which is what makes the engine's
/// recursion terminate and an unknown discriminant a refusal.
pub struct End;

/// One row and the rest of them.
///
/// A type, not a value: the list is consumed entirely by monomorphization,
/// so nothing of it survives into the binary but the comparisons a `match`
/// would have emitted.  Built by [`verb_table!`](crate::verb_table), never
/// written by hand.
pub struct Row<V, R>(PhantomData<(V, R)>);

/// The **wire** face of a verb table: what the codecs and the client read.
///
/// Implemented for [`End`] and [`Row`] only.  Every method is total over the
/// table, so a protocol's `wire.rs` delegates to it and names no verb.
pub trait VerbTable<W: VerbWire> {
    /// Lower `req` into `wire`; `false` when no row claimed it.
    fn encode_into(req: &W::Request, wire: &mut W::ReqWire) -> bool;

    /// Lift a record structurally, by discriminant.  Prefer
    /// [`decode`](Self::decode), which is the fail-closed form.
    fn raw_decode(wire: &W::ReqWire) -> Option<W::Request>;

    /// Grant channels this exact encoded request's reply may ride.
    fn grants(wire: &W::ReqWire) -> usize;

    /// Whether `rep` answers the verb `wire` asked.
    fn answers(wire: &W::ReqWire, rep: &W::Response) -> bool;

    /// The code of the first row `rep` is an answer to.
    ///
    /// For the protocol whose reply record echoes the verb it answers.
    /// **First**, deliberately: two verbs may share a reply shape (the
    /// internet row's `Query` and `Configure` both answer with an identity),
    /// and the echoed code is then the earlier row's — so the row list's
    /// order is part of that protocol's wire and is stated where the list is.
    fn code_answering(rep: &W::Response) -> Option<u8>;

    /// Lower a request.
    ///
    /// A request no row claims lowers to the blank record, whose zero
    /// discriminant is invalid — so a vocabulary this table does not cover
    /// is *refused* by the decoder rather than read as some other verb.
    /// Fail-closed, and no panic on a path a service must not panic on.
    fn encode(req: &W::Request) -> W::ReqWire {
        let mut wire = W::blank();
        Self::encode_into(req, &mut wire);
        wire
    }

    /// Lift a record, **fail-closed**.
    ///
    /// The structural lift is [`raw_decode`](Self::raw_decode); the
    /// re-encode comparison then rejects any record whose reserved bytes,
    /// unused words or off-verb fields are non-zero.
    fn decode(wire: &W::ReqWire) -> Option<W::Request> {
        let req = Self::raw_decode(wire)?;
        (Self::encode(&req) == *wire).then_some(req)
    }
}

impl<W: VerbWire> VerbTable<W> for End {
    #[inline]
    fn encode_into(_req: &W::Request, _wire: &mut W::ReqWire) -> bool {
        false
    }
    #[inline]
    fn raw_decode(_wire: &W::ReqWire) -> Option<W::Request> {
        None
    }
    #[inline]
    fn grants(_wire: &W::ReqWire) -> usize {
        0
    }
    #[inline]
    fn answers(_wire: &W::ReqWire, _rep: &W::Response) -> bool {
        false
    }
    #[inline]
    fn code_answering(_rep: &W::Response) -> Option<u8> {
        None
    }
}

impl<W: VerbWire, V: Verb<W>, R: VerbTable<W>> VerbTable<W> for Row<V, R> {
    #[inline]
    fn encode_into(req: &W::Request, wire: &mut W::ReqWire) -> bool {
        if V::encode(req, wire) {
            W::set_code(wire, V::CODE);
            true
        } else {
            R::encode_into(req, wire)
        }
    }
    #[inline]
    fn raw_decode(wire: &W::ReqWire) -> Option<W::Request> {
        if W::code_of(wire) == V::CODE {
            V::decode(wire)
        } else {
            R::raw_decode(wire)
        }
    }
    #[inline]
    fn grants(wire: &W::ReqWire) -> usize {
        if W::code_of(wire) == V::CODE {
            V::GRANTS
        } else {
            R::grants(wire)
        }
    }
    #[inline]
    fn answers(wire: &W::ReqWire, rep: &W::Response) -> bool {
        if W::code_of(wire) == V::CODE {
            V::answers(rep)
        } else {
            R::answers(wire, rep)
        }
    }
    #[inline]
    fn code_answering(rep: &W::Response) -> Option<u8> {
        if V::answers(rep) {
            Some(V::CODE)
        } else {
            R::code_answering(rep)
        }
    }
}

/// The **host** face of the same table: the arm that answers.
///
/// Separate from [`VerbTable`] because the host and its serve context are
/// parameters and the wire is not; the rows are the same rows.
pub trait ServeTable<W: VerbWire, H, C = ()> {
    /// Answer `req` from the row that claims it, or `None` when no row does
    /// — which a host turns into its own "unsupported verb" refusal rather
    /// than guessing.
    fn serve(host: &mut H, cx: &mut C, req: &W::Request) -> Option<W::Response>;

    /// The same, where the answer must be awaited.
    async fn serve_async(host: &mut H, cx: &mut C, req: &W::Request) -> Option<W::Response>;
}

impl<W: VerbWire, H, C> ServeTable<W, H, C> for End {
    #[inline]
    fn serve(_host: &mut H, _cx: &mut C, _req: &W::Request) -> Option<W::Response> {
        None
    }
    #[inline]
    async fn serve_async(_host: &mut H, _cx: &mut C, _req: &W::Request) -> Option<W::Response> {
        None
    }
}

impl<W: VerbWire, H, C, V: ServeVerb<W, H, C>, R: ServeTable<W, H, C>> ServeTable<W, H, C>
    for Row<V, R>
{
    #[inline]
    fn serve(host: &mut H, cx: &mut C, req: &W::Request) -> Option<W::Response> {
        match V::serve(host, cx, req) {
            Some(rep) => Some(rep),
            None => R::serve(host, cx, req),
        }
    }
    #[inline]
    async fn serve_async(host: &mut H, cx: &mut C, req: &W::Request) -> Option<W::Response> {
        match V::serve_async(host, cx, req).await {
            Some(rep) => Some(rep),
            None => R::serve_async(host, cx, req).await,
        }
    }
}

/// Declare a protocol's verb table, and assert its codes are distinct.
///
/// ```ignore
/// verb_table! {
///     /// The frame row's verbs.
///     pub type NetVerbs over NetWire = [Query, OpenVif, CloseVif];
/// }
/// ```
///
/// The assertion is the point, and it is the same one
/// [`kore_abi::svc`](kore_abi::svc) applies to refusal codes: two verbs with
/// different names and the same code *compile*, and the decoder built from
/// both reads one workstream's request as the other's — green on every gate,
/// wrong on the wire (`res/INET_MIGRATION.md` §2.1).  Here it is a build
/// error, so the collision is paid for by whoever caused it, in the merge
/// that caused it.
#[macro_export]
macro_rules! verb_table {
    ($(#[$meta:meta])* $vis:vis type $name:ident over $wire:ty = [$($verb:ty),+ $(,)?];) => {
        $(#[$meta])*
        $vis type $name = $crate::__verb_rows!($($verb,)+);

        /// Every verb code in the table above is distinct and non-zero.
        const _: () = {
            let codes = [$(<$verb as $crate::svc::Verb<$wire>>::CODE,)+];
            let mut i = 0;
            while i < codes.len() {
                // Zero is "no verb" on the wire, so no row may take it.
                assert!(
                    codes[i] != 0,
                    concat!("a verb code in ", stringify!($name), " is zero"),
                );
                let mut j = i + 1;
                while j < codes.len() {
                    assert!(
                        codes[i] != codes[j],
                        concat!("duplicate verb code in ", stringify!($name)),
                    );
                    j += 1;
                }
                i += 1;
            }
        };
    };
}

/// The row list, as the cons type [`verb_table!`](crate::verb_table) builds.
#[doc(hidden)]
#[macro_export]
macro_rules! __verb_rows {
    () => { $crate::svc::End };
    ($head:ty, $($tail:ty,)*) => {
        $crate::svc::Row<$head, $crate::__verb_rows!($($tail,)*)>
    };
}
