//! Bounded inner protocol for a generation-bound agent turn.
//!
//! KIR1/KIE1 remain the outer inference envelope. These records are their
//! payload, so host and guest share one exact tool vocabulary without passing
//! JSON or authority-bearing identifiers inside kore.

use core::str;

use zerocopy::byteorder::little_endian::{U16, U32, U64};
use zerocopy::{FromBytes, Immutable, IntoBytes, KnownLayout, Unaligned};

use crate::{RequestId, RouteId, TaskGeneration};

pub const AGENT_REQUEST_MAGIC: u32 = u32::from_le_bytes(*b"KAT1");
pub const AGENT_EVENT_MAGIC: u32 = u32::from_le_bytes(*b"KAE1");
pub const AGENT_SCHEMA_V2: u16 = 2;
pub const AGENT_SCHEMA_V3: u16 = 3;
pub const AGENT_HEADER_LEN: usize = 32;
pub const AGENT_ARGUMENT_HEADER_LEN: usize = 4;
pub const AGENT_TOOL_HEADER_LEN: usize = 24;
pub const AGENT_COMMAND_MAX: usize = 48;
/// Guest route/mailbox identities available to one resident broker. This
/// bounds policy storage and the HostLink service together; actual admission
/// additionally reserves native task, filesystem and memory capacity.
pub const AGENT_ROUTE_CAPACITY: usize = 16;
const _: () = assert!(AGENT_ROUTE_CAPACITY >= 2 && AGENT_ROUTE_CAPACITY <= u8::MAX as usize + 1);
pub const AGENT_ARGUMENTS_MAX: usize = 4;
pub const AGENT_TOOLS_MAX: usize = 4;
pub const AGENT_RESOURCES_MAX: usize = 16;
pub const AGENT_RESOURCE_PATH_MAX: usize = 64;
pub const AGENT_RESOURCE_HEADER_LEN: usize = 4;
pub const AGENT_TEXT_MAX: usize = crate::INFERENCE_PART_MAX - AGENT_HEADER_LEN;

/// Bounded borrowed user-visible answer. Construction validates the exact
/// UTF-8 byte bound used by KAE1; it contains no protocol wrapper or model
/// provenance claim.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AssistantMessage<'a> {
    text: &'a str,
}

impl<'a> AssistantMessage<'a> {
    pub fn new(text: &'a str) -> Option<Self> {
        (!text.is_empty() && text.len() <= AGENT_TEXT_MAX && !text.as_bytes().contains(&0))
            .then_some(Self { text })
    }

    pub const fn text(self) -> &'a str {
        self.text
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum AgentRequestKind {
    UserText = 1,
    ToolResult = 2,
    Cancel = 3,
}

impl AgentRequestKind {
    const fn from_wire(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::UserText),
            2 => Some(Self::ToolResult),
            3 => Some(Self::Cancel),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum AgentEventKind {
    AssistantText = 1,
    ToolCall = 2,
    Refusal = 3,
    Cancelled = 4,
    Completed = 5,
    /// Validated, terminal user-visible answer from the worker model.
    Respond = 6,
}

impl AgentEventKind {
    const fn from_wire(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::AssistantText),
            2 => Some(Self::ToolCall),
            3 => Some(Self::Refusal),
            4 => Some(Self::Cancelled),
            5 => Some(Self::Completed),
            6 => Some(Self::Respond),
            _ => None,
        }
    }

    pub const fn terminal(self) -> bool {
        !matches!(self, Self::AssistantText)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ToolResultKind {
    Output = 1,
    Refused = 2,
    Revoked = 3,
    Cancelled = 4,
    Failed = 5,
}

impl ToolResultKind {
    const fn from_wire(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Output),
            2 => Some(Self::Refused),
            3 => Some(Self::Revoked),
            4 => Some(Self::Cancelled),
            5 => Some(Self::Failed),
            _ => None,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[repr(u8)]
pub enum ToolArgumentKind {
    Utf8 = 1,
    Path = 2,
    U64 = 3,
    Bool = 4,
}

impl ToolArgumentKind {
    const fn from_wire(value: u8) -> Option<Self> {
        match value {
            1 => Some(Self::Utf8),
            2 => Some(Self::Path),
            3 => Some(Self::U64),
            4 => Some(Self::Bool),
            _ => None,
        }
    }

    pub const fn mask(self) -> u8 {
        1 << (self as u8 - 1)
    }
}

pub const TOOL_ARGUMENT_KIND_MASK: u8 = ToolArgumentKind::Utf8.mask()
    | ToolArgumentKind::Path.mask()
    | ToolArgumentKind::U64.mask()
    | ToolArgumentKind::Bool.mask();
pub const TOOL_EFFECT_OBSERVE: u16 = 1 << 0;
pub const TOOL_EFFECT_MUTATE: u16 = 1 << 1;
pub const TOOL_EFFECT_DISCOVER: u16 = 1 << 2;
pub const TOOL_EFFECT_SHARE: u16 = 1 << 3;
pub const TOOL_EFFECT_MASK: u16 =
    TOOL_EFFECT_OBSERVE | TOOL_EFFECT_MUTATE | TOOL_EFFECT_DISCOVER | TOOL_EFFECT_SHARE;
pub const TOOL_RESULT_PUBLIC: u8 = 0;
pub const TOOL_RESULT_WORKSPACE: u8 = 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ToolInputSchema {
    pub minimum: u8,
    pub maximum: u8,
    /// One accepted-kind bitmask per positional argument. A tool may accept
    /// several kinds at one position without naming a command in the runner.
    pub kinds: [u8; AGENT_ARGUMENTS_MAX],
    /// Bit `n` permits ASCII whitespace in argument `n` when it is textual.
    pub whitespace: u8,
}

impl ToolInputSchema {
    pub const NONE: Self = Self {
        minimum: 0,
        maximum: 0,
        kinds: [0; AGENT_ARGUMENTS_MAX],
        whitespace: 0,
    };

    pub const fn is_valid(self) -> bool {
        if self.minimum > self.maximum || self.maximum as usize > AGENT_ARGUMENTS_MAX {
            return false;
        }
        let active_mask = if self.maximum == 0 {
            0
        } else {
            (1_u8 << self.maximum) - 1
        };
        if self.whitespace & !active_mask != 0 {
            return false;
        }
        let mut index = 0;
        while index < AGENT_ARGUMENTS_MAX {
            let kinds = self.kinds[index];
            if index < self.maximum as usize {
                if kinds == 0 || kinds & !TOOL_ARGUMENT_KIND_MASK != 0 {
                    return false;
                }
            } else if kinds != 0 {
                return false;
            }
            index += 1;
        }
        true
    }

    pub const fn accepts(self, index: usize, kind: ToolArgumentKind) -> bool {
        index < self.maximum as usize && self.kinds[index] & kind.mask() != 0
    }
}

/// Capability-filtered inert description of one installed executable. This
/// record can describe a tool but cannot authorize its invocation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AgentToolDescriptor<'a> {
    pub name: &'a str,
    pub usage: &'a str,
    pub summary: &'a str,
    pub input: ToolInputSchema,
    pub effects: u16,
    pub result_sensitivity: u8,
    pub remaining_uses: u32,
}

/// Inert mapping from a broker-visible canonical workspace path to the exact
/// private-view path installed for this agent. It describes accessibility but
/// conveys no authority by itself.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AgentResourceDescriptor<'a> {
    pub path: &'a str,
    pub local_path: &'a str,
}

impl AgentResourceDescriptor<'_> {
    fn valid(&self) -> bool {
        !self.path.is_empty()
            && self.path.len() <= AGENT_RESOURCE_PATH_MAX
            && !self.local_path.is_empty()
            && self.local_path.len() <= AGENT_RESOURCE_PATH_MAX
            && !self.path.as_bytes().contains(&0)
            && !self.local_path.as_bytes().contains(&0)
            && canonical_relative_path(self.path)
            && (self.local_path == "/"
                || self
                    .local_path
                    .strip_prefix('/')
                    .is_some_and(canonical_relative_path))
    }

    pub fn encode(&self, output: &mut [u8]) -> Option<usize> {
        if !self.valid() {
            return None;
        }
        let length = AGENT_RESOURCE_HEADER_LEN
            .checked_add(self.path.len())?
            .checked_add(self.local_path.len())?;
        let output = output.get_mut(..length)?;
        output[0] = self.path.len() as u8;
        output[1] = self.local_path.len() as u8;
        output[2..AGENT_RESOURCE_HEADER_LEN].fill(0);
        let split = AGENT_RESOURCE_HEADER_LEN + self.path.len();
        output[AGENT_RESOURCE_HEADER_LEN..split].copy_from_slice(self.path.as_bytes());
        output[split..].copy_from_slice(self.local_path.as_bytes());
        Some(length)
    }
}

fn canonical_relative_path(path: &str) -> bool {
    !path.is_empty()
        && !path.starts_with('/')
        && !path.ends_with('/')
        && !path.as_bytes().contains(&b'\\')
        && path.split('/').all(|component| {
            !component.is_empty()
                && component != "."
                && component != ".."
                && !component.bytes().any(|byte| byte.is_ascii_control())
        })
}

pub struct AgentResources<'a> {
    input: &'a [u8],
}

pub fn decode_agent_resources(input: &[u8]) -> Option<AgentResources<'_>> {
    validate_resource_catalog(input)?;
    Some(AgentResources { input })
}

impl<'a> Iterator for AgentResources<'a> {
    type Item = AgentResourceDescriptor<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.input.is_empty() {
            return None;
        }
        let header = self.input.get(..AGENT_RESOURCE_HEADER_LEN)?;
        if header[2..].iter().any(|byte| *byte != 0) {
            return None;
        }
        let path_len = usize::from(header[0]);
        let local_len = usize::from(header[1]);
        let split = AGENT_RESOURCE_HEADER_LEN.checked_add(path_len)?;
        let end = split.checked_add(local_len)?;
        let descriptor = AgentResourceDescriptor {
            path: str::from_utf8(self.input.get(AGENT_RESOURCE_HEADER_LEN..split)?).ok()?,
            local_path: str::from_utf8(self.input.get(split..end)?).ok()?,
        };
        if !descriptor.valid() {
            return None;
        }
        self.input = &self.input[end..];
        Some(descriptor)
    }
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
struct AgentToolHeaderV2 {
    name_len: u8,
    usage_len: u8,
    summary_len: u8,
    minimum: u8,
    maximum: u8,
    result_sensitivity: u8,
    kinds: [u8; AGENT_ARGUMENTS_MAX],
    whitespace: u8,
    effects: U16,
    remaining_uses: U32,
    reserved: [u8; 7],
}

impl AgentToolDescriptor<'_> {
    fn valid(&self) -> bool {
        !(self.name.is_empty()
            || self.name.len() > AGENT_COMMAND_MAX
            || self.usage.is_empty()
            || self.usage.len() > u8::MAX as usize
            || self.summary.is_empty()
            || self.summary.len() > u8::MAX as usize
            || self.name.as_bytes().contains(&0)
            || self.usage.as_bytes().contains(&0)
            || self.summary.as_bytes().contains(&0)
            || !self.input.is_valid()
            || self.effects == 0
            || self.effects & !TOOL_EFFECT_MASK != 0
            || self.result_sensitivity > TOOL_RESULT_WORKSPACE
            || self.remaining_uses == 0)
    }

    pub fn encode(&self, output: &mut [u8]) -> Option<usize> {
        if !self.valid() {
            return None;
        }
        let length = AGENT_TOOL_HEADER_LEN
            .checked_add(self.name.len())?
            .checked_add(self.usage.len())?
            .checked_add(self.summary.len())?;
        let output = output.get_mut(..length)?;
        let header = AgentToolHeaderV2 {
            name_len: self.name.len() as u8,
            usage_len: self.usage.len() as u8,
            summary_len: self.summary.len() as u8,
            minimum: self.input.minimum,
            maximum: self.input.maximum,
            result_sensitivity: self.result_sensitivity,
            kinds: self.input.kinds,
            whitespace: self.input.whitespace,
            effects: U16::new(self.effects),
            remaining_uses: U32::new(self.remaining_uses),
            reserved: [0; 7],
        };
        output[..AGENT_TOOL_HEADER_LEN].copy_from_slice(header.as_bytes());
        let mut at = AGENT_TOOL_HEADER_LEN;
        for value in [self.name, self.usage, self.summary] {
            output[at..at + value.len()].copy_from_slice(value.as_bytes());
            at += value.len();
        }
        Some(at)
    }
}

pub struct AgentTools<'a> {
    remaining: u8,
    input: &'a [u8],
}

impl<'a> Iterator for AgentTools<'a> {
    type Item = AgentToolDescriptor<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let header =
            AgentToolHeaderV2::read_from_bytes(self.input.get(..AGENT_TOOL_HEADER_LEN)?).ok()?;
        if header.reserved != [0; 7] {
            return None;
        }
        let values_len = usize::from(header.name_len)
            .checked_add(usize::from(header.usage_len))?
            .checked_add(usize::from(header.summary_len))?;
        let end = AGENT_TOOL_HEADER_LEN.checked_add(values_len)?;
        let values = self.input.get(AGENT_TOOL_HEADER_LEN..end)?;
        let usage_at = usize::from(header.name_len);
        let summary_at = usage_at.checked_add(usize::from(header.usage_len))?;
        let descriptor = AgentToolDescriptor {
            name: str::from_utf8(&values[..usage_at]).ok()?,
            usage: str::from_utf8(&values[usage_at..summary_at]).ok()?,
            summary: str::from_utf8(&values[summary_at..]).ok()?,
            input: ToolInputSchema {
                minimum: header.minimum,
                maximum: header.maximum,
                kinds: header.kinds,
                whitespace: header.whitespace,
            },
            effects: header.effects.get(),
            result_sensitivity: header.result_sensitivity,
            remaining_uses: header.remaining_uses.get(),
        };
        if !descriptor.valid() {
            return None;
        }
        self.input = &self.input[end..];
        self.remaining -= 1;
        Some(descriptor)
    }
}

#[derive(
    Clone, Copy, Debug, Eq, FromBytes, Immutable, IntoBytes, KnownLayout, PartialEq, Unaligned,
)]
#[repr(C)]
struct AgentHeaderV1 {
    magic: U32,
    version: U16,
    kind: u8,
    flags: u8,
    turn: U64,
    call_id: U64,
    part: U16,
    argument_count: u8,
    result_kind: u8,
    command_len: U16,
    body_len: U16,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ToolArgument<'a> {
    Utf8(&'a str),
    Path(&'a str),
    U64(u64),
    Bool(bool),
}

impl ToolArgument<'_> {
    fn encoded_len(self) -> usize {
        AGENT_ARGUMENT_HEADER_LEN
            + match self {
                Self::Utf8(value) | Self::Path(value) => value.len(),
                Self::U64(_) => 8,
                Self::Bool(_) => 1,
            }
    }

    /// The inert wire type, for validation against an installed tool schema.
    pub const fn kind(self) -> ToolArgumentKind {
        match self {
            Self::Utf8(_) => ToolArgumentKind::Utf8,
            Self::Path(_) => ToolArgumentKind::Path,
            Self::U64(_) => ToolArgumentKind::U64,
            Self::Bool(_) => ToolArgumentKind::Bool,
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AgentRequestView<'a> {
    pub kind: AgentRequestKind,
    pub turn: u64,
    pub call_id: u64,
    pub result: Option<ToolResultKind>,
    /// Counted versioned descriptors from the calling task's installed executable
    /// scope. User-text and tool-result records may carry the current inert
    /// catalog so revocation and use depletion are visible on every round.
    pub tools: &'a [u8],
    /// Counted inert mappings for the exact workspace objects installed in
    /// the task-private filesystem view.
    pub resources: &'a [u8],
    pub text: &'a str,
}

impl AgentRequestView<'_> {
    pub fn encode(&self, output: &mut [u8]) -> Option<usize> {
        let tool_count = validate_tool_catalog(self.tools)?;
        let resource_count = validate_resource_catalog(self.resources)?;
        let valid = match self.kind {
            AgentRequestKind::UserText => {
                self.call_id == 0 && self.result.is_none() && !self.text.is_empty()
            }
            AgentRequestKind::ToolResult => self.call_id != 0 && self.result.is_some(),
            AgentRequestKind::Cancel => {
                self.call_id == 0
                    && self.result.is_none()
                    && self.text.is_empty()
                    && tool_count == 0
                    && resource_count == 0
            }
        };
        let body_len = self
            .tools
            .len()
            .checked_add(self.resources.len())?
            .checked_add(self.text.len())?;
        if self.turn == 0 || !valid || body_len > AGENT_TEXT_MAX {
            return None;
        }
        let length = AGENT_HEADER_LEN.checked_add(body_len)?;
        let output = output.get_mut(..length)?;
        let header = AgentHeaderV1 {
            magic: U32::new(AGENT_REQUEST_MAGIC),
            version: U16::new(AGENT_SCHEMA_V3),
            kind: self.kind as u8,
            flags: u8::from(tool_count != 0) | (u8::from(resource_count != 0) << 1),
            turn: U64::new(self.turn),
            call_id: U64::new(self.call_id),
            part: U16::new(self.resources.len() as u16),
            argument_count: tool_count,
            result_kind: self.result.map_or(0, |kind| kind as u8),
            command_len: U16::new(self.tools.len() as u16),
            body_len: U16::new(body_len as u16),
        };
        output[..AGENT_HEADER_LEN].copy_from_slice(header.as_bytes());
        let catalog_end = AGENT_HEADER_LEN + self.tools.len();
        output[AGENT_HEADER_LEN..catalog_end].copy_from_slice(self.tools);
        let resources_end = catalog_end + self.resources.len();
        output[catalog_end..resources_end].copy_from_slice(self.resources);
        output[resources_end..].copy_from_slice(self.text.as_bytes());
        Some(length)
    }

    pub fn tools(&self) -> AgentTools<'_> {
        AgentTools {
            remaining: validate_tool_catalog(self.tools).unwrap_or(0),
            input: self.tools,
        }
    }

    pub fn resources(&self) -> AgentResources<'_> {
        AgentResources {
            input: self.resources,
        }
    }
}

fn validate_tool_catalog(tools: &[u8]) -> Option<u8> {
    if tools.is_empty() {
        return Some(0);
    }
    let mut input = tools;
    let mut names = [[0_u8; AGENT_COMMAND_MAX]; AGENT_TOOLS_MAX];
    let mut lengths = [0_u8; AGENT_TOOLS_MAX];
    let mut count = 0_usize;
    while !input.is_empty() {
        if count == AGENT_TOOLS_MAX {
            return None;
        }
        let mut iterator = AgentTools {
            remaining: 1,
            input,
        };
        let tool = iterator.next()?;
        if names[..count]
            .iter()
            .zip(lengths[..count].iter())
            .any(|(prior, length)| prior[..usize::from(*length)] == *tool.name.as_bytes())
        {
            return None;
        }
        names[count][..tool.name.len()].copy_from_slice(tool.name.as_bytes());
        lengths[count] = tool.name.len() as u8;
        input = iterator.input;
        count += 1;
    }
    Some(count as u8)
}

fn validate_resource_catalog(resources: &[u8]) -> Option<u8> {
    let mut iterator = AgentResources { input: resources };
    let mut paths = [[0_u8; AGENT_RESOURCE_PATH_MAX]; AGENT_RESOURCES_MAX];
    let mut path_lengths = [0_u8; AGENT_RESOURCES_MAX];
    let mut locals = [[0_u8; AGENT_RESOURCE_PATH_MAX]; AGENT_RESOURCES_MAX];
    let mut local_lengths = [0_u8; AGENT_RESOURCES_MAX];
    let mut count = 0_usize;
    while !iterator.input.is_empty() {
        if count == AGENT_RESOURCES_MAX {
            return None;
        }
        let resource = iterator.next()?;
        if paths[..count]
            .iter()
            .zip(path_lengths[..count].iter())
            .any(|(prior, length)| prior[..usize::from(*length)] == *resource.path.as_bytes())
            || locals[..count]
                .iter()
                .zip(local_lengths[..count].iter())
                .any(|(prior, length)| {
                    prior[..usize::from(*length)] == *resource.local_path.as_bytes()
                })
        {
            return None;
        }
        paths[count][..resource.path.len()].copy_from_slice(resource.path.as_bytes());
        path_lengths[count] = resource.path.len() as u8;
        locals[count][..resource.local_path.len()].copy_from_slice(resource.local_path.as_bytes());
        local_lengths[count] = resource.local_path.len() as u8;
        count += 1;
    }
    Some(count as u8)
}

#[must_use]
pub fn decode_agent_request(input: &[u8]) -> Option<AgentRequestView<'_>> {
    let header = AgentHeaderV1::read_from_bytes(input.get(..AGENT_HEADER_LEN)?).ok()?;
    let body_len = header.body_len.get() as usize;
    let catalog_len = header.command_len.get() as usize;
    let resources_len = header.part.get() as usize;
    if header.magic.get() != AGENT_REQUEST_MAGIC
        || header.version.get() != AGENT_SCHEMA_V3
        || header.flags & !3 != 0
        || header.turn.get() == 0
        || header.argument_count as usize > AGENT_TOOLS_MAX
        || catalog_len.checked_add(resources_len)? > body_len
        || body_len > AGENT_TEXT_MAX
        || input.len() != AGENT_HEADER_LEN.checked_add(body_len)?
    {
        return None;
    }
    let kind = AgentRequestKind::from_wire(header.kind)?;
    let body = &input[AGENT_HEADER_LEN..];
    let tools = &body[..catalog_len];
    let resources_end = catalog_len + resources_len;
    let resources = &body[catalog_len..resources_end];
    let text = str::from_utf8(&body[resources_end..]).ok()?;
    let tool_count = validate_tool_catalog(tools)?;
    let resource_count = validate_resource_catalog(resources)?;
    if tool_count != header.argument_count
        || (header.flags & 1 != 0) != (tool_count != 0)
        || (header.flags & 2 != 0) != (resource_count != 0)
    {
        return None;
    }
    let result = if header.result_kind == 0 {
        None
    } else {
        Some(ToolResultKind::from_wire(header.result_kind)?)
    };
    let valid = match kind {
        AgentRequestKind::UserText => {
            header.call_id.get() == 0 && result.is_none() && !text.is_empty()
        }
        AgentRequestKind::ToolResult => header.call_id.get() != 0 && result.is_some(),
        AgentRequestKind::Cancel => {
            header.call_id.get() == 0
                && result.is_none()
                && text.is_empty()
                && tool_count == 0
                && resource_count == 0
        }
    };
    valid.then_some(AgentRequestView {
        kind,
        turn: header.turn.get(),
        call_id: header.call_id.get(),
        result,
        tools,
        resources,
        text,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AgentEventView<'a> {
    pub kind: AgentEventKind,
    pub turn: u64,
    pub part: u16,
    pub call_id: u64,
    pub command: &'a str,
    pub arguments: &'a [ToolArgument<'a>],
    pub text: &'a str,
}

impl AgentEventView<'_> {
    pub fn encode(&self, output: &mut [u8]) -> Option<usize> {
        if self.turn == 0 || self.arguments.len() > AGENT_ARGUMENTS_MAX {
            return None;
        }
        let shape = match self.kind {
            AgentEventKind::AssistantText => {
                self.call_id == 0
                    && self.command.is_empty()
                    && self.arguments.is_empty()
                    && !self.text.is_empty()
            }
            AgentEventKind::Respond => {
                self.call_id == 0
                    && self.command.is_empty()
                    && self.arguments.is_empty()
                    && AssistantMessage::new(self.text).is_some()
            }
            AgentEventKind::ToolCall => {
                self.call_id != 0
                    && !self.command.is_empty()
                    && self.command.len() <= AGENT_COMMAND_MAX
                    && self.text.is_empty()
            }
            AgentEventKind::Refusal => {
                self.call_id == 0 && self.command.is_empty() && self.arguments.is_empty()
            }
            AgentEventKind::Cancelled | AgentEventKind::Completed => {
                self.call_id == 0
                    && self.command.is_empty()
                    && self.arguments.is_empty()
                    && self.text.is_empty()
            }
        };
        if !shape {
            return None;
        }
        let mut body_len = self.command.len().checked_add(self.text.len())?;
        for argument in self.arguments {
            body_len = body_len.checked_add(argument.encoded_len())?;
        }
        if body_len > AGENT_TEXT_MAX {
            return None;
        }
        let length = AGENT_HEADER_LEN.checked_add(body_len)?;
        let output = output.get_mut(..length)?;
        let header = AgentHeaderV1 {
            magic: U32::new(AGENT_EVENT_MAGIC),
            version: U16::new(AGENT_SCHEMA_V2),
            kind: self.kind as u8,
            flags: u8::from(self.kind.terminal()),
            turn: U64::new(self.turn),
            call_id: U64::new(self.call_id),
            part: U16::new(self.part),
            argument_count: self.arguments.len() as u8,
            result_kind: 0,
            command_len: U16::new(self.command.len() as u16),
            body_len: U16::new(body_len as u16),
        };
        output[..AGENT_HEADER_LEN].copy_from_slice(header.as_bytes());
        let mut at = AGENT_HEADER_LEN;
        output[at..at + self.command.len()].copy_from_slice(self.command.as_bytes());
        at += self.command.len();
        for argument in self.arguments {
            let payload_len = argument.encoded_len() - AGENT_ARGUMENT_HEADER_LEN;
            output[at] = argument.kind() as u8;
            output[at + 1] = 0;
            output[at + 2..at + 4].copy_from_slice(&(payload_len as u16).to_le_bytes());
            at += AGENT_ARGUMENT_HEADER_LEN;
            match argument {
                ToolArgument::Utf8(value) | ToolArgument::Path(value) => {
                    output[at..at + value.len()].copy_from_slice(value.as_bytes())
                }
                ToolArgument::U64(value) => {
                    output[at..at + 8].copy_from_slice(&value.to_le_bytes())
                }
                ToolArgument::Bool(value) => output[at] = u8::from(*value),
            }
            at += payload_len;
        }
        output[at..at + self.text.len()].copy_from_slice(self.text.as_bytes());
        Some(length)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct DecodedAgentEvent<'a> {
    pub kind: AgentEventKind,
    pub turn: u64,
    pub part: u16,
    pub call_id: u64,
    pub command: &'a str,
    arguments: &'a [u8],
    argument_count: u8,
    pub text: &'a str,
}

impl DecodedAgentEvent<'_> {
    pub fn arguments(&self) -> ToolArguments<'_> {
        ToolArguments {
            bytes: self.arguments,
            remaining: self.argument_count,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct ToolArguments<'a> {
    bytes: &'a [u8],
    remaining: u8,
}

impl<'a> Iterator for ToolArguments<'a> {
    type Item = ToolArgument<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining == 0 {
            return None;
        }
        let header = self.bytes.get(..AGENT_ARGUMENT_HEADER_LEN)?;
        if header[1] != 0 {
            return None;
        }
        let kind = ToolArgumentKind::from_wire(header[0])?;
        let len = u16::from_le_bytes(header[2..4].try_into().ok()?) as usize;
        let end = AGENT_ARGUMENT_HEADER_LEN.checked_add(len)?;
        let argument = decode_argument(kind, self.bytes.get(AGENT_ARGUMENT_HEADER_LEN..end)?)?;
        self.bytes = &self.bytes[end..];
        self.remaining -= 1;
        Some(argument)
    }
}

fn decode_argument(kind: ToolArgumentKind, value: &[u8]) -> Option<ToolArgument<'_>> {
    match kind {
        ToolArgumentKind::Utf8 => Some(ToolArgument::Utf8(str::from_utf8(value).ok()?)),
        ToolArgumentKind::Path => {
            let path = str::from_utf8(value).ok()?;
            (!path.is_empty() && value.first() == Some(&b'/')).then_some(ToolArgument::Path(path))
        }
        ToolArgumentKind::U64 => Some(ToolArgument::U64(u64::from_le_bytes(
            value.try_into().ok()?,
        ))),
        ToolArgumentKind::Bool => match value {
            [0] => Some(ToolArgument::Bool(false)),
            [1] => Some(ToolArgument::Bool(true)),
            _ => None,
        },
    }
}

#[must_use]
pub fn decode_agent_event(input: &[u8]) -> Option<DecodedAgentEvent<'_>> {
    let header = AgentHeaderV1::read_from_bytes(input.get(..AGENT_HEADER_LEN)?).ok()?;
    let body_len = header.body_len.get() as usize;
    let command_len = header.command_len.get() as usize;
    if header.magic.get() != AGENT_EVENT_MAGIC
        || header.version.get() != AGENT_SCHEMA_V2
        || header.flags & !1 != 0
        || header.turn.get() == 0
        || header.result_kind != 0
        || header.argument_count as usize > AGENT_ARGUMENTS_MAX
        || command_len > AGENT_COMMAND_MAX
        || body_len > AGENT_TEXT_MAX
        || command_len > body_len
        || input.len() != AGENT_HEADER_LEN.checked_add(body_len)?
    {
        return None;
    }
    let kind = AgentEventKind::from_wire(header.kind)?;
    if (header.flags & 1 != 0) != kind.terminal() {
        return None;
    }
    let body = &input[AGENT_HEADER_LEN..];
    let command = str::from_utf8(&body[..command_len]).ok()?;
    let mut at = command_len;
    for _ in 0..header.argument_count {
        let arg = body.get(at..at.checked_add(AGENT_ARGUMENT_HEADER_LEN)?)?;
        if arg[1] != 0 {
            return None;
        }
        let argument_kind = ToolArgumentKind::from_wire(arg[0])?;
        let len = u16::from_le_bytes(arg[2..4].try_into().ok()?) as usize;
        let start = at.checked_add(AGENT_ARGUMENT_HEADER_LEN)?;
        let end = start.checked_add(len)?;
        decode_argument(argument_kind, body.get(start..end)?)?;
        at = end;
    }
    let arguments = &body[command_len..at];
    let text = str::from_utf8(&body[at..]).ok()?;
    let shape = match kind {
        AgentEventKind::AssistantText => {
            header.call_id.get() == 0
                && command.is_empty()
                && header.argument_count == 0
                && !text.is_empty()
        }
        AgentEventKind::Respond => {
            header.call_id.get() == 0
                && command.is_empty()
                && header.argument_count == 0
                && AssistantMessage::new(text).is_some()
        }
        AgentEventKind::ToolCall => {
            header.call_id.get() != 0 && !command.is_empty() && text.is_empty()
        }
        AgentEventKind::Refusal => {
            header.call_id.get() == 0 && command.is_empty() && header.argument_count == 0
        }
        AgentEventKind::Cancelled | AgentEventKind::Completed => {
            header.call_id.get() == 0
                && command.is_empty()
                && header.argument_count == 0
                && text.is_empty()
        }
    };
    shape.then_some(DecodedAgentEvent {
        kind,
        turn: header.turn.get(),
        part: header.part.get(),
        call_id: header.call_id.get(),
        command,
        arguments,
        argument_count: header.argument_count,
        text,
    })
}

const _: () = assert!(core::mem::size_of::<AgentHeaderV1>() == AGENT_HEADER_LEN);

pub const AGENT_CALL_HISTORY: usize = 16;
pub const AGENT_REQUEST_HISTORY: usize = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentStateError {
    WrongRoute,
    WrongGeneration,
    WrongRequest,
    DuplicateRequest,
    DuplicateCall,
    OutOfOrder,
    Late,
    WrongState,
    Capacity,
    Overflow,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum AgentTurnPhase {
    Ready,
    Inference {
        request: RequestId,
        turn: u64,
        next_part: u16,
    },
    AwaitingTool {
        turn: u64,
        call_id: u64,
    },
    Terminal(AgentEventKind),
    Retired,
}

/// Guest-side validator for one founded runner endpoint.
///
/// The route and generation are correlation checks around an already-held
/// endpoint; they never locate authority. Every mutating operation validates
/// against a temporary next state first, so a refusal cannot advance a turn,
/// part, request, or tool-call cursor.
pub struct AgentTurnMachine {
    route: RouteId,
    generation: TaskGeneration,
    phase: AgentTurnPhase,
    requests: heapless::Vec<RequestId, AGENT_REQUEST_HISTORY>,
    calls: heapless::Vec<u64, AGENT_CALL_HISTORY>,
    next_user_turn: u64,
}

impl AgentTurnMachine {
    pub const fn new(route: RouteId, generation: TaskGeneration) -> Option<Self> {
        if route.is_zero() {
            return None;
        }
        Some(Self {
            route,
            generation,
            phase: AgentTurnPhase::Ready,
            requests: heapless::Vec::new(),
            calls: heapless::Vec::new(),
            next_user_turn: 1,
        })
    }

    pub const fn phase(&self) -> AgentTurnPhase {
        self.phase
    }

    pub const fn next_user_turn(&self) -> u64 {
        self.next_user_turn
    }

    fn identity(&self, route: RouteId, generation: TaskGeneration) -> Result<(), AgentStateError> {
        if route != self.route {
            Err(AgentStateError::WrongRoute)
        } else if generation != self.generation {
            Err(AgentStateError::WrongGeneration)
        } else {
            Ok(())
        }
    }

    pub fn submit(
        &mut self,
        route: RouteId,
        generation: TaskGeneration,
        request: RequestId,
        message: AgentRequestView<'_>,
    ) -> Result<(), AgentStateError> {
        self.identity(route, generation)?;
        if request.is_zero() {
            return Err(AgentStateError::WrongRequest);
        }
        if self.requests.contains(&request) {
            return Err(AgentStateError::DuplicateRequest);
        }
        if self.requests.is_full() {
            return Err(AgentStateError::Capacity);
        }
        let next = match (self.phase, message.kind) {
            (AgentTurnPhase::Ready, AgentRequestKind::UserText)
                if message.turn == self.next_user_turn =>
            {
                AgentTurnPhase::Inference {
                    request,
                    turn: message.turn,
                    next_part: 0,
                }
            }
            (
                AgentTurnPhase::Terminal(AgentEventKind::Completed | AgentEventKind::Respond),
                AgentRequestKind::UserText,
            ) if message.turn == self.next_user_turn => AgentTurnPhase::Inference {
                request,
                turn: message.turn,
                next_part: 0,
            },
            (AgentTurnPhase::AwaitingTool { turn, call_id }, AgentRequestKind::ToolResult)
                if message.call_id == call_id =>
            {
                let expected = turn.checked_add(1).ok_or(AgentStateError::Overflow)?;
                if message.turn != expected {
                    return Err(AgentStateError::OutOfOrder);
                }
                AgentTurnPhase::Inference {
                    request,
                    turn: message.turn,
                    next_part: 0,
                }
            }
            (AgentTurnPhase::Inference { .. }, _) => return Err(AgentStateError::WrongState),
            (AgentTurnPhase::Terminal(_) | AgentTurnPhase::Retired, _) => {
                return Err(AgentStateError::Late);
            }
            _ => return Err(AgentStateError::OutOfOrder),
        };
        self.requests
            .push(request)
            .map_err(|_| AgentStateError::Capacity)?;
        self.phase = next;
        Ok(())
    }

    pub fn accept_event(
        &mut self,
        route: RouteId,
        generation: TaskGeneration,
        request: RequestId,
        event: DecodedAgentEvent<'_>,
    ) -> Result<(), AgentStateError> {
        self.identity(route, generation)?;
        let AgentTurnPhase::Inference {
            request: active,
            turn,
            next_part,
        } = self.phase
        else {
            return Err(match self.phase {
                AgentTurnPhase::Terminal(_) | AgentTurnPhase::Retired => AgentStateError::Late,
                _ => AgentStateError::WrongState,
            });
        };
        if request != active {
            return Err(AgentStateError::WrongRequest);
        }
        if event.turn != turn || event.part != next_part {
            return Err(AgentStateError::OutOfOrder);
        }
        let mut next_user_turn = self.next_user_turn;
        let next = match event.kind {
            AgentEventKind::AssistantText => AgentTurnPhase::Inference {
                request: active,
                turn,
                next_part: next_part.checked_add(1).ok_or(AgentStateError::Overflow)?,
            },
            AgentEventKind::ToolCall => {
                if self.calls.contains(&event.call_id) {
                    return Err(AgentStateError::DuplicateCall);
                }
                if self.calls.is_full() {
                    return Err(AgentStateError::Capacity);
                }
                AgentTurnPhase::AwaitingTool {
                    turn,
                    call_id: event.call_id,
                }
            }
            terminal @ (AgentEventKind::Refusal | AgentEventKind::Cancelled) => {
                AgentTurnPhase::Terminal(terminal)
            }
            AgentEventKind::Completed => {
                next_user_turn = turn.checked_add(1).ok_or(AgentStateError::Overflow)?;
                AgentTurnPhase::Terminal(AgentEventKind::Completed)
            }
            AgentEventKind::Respond => {
                next_user_turn = turn.checked_add(1).ok_or(AgentStateError::Overflow)?;
                AgentTurnPhase::Terminal(AgentEventKind::Respond)
            }
        };
        if event.kind == AgentEventKind::ToolCall {
            self.calls
                .push(event.call_id)
                .map_err(|_| AgentStateError::Capacity)?;
        }
        self.next_user_turn = next_user_turn;
        self.phase = next;
        Ok(())
    }

    pub fn parent_cancel(
        &mut self,
        route: RouteId,
        generation: TaskGeneration,
    ) -> Result<(), AgentStateError> {
        self.identity(route, generation)?;
        match self.phase {
            AgentTurnPhase::Ready
            | AgentTurnPhase::Inference { .. }
            | AgentTurnPhase::AwaitingTool { .. } => {
                self.phase = AgentTurnPhase::Terminal(AgentEventKind::Cancelled);
                Ok(())
            }
            AgentTurnPhase::Terminal(_) | AgentTurnPhase::Retired => Err(AgentStateError::Late),
        }
    }

    pub fn retire(&mut self) {
        self.phase = AgentTurnPhase::Retired;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tool_catalog<'a>(output: &'a mut [u8], names: &[&str]) -> &'a [u8] {
        let mut length = 0;
        for name in names {
            let descriptor = AgentToolDescriptor {
                name,
                usage: name,
                summary: "test tool",
                input: ToolInputSchema {
                    minimum: 0,
                    maximum: 1,
                    kinds: [ToolArgumentKind::Path.mask(), 0, 0, 0],
                    whitespace: 0,
                },
                effects: 1,
                result_sensitivity: 1,
                remaining_uses: 3,
            };
            length += descriptor.encode(&mut output[length..]).unwrap();
        }
        &output[..length]
    }

    fn resource_catalog<'a>(output: &'a mut [u8], resources: &[(&str, &str)]) -> &'a [u8] {
        let mut length = 0;
        for (path, local_path) in resources {
            length += AgentResourceDescriptor { path, local_path }
                .encode(&mut output[length..])
                .unwrap();
        }
        &output[..length]
    }

    #[test]
    fn a_resource_can_describe_the_installed_private_root() {
        let mut bytes = [0; 64];
        let descriptor = AgentResourceDescriptor {
            path: "project",
            local_path: "/",
        };
        let len = descriptor.encode(&mut bytes).unwrap();
        let decoded = decode_agent_resources(&bytes[..len])
            .unwrap()
            .next()
            .unwrap();
        assert_eq!(decoded, descriptor);
        for local_path in ["//", "/./", "/../", "/project/", ""] {
            assert!(
                AgentResourceDescriptor {
                    path: "project",
                    local_path
                }
                .encode(&mut bytes)
                .is_none()
            );
        }
    }

    #[test]
    fn exact_resource_catalog_round_trips_nested_duplicate_basenames() {
        let mut resources = [0_u8; 512];
        let resources = resource_catalog(
            &mut resources,
            &[
                ("alpha/data.txt", "/resource-0-3"),
                ("beta/data.txt", "/resource-0-9"),
            ],
        );
        let request = AgentRequestView {
            kind: AgentRequestKind::UserText,
            turn: 1,
            call_id: 0,
            result: None,
            tools: &[],
            resources,
            text: "compare both files",
        };
        let mut wire = [0; crate::INFERENCE_PART_MAX];
        let len = request.encode(&mut wire).unwrap();
        let decoded = decode_agent_request(&wire[..len]).unwrap();
        assert_eq!(
            decoded
                .resources()
                .map(|resource| (resource.path, resource.local_path))
                .collect::<heapless::Vec<_, 2>>(),
            [
                ("alpha/data.txt", "/resource-0-3"),
                ("beta/data.txt", "/resource-0-9"),
            ]
        );

        let mut duplicates = [0_u8; 512];
        let duplicates = resource_catalog(
            &mut duplicates,
            &[
                ("alpha/data.txt", "/resource-0-3"),
                ("beta/data.txt", "/resource-0-3"),
            ],
        );
        assert!(
            AgentRequestView {
                kind: AgentRequestKind::UserText,
                turn: 1,
                call_id: 0,
                result: None,
                tools: &[],
                resources: duplicates,
                text: "compare",
            }
            .encode(&mut wire)
            .is_none()
        );

        let mut malformed = [0_u8; 512];
        for (path, local_path) in [
            ("../alpha/data.txt", "/resource-0-3"),
            ("alpha/./data.txt", "/resource-0-3"),
            ("alpha\\data.txt", "/resource-0-3"),
            ("alpha/data.txt", "/../resource-0-3"),
            ("alpha/data.txt", "/resource-0-3/"),
        ] {
            assert!(
                AgentResourceDescriptor { path, local_path }
                    .encode(&mut malformed)
                    .is_none(),
                "non-canonical mapping {path:?} -> {local_path:?} must fail closed"
            );
        }
    }

    #[test]
    fn tool_descriptor_rejects_unknown_policy_values_and_exhausted_rows() {
        let descriptor = AgentToolDescriptor {
            name: "inspect-r7",
            usage: "inspect-r7",
            summary: "Inspect current state.",
            input: ToolInputSchema::NONE,
            effects: TOOL_EFFECT_OBSERVE,
            result_sensitivity: TOOL_RESULT_PUBLIC,
            remaining_uses: 1,
        };
        let mut wire = [0_u8; 256];
        assert!(descriptor.encode(&mut wire).is_some());
        assert!(
            AgentToolDescriptor {
                effects: TOOL_EFFECT_MASK | (1 << 15),
                ..descriptor
            }
            .encode(&mut wire)
            .is_none()
        );
        assert!(
            AgentToolDescriptor {
                result_sensitivity: TOOL_RESULT_WORKSPACE + 1,
                ..descriptor
            }
            .encode(&mut wire)
            .is_none()
        );
        assert!(
            AgentToolDescriptor {
                remaining_uses: 0,
                ..descriptor
            }
            .encode(&mut wire)
            .is_none()
        );
    }

    #[test]
    fn tool_call_round_trips_typed_arguments() {
        let args = [
            ToolArgument::Path("/private/input"),
            ToolArgument::Utf8("metadata"),
            ToolArgument::U64(17),
            ToolArgument::Bool(true),
        ];
        let event = AgentEventView {
            kind: AgentEventKind::ToolCall,
            turn: 3,
            part: 2,
            call_id: 91,
            command: "stat",
            arguments: &args,
            text: "",
        };
        let mut wire = [0; crate::INFERENCE_PART_MAX];
        let len = event.encode(&mut wire).unwrap();
        let decoded = decode_agent_event(&wire[..len]).unwrap();
        assert_eq!(decoded.command, "stat");
        assert_eq!(decoded.arguments().collect::<heapless::Vec<_, 4>>(), args);
    }

    #[test]
    fn zero_argument_tool_call_round_trips() {
        let event = AgentEventView {
            kind: AgentEventKind::ToolCall,
            turn: 1,
            part: 0,
            call_id: 1,
            command: "status-r9",
            arguments: &[],
            text: "",
        };
        let mut wire = [0; crate::INFERENCE_PART_MAX];
        let len = event.encode(&mut wire).unwrap();
        let decoded = decode_agent_event(&wire[..len]).unwrap();
        assert_eq!(decoded.command, "status-r9");
        assert_eq!(decoded.arguments().count(), 0);
    }

    #[test]
    fn bounded_respond_round_trips_as_terminal_assistant_message() {
        let message = AssistantMessage::new("Hello! How can I help?").unwrap();
        let event = AgentEventView {
            kind: AgentEventKind::Respond,
            turn: 1,
            part: 0,
            call_id: 0,
            command: "",
            arguments: &[],
            text: message.text(),
        };
        let mut wire = [0; crate::INFERENCE_PART_MAX];
        let len = event.encode(&mut wire).unwrap();
        let decoded = decode_agent_event(&wire[..len]).unwrap();
        assert_eq!(decoded.kind, AgentEventKind::Respond);
        assert_eq!(decoded.text, "Hello! How can I help?");
        assert!(decoded.kind.terminal());
        assert!(AssistantMessage::new("").is_none());
        assert!(AssistantMessage::new("hello\0world").is_none());
        let oversized = "x".repeat(AGENT_TEXT_MAX + 1);
        assert!(AssistantMessage::new(&oversized).is_none());
    }

    #[test]
    fn reserved_length_utf8_and_shape_errors_fail_closed() {
        let mut catalog = [0; 512];
        let catalog = tool_catalog(&mut catalog, &["inspect-a", "inspect-b"]);
        let request = AgentRequestView {
            kind: AgentRequestKind::UserText,
            turn: 1,
            call_id: 0,
            result: None,
            tools: catalog,
            resources: &[],
            text: "inspect",
        };
        let mut wire = [0; crate::INFERENCE_PART_MAX];
        let len = request.encode(&mut wire).unwrap();
        let decoded = decode_agent_request(&wire[..len]).unwrap();
        assert_eq!(
            decoded
                .tools()
                .map(|tool| tool.name)
                .collect::<heapless::Vec<_, 4>>(),
            ["inspect-a", "inspect-b"]
        );
        let descriptor = decoded.tools().next().unwrap();
        assert_eq!(descriptor.usage, "inspect-a");
        assert_eq!(descriptor.summary, "test tool");
        assert_eq!(descriptor.input.minimum, 0);
        assert_eq!(descriptor.input.maximum, 1);
        assert!(descriptor.input.accepts(0, ToolArgumentKind::Path));
        assert_eq!(descriptor.remaining_uses, 3);
        for index in [4, 5, 7, 24, 25, 26, 27, 28, 29, 30, 31] {
            let mut bad = wire[..len].to_vec();
            bad[index] ^= 0x80;
            assert!(decode_agent_request(&bad).is_none(), "byte {index}");
        }
        let mut bad = wire[..len].to_vec();
        bad[AGENT_HEADER_LEN] = 0xff;
        assert!(decode_agent_request(&bad).is_none());
        assert!(decode_agent_request(&wire[..len - 1]).is_none());
    }

    fn route(byte: u8) -> RouteId {
        RouteId([byte; 16])
    }

    fn generation(value: u64) -> TaskGeneration {
        TaskGeneration::new(value).unwrap()
    }

    fn decode_event<'a>(wire: &'a mut [u8], event: &AgentEventView<'_>) -> DecodedAgentEvent<'a> {
        let len = event.encode(wire).unwrap();
        decode_agent_event(&wire[..len]).unwrap()
    }

    #[test]
    fn turn_machine_rejects_replay_cross_route_and_stale_generation_without_advancing() {
        let mut machine = AgentTurnMachine::new(route(1), generation(7)).unwrap();
        let user = AgentRequestView {
            kind: AgentRequestKind::UserText,
            turn: 1,
            call_id: 0,
            result: None,
            tools: &[],
            resources: &[],
            text: "read it",
        };
        let request = crate::RequestId([9; 16]);
        assert_eq!(
            machine.submit(route(2), generation(7), request, user),
            Err(AgentStateError::WrongRoute)
        );
        assert_eq!(machine.phase(), AgentTurnPhase::Ready);
        machine
            .submit(route(1), generation(7), request, user)
            .unwrap();
        assert_eq!(
            machine.submit(route(1), generation(7), request, user),
            Err(AgentStateError::DuplicateRequest)
        );

        let text = AgentEventView {
            kind: AgentEventKind::AssistantText,
            turn: 1,
            part: 0,
            call_id: 0,
            command: "",
            arguments: &[],
            text: "checking",
        };
        let mut wire = [0; crate::INFERENCE_PART_MAX];
        let decoded = decode_event(&mut wire, &text);
        assert_eq!(
            machine.accept_event(route(1), generation(6), request, decoded),
            Err(AgentStateError::WrongGeneration)
        );
        assert_eq!(
            machine.phase(),
            AgentTurnPhase::Inference {
                request,
                turn: 1,
                next_part: 0
            }
        );
        machine
            .accept_event(route(1), generation(7), request, decoded)
            .unwrap();
        assert_eq!(
            machine.accept_event(route(1), generation(7), request, decoded),
            Err(AgentStateError::OutOfOrder)
        );
    }

    #[test]
    fn tool_round_continues_then_completes_and_late_records_fail_closed() {
        let mut machine = AgentTurnMachine::new(route(1), generation(1)).unwrap();
        let first = crate::RequestId([1; 16]);
        machine
            .submit(
                route(1),
                generation(1),
                first,
                AgentRequestView {
                    kind: AgentRequestKind::UserText,
                    turn: 1,
                    call_id: 0,
                    result: None,
                    tools: &[],
                    resources: &[],
                    text: "stat /private/a",
                },
            )
            .unwrap();
        let args = [ToolArgument::Path("/private/a")];
        let call = AgentEventView {
            kind: AgentEventKind::ToolCall,
            turn: 1,
            part: 0,
            call_id: 44,
            command: "stat",
            arguments: &args,
            text: "",
        };
        let mut call_wire = [0; crate::INFERENCE_PART_MAX];
        let call = decode_event(&mut call_wire, &call);
        machine
            .accept_event(route(1), generation(1), first, call)
            .unwrap();

        let second = crate::RequestId([2; 16]);
        machine
            .submit(
                route(1),
                generation(1),
                second,
                AgentRequestView {
                    kind: AgentRequestKind::ToolResult,
                    turn: 2,
                    call_id: 44,
                    result: Some(ToolResultKind::Output),
                    tools: &[],
                    resources: &[],
                    text: "size=7",
                },
            )
            .unwrap();
        let completed = AgentEventView {
            kind: AgentEventKind::Completed,
            turn: 2,
            part: 0,
            call_id: 0,
            command: "",
            arguments: &[],
            text: "",
        };
        let mut completed_wire = [0; crate::INFERENCE_PART_MAX];
        let completed = decode_event(&mut completed_wire, &completed);
        machine
            .accept_event(route(1), generation(1), second, completed)
            .unwrap();
        assert_eq!(
            machine.phase(),
            AgentTurnPhase::Terminal(AgentEventKind::Completed)
        );
        assert_eq!(
            machine.accept_event(route(1), generation(1), second, completed),
            Err(AgentStateError::Late)
        );
        let third = crate::RequestId([3; 16]);
        machine
            .submit(
                route(1),
                generation(1),
                third,
                AgentRequestView {
                    kind: AgentRequestKind::UserText,
                    turn: 3,
                    call_id: 0,
                    result: None,
                    tools: &[],
                    resources: &[],
                    text: "read the newly approved object",
                },
            )
            .unwrap();
        assert_eq!(
            machine.phase(),
            AgentTurnPhase::Inference {
                request: third,
                turn: 3,
                next_part: 0,
            }
        );
    }
}
