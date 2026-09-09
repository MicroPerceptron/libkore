use kore_abi::{
    cap::{ResourceKind, Rights as CapRights},
    session::SlotRef,
};
use kore_rt::cap::{ChanSendToken, ChannelRoot, verbs};

enum ViewEndpoint {
    Owned(ChanSendToken),
    Retiring(ChanSendToken),
    Rooted {
        root: ChannelRoot,
        endpoint: ChanSendToken,
    },
    Held {
        slot: SlotRef,
        rights: CapRights,
    },
}

impl ViewEndpoint {
    fn slot(&self) -> SlotRef {
        match self {
            Self::Owned(endpoint) | Self::Retiring(endpoint) => endpoint.slot(),
            Self::Rooted { endpoint, .. } => endpoint.slot(),
            Self::Held { slot, .. } => *slot,
        }
    }

    fn rights(&self) -> CapRights {
        match self {
            Self::Owned(endpoint) | Self::Retiring(endpoint) => endpoint.rights(),
            Self::Rooted { endpoint, .. } => endpoint.rights(),
            Self::Held { rights, .. } => *rights,
        }
    }
}

/// Affine authority to one native filesystem view/cwd binding.
pub struct FsViewCapability {
    endpoint: ViewEndpoint,
}

/// A recipient selector intentionally carries no invocation or delegation
/// authority. It can authenticate a task-view target for one grant request,
/// but cannot be substituted as that task's cwd endpoint.
pub const FS_GRANT_TARGET_RIGHTS: CapRights = CapRights::REVOKE;
pub const FS_GRANT_TARGET_DELEGATE_RIGHTS: CapRights = CapRights::DERIVE.union(CapRights::REVOKE);

impl FsViewCapability {
    pub const DELEGABLE_RIGHTS: CapRights = CapRights::ACCESS
        .union(CapRights::DERIVE)
        .union(CapRights::REVOKE);
    pub const INVOCATION_RIGHTS: CapRights = CapRights::ACCESS.union(CapRights::REVOKE);
    pub const INVOCATION_DELEGATE_RIGHTS: CapRights =
        Self::INVOCATION_RIGHTS.union(CapRights::DERIVE);

    pub fn from_endpoint(endpoint: ChanSendToken) -> Result<Self, ChanSendToken> {
        if !endpoint.rights().contains(CapRights::ACCESS) {
            return Err(endpoint);
        }
        Ok(Self {
            endpoint: ViewEndpoint::Owned(endpoint),
        })
    }

    pub fn from_held(slot: SlotRef) -> Option<Self> {
        let rights = verbs::capability_rights(slot, ResourceKind::ChanSend)?;
        rights.contains(CapRights::ACCESS).then_some(Self {
            endpoint: ViewEndpoint::Held { slot, rights },
        })
    }

    pub fn from_rooted(
        root: ChannelRoot,
        endpoint: ChanSendToken,
    ) -> Result<Self, (ChannelRoot, ChanSendToken)> {
        if !endpoint
            .rights()
            .contains(CapRights::ACCESS | CapRights::DERIVE)
        {
            return Err((root, endpoint));
        }
        Ok(Self {
            endpoint: ViewEndpoint::Rooted { root, endpoint },
        })
    }

    pub fn rights(&self) -> CapRights {
        self.endpoint.rights()
    }

    pub fn derive_invocation(&self) -> Option<ChanSendToken> {
        verbs::derive_owned_chan_send_with_rights(
            self.endpoint.slot(),
            Self::INVOCATION_DELEGATE_RIGHTS,
        )
    }

    pub fn derive_binding(&self) -> Option<ChanSendToken> {
        verbs::derive_owned_chan_send_with_rights(
            self.endpoint.slot(),
            Self::INVOCATION_DELEGATE_RIGHTS,
        )
    }

    pub fn into_endpoint(self) -> Result<ChanSendToken, Self> {
        let Self { endpoint } = self;
        match endpoint {
            ViewEndpoint::Owned(endpoint) => Ok(endpoint),
            endpoint => Err(Self { endpoint }),
        }
    }

    pub fn try_retire(self) -> Result<(), Self> {
        let Self { endpoint } = self;
        match endpoint {
            ViewEndpoint::Held { slot, rights } => {
                if verbs::retire_slot_outcome(slot).is_closed() {
                    Ok(())
                } else {
                    Err(Self {
                        endpoint: ViewEndpoint::Held { slot, rights },
                    })
                }
            }
            ViewEndpoint::Owned(endpoint) => {
                endpoint.try_retire().map(|_| ()).map_err(|endpoint| Self {
                    endpoint: ViewEndpoint::Owned(endpoint),
                })
            }
            ViewEndpoint::Retiring(endpoint) => {
                endpoint.try_retire().map(|_| ()).map_err(|endpoint| Self {
                    endpoint: ViewEndpoint::Retiring(endpoint),
                })
            }
            ViewEndpoint::Rooted { root, endpoint } => match root.try_retire() {
                Ok(_) => endpoint.try_retire().map(|_| ()).map_err(|endpoint| Self {
                    endpoint: ViewEndpoint::Retiring(endpoint),
                }),
                Err(root) => Err(Self {
                    endpoint: ViewEndpoint::Rooted { root, endpoint },
                }),
            },
        }
    }
}
