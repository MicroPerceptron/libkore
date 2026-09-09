//! The `Bundle` derive's full call chain: declaration -> `PROFILE` -> boot
//! lookup -> claim/capture -> typed endowment.

use kore_abi::{
    session::SlotRef,
    task::{BootBlock, Bootstrap},
};
use kore_rt::service::{BindError, Bundle};

const REQUIRED: u16 = 0x7f00;
const OPTIONAL: u16 = 0x7f01;
const CAPTURED: u16 = 0x7f02;

#[derive(Debug, kore_rt::Bundle)]
#[bundle(name = "derived-test")]
#[allow(dead_code)]
struct DerivedBundle {
    #[bundle(role = REQUIRED, index = 1)]
    required: kore_rt::cap::SupplyToken,
    #[bundle(role = OPTIONAL, optional)]
    optional: Option<kore_rt::cap::SupplyToken>,
    #[bundle(role = CAPTURED, index = 2, capture, optional)]
    captured: Option<SlotRef>,
}

#[derive(Debug, kore_rt::Bundle)]
#[bundle(name = "capture-test")]
struct CaptureBundle {
    #[bundle(role = CAPTURED, index = 2, capture, optional)]
    captured: Option<SlotRef>,
}

#[derive(Debug, PartialEq, Eq, kore_rt::Bundle)]
#[bundle(name = "empty-test")]
struct EmptyBundle;

#[test]
fn declaration_is_profile_and_fail_closed_resolution() {
    assert_eq!(DerivedBundle::PROFILE.name, "derived-test");
    assert_eq!(DerivedBundle::PROFILE.roles.len(), 2);
    assert_eq!(DerivedBundle::PROFILE.roles[0].role, REQUIRED);
    assert_eq!(DerivedBundle::PROFILE.roles[0].index, 1);
    assert_eq!(DerivedBundle::PROFILE.roles[1].role, OPTIONAL);
    assert_eq!(DerivedBundle::PROFILE.roles[1].index, 0);
    assert_eq!(DerivedBundle::WIRING.len(), 1);
    assert_eq!(DerivedBundle::WIRING[0].role, CAPTURED);
    assert_eq!(DerivedBundle::WIRING[0].index, 2);
    assert!(!DerivedBundle::WIRING[0].required);

    let mut block = BootBlock::new();
    assert!(block.push(CAPTURED, 2, SlotRef::from_wire(11)));
    let boot = Bootstrap::of(&block).expect("the test block validates");
    let bundle = CaptureBundle::resolve(&boot).expect("a captured handle resolves");
    assert_eq!(bundle.captured, Some(SlotRef::from_wire(11)));

    let empty = BootBlock::new();
    let boot = Bootstrap::of(&empty).expect("an empty block validates");
    assert_eq!(EmptyBundle::resolve(&boot), Ok(EmptyBundle));
}

#[test]
fn missing_and_present_but_bad_are_distinct() {
    let block = BootBlock::new();
    let boot = Bootstrap::of(&block).expect("an empty block validates");
    assert!(matches!(
        DerivedBundle::resolve(&boot),
        Err(BindError::MissingRole {
            role: REQUIRED,
            index: 1,
        })
    ));

    let mut block = BootBlock::new();
    assert!(block.push(REQUIRED, 1, SlotRef::from_wire(99)));
    let boot = Bootstrap::of(&block).expect("the test block validates");
    assert!(matches!(
        DerivedBundle::resolve(&boot),
        Err(BindError::BadHandle {
            role: REQUIRED,
            index: 1,
        })
    ));
}
