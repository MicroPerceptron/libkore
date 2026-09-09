#!/usr/bin/env python3
"""Reject duplicate Kore owner packages in cargo metadata JSON from stdin."""
import json
import sys


def violations(metadata: dict) -> list[str]:
    # Only resolved packages belong to the consumer's graph.
    reached = {node["id"] for node in metadata["resolve"]["nodes"]}
    owners: dict[str, set[str]] = {}
    for package in metadata["packages"]:
        if package["id"] not in reached:
            continue
        name = package["name"]
        if name == "kore" or name.startswith("kore-"):
            owners.setdefault(name, set()).add(package["id"])
    errors = []
    if "kore" in owners:
        errors.append("kernel package kore entered the application graph")
    for name, ids in sorted(owners.items()):
        if len(ids) != 1:
            errors.append(f"{name}: multiple owners: {', '.join(sorted(ids))}")
    if "kore-std" not in owners:
        errors.append("no standard library resolved; this did not test a consumer")
    return errors


if __name__ == "__main__":
    errors = violations(json.load(sys.stdin))
    if errors:
        sys.exit("\n".join(errors))
    print("owner graph: one package per Kore owner; no kernel dependency")
