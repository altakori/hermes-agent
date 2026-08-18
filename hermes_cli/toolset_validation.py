"""Validation for the ``platform_toolsets`` config section.

Pure, side-effect-free helpers so the logic is unit-testable without importing
the tool registry or launching Hermes (mirrors the decoupled-helper pattern used
elsewhere in the CLI).

Motivated by #38798: a config migration silently rewrote the valid toolset name
``hermes-cli`` to the non-existent ``hermes``. ``resolve_toolset('hermes')``
returns an empty list, so every tool silently disappeared with no error, warning,
or log entry — the agent degraded to text-only replies and the cause took
significant debugging to find. Surfacing invalid toolset names (and the
zero-tools end state) loudly turns that silent failure into an actionable one.
"""

from typing import Callable, List


def find_unknown_startup_toolsets(
    toolsets: object,
    config: object,
    is_valid_toolset: Callable[[str], bool],
) -> List[str]:
    """Return toolsets that are genuinely unknown during early startup.

    Plugin and MCP discovery happens after the CLI constructs the agent, so
    their live registry entries are not available to ``is_valid_toolset`` at
    this point.  The config records both kinds of deferred names; accept those
    while continuing to report entries that neither the registry nor config
    knows about.
    """
    if not isinstance(toolsets, (list, tuple, set)):
        return []

    deferred_names = set()
    if isinstance(config, dict):
        mcp_servers = config.get("mcp_servers")
        if isinstance(mcp_servers, dict):
            deferred_names.update(
                name for name in mcp_servers if isinstance(name, str) and name
            )

        known_plugins = config.get("known_plugin_toolsets")
        if isinstance(known_plugins, dict):
            for raw_names in known_plugins.values():
                names = raw_names if isinstance(raw_names, list) else [raw_names]
                deferred_names.update(
                    name for name in names if isinstance(name, str) and name
                )

    return [
        name
        for name in toolsets
        if isinstance(name, str)
        and not is_valid_toolset(name)
        and name not in deferred_names
    ]


def validate_platform_toolsets(
    platform_toolsets: object,
    is_valid_toolset: Callable[[str], bool],
) -> List[str]:
    """Return human-readable warnings for a ``platform_toolsets`` mapping.

    Two failure modes are reported:

    1. A toolset name that ``is_valid_toolset`` rejects — usually a corrupted or
       renamed entry. When ``hermes-<platform>`` would have been valid (the exact
       #38798 shape, where ``cli`` held ``hermes`` instead of ``hermes-cli``),
       the warning includes that as a suggestion.
    2. The mapping is non-empty but resolves to *zero* valid toolsets, so the
       agent would start with no tools at all.

    ``is_valid_toolset`` is injected (normally :func:`toolsets.validate_toolset`)
    so this function performs no imports or I/O and is testable in isolation.

    Args:
        platform_toolsets: The raw ``platform_toolsets`` value from config. Only
            ``dict`` values carry toolset entries; anything else yields no
            warnings (nothing to validate).
        is_valid_toolset: Predicate returning ``True`` for a known toolset name.

    Returns:
        A list of warning strings (empty when everything is valid).
    """
    warnings: List[str] = []
    if not isinstance(platform_toolsets, dict) or not platform_toolsets:
        return warnings

    valid_count = 0
    for platform, raw in platform_toolsets.items():
        names = raw if isinstance(raw, list) else [raw]
        for name in names:
            if not isinstance(name, str) or not name:
                continue
            if is_valid_toolset(name):
                valid_count += 1
                continue
            suggestion = f"hermes-{platform}"
            hint = (
                f" — did you mean '{suggestion}'?"
                if is_valid_toolset(suggestion)
                else ""
            )
            warnings.append(
                f"platform '{platform}' references unknown toolset "
                f"'{name}'{hint}"
            )

    if valid_count == 0:
        warnings.append(
            "platform_toolsets resolves to zero valid toolsets — the agent will "
            "have no tools. Run `hermes tools` to reconfigure."
        )
    return warnings
