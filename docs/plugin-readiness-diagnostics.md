# Plugin readiness diagnostics

Hermes reports plugin readiness through the existing `hermes plugins doctor`
validation path:

```bash
hermes plugins doctor <path-or-plugin-id>
hermes plugins doctor <path-or-plugin-id> --json
```

Readiness is a diagnostic summary of that one isolated Doctor run:

| State | Meaning |
| --- | --- |
| `ready` | Manifest discovery, import, registration, and contract checks completed without findings. |
| `degraded` | Validation completed, but one or more warnings indicate a declaration, dependency, or registration mismatch. |
| `unavailable` | An error prevented the plugin from satisfying the runtime contract. |
| `unknown` | No manifest was evaluated and no concrete failure was observed. This is primarily a defensive API state. |

The JSON response includes `readiness`, the existing `ok` error gate, manifest
identity, findings, and exact tool/hook registrations. Stdout remains one valid
JSON document even if plugin import or registration code prints; that plugin
output is preserved on stderr. `ok` remains backward compatible: warnings
produce `degraded` while `ok` stays `true`; `--ci` exits non-zero only for
`unavailable` reports caused by errors.

## Design boundary

Readiness is deliberately:

- derived from the current Doctor report rather than stored in a registry;
- diagnostic-only and unable to enable, disable, retry, or quarantine plugins;
- isolated from the active profile by Plugin Doctor's temporary `HERMES_HOME`;
- not a claim that an external provider, credential, network, or upstream API is
  currently healthy.

This keeps readiness compatible with the current ownership ledger, unload
lifecycle, deferred platform loading, and profile isolation. Live-provider
circuit breakers and persistent health state remain out of scope until concrete
operator incidents justify a separate design.
