; scripts/installer_hooks.nsh — master NSIS hook file for cargo-packager.
;
; cargo-packager `nsis.installer-hooks` accepts ONE .nsh path. We point it
; here, and this file `!include`s the actual per-step macros. The two
; cargo-packager-recognised macro names (PREINSTALL / POSTINSTALL) are
; defined at the bottom — those are what the cargo-packager NSIS template
; calls automatically at the matching install phases.
;
; PRE-install runs BEFORE files are extracted. Used for VC++ Redistributable
; because our .exe imports VCRUNTIME140.dll at load time; install would
; fail on pristine Windows without the runtime.

!include "${PROJECTDIR}\scripts\vc_redist_install.nsh"

; ── cargo-packager hook: PRE-install ──────────────────────────────────────
; Ensure Visual C++ 2015-2022 redistributable is present before NeoEthos
; binaries are dropped onto disk. Without this, pristine Windows installs
; fail with a "VCRUNTIME140.dll not found" loader error the first time
; NeoEthos.exe is double-clicked.
!macro NSIS_HOOK_PREINSTALL
    !insertmacro NEOETHOS_ENSURE_VC_REDIST
!macroend

; ── cargo-packager hook: POST-install ─────────────────────────────────────
; No post-install work today. The AI Helper uses the operator's existing
; ChatGPT subscription (neoethos-codex OAuth) so there is nothing to
; download after files land on disk. The macro is left in place because
; cargo-packager's NSIS template references it unconditionally.
!macro NSIS_HOOK_POSTINSTALL
!macroend
