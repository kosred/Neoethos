; scripts/installer_hooks.nsh — master NSIS hook file for cargo-packager.
;
; cargo-packager `nsis.installer-hooks` accepts ONE .nsh path. We point it
; here, and this file `!include`s the actual per-step macros (vc_redist,
; gemma_model). The two cargo-packager-recognised macro names (PREINSTALL /
; POSTINSTALL) are defined at the bottom — those are what the cargo-packager
; NSIS template calls automatically at the matching install phases.
;
; Order matters:
;   * PRE-install runs BEFORE files are extracted. Used for VC++ Redistributable
;     because our .exe imports VCRUNTIME140.dll at load time; install would
;     fail on pristine Windows without the runtime.
;   * POST-install runs AFTER $INSTDIR is populated. Used for the Gemma GGUF
;     fetch because we write into $INSTDIR\resources\models\ which is created
;     by the file-extraction step.

!include "${PROJECTDIR}\scripts\vc_redist_install.nsh"
!include "${PROJECTDIR}\scripts\gemma_model_install.nsh"

; ── cargo-packager hook: PRE-install ──────────────────────────────────────
; Ensure Visual C++ 2015-2022 redistributable is present before NeoEthos
; binaries are dropped onto disk. Without this, pristine Windows installs
; fail with a "VCRUNTIME140.dll not found" loader error the first time
; NeoEthos.exe is double-clicked.
!macro NSIS_HOOK_PREINSTALL
    !insertmacro NEOETHOS_ENSURE_VC_REDIST
!macroend

; ── cargo-packager hook: POST-install ─────────────────────────────────────
; After the Lite bundle (Flutter shell + Rust backend + config) is on disk,
; fetch the 5 GiB Gemma 4 GGUF from HuggingFace into the same install dir.
; If the download fails OR the user cancels, NeoEthos still installs cleanly
; and the Flutter AI Helper screen offers a first-launch re-download path.
!macro NSIS_HOOK_POSTINSTALL
    !insertmacro NEOETHOS_FETCH_GEMMA_MODEL
!macroend
