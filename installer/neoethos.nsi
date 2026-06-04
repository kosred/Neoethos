; installer/neoethos.nsi
;
; NSIS installer for NeoEthos — wraps the `dist/NeoEthos/` bundle that
; `scripts/make-release-bundle.ps1` produces into a single
; `NeoEthos-Setup-<version>.exe`.
;
; End-user workflow:
;   1. Download NeoEthos-Setup-0.4.20.exe (one file).
;   2. Double-click → installer runs.
;   3. Files land in `%ProgramFiles%\NeoEthos\`.
;   4. Start Menu folder "NeoEthos" + Desktop shortcut both point to
;      `NeoEthos.exe` (the Flutter shell).
;   5. User clicks Desktop / Start-menu → app launches → backend
;      spawns automatically.
;   6. User NEVER sees `bin\neoethos-app.exe`; it's installed but
;      the folder is Hidden+System.
;
; To compile:
;   1. `pwsh scripts/make-release-bundle.ps1` (produces dist/NeoEthos/)
;   2. `makensis installer/neoethos.nsi` (produces dist/NeoEthos-Setup-*.exe)
;
; Or use the wrapper: `pwsh scripts/build-installer.ps1`

;==============================================================================
; Configuration
;==============================================================================

!define PRODUCT_NAME      "NeoEthos"
!define PRODUCT_VERSION   "0.4.41"
!define PRODUCT_PUBLISHER "Konstantinos Red"
!define PRODUCT_WEB_SITE  "https://github.com/kosred/neoethos"
!define PRODUCT_REGKEY    "Software\${PRODUCT_NAME}"
!define UNINST_REGKEY     "Software\Microsoft\Windows\CurrentVersion\Uninstall\${PRODUCT_NAME}"

Name "${PRODUCT_NAME} ${PRODUCT_VERSION}"
OutFile "..\dist\${PRODUCT_NAME}-Setup-${PRODUCT_VERSION}.exe"
InstallDir "$PROGRAMFILES64\${PRODUCT_NAME}"
InstallDirRegKey HKLM "${PRODUCT_REGKEY}" "InstallDir"

; Request admin elevation so we can write to Program Files. Without
; this NSIS would silently fall back to per-user install which is
; harder to uninstall through Add/Remove Programs.
RequestExecutionLevel admin

; LZMA gives ~30% better compression than the default zlib at the cost
; of a slower build (one-time pain for the developer, one-time saving
; for every downloader).
SetCompressor /SOLID lzma

;==============================================================================
; Modern UI 2 — wizard look-and-feel
;==============================================================================

!include "MUI2.nsh"
!include "FileFunc.nsh"
!include "LogicLib.nsh"

!define MUI_ABORTWARNING
!define MUI_ICON   "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_UNICON "${NSISDIR}\Contrib\Graphics\Icons\modern-uninstall.ico"

; ── Pages ────────────────────────────────────────────────────────────────────
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_LICENSE "..\LICENSE"
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES

; Finish page — show a tip pointing at the Start Menu shortcut
; INSTEAD of the auto-launch checkbox. Auto-launch from the
; installer inherits the installer's admin token, which makes
; NeoEthos run at High integrity → Windows UIPI then blocks
; input from any normal-integrity process (drag-drop from File
; Explorer, automation tools, etc.) and files written by the app
; land owned by admin, unreadable to the standard user. The user
; launches from Start Menu / Desktop shortcut → app runs at the
; user's normal token. Task #177.
!define MUI_FINISHPAGE_TEXT \
    "Setup is complete.$\r$\n$\r$\nLaunch NeoEthos from the Start \
Menu (or the Desktop shortcut).$\r$\n$\r$\nDo NOT relaunch this \
installer to start the app — that would run NeoEthos with \
administrator privileges and break some Windows features."
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "English"

;==============================================================================
; Version info shown in the .exe Properties dialog
;==============================================================================

VIProductVersion "${PRODUCT_VERSION}.0"
VIAddVersionKey "ProductName"      "${PRODUCT_NAME}"
VIAddVersionKey "ProductVersion"   "${PRODUCT_VERSION}"
VIAddVersionKey "FileVersion"      "${PRODUCT_VERSION}"
VIAddVersionKey "CompanyName"      "${PRODUCT_PUBLISHER}"
VIAddVersionKey "LegalCopyright"   "© 2024-2026 ${PRODUCT_PUBLISHER}"
VIAddVersionKey "FileDescription"  "NeoEthos installer"

;==============================================================================
; Install
;==============================================================================

Section "NeoEthos" SecMain
    SectionIn RO  ; required — user can't unselect

    SetOutPath "$INSTDIR"

    ; Bundle root: NeoEthos.exe + flutter_windows.dll + data/ + config.yaml
    ; + assets/ + resources/ + LICENSE + README.md + bin/ (hidden)
    ; The `make-release-bundle.ps1` script lays these out in dist/NeoEthos/
    ; and applies the Hidden+System attributes to bin/ before we get here.
    ; /r preserves subfolders and their attributes.
    File /r "..\dist\NeoEthos\*.*"

    ; Re-apply Hidden+System to bin/ in the install dir. NSIS's /r
    ; preserves attributes on most files but is inconsistent on
    ; directory attributes across Windows versions — belt + braces.
    SetFileAttributes "$INSTDIR\bin" HIDDEN|SYSTEM

    ; Register uninstaller in Add/Remove Programs so the user can
    ; uninstall cleanly via Settings → Apps. Without this, the
    ; uninstaller still works but is invisible.
    WriteRegStr HKLM "${PRODUCT_REGKEY}" "InstallDir" "$INSTDIR"
    WriteRegStr HKLM "${PRODUCT_REGKEY}" "Version"    "${PRODUCT_VERSION}"

    WriteRegStr HKLM "${UNINST_REGKEY}" "DisplayName"     "${PRODUCT_NAME}"
    WriteRegStr HKLM "${UNINST_REGKEY}" "DisplayVersion"  "${PRODUCT_VERSION}"
    WriteRegStr HKLM "${UNINST_REGKEY}" "Publisher"       "${PRODUCT_PUBLISHER}"
    WriteRegStr HKLM "${UNINST_REGKEY}" "URLInfoAbout"    "${PRODUCT_WEB_SITE}"
    WriteRegStr HKLM "${UNINST_REGKEY}" "DisplayIcon"     "$INSTDIR\NeoEthos.exe"
    WriteRegStr HKLM "${UNINST_REGKEY}" "InstallLocation" "$INSTDIR"
    WriteRegStr HKLM "${UNINST_REGKEY}" "UninstallString" "$INSTDIR\Uninstall.exe"
    WriteRegDWORD HKLM "${UNINST_REGKEY}" "NoModify" 1
    WriteRegDWORD HKLM "${UNINST_REGKEY}" "NoRepair" 1

    ; Calculate install size for the Programs & Features panel.
    ${GetSize} "$INSTDIR" "/S=0K" $0 $1 $2
    IntFmt $0 "0x%08X" $0
    WriteRegDWORD HKLM "${UNINST_REGKEY}" "EstimatedSize" "$0"

    WriteUninstaller "$INSTDIR\Uninstall.exe"
SectionEnd

Section "Start Menu shortcut" SecStartMenu
    ; Always points at NeoEthos.exe (the Flutter shell), NEVER the backend.
    ; This is the only entry-point the operator should ever click.
    CreateDirectory "$SMPROGRAMS\${PRODUCT_NAME}"
    CreateShortcut  "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME}.lnk" \
                    "$INSTDIR\NeoEthos.exe" "" \
                    "$INSTDIR\NeoEthos.exe" 0
    CreateShortcut  "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall.lnk" \
                    "$INSTDIR\Uninstall.exe"
SectionEnd

Section "Desktop shortcut" SecDesktop
    ; Optional — user can untick on the Components page. Some
    ; operators (kiosk installs, USB-portable use) don't want a
    ; desktop icon.
    CreateShortcut "$DESKTOP\${PRODUCT_NAME}.lnk" \
                   "$INSTDIR\NeoEthos.exe" "" \
                   "$INSTDIR\NeoEthos.exe" 0
SectionEnd

; Section descriptions shown when the user hovers over the
; Components page entries.
!insertmacro MUI_FUNCTION_DESCRIPTION_BEGIN
!insertmacro MUI_DESCRIPTION_TEXT ${SecMain}      "Core NeoEthos files (required)."
!insertmacro MUI_DESCRIPTION_TEXT ${SecStartMenu} "Start Menu shortcut to NeoEthos."
!insertmacro MUI_DESCRIPTION_TEXT ${SecDesktop}   "Desktop shortcut to NeoEthos."
!insertmacro MUI_FUNCTION_DESCRIPTION_END

;==============================================================================
; Uninstall
;==============================================================================

Section "Uninstall"
    ; Tear down everything the install put on disk + in the registry.
    ; The user's data dir (%LOCALAPPDATA%\neoethos\) is INTENTIONALLY
    ; left alone — that's where their broker_credentials.toml and
    ; OAuth token bundle live. Wiping it on uninstall is hostile;
    ; the operator can rm it manually if they really want.

    Delete "$INSTDIR\Uninstall.exe"
    Delete "$INSTDIR\NeoEthos.exe"
    Delete "$INSTDIR\flutter_windows.dll"
    Delete "$INSTDIR\config.yaml"
    Delete "$INSTDIR\LICENSE"
    Delete "$INSTDIR\README.md"

    RMDir /r "$INSTDIR\data"
    RMDir /r "$INSTDIR\assets"
    RMDir /r "$INSTDIR\resources"
    RMDir /r "$INSTDIR\bin"

    ; $INSTDIR itself — only remove if empty. If the user dropped
    ; their own files in there we don't nuke them.
    RMDir "$INSTDIR"

    ; Shortcuts
    Delete "$SMPROGRAMS\${PRODUCT_NAME}\${PRODUCT_NAME}.lnk"
    Delete "$SMPROGRAMS\${PRODUCT_NAME}\Uninstall.lnk"
    RMDir  "$SMPROGRAMS\${PRODUCT_NAME}"
    Delete "$DESKTOP\${PRODUCT_NAME}.lnk"

    ; Registry
    DeleteRegKey HKLM "${UNINST_REGKEY}"
    DeleteRegKey HKLM "${PRODUCT_REGKEY}"
SectionEnd
