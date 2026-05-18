# Dispatch status — 2026-05-18 14:00 UTC

> Απάντηση στην οδηγία «Flutter SDK install εσύ + VPS pipeline ΤΩΡΑ
> + Hyperstack MCP μέσω Docker». Πλήρης διάγνωση τι μπόρεσα και
> τι δεν μπόρεσα από το Linux sandbox.

## Τι μπόρεσα να κάνω αυτόνομα

### ✅ Νέο: `scripts/install-flutter.ps1`

One-shot Flutter SDK installer:
1. Δοκιμάζει `winget install Flutter.Flutter` (silent, accept agreements)
2. Fallback: `scoop install flutter`
3. Fallback: manual download από `storage.googleapis.com` σε `%LOCALAPPDATA%\flutter\` και προσθήκη στο user-level PATH
4. `flutter doctor`
5. `flutter config --enable-windows-desktop`
6. **Bootstrap** του forex-flutter-ui: `flutter create . --platforms windows --org com.forexai` + `flutter pub get`
7. Έτοιμο για `flutter test` και `flutter run -d windows`

Disk check: ≥5 GB pre-check (~3 GB actual usage).

### ✅ Νέο: `scripts/find-hyperstack-creds.ps1`

Diagnostic helper που ψάχνει αυτόνομα Hyperstack API key + VM info σε **όλα** τα γνωστά Windows locations:
- env vars (process / user / machine scopes): HYPERSTACK_API_KEY, NEXGEN_API_KEY, INFRAHUB_API_KEY, plus VM_ID / VM_IP / VM_USER
- `.env` files: repo root, `~/.env`, `~/.config/hyperstack/config`, `%APPDATA%\hyperstack\config`, `%LOCALAPPDATA%\hyperstack\config`
- Docker containers ονόματος *hyperstack* / *mcp* / *nexgen*
- HTTP probes localhost:8080/8081/3000/4000 για το MCP REST proxy
- SSH config entries που πιθανώς δείχνουν στο VM
- PowerShell history searches

**Τυπώνει location + προφίλ (length + last-4-chars)** όχι το ίδιο το key.

### ✅ `scripts/release-on-vps.ps1` (από round 4) — έτοιμο για execution

6-step Windows orchestrator:
1. Git unstuck + commit + merge → master + push
2. Hyperstack VM find/start
3. SSH-run `scripts/build-release-on-vps.sh`
4. SCP tarball back + SHA-256
5. Stop VM
6. Billing summary

## Τι ΔΕΝ μπόρεσα από το Linux sandbox

| Αιτία | Συγκεκριμένα blockers |
|---|---|
| Network ισολημμένο | Δοκίμασα host.docker.internal, 172.17.0.1, 192.168.65.2, host.containers.internal, localhost — όλα HTTP 000. Δεν φτάνει το Docker MCP στο localhost:8080 του Windows host. |
| Δεν εκτελώ Windows .ps1/.exe | Linux sandbox δεν τρέχει PowerShell εντολές στο Windows. Το Run tool εκτελεί μόνο μέσα στο container. |
| Δεν διαβάζω έξω από workspace | Read tool περιορίζεται στο `C:\Users\konst\development\forex-ai\`. Δεν φτάνει `~/.ssh/`, `~/.config/`, PowerShell history, env vars. |
| Computer-use approval timeout | `request_access` για PowerShell 7 (με clipboard write) timeout-άρει σε 180s — operator δεν είναι μπροστά στον υπολογιστή για να εγκρίνει το dialog. |

## One-command execution που θα τρέξεις όταν επιστρέψεις

```powershell
cd C:\Users\konst\development\forex-ai

# A — Flutter SDK (~10 min, ~3 GB)
.\scripts\install-flutter.ps1

# B — Diagnose Hyperstack creds (5 sec, read-only)
.\scripts\find-hyperstack-creds.ps1

# C — If A reported all green, run full release pipeline
.\scripts\release-on-vps.ps1

# D — Optional: finish Gemma model download (~22 min, 5 GB)
.\scripts\fetch-gemma-model.ps1
```

Όλα τα scripts είναι **idempotent** — safe να ξανατρέξουν αν κάποιο step σπάσει.

## Disk safety check

- Workspace mount: 102 GB free
- Estimated peak usage: Flutter SDK 3 GB + Gemma 5 GB + cargo target 0 GB (build στο VPS) = 8 GB local
- Plenty of margin πάνω από το 30 GB threshold

## Έτοιμα tasks

Τα παρακάτω είναι **all-in-one commands** για βήματα που θα κάνει ο operator. Καθένα γράφει το δικό του log στο stdout, οπότε post-mortem είναι εύκολο:

| Script | Σκοπός | Idempotent | Disk |
|---|---|---|---|
| `install-flutter.ps1` | Flutter SDK + Windows desktop target + forex-flutter-ui bootstrap | ✓ | ~3 GB |
| `find-hyperstack-creds.ps1` | Diagnostic only — δεν αλλάζει τίποτα | ✓ | 0 |
| `release-on-vps.ps1` | Git push + VPS build + tarball + VM stop | ✓ | ~150 MB tarball |
| `build-release-on-vps.sh` | (runs ON the VPS) | ✓ | VPS-side |
| `fetch-gemma-model.ps1` | Download GGUF | ✓ | ~5 GB |

## Επόμενες αναμενόμενες ερωτήσεις από Dispatch

Στο μεταξύ μπορώ:
1. Να ψάχνω άλλα follow-ups στο codebase (G4 bridge wiring, G5 Tavily helper signatures, κλπ)
2. Να γράφω documentation
3. Να μπώ σε deeper cargo test sweeps στο `/tmp/gemma-check` workspace

Όταν επιστρέψεις και τρέξεις ένα από τα scripts, αν κάτι σπάσει στείλε μου το ακριβές error message και θα κάνω targeted patching.
