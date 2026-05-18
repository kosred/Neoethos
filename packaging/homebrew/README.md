# Homebrew distribution for forex-ai

This directory holds the canonical Cask formula for installing forex-ai on
macOS via Homebrew. Two delivery paths are supported; pick the one your
project's current notability allows.

## Path A — official `Homebrew/homebrew-cask`

The official tap accepts Cask submissions for free, but Homebrew's
`Acceptable-Casks` rules gate notability:

> "App is too obscure. Examples:
> - An app from a code repository that is not notable enough (under
>   30 forks, 30 watchers, 75 stars).
> - For self-submitted casks where the PR author is the owner of the
>   repository, higher thresholds apply (under 90 forks, 90 watchers,
>   225 stars)."
>
> — `docs/Acceptable-Casks.md` (quoted via
>    `docs/audits/research/installer_no_paid_certs_strategy.md` §2.2).

**Install command, once merged:**

```sh
brew install --cask forex-ai
```

Submission flow:

1. Fork `Homebrew/homebrew-cask`.
2. Copy `Casks/forex-ai.rb` into `Casks/f/forex-ai.rb` of your fork.
3. Replace `TODO(release-time)` markers (SHA-256) with the values from
   `https://github.com/kosred/forex-ai/releases/download/vX.Y.Z/forex-ai-X.Y.Z-macos-universal.tar.gz.sha256`.
4. Run `brew audit --new --cask forex-ai` and `brew style forex-ai`.
5. Open a PR. Review takes ~1-3 days for a well-formed PR.

## Path B — third-party tap (until notability gate clears)

Before the kosred/forex-ai repo crosses the 225-star self-submission
threshold, ship the cask through a project-owned tap so users can install
without waiting on Homebrew/cask review:

```sh
brew tap kosred/forex-ai https://github.com/kosred/homebrew-forex-ai
brew install --cask forex-ai
```

The third-party tap is a separate GitHub repo named
`homebrew-<name>` (Homebrew's auto-discovery convention) containing a
top-level `Casks/forex-ai.rb` identical to the file in this directory.

Setup steps (one-time):

1. Create `https://github.com/kosred/homebrew-forex-ai` (public, empty).
2. Copy `Casks/forex-ai.rb` into the new repo at `Casks/forex-ai.rb`.
3. Tag releases in the new repo to match `forex-ai` versions
   (`v0.4.7`, `v0.4.6`, ...).
4. The CI workflow `.github/workflows/release.yml` job
   `publish-homebrew-tap` automates the bump on every tagged release.

## Path C — direct `.tar.gz` + `xattr -dr`

If Homebrew is unavailable (corporate-locked macOS, etc.), the GitHub
Releases page exposes the same `forex-ai-X.Y.Z-macos-universal.tar.gz`
artefact. Users extract it and remove the quarantine attribute manually:

```sh
tar -xzf forex-ai-0.4.7-macos-universal.tar.gz
xattr -dr com.apple.quarantine forex-app forex-cli
./forex-app
```

This is the manual equivalent of what the Cask install does internally
(`xattr -d com.apple.quarantine` per
`Library/Homebrew/cask/quarantine.rb`, cited in strategy doc §2.2).

## Notability roadmap

| Threshold | Star count | Status |
|-----------|------------|--------|
| `Acceptable-Casks` (third-party submitter) | 75 | TODO(release-time) |
| `Acceptable-Casks` (self-submitted) | 225 | TODO(release-time) |

While both thresholds are unmet, Path B is the only sustainable channel.
