# Homebrew Cask manifest for forex-ai.
#
# Spec:
#   - Cask cookbook: https://docs.brew.sh/Cask-Cookbook
#   - Acceptable casks: https://docs.brew.sh/Acceptable-Casks
# Strategy ref: docs/audits/research/installer_no_paid_certs_strategy.md §2.2
#
# Submission target:
#   - Primary: PR to Homebrew/homebrew-cask (requires repo notability:
#     >= 75 stars, or >= 225 for self-submitted casks).
#   - Fallback (until kosred/forex-ai meets notability): local tap at
#     https://github.com/kosred/homebrew-forex-ai — see ../README.md.
#
# All TODO(release-time) markers populated by .github/workflows/release.yml.

cask "forex-ai" do
  version "0.4.8"
  # TODO(release-time): SHA-256 of the uploaded universal tarball. Brew CI
  # rejects the cask if the file on disk does not match this checksum.
  sha256 "TODO_SHA256_AT_RELEASE_TIME"

  url "https://github.com/kosred/forex-ai/releases/download/v#{version}/forex-ai-#{version}-macos-universal.tar.gz"
  name "forex-ai"
  desc "Quantitative trading workspace for the cTrader Open API"
  homepage "https://github.com/kosred/forex-ai"

  livecheck do
    url :url
    strategy :github_latest
  end

  binary "forex-app"
  binary "forex-cli"

  # `zap` defines best-effort cleanup paths for `brew uninstall --zap`. These
  # locations come from installer_infrastructure_spec.md §8.3 (macOS XDG-style
  # layout) and match what the in-app wizard writes on first launch.
  zap trash: [
    "~/Library/Application Support/forex-ai",
    "~/Library/Logs/forex-ai",
    "~/Library/Preferences/com.kosred.forex-ai.plist",
  ]
end
