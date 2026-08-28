import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import test from "node:test";

function readJson(relativePath: string): Record<string, any> {
  const url = new URL(relativePath, import.meta.url);
  assert.ok(existsSync(url), `missing required Tauri config ${relativePath}`);
  return JSON.parse(readFileSync(url, "utf8"));
}

function readOptionalJson(relativePath: string): Record<string, any> | undefined {
  const url = new URL(relativePath, import.meta.url);
  return existsSync(url) ? JSON.parse(readFileSync(url, "utf8")) : undefined;
}

test("ordinary cargo builds never require release-generated bundle resources", () => {
  const base = readJson("../src-tauri/tauri.conf.json");
  const windows = readOptionalJson("../src-tauri/tauri.windows.conf.json");
  const linux = readJson("../src-tauri/tauri.linux.conf.json");

  assert.equal(
    base.bundle?.resources,
    undefined,
    "base cargo test/check config must not resolve generated resources",
  );
  assert.equal(
    base.bundle?.externalBin,
    undefined,
    "base cargo test/check config must not resolve generated sidecars",
  );
  assert.equal(
    windows?.bundle?.resources,
    undefined,
    "Windows cargo test/check must not resolve release-generated sidecars or DLLs",
  );
  assert.equal(
    linux.bundle?.resources,
    undefined,
    "Linux cargo test/check must not resolve release-generated sidecars or shared libraries",
  );
});

test("release-only overlays enumerate every required sidecar and native runtime", () => {
  const windows = readJson("../src-tauri/tauri.release.windows.conf.json");
  const linux = readJson("../src-tauri/tauri.release.linux.conf.json");

  assert.deepEqual(Object.keys(windows.bundle.resources).sort(), [
    "../../mcp/target/release/neoethos-mcp.exe",
    "../../mesh/target/release/neoethos-mesh.exe",
    "../../target/release/catboostmodel.dll",
    "../../target/release/xgboost.dll",
  ]);
  assert.deepEqual(Object.keys(linux.bundle.resources).sort(), [
    "../../mcp/target/release/neoethos-mcp",
    "../../mesh/target/release/neoethos-mesh",
    "../../target/release/libcatboostmodel.so",
    "../../target/release/libxgboost.so",
  ]);
});

test("release workflow stages and verifies native runtimes before applying the overlay", () => {
  const workflow = readFileSync(
    new URL("../../.github/workflows/release-desktop.yml", import.meta.url),
    "utf8",
  );

  assert.match(
    workflow,
    /cargo build --release -p neoethos-app --lib --no-default-features/,
  );
  assert.match(workflow, /target\\release\\xgboost\.dll/);
  assert.match(workflow, /target\/release\/libxgboost\.so/);
  assert.match(workflow, /target\\release\\catboostmodel\.dll/);
  assert.match(workflow, /target\/release\/libcatboostmodel\.so/);
  assert.doesNotMatch(workflow, /target[\\/]release[\\/]deps[\\/](?:lib)?xgboost/);
  assert.match(
    workflow,
    /args:\s*>-\s*\n\s*--config src-tauri\/tauri\.release\.\$\{\{ matrix\.target \}\}\.conf\.json/,
  );

  const upload = workflow.indexOf("- name: Build + upload the desktop GUI");
  const prerequisites = [
    "- name: Build desktop native runtime prerequisites",
    "- name: Stage and verify Windows native runtimes",
    "- name: Stage and verify Linux native runtimes",
    "- name: Build MCP sidecar",
    "- name: Build mesh sidecar",
  ];
  assert.ok(upload >= 0, "desktop upload step must exist");
  for (const step of prerequisites) {
    const index = workflow.indexOf(step);
    assert.ok(index >= 0 && index < upload, `${step} must precede bundle upload`);
  }
  assert.ok(
    workflow.indexOf("--config src-tauri/tauri.release.", upload) > upload,
    "the release overlay must be applied by the post-staging bundle step",
  );
});

test("Linux packages can load native runtimes and resolve both sidecars", () => {
  const base = readJson("../src-tauri/tauri.conf.json");
  const buildScript = readFileSync(
    new URL("../src-tauri/build.rs", import.meta.url),
    "utf8",
  );
  const shell = readFileSync(
    new URL("../src-tauri/src/lib.rs", import.meta.url),
    "utf8",
  );
  const workflow = readFileSync(
    new URL("../../.github/workflows/release-desktop.yml", import.meta.url),
    "utf8",
  );

  assert.equal(base.productName, "NeoEthos");
  assert.match(
    buildScript,
    new RegExp(`\\$ORIGIN/\\.\\./lib/${base.productName}`),
  );
  assert.match(buildScript, /NEOETHOS_TAURI_RELEASE_BUNDLE/);
  assert.match(workflow, /NEOETHOS_TAURI_RELEASE_BUNDLE:\s*["']?1["']?/);
  const bundle = workflow.indexOf("- name: Build + upload the desktop GUI");
  const installedCheck = workflow.indexOf(
    "- name: Verify installed Linux runtime layout",
  );
  assert.ok(
    installedCheck > bundle,
    "installed deb/rpm verification must run after the bundle exists",
  );
  assert.match(workflow, /dpkg-deb -x/);
  assert.match(workflow, /rpm2cpio/);
  assert.match(workflow, /test -x .*neoethos-mcp/);
  assert.match(workflow, /test -x .*neoethos-mesh/);
  assert.match(workflow, /readelf -d/);
  assert.match(workflow, /grep -Fq '\(RUNPATH\)'/);
  assert.match(workflow, /grep -Fq '\$ORIGIN\/\.\.\/lib\/NeoEthos'/);
  assert.match(workflow, /if ! env -u LD_LIBRARY_PATH ldd .* 2>&1; then/);
  assert.match(workflow, /realpath "\$resolved"/);
  assert.match(workflow, /realpath "\$resources\/\$library"/);
  assert.doesNotMatch(workflow, /ldd .*\|.*grep/);
  assert.match(shell, /resource_dir\(\)/);
  assert.match(shell, /resource_dir\.join\(bin_name\)/);
  assert.match(shell, /mcp_sidecar::start\([^)]*resource_dir/);
  assert.match(shell, /mesh_sidecar::start\([^)]*resource_dir/);
});
