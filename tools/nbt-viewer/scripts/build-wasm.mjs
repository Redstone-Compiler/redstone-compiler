import { mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';

const scriptDir = dirname(fileURLToPath(import.meta.url));
const viewerDir = resolve(scriptDir, '..');
const repoDir = resolve(viewerDir, '..', '..');
const outDir = resolve(viewerDir, 'public', 'wasm', 'nbt-sim');
const wasmInput = resolve(repoDir, 'target', 'wasm32-unknown-unknown', 'release', 'nbt_sim_wasm.wasm');

function run(command, args, cwd = repoDir) {
  const result = spawnSync(command, args, {
    cwd,
    shell: process.platform === 'win32',
    stdio: 'inherit',
  });

  if (result.status !== 0) {
    process.exit(result.status ?? 1);
  }
}

mkdirSync(outDir, { recursive: true });

run('cargo', ['build', '-p', 'nbt-sim-wasm', '--target', 'wasm32-unknown-unknown', '--release']);
run('wasm-bindgen', [
  wasmInput,
  '--target',
  'web',
  '--out-dir',
  outDir,
  '--out-name',
  'nbt_sim_wasm',
]);
