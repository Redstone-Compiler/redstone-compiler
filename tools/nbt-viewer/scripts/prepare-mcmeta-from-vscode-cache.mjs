import { copyFile, mkdir } from 'node:fs/promises';
import { join } from 'node:path';

const version = process.argv[2] ?? '1.18.2';
const sourceRoot = process.env.MCMETA_SOURCE_DIR ?? defaultSourceRoot();
const targetRoot = join(process.cwd(), 'public', 'mcmeta');
const files = ['assets', 'atlas', 'blocks', 'uvmapping'];

await mkdir(targetRoot, { recursive: true });

for (const file of files) {
  await copyFile(join(sourceRoot, `${version}-${file}`), join(targetRoot, `${version}-${file}`));
}

await copyFile(join(sourceRoot, 'versions'), join(targetRoot, 'versions'));
console.log(`Copied vscode-nbt mcmeta cache for ${version} to ${targetRoot}`);

function defaultSourceRoot() {
  const localAppData = process.env.LOCALAPPDATA;
  if (!localAppData) {
    throw new Error('LOCALAPPDATA is not set. Set MCMETA_SOURCE_DIR to a mcmeta cache directory.');
  }

  return join(localAppData, 'vscode-nbt-nodejs', 'Cache', 'mcmeta');
}
