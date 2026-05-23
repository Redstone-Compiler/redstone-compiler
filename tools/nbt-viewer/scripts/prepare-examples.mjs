import { copyFile, mkdir, readdir, stat, writeFile } from 'node:fs/promises';
import { join } from 'node:path';

const repoRoot = join(process.cwd(), '..', '..');
const sourceRoot = join(repoRoot, 'test');
const targetRoot = join(process.cwd(), 'public', 'examples');

await mkdir(targetRoot, { recursive: true });

const entries = await readdir(sourceRoot, { withFileTypes: true });
const examples = [];

for (const entry of entries) {
  if (!entry.isFile() || !entry.name.endsWith('.nbt')) continue;

  const source = join(sourceRoot, entry.name);
  const target = join(targetRoot, entry.name);
  const info = await stat(source);
  await copyFile(source, target);
  examples.push({
    name: entry.name,
    file: entry.name,
    path: `examples/${entry.name}`,
    size: info.size,
  });
}

examples.sort((a, b) => a.name.localeCompare(b.name));
await writeFile(join(targetRoot, 'manifest.json'), `${JSON.stringify(examples, null, 2)}\n`);
console.log(`Prepared ${examples.length} NBT examples in ${targetRoot}`);
