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
  const outputsName = entry.name.replace(/\.nbt$/, '.outputs.json');
  const outputsSource = join(sourceRoot, outputsName);
  const outputsTarget = join(targetRoot, outputsName);
  const info = await stat(source);
  await copyFile(source, target);
  const example = {
    name: entry.name,
    file: entry.name,
    path: `examples/${entry.name}`,
    size: info.size,
  };
  try {
    await stat(outputsSource);
    await copyFile(outputsSource, outputsTarget);
    example.outputsPath = `examples/${outputsName}`;
  } catch (error) {
    if (error?.code !== 'ENOENT') throw error;
  }
  examples.push(example);
}

examples.sort((a, b) => a.name.localeCompare(b.name));
await writeFile(join(targetRoot, 'manifest.json'), `${JSON.stringify(examples, null, 2)}\n`);
console.log(`Prepared ${examples.length} NBT examples in ${targetRoot}`);
