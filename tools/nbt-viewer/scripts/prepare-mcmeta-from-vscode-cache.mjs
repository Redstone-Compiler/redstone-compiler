import { gunzipSync } from 'node:zlib';
import { copyFile, mkdir, writeFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { join } from 'node:path';

const MCMETA = 'https://raw.githubusercontent.com/misode/mcmeta';
const version = process.argv[2] ?? '1.18.2';
const sourceRoot = process.env.MCMETA_SOURCE_DIR ?? defaultSourceRoot();
const targetRoot = join(process.cwd(), 'public', 'mcmeta');
const files = ['assets', 'atlas', 'blocks', 'uvmapping'];

await mkdir(targetRoot, { recursive: true });

if (sourceRoot && existsSync(sourceRoot)) {
  await copyLocalCache();
} else {
  await downloadMcmeta();
}

async function copyLocalCache() {
  for (const file of files) {
    await copyFile(join(sourceRoot, `${version}-${file}`), join(targetRoot, `${version}-${file}`));
  }

  await copyFile(join(sourceRoot, 'versions'), join(targetRoot, 'versions'));
  console.log(`Copied vscode-nbt mcmeta cache for ${version} to ${targetRoot}`);
}

async function downloadMcmeta() {
  console.log(`Downloading mcmeta assets for ${version} to ${targetRoot}`);

  const [versions, blocks, atlas, uvmapping, assetsArchive] = await Promise.all([
    fetchBytes(`${MCMETA}/summary/versions/data.min.json`),
    fetchBytes(`${MCMETA}/${version}-summary/blocks/data.min.json`),
    fetchBytes(`${MCMETA}/${version}-atlas/all/atlas.png`),
    fetchBytes(`${MCMETA}/${version}-atlas/all/data.min.json`),
    fetchBytes(`https://github.com/misode/mcmeta/tarball/${version}-assets-json`),
  ]);

  await Promise.all([
    writeFile(join(targetRoot, 'versions'), versions),
    writeFile(join(targetRoot, `${version}-blocks`), wrapJson('stringifiedBlocks', blocks)),
    writeFile(join(targetRoot, `${version}-atlas`), atlas),
    writeFile(join(targetRoot, `${version}-uvmapping`), wrapJson('stringifiedUvmapping', uvmapping)),
    writeFile(join(targetRoot, `${version}-assets`), await buildAssetsJson(assetsArchive)),
  ]);

  console.log(`Downloaded mcmeta assets for ${version} to ${targetRoot}`);
}

async function fetchBytes(url) {
  const response = await fetch(url, { redirect: 'follow' });
  if (!response.ok) {
    throw new Error(`Failed to download ${url}: ${response.status} ${response.statusText}`);
  }

  return Buffer.from(await response.arrayBuffer());
}

async function buildAssetsJson(archive) {
  const entries = readTarGz(archive, path => {
    return path.includes('/assets/minecraft/models/') || path.includes('/assets/minecraft/blockstates/');
  });

  const blockstates = filterJsonEntries(entries, 'blockstates');
  const models = filterJsonEntries(entries, 'models');
  return wrapJson('stringifiedAssets', Buffer.from(JSON.stringify({ blockstates, models })));
}

function filterJsonEntries(entries, type) {
  const pattern = RegExp(`/assets/minecraft/${type}/([a-z0-9/_]+)\\.json$`);
  return Object.fromEntries(
    entries.flatMap(({ path, data }) => {
      const match = path.match(pattern);
      return match ? [[match[1], JSON.parse(data.toString('utf-8'))]] : [];
    }),
  );
}

function readTarGz(archive, filter) {
  const buffer = gunzipSync(archive);
  const entries = [];
  let offset = 0;

  while (offset + 512 <= buffer.length) {
    const header = buffer.subarray(offset, offset + 512);
    if (header.every(byte => byte === 0)) break;

    const name = readTarString(header, 0, 100);
    const prefix = readTarString(header, 345, 155);
    const path = prefix ? `${prefix}/${name}` : name;
    const size = Number.parseInt(readTarString(header, 124, 12).trim() || '0', 8);
    const dataStart = offset + 512;
    const dataEnd = dataStart + size;

    if (filter(path)) {
      entries.push({ path, data: buffer.subarray(dataStart, dataEnd) });
    }

    offset = dataStart + Math.ceil(size / 512) * 512;
  }

  return entries;
}

function readTarString(buffer, start, length) {
  const bytes = buffer.subarray(start, start + length);
  const end = bytes.indexOf(0);
  return bytes.subarray(0, end === -1 ? bytes.length : end).toString('utf-8');
}

function wrapJson(variableName, data) {
  return Buffer.from(`const ${variableName} = \`${data.toString('utf-8')}\``);
}

function defaultSourceRoot() {
  const localAppData = process.env.LOCALAPPDATA;
  if (!localAppData) return undefined;

  return join(localAppData, 'vscode-nbt-nodejs', 'Cache', 'mcmeta');
}
