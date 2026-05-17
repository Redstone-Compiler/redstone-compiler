import { BlockDefinition, BlockModel, TextureAtlas } from 'deepslate/render';
import { Identifier } from 'deepslate/core';

type BlocksData = Record<string, [Record<string, string[]>, Record<string, string>]>;
type McmetaAssets = {
  blockstates: Record<string, unknown>;
  models: Record<string, unknown>;
  textures: Record<string, [number, number, number, number]>;
};

function extractStringifiedJson(source: string, variableName: string): unknown {
  const prefix = `const ${variableName} = \``;
  if (!source.startsWith(prefix) || !source.endsWith('`')) {
    throw new Error(`Unexpected mcmeta cache format for ${variableName}.`);
  }

  return JSON.parse(source.slice(prefix.length, -1));
}

async function fetchStringifiedJson<T>(url: string, variableName: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to load ${url}: ${response.status}`);
  }
  return extractStringifiedJson(await response.text(), variableName) as T;
}

async function loadImageData(url: string): Promise<ImageData> {
  const image = new Image();
  image.decoding = 'async';
  image.src = url;
  await image.decode();

  const canvas = document.createElement('canvas');
  canvas.width = upperPowerOfTwo(image.width);
  canvas.height = upperPowerOfTwo(image.height);

  const context = canvas.getContext('2d');
  if (!context) throw new Error('2D canvas is unavailable.');
  context.drawImage(image, 0, 0);
  return context.getImageData(0, 0, canvas.width, canvas.height);
}

function upperPowerOfTwo(value: number): number {
  let result = 1;
  while (result < value) result *= 2;
  return result;
}

export class MinecraftResources {
  private readonly blocks = new Map<string, { properties: Record<string, string[]>; default: Record<string, string> }>();
  private readonly blockDefinitions: Record<string, BlockDefinition> = {};
  private readonly blockModels: Record<string, BlockModel> = {};
  private readonly textureAtlas: TextureAtlas;

  private constructor(blocks: BlocksData, assets: McmetaAssets, atlas: TextureAtlas) {
    for (const [key, value] of Object.entries(blocks)) {
      this.blocks.set(Identifier.create(key).toString(), {
        properties: value[0],
        default: value[1],
      });
    }

    for (const [key, value] of Object.entries(assets.blockstates)) {
      this.blockDefinitions[Identifier.create(key).toString()] = BlockDefinition.fromJson(value);
    }

    for (const [key, value] of Object.entries(assets.models)) {
      this.blockModels[Identifier.create(key).toString()] = BlockModel.fromJson(value);
    }

    for (const model of Object.values(this.blockModels)) {
      model.flatten(this);
    }

    this.textureAtlas = atlas;
  }

  static async load(version = '1.18.2'): Promise<MinecraftResources> {
    const [blocks, assets, uvMapping, atlasImage] = await Promise.all([
      fetchStringifiedJson<BlocksData>(`/mcmeta/${version}-blocks`, 'stringifiedBlocks'),
      fetchStringifiedJson<Omit<McmetaAssets, 'textures'>>(`/mcmeta/${version}-assets`, 'stringifiedAssets'),
      fetchStringifiedJson<Record<string, [number, number, number, number]>>(
        `/mcmeta/${version}-uvmapping`,
        'stringifiedUvmapping',
      ),
      loadImageData(`/mcmeta/${version}-atlas`),
    ]);

    const idMap: Record<string, [number, number, number, number]> = {};
    for (const [key, [u, v, width, height]] of Object.entries(uvMapping)) {
      const adjustedHeight = width !== height && key.startsWith('block/') ? width : height;
      idMap[Identifier.create(key).toString()] = [
        u / atlasImage.width,
        v / atlasImage.height,
        (u + width) / atlasImage.width,
        (v + adjustedHeight) / atlasImage.height,
      ];
    }

    return new MinecraftResources(
      blocks,
      { ...assets, textures: uvMapping },
      new TextureAtlas(atlasImage, idMap),
    );
  }

  getBlockDefinition(id: Identifier): BlockDefinition | null {
    return this.blockDefinitions[id.toString()] ?? null;
  }

  getBlockModel(id: Identifier): BlockModel | null {
    return this.blockModels[id.toString()] ?? null;
  }

  getTextureAtlas(): ImageData {
    return this.textureAtlas.getTextureAtlas();
  }

  getTextureUV(id: Identifier): [number, number, number, number] {
    return this.textureAtlas.getTextureUV(id);
  }

  getPixelSize(): number {
    return this.textureAtlas.getPixelSize();
  }

  getBlockFlags(id: Identifier): { opaque?: boolean; semi_transparent?: boolean; self_culling?: boolean } {
    const name = id.toString();
    const nonOpaque =
      name.includes('air') ||
      name.includes('torch') ||
      name.includes('redstone_wire') ||
      name.includes('lever') ||
      name.includes('repeater') ||
      name.includes('button') ||
      name.includes('rail') ||
      name.includes('glass') ||
      name.includes('water') ||
      name.includes('lava');

    return {
      opaque: !nonOpaque,
      semi_transparent: name.includes('glass') || name.includes('water') || name.includes('lava'),
      self_culling: true,
    };
  }

  getBlockProperties(id: Identifier): Record<string, string[]> | null {
    return this.blocks.get(id.toString())?.properties ?? null;
  }

  getDefaultBlockProperties(id: Identifier): Record<string, string> | null {
    return this.blocks.get(id.toString())?.default ?? null;
  }
}
