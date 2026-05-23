# Redstone NBT Viewer

A lightweight local web viewer for Minecraft NBT files. It can render
structure-like block data in 3D and browse multiple local NBT files from a
selected folder.

## Setup

Install dependencies from this directory:

```powershell
npm.cmd install
```

Prepare Minecraft block assets:

```powershell
npm.cmd run prepare:mcmeta
```

Build the optional Rust simulator WASM package:

```powershell
npm.cmd run build:wasm
```

Start the development server:

```powershell
npm.cmd run dev
```

Vite prints the local URL, usually `http://127.0.0.1:5173/` or the next free
port.

## Minecraft Asset Cache

The viewer does not download Minecraft block assets automatically. It serves
block models, states, and textures from:

```text
tools/nbt-viewer/public/mcmeta
```

That directory is ignored by git because it is a generated local asset cache.
The included `prepare:mcmeta` script copies the same cached mcmeta artifacts used
by the VS Code NBT viewer extension from:

```text
%LOCALAPPDATA%/vscode-nbt-nodejs/Cache/mcmeta
```

If that source cache is missing, open a Minecraft NBT file once in the VS Code
NBT viewer extension so it can populate its cache, then run
`npm.cmd run prepare:mcmeta` again.

## Usage

- Use `Open NBT` to load a single local `.nbt`, `.dat`, `.schem`,
  `.schematic`, `.litematic`, or `.mcstructure` file.
- Use `Open Folder` to browse supported NBT-like files from a local directory.
- Use the mouse to rotate the 3D view, the wheel to zoom, and `W/A/S/D`,
  `Space`, and `Shift` to move the camera.

All files are opened locally in the browser. The app does not upload selected
NBT files to a server.

## Build

```powershell
npm.cmd run build:wasm
npm.cmd run build
```

## GitHub Pages

The viewer can be deployed as a static GitHub Pages artifact. The deployment
workflow builds the Rust simulator WASM first, then runs the Vite production
build so these generated files are included under `dist/wasm/nbt-sim`.

For repository Pages, build with the repository subpath as the Vite base:

```powershell
$env:VITE_BASE_PATH = "/redstone-compiler/"
npm.cmd run build:wasm
npm.cmd run build
```

The GitHub Actions workflow uses the same base path and publishes
`tools/nbt-viewer/dist`.
