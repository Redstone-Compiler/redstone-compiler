# Redstone Compiler Project

Toolkit for Redstone Transfer Level Desgin

## How to use

Currently, you can use only unittests.

## Compiler Stack

```
HDL -> Synthesis -> Cluster -> Logic Graph -> Place and Route -> Synthesis -> World -> NBT
```

- Synthesis: Converts HDL to a compilable netlist graph. Inserts buffers to match the clock timing of each module, and generates and connects gates in a form that is easy to compile into a Minecraft redstone circuit.
- Cluster: Slice and cluster large modules to an appropriate size for compiling, taking into account the locality and size of each piece.
- Place And Route: Compile the clustered graph to world3d, from module to redstone, in a top-down approach. See [Place And Route](docs/place_and_route.md)
- World, World3d: `World` and `World3D` are collections of blocks and positions, designed to correspond exactly to the minecraft world. See [world/mod.rs](https://github.com/Redstone-Compiler/redstone-compiler/blob/master/src/world/mod.rs) if you want more details.
- NBT: A blueprint format that can be imported into Minecraft. You can import nbt using [MCEdit](https://www.mcedit.net/), [Litematica](https://www.curseforge.com/minecraft/mc-mods/litematica) or etc.
