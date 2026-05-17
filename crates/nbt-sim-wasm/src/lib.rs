use redstone_compiler::nbt::{NBTRoot, ToNBT};
use redstone_compiler::world::block::BlockKind;
use redstone_compiler::world::position::Position;
use redstone_compiler::world::simulator::Simulator;
use serde::Serialize;
use wasm_bindgen::prelude::*;

#[derive(Serialize)]
struct SwitchInfo {
    pos: [usize; 3],
    is_on: bool,
}

#[wasm_bindgen]
pub struct NbtSimulator {
    sim: Simulator,
}

#[wasm_bindgen]
impl NbtSimulator {
    #[wasm_bindgen(constructor)]
    pub fn new(gzip_nbt: &[u8]) -> Result<NbtSimulator, JsValue> {
        let nbt = NBTRoot::from_gzip_bytes(gzip_nbt).map_err(to_js_error)?;
        let world = nbt.to_world();
        let sim = Simulator::from(&world).map_err(to_js_error)?;

        Ok(Self { sim })
    }

    pub fn switches(&self) -> Result<JsValue, JsValue> {
        let switches = self
            .sim
            .world()
            .iter_block()
            .into_iter()
            .filter_map(|(pos, block)| {
                let BlockKind::Switch { is_on } = block.kind else {
                    return None;
                };

                Some(SwitchInfo {
                    pos: [pos.0, pos.1, pos.2],
                    is_on,
                })
            })
            .collect::<Vec<_>>();

        serde_wasm_bindgen::to_value(&switches).map_err(to_js_error)
    }

    pub fn toggle_switch(
        &mut self,
        x: usize,
        y: usize,
        z: usize,
        is_on: bool,
    ) -> Result<JsValue, JsValue> {
        self.sim
            .change_state(vec![(Position(x, y, z), is_on)])
            .map_err(to_js_error)?;
        self.structure()
    }

    pub fn structure(&self) -> Result<JsValue, JsValue> {
        let nbt = self.sim.world().to_nbt();
        serde_wasm_bindgen::to_value(&nbt).map_err(to_js_error)
    }

    pub fn gzip_nbt(&self) -> Result<Vec<u8>, JsValue> {
        self.sim
            .world()
            .to_nbt()
            .to_gzip_bytes()
            .map_err(to_js_error)
    }
}

fn to_js_error(error: impl std::fmt::Display) -> JsValue {
    JsValue::from_str(&error.to_string())
}
